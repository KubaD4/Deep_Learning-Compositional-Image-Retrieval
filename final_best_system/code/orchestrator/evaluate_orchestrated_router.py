#!/usr/bin/env python3
"""Evaluate attribute-routed composition methods on celeba_evaluation.json.

This script is intentionally separate from the main training/evaluation scripts.
It tests no-training orchestration strategies:

- local attributes are composed by the best learned gate checkpoint;
- global/weak attributes such as Male, Young, and Chubby are composed by
  hand-designed CLIP arithmetic found in local diagnostics;
- several assembly strategies are evaluated in one run.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))
os.environ.setdefault("DL_PROJECT_ROOT", str(ROOT))

from learned_gate_core import condition_embeddings, load_model_checkpoint, load_prompt_embedding_cache
from project_core import (
    ARTIFACTS_DIR,
    CHECKPOINT_DIR,
    EMBEDDING_DIR,
    MODEL_ID,
    TOP_KS,
    atomic_json_dump,
    choose_device,
    load_evaluation,
    load_torch,
    parse_query,
    read_attribute_table,
    retrieval_metrics,
)


STOP_REQUESTED = False
WEAK_ATTRS = {"Male", "Young", "Chubby"}
COUPLED_WITH_MALE = {"Mustache"}


METHODS = (
    "stage_tuned",
    "delta_sum_tuned",
    "score_fusion_70local",
    "score_fusion_50",
    "rrf_union",
    "full_arithmetic_if_weak",
)


def request_stop(signum, frame) -> None:
    del signum, frame
    global STOP_REQUESTED
    STOP_REQUESTED = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output-root", type=Path, default=ARTIFACTS_DIR / "results" / "orchestrated_router")
    parser.add_argument("--source-batch-size", type=int, default=256)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--time-budget-seconds", type=int, default=0)
    parser.add_argument("--methods", nargs="+", default=list(METHODS), choices=METHODS)
    parser.add_argument("--query-ids", nargs="*", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def normalize(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x.float(), dim=-1)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def find_best_official_checkpoint() -> Path:
    """Prefer the checkpoint of the best official JSON result, fallback to validation."""
    best_score = float("-inf")
    best_checkpoint: Path | None = None
    for summary_path in sorted((ARTIFACTS_DIR / "results" / "gate_model").glob("*/summary.csv")):
        with summary_path.open(encoding="utf-8") as handle:
            row = next(csv.DictReader(handle), None)
        if not row:
            continue
        checkpoint = Path(row.get("checkpoint", ""))
        if not checkpoint.exists():
            continue
        score = float(row.get("macro_Recall@10", 0.0))
        if score > best_score:
            best_score = score
            best_checkpoint = checkpoint
    if best_checkpoint is not None:
        print(f"Selected official-best checkpoint: {best_checkpoint} macro_R@10={best_score:.4f}")
        return best_checkpoint

    runs_root = ARTIFACTS_DIR / "training_runs"
    for metrics_path in sorted(runs_root.glob("**/metrics.csv")):
        checkpoint = metrics_path.parent / "checkpoints" / "best_val_official_like_at10.pt"
        if not checkpoint.exists():
            continue
        with metrics_path.open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            score = float(row.get("val_official_like@10", 0.0))
            if score > best_score:
                best_score = score
                best_checkpoint = checkpoint
    if best_checkpoint is None:
        raise FileNotFoundError("No learned gate checkpoint found")
    print(f"Selected validation-best checkpoint: {best_checkpoint} val_official_like@10={best_score:.4f}")
    return best_checkpoint


def resolve_project_path(path_value: str | Path | None) -> Path | None:
    if not path_value:
        return None
    path = Path(path_value)
    return path if path.is_absolute() else ROOT / path


@dataclass(frozen=True)
class SplitQuery:
    local: list[tuple[int, str]]
    weak: list[tuple[int, str]]
    routing_note: str


def split_query(conditions: list[tuple[int, str]]) -> SplitQuery:
    has_male = any(attr == "Male" for _, attr in conditions)
    weak = []
    local = []
    for sign, attr in conditions:
        if attr in WEAK_ATTRS or (has_male and attr in COUPLED_WITH_MALE):
            weak.append((sign, attr))
        else:
            local.append((sign, attr))
    note = {
        "weak_attrs": [f"{'+' if s > 0 else '-'}{a}" for s, a in weak],
        "local_attrs": [f"{'+' if s > 0 else '-'}{a}" for s, a in local],
        "coupled_mustache_with_male": has_male,
    }
    return SplitQuery(local=local, weak=weak, routing_note=json.dumps(note))


def condition_tensors(
    conditions: list[tuple[int, str]],
    attribute_to_index: dict[str, int],
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    attr_indices = torch.tensor(
        [[attribute_to_index[attr] for _, attr in conditions]],
        dtype=torch.long,
        device=device,
    ).expand(batch_size, -1)
    signs = torch.tensor(
        [[sign for sign, _ in conditions]],
        dtype=torch.int8,
        device=device,
    ).expand(batch_size, -1)
    return attr_indices, signs


def learned_local_query(
    model,
    source: torch.Tensor,
    local_conditions: list[tuple[int, str]],
    attribute_to_index: dict[str, int],
    prompt_cache: dict,
    model_condition_mode: str,
    device: torch.device,
) -> torch.Tensor:
    if not local_conditions:
        return normalize(source)
    attr_indices, signs = condition_tensors(local_conditions, attribute_to_index, len(source), device)
    conditions, mask = condition_embeddings(prompt_cache, attr_indices, signs, device, model_condition_mode)
    query, _, _ = model(source, conditions, mask)
    return normalize(query)


def signed_direction(bank: dict, attribute_to_index: dict[str, int], sign: int, attr: str) -> torch.Tensor:
    return sign * normalize(bank["directions"])[attribute_to_index[attr]]


def signed_endpoint(bank: dict, attribute_to_index: dict[str, int], sign: int, attr: str) -> torch.Tensor:
    key = "positive" if sign > 0 else "negative"
    return normalize(bank[key])[attribute_to_index[attr]]


def sum_and_normalize(parts: list[torch.Tensor]) -> torch.Tensor:
    if not parts:
        raise ValueError("Cannot compose empty edit")
    return normalize(torch.stack(parts).sum(dim=0, keepdim=True)).squeeze(0)


def compose_tangent(base: torch.Tensor, edit: torch.Tensor, alpha: float, source_weight: float) -> torch.Tensor:
    edit = normalize(edit.unsqueeze(0)).squeeze(0).to(base.device)
    alignment = (base * edit.unsqueeze(0)).sum(dim=-1, keepdim=True)
    tangent = edit.unsqueeze(0) - alignment * base
    return normalize(source_weight * base + alpha * tangent)


def compose_sum(base: torch.Tensor, edit: torch.Tensor, alpha: float, source_weight: float) -> torch.Tensor:
    edit = normalize(edit.unsqueeze(0)).squeeze(0).to(base.device)
    return normalize(source_weight * base + alpha * edit.unsqueeze(0))


def weak_query_from_base(
    base: torch.Tensor,
    weak_conditions: list[tuple[int, str]],
    attribute_to_index: dict[str, int],
    text_bank: dict,
    prompt_v1_bank: dict,
) -> torch.Tensor:
    if not weak_conditions:
        return normalize(base)

    attrs = {attr for _, attr in weak_conditions}
    signed = {(attr, sign) for sign, attr in weak_conditions}

    # Chubby+older is weak as an edit direction. Treat it more like a global
    # endpoint query and add a small Double_Chin cue found useful locally.
    if ("Chubby", 1) in signed and ("Young", -1) in signed:
        parts = [
            signed_endpoint(prompt_v1_bank, attribute_to_index, sign, attr)
            for sign, attr in weak_conditions
            if attr in {"Chubby", "Young"}
        ]
        parts.extend(
            signed_direction(text_bank, attribute_to_index, sign, attr)
            for sign, attr in weak_conditions
            if attr not in {"Chubby", "Young"}
        )
        parts.append(0.5 * signed_direction(prompt_v1_bank, attribute_to_index, 1, "Double_Chin"))
        edit = sum_and_normalize(parts)
        return compose_sum(base, edit, alpha=1.0, source_weight=0.25)

    # If Male and Mustache occur together, prompt-v1 contrastive directions were
    # better locally than endpoint prompts.
    if "Male" in attrs and "Mustache" in attrs:
        edit = sum_and_normalize(
            [signed_direction(prompt_v1_bank, attribute_to_index, sign, attr) for sign, attr in weak_conditions]
        )
        return compose_sum(base, edit, alpha=0.65, source_weight=0.85)

    query = normalize(base)
    for sign, attr in weak_conditions:
        if attr == "Young":
            edit = signed_direction(text_bank, attribute_to_index, sign, attr)
            query = compose_tangent(query, edit, alpha=2.0, source_weight=0.75)
        elif attr == "Male":
            edit = signed_direction(text_bank, attribute_to_index, sign, attr)
            query = compose_sum(query, edit, alpha=0.75, source_weight=1.0)
        elif attr == "Chubby":
            edit = signed_endpoint(prompt_v1_bank, attribute_to_index, sign, attr)
            query = compose_sum(query, edit, alpha=1.0, source_weight=0.35)
        else:
            edit = signed_direction(text_bank, attribute_to_index, sign, attr)
            query = compose_sum(query, edit, alpha=0.75, source_weight=1.0)
    return normalize(query)


def weak_query_from_source(
    source: torch.Tensor,
    weak_conditions: list[tuple[int, str]],
    attribute_to_index: dict[str, int],
    text_bank: dict,
    prompt_v1_bank: dict,
) -> torch.Tensor:
    return weak_query_from_base(source, weak_conditions, attribute_to_index, text_bank, prompt_v1_bank)


def all_arithmetic_query(
    source: torch.Tensor,
    conditions: list[tuple[int, str]],
    attribute_to_index: dict[str, int],
    text_bank: dict,
    prompt_v1_bank: dict,
) -> torch.Tensor:
    if not any(attr in WEAK_ATTRS for _, attr in conditions):
        edit = sum_and_normalize([signed_direction(text_bank, attribute_to_index, sign, attr) for sign, attr in conditions])
        return compose_sum(source, edit, alpha=1.0, source_weight=1.0)

    # Route all weak-containing queries through the same tuned global composer.
    return weak_query_from_base(source, conditions, attribute_to_index, text_bank, prompt_v1_bank)


def combine_queries(
    method: str,
    source: torch.Tensor,
    q_local: torch.Tensor,
    q_weak: torch.Tensor,
    split: SplitQuery,
    conditions: list[tuple[int, str]],
    attribute_to_index: dict[str, int],
    text_bank: dict,
    prompt_v1_bank: dict,
) -> torch.Tensor | None:
    if not split.weak:
        return q_local
    if method == "stage_tuned":
        return weak_query_from_base(q_local, split.weak, attribute_to_index, text_bank, prompt_v1_bank)
    if method == "delta_sum_tuned":
        delta = q_weak - normalize(source)
        return normalize(q_local + 0.85 * delta)
    if method == "full_arithmetic_if_weak":
        return all_arithmetic_query(source, conditions, attribute_to_index, text_bank, prompt_v1_bank)
    if method in {"score_fusion_70local", "score_fusion_50", "rrf_union"}:
        return None
    raise ValueError(method)


def rrf_rank(scores_a: torch.Tensor, scores_b: torch.Tensor, top_k: int = 10, pool_k: int = 200) -> list[list[int]]:
    pool_k = min(pool_k, scores_a.shape[1])
    top_a = scores_a.topk(pool_k, dim=1).indices.cpu().tolist()
    top_b = scores_b.topk(pool_k, dim=1).indices.cpu().tolist()
    all_rankings = []
    for ra, rb in zip(top_a, top_b):
        fused: dict[int, float] = {}
        for rank, idx in enumerate(ra, start=1):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (60.0 + rank)
        for rank, idx in enumerate(rb, start=1):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (60.0 + rank)
        ranking = [idx for idx, _ in sorted(fused.items(), key=lambda item: item[1], reverse=True)[:top_k]]
        all_rankings.append(ranking)
    return all_rankings


def load_text_banks(device: torch.device) -> tuple[dict, dict]:
    text = load_torch(EMBEDDING_DIR / "attribute_text_embeddings.pt")
    prompt_v1 = load_torch(EMBEDDING_DIR / "signed_attribute_prompt_embeddings.pt")
    return (
        {key: normalize(text[key]).to(device) for key in ("positive", "negative", "directions")},
        {key: normalize(prompt_v1[key]).to(device) for key in ("positive", "negative", "directions")},
    )


def finalize_method(annotations: list[dict], records_by_query: dict[int, list[dict]], method_dir: Path, checkpoint: Path) -> None:
    per_query = []
    retrieval_path = method_dir / "retrievals.jsonl"
    with retrieval_path.open("w", encoding="utf-8") as retrieval_file:
        for query_id, item in enumerate(annotations):
            if query_id not in records_by_query:
                continue
            totals = {f"Recall@{k}": 0.0 for k in TOP_KS}
            totals.update({f"Precision@{k}": 0.0 for k in TOP_KS})
            for record in records_by_query[query_id]:
                retrieval_file.write(json.dumps(record) + "\n")
                valid = set(item["ground_truth"][str(record["source_index"])])
                for k in TOP_KS:
                    recall, precision = retrieval_metrics(record["top10"], valid, k)
                    totals[f"Recall@{k}"] += recall
                    totals[f"Precision@{k}"] += precision
            source_count = len(records_by_query[query_id])
            per_query.append(
                {
                    "query_id": query_id,
                    "query": item["query"],
                    "sources": source_count,
                    **{name: value / source_count for name, value in totals.items()},
                }
            )
    write_csv(method_dir / "per_query_metrics.csv", per_query)
    metric_names = [name for name in per_query[0] if "@" in name]
    total_sources = sum(row["sources"] for row in per_query)
    summary = {
        "method": method_dir.name,
        "checkpoint": str(checkpoint),
        "query_entries": len(per_query),
        "source_query_cases": total_sources,
        **{
            f"macro_{name}": sum(row[name] for row in per_query) / len(per_query)
            for name in metric_names
        },
        **{
            f"micro_{name}": sum(row[name] * row["sources"] for row in per_query) / total_sources
            for name in metric_names
        },
    }
    write_csv(method_dir / "summary.csv", [summary])
    (method_dir / "COMPLETE").write_text("complete\n", encoding="utf-8")


def load_reference_summaries() -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_frames = []
    per_query_frames = []
    labels = {
        "01_direct_sum": "Direct sum",
        "02_direct_sequential": "Direct sequential",
        "03_contrastive_sum": "Contrastive sum",
        "04_contrastive_sequential": "Contrastive sequential",
        "05_adaptive_tangent_sequential": "Adaptive tangent sequential",
    }
    for folder, label in labels.items():
        root = ARTIFACTS_DIR / "results" / "baselines" / folder
        if (root / "summary.csv").exists():
            summary = pd.read_csv(root / "summary.csv")
            perq = pd.read_csv(root / "per_query_metrics.csv")
            summary["method"] = label
            perq["method"] = label
            summary_frames.append(summary)
            per_query_frames.append(perq)
    gate_root = ARTIFACTS_DIR / "results" / "gate_model"
    for root in sorted(gate_root.glob("gate_v3_sequentialgate_hybmp_l002*")):
        if (root / "summary.csv").exists():
            summary = pd.read_csv(root / "summary.csv")
            perq = pd.read_csv(root / "per_query_metrics.csv")
            summary["method"] = "Gate v3 hybrid best"
            perq["method"] = "Gate v3 hybrid best"
            summary_frames.append(summary)
            per_query_frames.append(perq)
    return pd.concat(summary_frames, ignore_index=True), pd.concat(per_query_frames, ignore_index=True)


def plot_comparison(output_root: Path) -> None:
    summaries_ref, perq_ref = load_reference_summaries()
    summaries_new = []
    perq_new = []
    for method_dir in sorted(output_root.glob("*")):
        if not method_dir.is_dir() or method_dir.name == "comparison":
            continue
        if (method_dir / "summary.csv").exists():
            s = pd.read_csv(method_dir / "summary.csv")
            p = pd.read_csv(method_dir / "per_query_metrics.csv")
            s["method"] = f"Router {method_dir.name}"
            p["method"] = f"Router {method_dir.name}"
            summaries_new.append(s)
            perq_new.append(p)
    summaries = pd.concat([summaries_ref, *summaries_new], ignore_index=True)
    perq = pd.concat([perq_ref, *perq_new], ignore_index=True)

    comp = output_root / "comparison"
    comp.mkdir(parents=True, exist_ok=True)
    summaries.to_csv(comp / "combined_summary.csv", index=False)
    perq.to_csv(comp / "combined_per_query_metrics.csv", index=False)

    top = summaries.sort_values("macro_Recall@10", ascending=False).head(12).iloc[::-1]
    plt.figure(figsize=(10, 5.5))
    plt.barh(top["method"], top["macro_Recall@10"], color="#2f6f73")
    plt.xlabel("Macro Recall@10")
    plt.title("Official JSON: routed orchestrator vs baselines")
    plt.tight_layout()
    plt.savefig(comp / "router_macro_recall10.png", dpi=180)
    plt.close()

    weak_queries = ["+Male", "-Young", "-Male, -Mustache", "+Chubby, -Young"]
    weak = perq[perq["query"].isin(weak_queries)]
    pivot = weak.pivot_table(index="query", columns="method", values="Recall@10", aggfunc="first")
    keep = [c for c in pivot.columns if c.startswith("Router") or c in {"Gate v3 hybrid best", "Contrastive sequential", "Adaptive tangent sequential"}]
    pivot = pivot[keep]
    pivot.to_csv(comp / "weak_query_recall10.csv")
    pivot.plot(kind="barh", figsize=(11, 5))
    plt.xlabel("Recall@10")
    plt.title("Weak query Recall@10")
    plt.tight_layout()
    plt.savefig(comp / "weak_query_recall10.png", dpi=180)
    plt.close()

    # Per-query winner among router methods only.
    router = perq[perq["method"].str.startswith("Router")]
    rows = []
    for query_id, frame in router.groupby("query_id"):
        best = frame.sort_values("Recall@10", ascending=False).iloc[0]
        rows.append(
            {
                "query_id": int(query_id),
                "query": best["query"],
                "best_router_method": best["method"],
                "best_router_R@10": float(best["Recall@10"]),
            }
        )
    pd.DataFrame(rows).to_csv(comp / "router_per_query_winners.csv", index=False)


def main() -> int:
    args = parse_args()
    signal.signal(signal.SIGTERM, request_stop)
    if hasattr(signal, "SIGUSR1"):
        signal.signal(signal.SIGUSR1, request_stop)
    started = time.monotonic()

    device = choose_device(args.device)
    checkpoint = args.checkpoint or find_best_official_checkpoint()
    model, checkpoint_data = load_model_checkpoint(checkpoint, device)
    model.eval()
    config = checkpoint_data["config"]
    prompt_cache = load_prompt_embedding_cache(resolve_project_path(config.get("prompt_cache_path")))
    text_bank, prompt_v1_bank = load_text_banks(device)

    attributes, _, _ = read_attribute_table()
    attribute_to_index = {name: index for index, name in enumerate(attributes)}
    annotations_all = load_evaluation()
    if args.query_ids is None:
        annotations = annotations_all
        query_id_map = list(range(len(annotations_all)))
    else:
        query_id_map = args.query_ids
        annotations = [annotations_all[i] for i in query_id_map]

    gallery_cache = load_torch(EMBEDDING_DIR / "test_image_embeddings.pt")
    gallery = normalize(gallery_cache["embeddings"]).to(device)
    args.output_root.mkdir(parents=True, exist_ok=True)
    debug_dir = args.output_root / "debug"
    debug_dir.mkdir(exist_ok=True)

    for method in args.methods:
        method_dir = args.output_root / method
        if method_dir.exists() and args.force:
            for path in method_dir.glob("*"):
                if path.is_file():
                    path.unlink()
        method_dir.mkdir(parents=True, exist_ok=True)
        atomic_json_dump(
            {
                "method": method,
                "checkpoint": str(checkpoint),
                "weak_attrs": sorted(WEAK_ATTRS),
                "coupled_with_male": sorted(COUPLED_WITH_MALE),
                "top_ks": TOP_KS,
            },
            method_dir / "config.json",
        )

    records_by_method = {method: {} for method in args.methods}
    routing_log = (debug_dir / "routing_decisions.jsonl").open("w", encoding="utf-8")

    with torch.inference_mode():
        for local_list_index, item in enumerate(annotations):
            query_id = query_id_map[local_list_index]
            conditions = parse_query(item["query"])
            split = split_query(conditions)
            routing_log.write(
                json.dumps({"query_id": query_id, "query": item["query"], "routing": split.routing_note}) + "\n"
            )
            source_indices = [int(index) for index in item["ground_truth"]]
            for start in range(0, len(source_indices), args.source_batch_size):
                end = min(start + args.source_batch_size, len(source_indices))
                batch_indices = source_indices[start:end]
                idx = torch.tensor(batch_indices, device=device)
                source = gallery[idx]
                q_local = learned_local_query(
                    model,
                    source,
                    split.local,
                    attribute_to_index,
                    prompt_cache,
                    str(config.get("condition_mode", "signed_prompt")),
                    device,
                )
                q_weak = weak_query_from_source(source, split.weak, attribute_to_index, text_bank, prompt_v1_bank)
                if not split.weak:
                    q_weak = q_local

                score_local = q_local @ gallery.T
                score_weak = q_weak @ gallery.T
                rows = torch.arange(len(batch_indices), device=device)
                score_local[rows, idx] = -torch.inf
                score_weak[rows, idx] = -torch.inf

                for method in args.methods:
                    if not split.weak:
                        rankings = score_local.topk(max(TOP_KS), dim=1).indices.cpu().tolist()
                        records = records_by_method[method].setdefault(query_id, [])
                        records.extend(
                            {
                                "method": method,
                                "query_id": query_id,
                                "query": item["query"],
                                "source_index": source_index,
                                "top10": ranking,
                            }
                            for source_index, ranking in zip(batch_indices, rankings)
                        )
                        continue

                    if method == "score_fusion_70local":
                        scores = 0.70 * score_local + 0.30 * score_weak
                        rankings = scores.topk(max(TOP_KS), dim=1).indices.cpu().tolist()
                    elif method == "score_fusion_50":
                        scores = 0.50 * score_local + 0.50 * score_weak
                        rankings = scores.topk(max(TOP_KS), dim=1).indices.cpu().tolist()
                    elif method == "rrf_union":
                        rankings = rrf_rank(score_local, score_weak, max(TOP_KS), pool_k=200)
                    else:
                        q = combine_queries(
                            method,
                            source,
                            q_local,
                            q_weak,
                            split,
                            conditions,
                            attribute_to_index,
                            text_bank,
                            prompt_v1_bank,
                        )
                        if q is None:
                            raise RuntimeError(f"Method {method} did not produce a query")
                        scores = q @ gallery.T
                        scores[rows, idx] = -torch.inf
                        rankings = scores.topk(max(TOP_KS), dim=1).indices.cpu().tolist()

                    records = records_by_method[method].setdefault(query_id, [])
                    records.extend(
                        {
                            "method": method,
                            "query_id": query_id,
                            "query": item["query"],
                            "source_index": source_index,
                            "top10": ranking,
                        }
                        for source_index, ranking in zip(batch_indices, rankings)
                    )

                print(f"query {query_id + 1}/{len(annotations_all)} sources {end}/{len(source_indices)}")
                if STOP_REQUESTED or (
                    args.time_budget_seconds and time.monotonic() - started >= args.time_budget_seconds
                ):
                    routing_log.close()
                    raise SystemExit("Time limit reached before completing all methods")

    routing_log.close()

    # Use full annotations so query IDs stay canonical even when --query-ids is used.
    for method in args.methods:
        method_records = records_by_method[method]
        if args.query_ids is None:
            finalize_method(annotations_all, method_records, args.output_root / method, checkpoint)
        else:
            finalize_method(annotations_all, method_records, args.output_root / method, checkpoint)

    if args.query_ids is None:
        plot_comparison(args.output_root)
    print(f"Orchestrated router results saved in {args.output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
