#!/usr/bin/env python3
"""Compare learned model, arithmetic-only, and model+arithmetic blends.

This evaluator is a follow-up to the routed orchestrator. It asks a simpler
question: does arithmetic work better by itself, or mainly as a correction to
the learned gate?

The main families are intentionally global/fixed:

- learned model only;
- generic CLIP arithmetic only;
- tuned arithmetic only, using the weak-attribute rules from the local diagnosis;
- vector-delta blends: q = normalize(q_model + beta * (q_sum - z_source));
- score-level blends: s = (1-w) * s_model + w * s_sum;
- reciprocal-rank fusion.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import sys
import time
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
sys.path.insert(0, str(ROOT / "orchestrator"))
os.environ.setdefault("DL_PROJECT_ROOT", str(ROOT))

import evaluate_orchestrated_router as router
from learned_gate_core import condition_embeddings, load_model_checkpoint, load_prompt_embedding_cache
from project_core import (
    ARTIFACTS_DIR,
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


METHODS = (
    "model_only",
    "generic_sum_only",
    "generic_tangent_seq",
    "tuned_sum_only",
    "model_plus_generic_delta_025",
    "model_plus_generic_delta_050",
    "model_plus_generic_delta_100",
    "model_plus_tuned_delta_025",
    "model_plus_tuned_delta_050",
    "model_plus_tuned_delta_100",
    "score_fusion_generic_25sum",
    "score_fusion_generic_50sum",
    "score_fusion_generic_75sum",
    "score_fusion_tuned_25sum",
    "score_fusion_tuned_50sum",
    "score_fusion_tuned_75sum",
    "rrf_model_generic",
    "rrf_model_tuned",
)


def request_stop(signum, frame) -> None:
    del signum, frame
    global STOP_REQUESTED
    STOP_REQUESTED = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output-root", type=Path, default=ARTIFACTS_DIR / "results" / "sum_model_blends")
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


def condition_tensors(conditions, attribute_to_index, batch_size, device):
    attrs = torch.tensor(
        [[attribute_to_index[attr] for _, attr in conditions]],
        dtype=torch.long,
        device=device,
    ).expand(batch_size, -1)
    signs = torch.tensor([[sign for sign, _ in conditions]], dtype=torch.int8, device=device).expand(batch_size, -1)
    return attrs, signs


def model_query(model, source, conditions, attribute_to_index, prompt_cache, condition_mode, device):
    attrs, signs = condition_tensors(conditions, attribute_to_index, len(source), device)
    cond, mask = condition_embeddings(prompt_cache, attrs, signs, device, condition_mode)
    query, _, _ = model(source, cond, mask)
    return normalize(query)


def signed_text_direction(text_bank, attribute_to_index, sign, attr):
    return sign * normalize(text_bank["directions"])[attribute_to_index[attr]]


def generic_sum_query(source, conditions, attribute_to_index, text_bank, alpha=1.0, source_weight=1.0):
    edit = torch.stack(
        [signed_text_direction(text_bank, attribute_to_index, sign, attr) for sign, attr in conditions]
    ).sum(dim=0)
    return router.compose_sum(source, edit, alpha=alpha, source_weight=source_weight)


def generic_tangent_seq_query(source, conditions, attribute_to_index, text_bank):
    query = normalize(source)
    for sign, attr in conditions:
        direction = signed_text_direction(text_bank, attribute_to_index, sign, attr)
        # This is the best global-ish setting found for Young and still
        # reasonable for other attributes: stronger tangent edit, some source.
        query = router.compose_tangent(query, direction, alpha=1.5, source_weight=0.75)
    return normalize(query)


def tuned_sum_query(source, conditions, attribute_to_index, text_bank, prompt_v1_bank):
    return router.all_arithmetic_query(source, conditions, attribute_to_index, text_bank, prompt_v1_bank)


def vector_delta(model_q, sum_q, source, beta):
    return normalize(model_q + beta * (sum_q - normalize(source)))


def finalize_method(annotations, records_by_query, method_dir: Path, checkpoint: Path) -> None:
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


def load_references():
    summaries = []
    per_queries = []
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
            per_query = pd.read_csv(root / "per_query_metrics.csv")
            summary["method"] = label
            per_query["method"] = label
            summaries.append(summary)
            per_queries.append(per_query)
    for root in sorted((ARTIFACTS_DIR / "results" / "gate_model").glob("gate_v3_sequentialgate_hybmp_l002*")):
        if (root / "summary.csv").exists():
            summary = pd.read_csv(root / "summary.csv")
            per_query = pd.read_csv(root / "per_query_metrics.csv")
            summary["method"] = "Gate v3 hybrid best"
            per_query["method"] = "Gate v3 hybrid best"
            summaries.append(summary)
            per_queries.append(per_query)
    return pd.concat(summaries, ignore_index=True), pd.concat(per_queries, ignore_index=True)


def plot_outputs(output_root: Path) -> None:
    ref_summary, ref_perq = load_references()
    summaries = [ref_summary]
    per_queries = [ref_perq]
    for method_dir in sorted(output_root.glob("*")):
        if not method_dir.is_dir() or method_dir.name == "comparison":
            continue
        if (method_dir / "summary.csv").exists():
            summary = pd.read_csv(method_dir / "summary.csv")
            per_query = pd.read_csv(method_dir / "per_query_metrics.csv")
            summary["method"] = f"Blend {method_dir.name}"
            per_query["method"] = f"Blend {method_dir.name}"
            summaries.append(summary)
            per_queries.append(per_query)

    summary = pd.concat(summaries, ignore_index=True)
    perq = pd.concat(per_queries, ignore_index=True)
    comp = output_root / "comparison"
    comp.mkdir(parents=True, exist_ok=True)
    summary.to_csv(comp / "combined_summary.csv", index=False)
    perq.to_csv(comp / "combined_per_query_metrics.csv", index=False)

    top = summary.sort_values("macro_Recall@10", ascending=False).head(16).iloc[::-1]
    plt.figure(figsize=(10, 7))
    plt.barh(top["method"], top["macro_Recall@10"], color="#2f6f73")
    plt.xlabel("Macro Recall@10")
    plt.title("Model vs arithmetic-only vs model+sum blends")
    plt.tight_layout()
    plt.savefig(comp / "macro_recall10_top_methods.png", dpi=180)
    plt.close()

    weak_queries = ["+Male", "-Young", "-Male, -Mustache", "+Chubby, -Young"]
    weak = perq[perq["query"].isin(weak_queries)]
    focus_methods = [
        "Gate v3 hybrid best",
        "Blend model_only",
        "Blend tuned_sum_only",
        "Blend generic_sum_only",
        "Blend model_plus_tuned_delta_050",
        "Blend score_fusion_tuned_50sum",
        "Blend rrf_model_tuned",
    ]
    pivot = weak.pivot_table(index="query", columns="method", values="Recall@10", aggfunc="first")
    keep = [method for method in focus_methods if method in pivot.columns]
    pivot = pivot[keep]
    pivot.to_csv(comp / "weak_query_recall10.csv")
    pivot.plot(kind="barh", figsize=(12, 5.5))
    plt.xlabel("Recall@10")
    plt.title("Weak query performance")
    plt.tight_layout()
    plt.savefig(comp / "weak_query_recall10.png", dpi=180)
    plt.close()

    rows = []
    blend_only = perq[perq["method"].str.startswith("Blend")]
    for query_id, frame in blend_only.groupby("query_id"):
        best = frame.sort_values("Recall@10", ascending=False).iloc[0]
        rows.append(
            {
                "query_id": int(query_id),
                "query": best["query"],
                "best_blend_method": best["method"],
                "best_blend_R@10": float(best["Recall@10"]),
            }
        )
    pd.DataFrame(rows).to_csv(comp / "blend_per_query_winners.csv", index=False)


def main() -> int:
    args = parse_args()
    signal.signal(signal.SIGTERM, request_stop)
    if hasattr(signal, "SIGUSR1"):
        signal.signal(signal.SIGUSR1, request_stop)
    started = time.monotonic()
    device = choose_device(args.device)
    checkpoint = args.checkpoint or router.find_best_official_checkpoint()
    model, checkpoint_data = load_model_checkpoint(checkpoint, device)
    model.eval()
    config = checkpoint_data["config"]
    prompt_cache = load_prompt_embedding_cache(router.resolve_project_path(config.get("prompt_cache_path")))
    text_bank, prompt_v1_bank = router.load_text_banks(device)
    attributes, _, _ = read_attribute_table()
    attribute_to_index = {name: idx for idx, name in enumerate(attributes)}
    annotations_all = load_evaluation()
    if args.query_ids is None:
        query_ids = list(range(len(annotations_all)))
    else:
        query_ids = args.query_ids

    gallery_cache = load_torch(EMBEDDING_DIR / "test_image_embeddings.pt")
    gallery = normalize(gallery_cache["embeddings"]).to(device)

    if args.force and args.output_root.exists():
        for method_dir in args.output_root.glob("*"):
            if method_dir.is_dir():
                for path in method_dir.glob("*"):
                    if path.is_file():
                        path.unlink()
    args.output_root.mkdir(parents=True, exist_ok=True)

    atomic_json_dump(
        {
            "checkpoint": str(checkpoint),
            "methods": args.methods,
            "model_id": MODEL_ID,
            "notes": (
                "Compares learned model, arithmetic-only, vector delta correction, "
                "score fusion, and reciprocal-rank fusion."
            ),
        },
        args.output_root / "experiment_config.json",
    )

    records = {method: {} for method in args.methods}

    with torch.inference_mode():
        for query_id in query_ids:
            item = annotations_all[query_id]
            conditions = parse_query(item["query"])
            source_indices = [int(index) for index in item["ground_truth"]]
            for start in range(0, len(source_indices), args.source_batch_size):
                end = min(start + args.source_batch_size, len(source_indices))
                batch_indices = source_indices[start:end]
                idx = torch.tensor(batch_indices, device=device)
                source = gallery[idx]

                q_model = model_query(
                    model,
                    source,
                    conditions,
                    attribute_to_index,
                    prompt_cache,
                    str(config.get("condition_mode", "signed_prompt")),
                    device,
                )
                q_generic = generic_sum_query(source, conditions, attribute_to_index, text_bank)
                q_generic_tangent = generic_tangent_seq_query(source, conditions, attribute_to_index, text_bank)
                q_tuned = tuned_sum_query(source, conditions, attribute_to_index, text_bank, prompt_v1_bank)

                score_model = q_model @ gallery.T
                score_generic = q_generic @ gallery.T
                score_tuned = q_tuned @ gallery.T
                rows = torch.arange(len(batch_indices), device=device)
                for scores in (score_model, score_generic, score_tuned):
                    scores[rows, idx] = -torch.inf

                query_vectors = {
                    "model_only": q_model,
                    "generic_sum_only": q_generic,
                    "generic_tangent_seq": q_generic_tangent,
                    "tuned_sum_only": q_tuned,
                    "model_plus_generic_delta_025": vector_delta(q_model, q_generic, source, 0.25),
                    "model_plus_generic_delta_050": vector_delta(q_model, q_generic, source, 0.50),
                    "model_plus_generic_delta_100": vector_delta(q_model, q_generic, source, 1.00),
                    "model_plus_tuned_delta_025": vector_delta(q_model, q_tuned, source, 0.25),
                    "model_plus_tuned_delta_050": vector_delta(q_model, q_tuned, source, 0.50),
                    "model_plus_tuned_delta_100": vector_delta(q_model, q_tuned, source, 1.00),
                }

                for method in args.methods:
                    if method in query_vectors:
                        q = query_vectors[method]
                        scores = q @ gallery.T
                        scores[rows, idx] = -torch.inf
                        rankings = scores.topk(max(TOP_KS), dim=1).indices.cpu().tolist()
                    elif method == "score_fusion_generic_25sum":
                        rankings = (0.75 * score_model + 0.25 * score_generic).topk(max(TOP_KS), dim=1).indices.cpu().tolist()
                    elif method == "score_fusion_generic_50sum":
                        rankings = (0.50 * score_model + 0.50 * score_generic).topk(max(TOP_KS), dim=1).indices.cpu().tolist()
                    elif method == "score_fusion_generic_75sum":
                        rankings = (0.25 * score_model + 0.75 * score_generic).topk(max(TOP_KS), dim=1).indices.cpu().tolist()
                    elif method == "score_fusion_tuned_25sum":
                        rankings = (0.75 * score_model + 0.25 * score_tuned).topk(max(TOP_KS), dim=1).indices.cpu().tolist()
                    elif method == "score_fusion_tuned_50sum":
                        rankings = (0.50 * score_model + 0.50 * score_tuned).topk(max(TOP_KS), dim=1).indices.cpu().tolist()
                    elif method == "score_fusion_tuned_75sum":
                        rankings = (0.25 * score_model + 0.75 * score_tuned).topk(max(TOP_KS), dim=1).indices.cpu().tolist()
                    elif method == "rrf_model_generic":
                        rankings = router.rrf_rank(score_model, score_generic, max(TOP_KS), pool_k=200)
                    elif method == "rrf_model_tuned":
                        rankings = router.rrf_rank(score_model, score_tuned, max(TOP_KS), pool_k=200)
                    else:
                        raise ValueError(method)

                    rows_for_method = records[method].setdefault(query_id, [])
                    rows_for_method.extend(
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
                    raise SystemExit("Time budget reached before completion")

    for method in args.methods:
        method_dir = args.output_root / method
        method_dir.mkdir(parents=True, exist_ok=True)
        atomic_json_dump({"method": method, "checkpoint": str(checkpoint)}, method_dir / "config.json")
        finalize_method(annotations_all, records[method], method_dir, checkpoint)

    if args.query_ids is None:
        plot_outputs(args.output_root)
    print(f"Sum/model blend results saved in {args.output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
