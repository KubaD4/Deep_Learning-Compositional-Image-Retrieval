#!/usr/bin/env python3
"""Probe filter component ablation.

This mirrors the oracle component ablation, but uses the calibrated CelebA
attribute probe instead of ground-truth attributes.

It answers:
  - does predicted query filtering help?
  - does predicted non-query Hamming filtering help?
  - does combining both help?

No training is performed here. The official JSON is used only for evaluation.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = ROOT.parent
EXPERIMENTAL = ROOT / "experimental"
SCRIPTS = ROOT / "scripts"
ORCH = ROOT / "orchestrator"
sys.path.insert(0, str(EXPERIMENTAL))
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(ORCH))
os.environ.setdefault("DL_PROJECT_ROOT", str(PACKAGE_ROOT))

import evaluate_sum_model_blends as blends  # noqa: E402
import probe_reranker_v1 as v1  # noqa: E402
import probe_reranker_v2_calibrated as v2  # noqa: E402
from learned_gate_core import load_model_checkpoint, load_prompt_embedding_cache, progress_line  # noqa: E402
from project_core import (  # noqa: E402
    ARTIFACTS_DIR,
    MODEL_ID,
    TOP_KS,
    choose_device,
    load_evaluation,
    load_torch,
    parse_query,
    read_attribute_table,
    retrieval_metrics,
)


METHODS = {
    "baseline_q_final": "No probe filtering; use q_final top-k directly.",
    "probe_query_only": "Keep candidates satisfying query according to calibrated probe.",
    "probe_hamming_only": "Keep candidates with predicted non-query Hamming<=max_hamming.",
    "probe_query_and_hamming": "Keep candidates satisfying both predicted query and predicted Hamming constraints.",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-results", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--probe-checkpoint", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--source-batch-size", type=int, default=256)
    parser.add_argument("--probe-batch-size", type=int, default=2048)
    parser.add_argument("--top-pool", type=int, default=500)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--max-hamming", type=float, default=2.0)
    parser.add_argument("--threshold-objective", default="accuracy")
    parser.add_argument("--query-ids", nargs="*", type=int, default=None)
    parser.add_argument("--max-sources-per-query", type=int, default=0)
    parser.add_argument("--time-budget-seconds", type=int, default=0)
    parser.add_argument(
        "--fill-to-k",
        action="store_true",
        help="Fill filtered lists with original q_final candidates if fewer than top-k pass.",
    )
    return parser.parse_args()


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def latest_probe_results() -> Path:
    candidates = sorted((ARTIFACTS_DIR / "results" / "probe_reranker_v2_calibrated").glob("probe_reranker_v2_calibrated_*"))
    if not candidates:
        raise FileNotFoundError("No probe_reranker_v2_calibrated results found. Run job 59 first.")
    return candidates[-1]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_experiment_config(probe_results: Path) -> dict[str, Any]:
    path = probe_results / "experiment_config.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_checkpoint_and_beta(args: argparse.Namespace, probe_results: Path, progress: Path) -> tuple[Path, float]:
    config = load_experiment_config(probe_results)
    if args.checkpoint is not None:
        checkpoint = args.checkpoint
    elif config.get("checkpoint"):
        checkpoint = Path(config["checkpoint"])
    else:
        checkpoint, _, _ = v1.scan_best_v7_checkpoint()
        if checkpoint is None:
            raise FileNotFoundError("Could not resolve checkpoint; pass --checkpoint")
    if args.beta is not None:
        beta = float(args.beta)
    elif config.get("beta") is not None:
        beta = float(config["beta"])
    else:
        beta = 1.25
    progress_line(progress, f"CHECKPOINT {checkpoint} beta={beta}")
    return checkpoint, beta


def resolve_probe_checkpoint(args: argparse.Namespace, probe_results: Path, progress: Path) -> Path:
    if args.probe_checkpoint is not None:
        probe_checkpoint = args.probe_checkpoint
    else:
        probe_checkpoint = probe_results / "probe" / "best_probe.pt"
    if not probe_checkpoint.exists():
        raise FileNotFoundError(f"Missing probe checkpoint: {probe_checkpoint}")
    progress_line(progress, f"PROBE_CHECKPOINT {probe_checkpoint}")
    return probe_checkpoint


def make_filtered_lists(
    method: str,
    top_indices: torch.Tensor,
    top_scores: torch.Tensor,
    source_probs: torch.Tensor,
    candidate_probs: torch.Tensor,
    attr_indices: torch.Tensor,
    signs: torch.Tensor,
    thresholds: torch.Tensor,
    top_k: int,
    max_hamming: float,
    fill_to_k: bool,
) -> tuple[list[list[int]], list[float], list[float], list[float]]:
    if method == "baseline_q_final":
        return (
            top_indices[:, :top_k].cpu().tolist(),
            [float(top_k)] * int(top_indices.shape[0]),
            [0.0] * int(top_indices.shape[0]),
            [0.0] * int(top_indices.shape[0]),
        )

    query_ok = v2.calibrated_query_ok(candidate_probs, attr_indices, signs, thresholds)
    query_margin = v2.calibrated_query_margin(candidate_probs, attr_indices, signs, thresholds)
    hard_ham, _, _ = v2.calibrated_hamming(source_probs, candidate_probs, attr_indices, thresholds)
    hamming_ok = hard_ham <= float(max_hamming)

    if method == "probe_query_only":
        keep = query_ok
    elif method == "probe_hamming_only":
        keep = hamming_ok
    elif method == "probe_query_and_hamming":
        keep = query_ok & hamming_ok
    else:
        raise ValueError(method)

    rankings = []
    kept_counts = []
    avg_hamming_kept = []
    avg_query_margin_kept = []
    for row in range(top_indices.shape[0]):
        local = torch.nonzero(keep[row], as_tuple=False).flatten()
        kept_counts.append(float(len(local)))
        if len(local):
            selected = top_indices[row, local[:top_k]].tolist()
            avg_hamming_kept.append(float(hard_ham[row, local].float().mean().cpu()))
            avg_query_margin_kept.append(float(query_margin[row, local].float().mean().cpu()))
        else:
            selected = []
            avg_hamming_kept.append(float("nan"))
            avg_query_margin_kept.append(float("nan"))

        if fill_to_k and len(selected) < top_k:
            selected_set = set(selected)
            for idx in top_indices[row].tolist():
                if idx not in selected_set:
                    selected.append(idx)
                    selected_set.add(idx)
                if len(selected) >= top_k:
                    break
        rankings.append(selected[:top_k])
    return rankings, kept_counts, avg_hamming_kept, avg_query_margin_kept


def summarize(
    annotations: list[dict[str, Any]],
    records_by_query: dict[int, list[dict[str, Any]]],
    method: str,
    checkpoint: Path,
    probe_checkpoint: Path,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    per_query = []
    for query_id, records in sorted(records_by_query.items()):
        item = annotations[query_id]
        totals = {f"Recall@{k}": 0.0 for k in TOP_KS}
        totals.update({f"Precision@{k}": 0.0 for k in TOP_KS})
        kept_sum = 0.0
        hamming_sum = 0.0
        margin_sum = 0.0
        hamming_count = 0
        margin_count = 0
        for record in records:
            valid = set(item["ground_truth"][str(record["source_index"])])
            kept_sum += float(record["kept_in_pool"])
            if record["avg_pred_hamming_kept"] == record["avg_pred_hamming_kept"]:
                hamming_sum += float(record["avg_pred_hamming_kept"])
                hamming_count += 1
            if record["avg_query_margin_kept"] == record["avg_query_margin_kept"]:
                margin_sum += float(record["avg_query_margin_kept"])
                margin_count += 1
            for k in TOP_KS:
                recall, precision = retrieval_metrics(record["top10"], valid, k)
                totals[f"Recall@{k}"] += recall
                totals[f"Precision@{k}"] += precision
        source_count = len(records)
        per_query.append(
            {
                "method": method,
                "query_id": query_id,
                "query": item["query"],
                "sources": source_count,
                "avg_kept_in_pool": kept_sum / source_count,
                "avg_pred_hamming_kept": hamming_sum / max(1, hamming_count),
                "avg_query_margin_kept": margin_sum / max(1, margin_count),
                **{name: value / source_count for name, value in totals.items()},
            }
        )

    metric_names = [name for name in per_query[0] if "@" in name]
    total_sources = sum(row["sources"] for row in per_query)
    summary = {
        "method": method,
        "description": METHODS[method],
        "checkpoint": str(checkpoint),
        "probe_checkpoint": str(probe_checkpoint),
        "model_id": MODEL_ID,
        "top_pool": args.top_pool,
        "top_k": args.top_k,
        "max_hamming": args.max_hamming,
        "threshold_objective": args.threshold_objective,
        "fill_to_k": args.fill_to_k,
        "query_entries": len(per_query),
        "source_query_cases": total_sources,
        "macro_avg_kept_in_pool": sum(row["avg_kept_in_pool"] for row in per_query) / len(per_query),
        "macro_avg_pred_hamming_kept": sum(row["avg_pred_hamming_kept"] for row in per_query) / len(per_query),
        "macro_avg_query_margin_kept": sum(row["avg_query_margin_kept"] for row in per_query) / len(per_query),
        **{f"macro_{name}": sum(row[name] for row in per_query) / len(per_query) for name in metric_names},
        **{f"micro_{name}": sum(row[name] * row["sources"] for row in per_query) / total_sources for name in metric_names},
    }
    return summary, per_query


def write_outputs(
    output_root: Path,
    annotations: list[dict[str, Any]],
    all_records: dict[str, dict[int, list[dict[str, Any]]]],
    checkpoint: Path,
    probe_checkpoint: Path,
    args: argparse.Namespace,
) -> None:
    summaries = []
    all_per_query = []
    for method, records_by_query in all_records.items():
        method_dir = output_root / method
        method_dir.mkdir(parents=True, exist_ok=True)
        summary, per_query = summarize(annotations, records_by_query, method, checkpoint, probe_checkpoint, args)
        summaries.append(summary)
        all_per_query.extend(per_query)
        write_csv(method_dir / "summary.csv", [summary])
        write_csv(method_dir / "per_query_metrics.csv", per_query)
        with (method_dir / "retrievals.jsonl").open("w", encoding="utf-8") as handle:
            for records in records_by_query.values():
                for record in records:
                    handle.write(json.dumps(record) + "\n")
        (method_dir / "COMPLETE").write_text("complete\n", encoding="utf-8")

    summaries.sort(key=lambda row: row["macro_Recall@10"], reverse=True)
    write_csv(output_root / "combined_summary.csv", summaries)
    write_csv(output_root / "combined_per_query_metrics.csv", all_per_query)
    best = summaries[0]
    (output_root / "BEST_PROBE_FILTER_COMPONENT.txt").write_text(
        "\n".join(
            [
                f"method={best['method']}",
                f"macro_Recall@10={best['macro_Recall@10']}",
                f"micro_Recall@10={best['micro_Recall@10']}",
                f"macro_Precision@10={best['macro_Precision@10']}",
                f"avg_kept_in_pool={best['macro_avg_kept_in_pool']}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (output_root / "config.json").write_text(json.dumps(vars(args), indent=2, default=str) + "\n", encoding="utf-8")

    labels = {
        "baseline_q_final": "q_final",
        "probe_query_only": "query only",
        "probe_hamming_only": "hamming only",
        "probe_query_and_hamming": "query+hamming",
    }
    order = ["baseline_q_final", "probe_query_only", "probe_hamming_only", "probe_query_and_hamming"]
    by_method = {row["method"]: row for row in summaries}
    for metric, title, filename in [
        ("macro_Recall@10", "Probe component ablation: Macro Recall@10", "macro_recall10_components.png"),
        ("macro_Precision@10", "Probe component ablation: Macro Precision@10", "macro_precision10_components.png"),
        ("macro_avg_kept_in_pool", "Average candidates kept from top-500", "avg_kept_components.png"),
    ]:
        values = [float(by_method[method][metric]) for method in order]
        plt.figure(figsize=(8, 4.5))
        colors = ["#777777", "#4c78a8", "#f58518", "#54a24b"]
        bars = plt.bar([labels[m] for m in order], values, color=colors)
        plt.title(title)
        plt.ylabel(metric)
        plt.xticks(rotation=15, ha="right")
        for bar, value in zip(bars, values):
            text = f"{value:.3f}" if value < 10 else f"{value:.1f}"
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), text, ha="center", va="bottom", fontsize=9)
        plt.tight_layout()
        plt.savefig(output_root / filename, dpi=180)
        plt.close()


def main() -> int:
    args = parse_args()
    device = choose_device(args.device)
    probe_results = args.probe_results or latest_probe_results()
    output_root = args.output_root or (
        ARTIFACTS_DIR / "results" / "probe_filter_component_ablation" / f"probe_filter_component_ablation_{timestamp()}"
    )
    output_root.mkdir(parents=True, exist_ok=True)
    progress = output_root / "progress.txt"
    progress_line(progress, f"START probe_filter_component_ablation output={output_root}")
    progress_line(progress, f"probe_results={probe_results}")
    progress_line(progress, f"args={json.dumps(vars(args), default=str, sort_keys=True)}")

    checkpoint, beta = resolve_checkpoint_and_beta(args, probe_results, progress)
    probe_checkpoint = resolve_probe_checkpoint(args, probe_results, progress)
    thresholds_cache = load_torch(probe_results / "probe" / "calibrated_thresholds.pt")
    thresholds = thresholds_cache["thresholds"][args.threshold_objective].float().to(device)

    model, checkpoint_data = load_model_checkpoint(checkpoint, device)
    model.eval()
    config = checkpoint_data["config"]
    prompt_cache = load_prompt_embedding_cache(blends.router.resolve_project_path(config.get("prompt_cache_path")))
    text_bank, _ = blends.router.load_text_banks(device)

    probe, probe_attributes, _ = v1.load_probe(probe_checkpoint, device)
    attributes, _, _ = read_attribute_table()
    if attributes != probe_attributes:
        raise RuntimeError("Probe attributes do not match CelebA attributes")
    attribute_to_index = {name: idx for idx, name in enumerate(attributes)}
    annotations = load_evaluation()

    gallery_cache = v1.load_split_embeddings("test")
    gallery = F.normalize(gallery_cache["embeddings"].float(), dim=-1).to(device)
    probs_path = probe_results / "probe" / "test_probe_probs.pt"
    if probs_path.exists():
        gallery_probs = load_torch(probs_path)["probs"].float().to(device)
        progress_line(progress, f"REUSE test probe probs {probs_path}")
    else:
        progress_line(progress, "PROBE predicting test attributes")
        gallery_probs = v1.predict_probe_probs(probe, gallery_cache["embeddings"], device, int(args.probe_batch_size)).to(device)

    query_ids = list(range(len(annotations))) if args.query_ids is None else args.query_ids
    records: dict[str, dict[int, list[dict[str, Any]]]] = {method: {} for method in METHODS}
    started = time.monotonic()
    with torch.inference_mode():
        for query_id in query_ids:
            item = annotations[query_id]
            conditions = parse_query(item["query"])
            attr_indices_one, signs_one = v1.query_tensors(conditions, attribute_to_index, device)
            attr_indices = attr_indices_one[0]
            signs = signs_one[0]
            source_indices = [int(index) for index in item["ground_truth"]]
            if args.max_sources_per_query:
                source_indices = source_indices[: int(args.max_sources_per_query)]
            for start in range(0, len(source_indices), int(args.source_batch_size)):
                end = min(start + int(args.source_batch_size), len(source_indices))
                batch_indices = source_indices[start:end]
                idx = torch.tensor(batch_indices, device=device)
                source = gallery[idx]
                q_model = blends.model_query(
                    model,
                    source,
                    conditions,
                    attribute_to_index,
                    prompt_cache,
                    str(config.get("condition_mode", "signed_prompt")),
                    device,
                )
                q_generic = blends.generic_sum_query(source, conditions, attribute_to_index, text_bank)
                q_final = blends.vector_delta(q_model, q_generic, source, beta)
                scores = q_final @ gallery.T
                rows = torch.arange(len(batch_indices), device=device)
                scores[rows, idx] = -torch.inf
                top_scores, top_indices = scores.topk(int(args.top_pool), dim=1)
                source_probs = gallery_probs[idx]
                candidate_probs = gallery_probs[top_indices]

                for method in METHODS:
                    rankings, kept_counts, avg_ham, avg_margin = make_filtered_lists(
                        method,
                        top_indices,
                        top_scores,
                        source_probs,
                        candidate_probs,
                        attr_indices,
                        signs,
                        thresholds,
                        int(args.top_k),
                        float(args.max_hamming),
                        bool(args.fill_to_k),
                    )
                    records[method].setdefault(query_id, []).extend(
                        {
                            "method": method,
                            "query_id": query_id,
                            "query": item["query"],
                            "source_index": source_index,
                            "top10": ranking,
                            "kept_in_pool": kept,
                            "avg_pred_hamming_kept": ham,
                            "avg_query_margin_kept": margin,
                        }
                        for source_index, ranking, kept, ham, margin in zip(batch_indices, rankings, kept_counts, avg_ham, avg_margin)
                    )
                progress_line(progress, f"EVAL query={query_id + 1}/{len(annotations)} sources={end}/{len(source_indices)}")
                if args.time_budget_seconds and time.monotonic() - started >= int(args.time_budget_seconds):
                    raise SystemExit("Time budget reached")

    write_outputs(output_root, annotations, records, checkpoint, probe_checkpoint, args)
    progress_line(progress, f"COMPLETE output={output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
