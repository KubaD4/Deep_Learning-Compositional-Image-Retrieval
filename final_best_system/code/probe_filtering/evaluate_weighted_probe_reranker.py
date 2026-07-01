#!/usr/bin/env python3
"""Weighted probe reranker for the final hybrid retrieval system.

This script is a no-training follow-up to the calibrated probe reranker.

Motivation
----------
The oracle experiment showed that the official non-query Hamming constraint is
very powerful, but the learned probe is not reliable enough to use Hamming as a
pure hard delete rule. This script keeps the current best hybrid retrieval
vector fixed, then evaluates safer alternatives:

1. promote candidates that satisfy the query according to the calibrated probe;
2. use predicted Hamming as a soft penalty instead of hard deletion;
3. down-weight noisy attributes in the Hamming penalty using probe reliability.

The official JSON is used only for final metric computation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - plotting is optional on minimal envs.
    plt = None


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
import embedding_probe_loader as probe_loader  # noqa: E402
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


@dataclass(frozen=True)
class RerankMethod:
    name: str
    kind: str
    objective: str = "accuracy"
    reliability: str = "uniform"
    hamming_mode: str = "prob"
    lambda_query: float = 0.0
    lambda_hamming: float = 0.0
    lambda_source: float = 0.0
    max_hamming: float = 2.0
    fill_to_k: bool = True


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
    parser.add_argument("--query-ids", nargs="*", type=int, default=None)
    parser.add_argument("--max-sources-per-query", type=int, default=0)
    parser.add_argument("--time-budget-seconds", type=int, default=0)
    parser.add_argument("--threshold-objectives", nargs="+", default=["accuracy", "f1"])
    parser.add_argument("--min-reliability", type=float, default=0.15)
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
    probe_checkpoint = args.probe_checkpoint or (probe_results / "probe" / "best_probe.pt")
    if not probe_checkpoint.exists():
        raise FileNotFoundError(f"Missing probe checkpoint: {probe_checkpoint}")
    progress_line(progress, f"PROBE_CHECKPOINT {probe_checkpoint}")
    return probe_checkpoint


def load_reliability_table(probe_results: Path, attributes: list[str], objective: str, min_value: float) -> dict[str, torch.Tensor]:
    by_mode = {
        "uniform": torch.ones(len(attributes), dtype=torch.float32),
    }

    per_attr_csv = probe_results / "probe" / "calibrated_thresholds_per_attribute.csv"
    if per_attr_csv.exists():
        rows = list(csv.DictReader(per_attr_csv.open(encoding="utf-8")))
        objective_rows = [row for row in rows if row["objective"] == objective]
        for metric in ["f1", "accuracy", "balanced_accuracy", "precision"]:
            values = {row["attribute"]: float(row[metric]) for row in objective_rows}
            if values:
                tensor = torch.tensor([values.get(attr, 1.0) for attr in attributes], dtype=torch.float32)
                by_mode[metric] = tensor.clamp(float(min_value), 1.0)

    # Prefer the validation error-pattern analysis if it was copied under the
    # same result folder. It directly measures calibrated prediction quality.
    for split in ["valid", "test"]:
        error_csv = probe_results / "probe_error_patterns" / split / "overall_attribute_errors.csv"
        if not error_csv.exists():
            continue
        rows = list(csv.DictReader(error_csv.open(encoding="utf-8")))
        values = {row["attribute"]: float(row["f1"]) for row in rows}
        tensor = torch.tensor([values.get(attr, 1.0) for attr in attributes], dtype=torch.float32)
        by_mode[f"{split}_f1"] = tensor.clamp(float(min_value), 1.0)

    return by_mode


def weighted_hamming(
    source_probs: torch.Tensor,
    candidate_probs: torch.Tensor,
    attr_indices: torch.Tensor,
    thresholds: torch.Tensor,
    weights: torch.Tensor,
    mode: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    nonquery = torch.ones(source_probs.shape[-1], dtype=torch.bool, device=source_probs.device)
    nonquery[attr_indices.long()] = False

    local_weights = weights.to(source_probs.device)[nonquery].view(1, 1, -1)
    ps = source_probs[:, None, nonquery]
    pc = candidate_probs[:, :, nonquery]

    if mode == "hard":
        source_bits = source_probs >= thresholds[None, :]
        cand_bits = candidate_probs >= thresholds[None, None, :]
        diff = (source_bits[:, None, nonquery] != cand_bits[:, :, nonquery]).float()
    elif mode == "prob":
        # Expected mismatch probability for independent Bernoulli predictions.
        diff = ps * (1.0 - pc) + (1.0 - ps) * pc
    else:
        raise ValueError(f"Unknown hamming mode: {mode}")

    raw = (diff * local_weights).sum(dim=-1)
    norm = raw / local_weights.sum().clamp_min(1e-8)
    return raw, norm


def build_methods(objectives: list[str]) -> list[RerankMethod]:
    methods = [
        RerankMethod("baseline_q_final", "baseline"),
    ]
    for objective in objectives:
        suffix = objective.replace("_", "")
        methods.extend(
            [
                RerankMethod(f"current_A_query_hardh2_{suffix}", "promote_query_hardh2", objective=objective),
                RerankMethod(f"query_promote_{suffix}", "promote_query", objective=objective),
            ]
        )

    soft_grid = [
        ("uniform", "prob", 0.10, 0.02, 0.00),
        ("f1", "prob", 0.10, 0.02, 0.00),
        ("f1", "prob", 0.20, 0.02, 0.00),
        ("f1", "prob", 0.20, 0.05, 0.00),
        ("f1", "prob", 0.20, 0.05, 0.05),
        ("accuracy", "prob", 0.20, 0.05, 0.00),
        ("f1", "hard", 0.20, 0.02, 0.00),
        ("f1", "hard", 0.20, 0.05, 0.05),
    ]
    for objective in objectives:
        suffix = objective.replace("_", "")
        for reliability, hmode, lq, lh, ls in soft_grid:
            name = f"soft_{suffix}_{reliability}_{hmode}_lq{lq:.2f}_lh{lh:.2f}_ls{ls:.2f}".replace(".", "p")
            methods.append(
                RerankMethod(
                    name,
                    "soft",
                    objective=objective,
                    reliability=reliability,
                    hamming_mode=hmode,
                    lambda_query=lq,
                    lambda_hamming=lh,
                    lambda_source=ls,
                )
            )
            name = f"queryhard_soft_{suffix}_{reliability}_{hmode}_lh{lh:.2f}_ls{ls:.2f}".replace(".", "p")
            methods.append(
                RerankMethod(
                    name,
                    "queryhard_soft",
                    objective=objective,
                    reliability=reliability,
                    hamming_mode=hmode,
                    lambda_hamming=lh,
                    lambda_source=ls,
                )
            )
    return methods


def select_rankings(
    method: RerankMethod,
    top_indices: torch.Tensor,
    top_scores: torch.Tensor,
    source: torch.Tensor,
    gallery_top: torch.Tensor,
    source_probs: torch.Tensor,
    candidate_probs: torch.Tensor,
    attr_indices: torch.Tensor,
    signs: torch.Tensor,
    thresholds: torch.Tensor,
    reliability_weights: torch.Tensor,
    top_k: int,
) -> tuple[list[list[int]], list[float], list[float], list[float]]:
    if method.kind == "baseline":
        return (
            top_indices[:, :top_k].cpu().tolist(),
            [float(top_k)] * int(top_indices.shape[0]),
            [0.0] * int(top_indices.shape[0]),
            [0.0] * int(top_indices.shape[0]),
        )

    query_ok = v2.calibrated_query_ok(candidate_probs, attr_indices, signs, thresholds)
    query_margin = v2.calibrated_query_margin(candidate_probs, attr_indices, signs, thresholds)
    hard_ham, _ = weighted_hamming(source_probs, candidate_probs, attr_indices, thresholds, torch.ones_like(reliability_weights), "hard")
    weighted_ham_raw, weighted_ham_norm = weighted_hamming(
        source_probs,
        candidate_probs,
        attr_indices,
        thresholds,
        reliability_weights,
        method.hamming_mode,
    )
    source_sim = (source[:, None, :] * gallery_top).sum(dim=-1)

    if method.kind == "promote_query":
        keep = query_ok
        score = top_scores.masked_fill(~keep, -torch.inf)
    elif method.kind == "promote_query_hardh2":
        keep = query_ok & (hard_ham <= float(method.max_hamming))
        score = top_scores.masked_fill(~keep, -torch.inf)
    elif method.kind == "soft":
        keep = torch.ones_like(top_scores, dtype=torch.bool)
        score = (
            top_scores
            + float(method.lambda_query) * query_margin
            - float(method.lambda_hamming) * weighted_ham_norm
            + float(method.lambda_source) * source_sim
        )
    elif method.kind == "queryhard_soft":
        keep = query_ok
        score = (
            top_scores
            - float(method.lambda_hamming) * weighted_ham_norm
            + float(method.lambda_source) * source_sim
        ).masked_fill(~keep, -torch.inf)
    else:
        raise ValueError(method.kind)

    rankings: list[list[int]] = []
    kept_counts: list[float] = []
    avg_hamming: list[float] = []
    avg_query_margin: list[float] = []
    for row in range(score.shape[0]):
        finite = torch.isfinite(score[row])
        kept_counts.append(float(finite.sum().detach().cpu()))
        if bool(finite.any()):
            avg_hamming.append(float(weighted_ham_raw[row, finite].float().mean().detach().cpu()))
            avg_query_margin.append(float(query_margin[row, finite].float().mean().detach().cpu()))
            ordered_local = score[row].topk(min(top_k, int(finite.sum().item()))).indices
            selected = top_indices[row, ordered_local].tolist()
        else:
            avg_hamming.append(float("nan"))
            avg_query_margin.append(float("nan"))
            selected = []

        if method.fill_to_k and len(selected) < top_k:
            selected_set = set(selected)
            for idx in top_indices[row].tolist():
                if idx not in selected_set:
                    selected.append(idx)
                    selected_set.add(idx)
                if len(selected) >= top_k:
                    break
        rankings.append(selected[:top_k])
    return rankings, kept_counts, avg_hamming, avg_query_margin


def summarize(
    annotations: list[dict[str, Any]],
    records_by_query: dict[int, list[dict[str, Any]]],
    method: RerankMethod,
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
            if not math.isnan(float(record["avg_pred_hamming_kept"])):
                hamming_sum += float(record["avg_pred_hamming_kept"])
                hamming_count += 1
            if not math.isnan(float(record["avg_query_margin_kept"])):
                margin_sum += float(record["avg_query_margin_kept"])
                margin_count += 1
            for k in TOP_KS:
                recall, precision = retrieval_metrics(record["top10"], valid, k)
                totals[f"Recall@{k}"] += recall
                totals[f"Precision@{k}"] += precision
        source_count = len(records)
        per_query.append(
            {
                "method": method.name,
                "query_id": query_id,
                "query": item["query"],
                "sources": source_count,
                "avg_kept_in_pool": kept_sum / max(1, source_count),
                "avg_pred_hamming_kept": hamming_sum / max(1, hamming_count),
                "avg_query_margin_kept": margin_sum / max(1, margin_count),
                **{name: value / source_count for name, value in totals.items()},
            }
        )

    metric_names = [name for name in per_query[0] if "@" in name]
    total_sources = sum(row["sources"] for row in per_query)
    summary = {
        "method": method.name,
        "kind": method.kind,
        "objective": method.objective,
        "reliability": method.reliability,
        "hamming_mode": method.hamming_mode,
        "lambda_query": method.lambda_query,
        "lambda_hamming": method.lambda_hamming,
        "lambda_source": method.lambda_source,
        "checkpoint": str(checkpoint),
        "probe_checkpoint": str(probe_checkpoint),
        "model_id": MODEL_ID,
        "top_pool": args.top_pool,
        "top_k": args.top_k,
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
    method_records: dict[str, dict[int, list[dict[str, Any]]]],
    methods_by_name: dict[str, RerankMethod],
    checkpoint: Path,
    probe_checkpoint: Path,
    args: argparse.Namespace,
) -> None:
    summaries = []
    all_per_query = []
    for name, records_by_query in method_records.items():
        method = methods_by_name[name]
        method_dir = output_root / name
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
    (output_root / "BEST_WEIGHTED_PROBE_RERANKER_METHOD.txt").write_text(
        "\n".join(
            [
                f"method={best['method']}",
                f"kind={best['kind']}",
                f"objective={best['objective']}",
                f"reliability={best['reliability']}",
                f"hamming_mode={best['hamming_mode']}",
                f"lambda_query={best['lambda_query']}",
                f"lambda_hamming={best['lambda_hamming']}",
                f"lambda_source={best['lambda_source']}",
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

    plot_summary(output_root, summaries)


def plot_summary(output_root: Path, summaries: list[dict[str, Any]]) -> None:
    if plt is None:
        (output_root / "PLOTS_SKIPPED.txt").write_text(
            "matplotlib is not available in this environment; CSV outputs are complete.\n",
            encoding="utf-8",
        )
        return
    top = summaries[: min(16, len(summaries))]
    for metric, title, filename in [
        ("macro_Recall@10", "Weighted probe reranker: Macro Recall@10", "macro_recall10_weighted_probe.png"),
        ("macro_Precision@10", "Weighted probe reranker: Macro Precision@10", "macro_precision10_weighted_probe.png"),
        ("macro_avg_kept_in_pool", "Weighted probe reranker: average finite kept candidates", "avg_kept_weighted_probe.png"),
    ]:
        names = [row["method"] for row in top]
        values = [float(row[metric]) for row in top]
        plt.figure(figsize=(12, max(5, 0.35 * len(names))))
        bars = plt.barh(range(len(names)), values, color="#4c78a8")
        plt.yticks(range(len(names)), names, fontsize=8)
        plt.gca().invert_yaxis()
        plt.xlabel(metric)
        plt.title(title)
        for bar, value in zip(bars, values):
            text = f"{value:.3f}" if value < 10 else f"{value:.1f}"
            plt.text(value, bar.get_y() + bar.get_height() / 2, f" {text}", va="center", fontsize=8)
        plt.tight_layout()
        plt.savefig(output_root / filename, dpi=180)
        plt.close()


def main() -> int:
    args = parse_args()
    device = choose_device(args.device)
    probe_results = args.probe_results or latest_probe_results()
    output_root = args.output_root or (
        ARTIFACTS_DIR / "results" / "weighted_probe_reranker" / f"weighted_probe_reranker_{timestamp()}"
    )
    output_root.mkdir(parents=True, exist_ok=True)
    progress = output_root / "progress.txt"
    progress_line(progress, f"START weighted_probe_reranker output={output_root}")
    progress_line(progress, f"probe_results={probe_results}")
    progress_line(progress, f"args={json.dumps(vars(args), default=str, sort_keys=True)}")

    checkpoint, beta = resolve_checkpoint_and_beta(args, probe_results, progress)
    probe_checkpoint = resolve_probe_checkpoint(args, probe_results, progress)

    thresholds_cache = load_torch(probe_results / "probe" / "calibrated_thresholds.pt")
    threshold_by_objective = {
        objective: thresholds_cache["thresholds"][objective].float().to(device)
        for objective in args.threshold_objectives
    }

    model, checkpoint_data = load_model_checkpoint(checkpoint, device)
    model.eval()
    config = checkpoint_data["config"]
    prompt_cache = load_prompt_embedding_cache(blends.router.resolve_project_path(config.get("prompt_cache_path")))
    text_bank, _ = blends.router.load_text_banks(device)

    probe, probe_attributes, _ = probe_loader.load_embedding_probe(probe_checkpoint, device)
    attributes, _, _ = read_attribute_table()
    if attributes != probe_attributes:
        raise RuntimeError("Probe attributes do not match CelebA attributes")
    attribute_to_index = {name: idx for idx, name in enumerate(attributes)}

    reliability_by_objective = {
        objective: load_reliability_table(probe_results, attributes, objective, float(args.min_reliability))
        for objective in args.threshold_objectives
    }
    methods = build_methods(list(args.threshold_objectives))
    methods_by_name = {method.name: method for method in methods}
    progress_line(progress, f"METHODS {len(methods)}")

    annotations = load_evaluation()
    gallery_cache = v1.load_split_embeddings("test")
    gallery = F.normalize(gallery_cache["embeddings"].float(), dim=-1).to(device)
    probs_path = probe_results / "probe" / "test_probe_probs.pt"
    if probs_path.exists():
        gallery_probs = load_torch(probs_path)["probs"].float().to(device)
        progress_line(progress, f"REUSE test probe probs {probs_path}")
    else:
        progress_line(progress, "PROBE predicting test attributes")
        gallery_probs = probe_loader.predict_embedding_probe_probs(
            probe,
            gallery_cache["embeddings"],
            device,
            int(args.probe_batch_size),
        ).to(device)

    query_ids = list(range(len(annotations))) if args.query_ids is None else args.query_ids
    method_records: dict[str, dict[int, list[dict[str, Any]]]] = {method.name: {} for method in methods}
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
                gallery_top = gallery[top_indices]

                for method in methods:
                    thresholds = threshold_by_objective[method.objective]
                    reliability_options = reliability_by_objective[method.objective]
                    reliability_weights = reliability_options.get(method.reliability)
                    if reliability_weights is None:
                        reliability_weights = reliability_options["uniform"]
                    rankings, kept_counts, avg_ham, avg_margin = select_rankings(
                        method,
                        top_indices,
                        top_scores,
                        source,
                        gallery_top,
                        source_probs,
                        candidate_probs,
                        attr_indices,
                        signs,
                        thresholds,
                        reliability_weights.to(device),
                        int(args.top_k),
                    )
                    method_records[method.name].setdefault(query_id, []).extend(
                        {
                            "method": method.name,
                            "query_id": query_id,
                            "query": item["query"],
                            "source_index": source_index,
                            "top10": ranking,
                            "kept_in_pool": kept,
                            "avg_pred_hamming_kept": ham,
                            "avg_query_margin_kept": margin,
                        }
                        for source_index, ranking, kept, ham, margin in zip(
                            batch_indices, rankings, kept_counts, avg_ham, avg_margin
                        )
                    )

                progress_line(progress, f"EVAL query={query_id + 1}/{len(annotations)} sources={end}/{len(source_indices)}")
                if args.time_budget_seconds and time.monotonic() - started > float(args.time_budget_seconds):
                    raise SystemExit("Time budget reached during weighted probe reranker evaluation")

    write_outputs(output_root, annotations, method_records, methods_by_name, checkpoint, probe_checkpoint, args)
    progress_line(progress, f"COMPLETE weighted_probe_reranker output={output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
