#!/usr/bin/env python3
"""Calibrated CelebA-probe reranker for the final hybrid system.

This is an experimental follow-up to `probe_reranker_v1.py`.

Main difference:
v1 used a fixed threshold of 0.5 for every CelebA attribute.
v2 calibrates one threshold per attribute on the validation split, then tests
whether calibrated query satisfaction and calibrated Hamming estimates improve
top-pool reranking.

The official JSON is still used only for final evaluation metrics.
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
from datetime import datetime
from pathlib import Path

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
import embedding_probe_loader as probe_loader  # noqa: E402
import probe_reranker_v1 as v1  # noqa: E402
from learned_gate_core import load_model_checkpoint, load_prompt_embedding_cache, progress_line  # noqa: E402
from project_core import (  # noqa: E402
    ARTIFACTS_DIR,
    MODEL_ID,
    TOP_KS,
    atomic_json_dump,
    atomic_torch_save,
    choose_device,
    load_evaluation,
    load_torch,
    parse_query,
    read_attribute_table,
    retrieval_metrics,
)


STOP_REQUESTED = False


def request_stop(signum, frame) -> None:
    del signum, frame
    global STOP_REQUESTED
    STOP_REQUESTED = True
    v1.STOP_REQUESTED = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--probe-checkpoint", type=Path, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--source-batch-size", type=int, default=256)
    parser.add_argument("--probe-batch-size", type=int, default=2048)
    parser.add_argument("--probe-epochs", type=int, default=8)
    parser.add_argument("--probe-max-steps", type=int, default=0)
    parser.add_argument("--probe-lr", type=float, default=3e-4)
    parser.add_argument("--probe-weight-decay", type=float, default=1e-4)
    parser.add_argument("--probe-hidden-dims", type=str, default="1024,512")
    parser.add_argument("--probe-dropout", type=float, default=0.1)
    parser.add_argument("--probe-hpsearch", action="store_true")
    parser.add_argument("--probe-use-flip", action="store_true")
    parser.add_argument("--max-probe-configs", type=int, default=0)
    parser.add_argument("--flip-batch-size", type=int, default=256)
    parser.add_argument("--flip-workers", type=int, default=8)
    parser.add_argument("--random-probe-smoke", action="store_true")
    parser.add_argument("--top-pool", type=int, default=500)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--query-ids", nargs="*", type=int, default=None)
    parser.add_argument("--max-sources-per-query", type=int, default=0)
    parser.add_argument("--time-budget-seconds", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-probe-training", action="store_true")
    parser.add_argument("--threshold-objectives", nargs="+", default=["f1", "balanced_accuracy", "accuracy"])
    parser.add_argument("--threshold-min", type=float, default=0.05)
    parser.add_argument("--threshold-max", type=float, default=0.95)
    parser.add_argument("--threshold-step", type=float, default=0.025)
    return parser.parse_args()


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def binary_metrics(preds: torch.Tensor, labels: torch.Tensor) -> dict[str, torch.Tensor]:
    truth = labels.bool()
    pred = preds.bool()
    eps = 1e-8
    tp = (pred & truth).sum(dim=0).float()
    fp = (pred & ~truth).sum(dim=0).float()
    fn = (~pred & truth).sum(dim=0).float()
    tn = (~pred & ~truth).sum(dim=0).float()
    accuracy = (tp + tn) / (tp + tn + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    specificity = tn / (tn + fp + eps)
    balanced_accuracy = 0.5 * (recall + specificity)
    f1 = 2 * precision * recall / (precision + recall + eps)
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "balanced_accuracy": balanced_accuracy,
        "f1": f1,
    }


def calibrate_probe_thresholds(
    args: argparse.Namespace,
    output_root: Path,
    progress: Path,
    device: torch.device,
    probe_path: Path,
) -> dict[str, torch.Tensor]:
    probe, probe_attributes, _ = probe_loader.load_embedding_probe(probe_path, device)
    valid_cache = v1.load_split_embeddings("valid")
    attributes, valid_labels = v1.attrs_for_filenames(valid_cache["filenames"])
    if attributes != probe_attributes:
        raise RuntimeError("Probe attributes do not match CelebA attributes")

    probs_path = output_root / "probe" / "valid_probe_probs.pt"
    if probs_path.exists() and not args.force:
        valid_probs = load_torch(probs_path)["probs"].float()
        progress_line(progress, f"REUSE valid probe probs {probs_path}")
    else:
        progress_line(progress, "CALIBRATION predicting validation attributes")
        valid_probs = probe_loader.predict_embedding_probe_probs(
            probe,
            valid_cache["embeddings"],
            device,
            int(args.probe_batch_size),
        )
        atomic_torch_save({"model_id": MODEL_ID, "attributes": attributes, "probs": valid_probs}, probs_path)

    thresholds = torch.arange(
        float(args.threshold_min),
        float(args.threshold_max) + 0.5 * float(args.threshold_step),
        float(args.threshold_step),
    ).clamp(0.0, 1.0)
    objective_set = set(args.threshold_objectives)
    valid_objectives = {"f1", "balanced_accuracy", "accuracy", "precision_recall_mid"}
    unknown = objective_set - valid_objectives
    if unknown:
        raise ValueError(f"Unknown threshold objectives: {sorted(unknown)}")

    threshold_by_objective: dict[str, torch.Tensor] = {}
    rows = []
    for objective in args.threshold_objectives:
        chosen = []
        for attr_idx, attr_name in enumerate(attributes):
            best_score = -1.0
            best_threshold = 0.5
            best_metrics = None
            labels = valid_labels[:, attr_idx : attr_idx + 1]
            scores = valid_probs[:, attr_idx : attr_idx + 1]
            for threshold in thresholds:
                metrics = binary_metrics(scores >= threshold, labels)
                if objective == "precision_recall_mid":
                    value = torch.minimum(metrics["precision"], metrics["recall"])[0]
                else:
                    value = metrics[objective][0]
                if float(value) > best_score:
                    best_score = float(value)
                    best_threshold = float(threshold)
                    best_metrics = metrics
            assert best_metrics is not None
            chosen.append(best_threshold)
            rows.append(
                {
                    "objective": objective,
                    "attribute": attr_name,
                    "threshold": best_threshold,
                    "score": best_score,
                    "accuracy": float(best_metrics["accuracy"][0]),
                    "precision": float(best_metrics["precision"][0]),
                    "recall": float(best_metrics["recall"][0]),
                    "balanced_accuracy": float(best_metrics["balanced_accuracy"][0]),
                    "f1": float(best_metrics["f1"][0]),
                    "positive_rate": float(labels.float().mean()),
                }
            )
        threshold_by_objective[objective] = torch.tensor(chosen, dtype=torch.float32)

    write_csv(output_root / "probe" / "calibrated_thresholds_per_attribute.csv", rows)

    summary_rows = []
    for objective, threshold_tensor in threshold_by_objective.items():
        preds = valid_probs >= threshold_tensor[None, :]
        metrics = binary_metrics(preds, valid_labels)
        summary_rows.append(
            {
                "objective": objective,
                "macro_accuracy": float(metrics["accuracy"].mean()),
                "macro_balanced_accuracy": float(metrics["balanced_accuracy"].mean()),
                "macro_precision": float(metrics["precision"].mean()),
                "macro_recall": float(metrics["recall"].mean()),
                "macro_f1": float(metrics["f1"].mean()),
                "micro_accuracy": float((preds.bool() == valid_labels.bool()).float().mean()),
            }
        )
    write_csv(output_root / "probe" / "calibrated_threshold_summary.csv", summary_rows)
    atomic_torch_save(
        {"model_id": MODEL_ID, "attributes": attributes, "thresholds": threshold_by_objective},
        output_root / "probe" / "calibrated_thresholds.pt",
    )
    best_line = max(summary_rows, key=lambda row: float(row["macro_f1"]))
    progress_line(progress, f"CALIBRATION complete best_by_macro_f1={best_line}")
    return threshold_by_objective


@dataclass(frozen=True)
class CalibratedMethod:
    name: str
    kind: str
    objective: str = "f1"
    lambda_query: float = 0.0
    lambda_hamming: float = 0.0
    lambda_source: float = 0.0
    hard_hamming_limit: float = 2.0


def calibrated_methods(objectives: list[str]) -> list[CalibratedMethod]:
    methods = [CalibratedMethod("baseline_q_final_top10", "baseline")]
    for objective in objectives:
        suffix = objective.replace("_", "")
        methods.extend(
            [
                CalibratedMethod(f"A_cal_query_only_{suffix}", "query_only", objective=objective),
                CalibratedMethod(f"A_cal_query_hardh2_{suffix}", "query_hard_hamming", objective=objective),
                CalibratedMethod(
                    f"B_cal_soft_lq010_lh010_ls005_{suffix}",
                    "soft",
                    objective=objective,
                    lambda_query=0.10,
                    lambda_hamming=0.10,
                    lambda_source=0.05,
                ),
                CalibratedMethod(
                    f"B_cal_soft_lq015_lh010_ls005_{suffix}",
                    "soft",
                    objective=objective,
                    lambda_query=0.15,
                    lambda_hamming=0.10,
                    lambda_source=0.05,
                ),
                CalibratedMethod(
                    f"C_cal_hybrid_lh010_ls005_{suffix}",
                    "hybrid",
                    objective=objective,
                    lambda_hamming=0.10,
                    lambda_source=0.05,
                ),
                CalibratedMethod(
                    f"C_cal_hybrid_lh015_ls005_{suffix}",
                    "hybrid",
                    objective=objective,
                    lambda_hamming=0.15,
                    lambda_source=0.05,
                ),
            ]
        )
    return methods


def calibrated_query_ok(
    candidate_probs: torch.Tensor,
    attr_indices: torch.Tensor,
    signs: torch.Tensor,
    thresholds: torch.Tensor,
) -> torch.Tensor:
    ok = torch.ones(candidate_probs.shape[:2], dtype=torch.bool, device=candidate_probs.device)
    for pos in range(attr_indices.numel()):
        attr = int(attr_indices[pos])
        sign = int(signs[pos])
        prob = candidate_probs[:, :, attr]
        threshold = thresholds[attr]
        if sign > 0:
            ok &= prob >= threshold
        else:
            ok &= prob <= threshold
    return ok


def calibrated_query_margin(
    candidate_probs: torch.Tensor,
    attr_indices: torch.Tensor,
    signs: torch.Tensor,
    thresholds: torch.Tensor,
) -> torch.Tensor:
    parts = []
    for pos in range(attr_indices.numel()):
        attr = int(attr_indices[pos])
        sign = int(signs[pos])
        prob = candidate_probs[:, :, attr]
        threshold = thresholds[attr]
        parts.append(prob - threshold if sign > 0 else threshold - prob)
    return torch.stack(parts, dim=-1).mean(dim=-1)


def calibrated_hamming(
    source_probs: torch.Tensor,
    candidate_probs: torch.Tensor,
    attr_indices: torch.Tensor,
    thresholds: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    nonquery = torch.ones(source_probs.shape[-1], dtype=torch.bool, device=source_probs.device)
    nonquery[attr_indices.long()] = False
    source_bits = source_probs >= thresholds[None, :]
    cand_bits = candidate_probs >= thresholds[None, None, :]
    hard = (source_bits[:, None, nonquery] != cand_bits[:, :, nonquery]).float().sum(dim=-1)
    ps = source_probs[:, None, nonquery]
    pc = candidate_probs[:, :, nonquery]
    soft_attr_diff = ps * (1.0 - pc) + (1.0 - ps) * pc
    soft_raw = soft_attr_diff.sum(dim=-1)
    soft_norm = soft_attr_diff.mean(dim=-1)
    return hard, soft_raw, soft_norm


def rerank_calibrated(
    method: CalibratedMethod,
    top_indices: torch.Tensor,
    top_scores: torch.Tensor,
    source: torch.Tensor,
    gallery: torch.Tensor,
    source_probs: torch.Tensor,
    gallery_probs: torch.Tensor,
    attr_indices: torch.Tensor,
    signs: torch.Tensor,
    thresholds: torch.Tensor,
    top_k: int,
) -> tuple[list[list[int]], list[float], list[float], list[float]]:
    if method.kind == "baseline":
        return top_indices[:, :top_k].cpu().tolist(), [float(top_k)] * int(top_indices.shape[0]), [0.0] * int(top_indices.shape[0]), [0.0] * int(top_indices.shape[0])

    candidate_probs = gallery_probs[top_indices].to(top_scores.device)
    source_probs = source_probs.to(top_scores.device)
    thresholds = thresholds.to(top_scores.device)
    attr_indices = attr_indices.to(top_scores.device)
    signs = signs.to(top_scores.device)
    query_ok = calibrated_query_ok(candidate_probs, attr_indices, signs, thresholds)
    query_margin = calibrated_query_margin(candidate_probs, attr_indices, signs, thresholds)
    hard_ham, _, soft_ham_norm = calibrated_hamming(source_probs, candidate_probs, attr_indices, thresholds)
    source_sim = (source[:, None, :] * gallery[top_indices]).sum(dim=-1)

    if method.kind == "query_only":
        keep = query_ok
        score = top_scores.masked_fill(~keep, -torch.inf)
    elif method.kind == "query_hard_hamming":
        keep = query_ok & (hard_ham <= float(method.hard_hamming_limit))
        score = top_scores.masked_fill(~keep, -torch.inf)
    elif method.kind == "soft":
        keep = torch.ones_like(top_scores, dtype=torch.bool)
        score = (
            top_scores
            + float(method.lambda_query) * query_margin
            - float(method.lambda_hamming) * soft_ham_norm
            + float(method.lambda_source) * source_sim
        )
    elif method.kind == "hybrid":
        keep = query_ok
        score = (
            top_scores
            - float(method.lambda_hamming) * soft_ham_norm
            + float(method.lambda_source) * source_sim
        ).masked_fill(~keep, -torch.inf)
    else:
        raise ValueError(method.kind)

    rankings = []
    kept_counts = []
    avg_hamming = []
    avg_query_margin = []
    for row in range(score.shape[0]):
        finite = torch.isfinite(score[row])
        kept_counts.append(float(finite.sum().detach().cpu()))
        avg_hamming.append(float(hard_ham[row, finite].float().mean().detach().cpu()) if bool(finite.any()) else float("nan"))
        avg_query_margin.append(float(query_margin[row, finite].float().mean().detach().cpu()) if bool(finite.any()) else float("nan"))
        if bool(finite.any()):
            ordered_local = score[row].topk(min(top_k, int(finite.sum().item()))).indices
            selected = top_indices[row, ordered_local].tolist()
        else:
            selected = []
        if len(selected) < top_k:
            selected_set = set(selected)
            for idx in top_indices[row].tolist():
                if idx not in selected_set:
                    selected.append(idx)
                    selected_set.add(idx)
                if len(selected) >= top_k:
                    break
        rankings.append(selected[:top_k])
    return rankings, kept_counts, avg_hamming, avg_query_margin


def finalize_method(
    annotations: list[dict],
    records_by_query: dict[int, list[dict]],
    method_dir: Path,
    checkpoint: Path,
    probe_path: Path,
    extra_config: dict,
) -> None:
    per_query = []
    retrieval_path = method_dir / "retrievals.jsonl"
    with retrieval_path.open("w", encoding="utf-8") as retrieval_file:
        for query_id, item in enumerate(annotations):
            if query_id not in records_by_query:
                continue
            totals = {f"Recall@{k}": 0.0 for k in TOP_KS}
            totals.update({f"Precision@{k}": 0.0 for k in TOP_KS})
            kept_sum = 0.0
            hamming_sum = 0.0
            margin_sum = 0.0
            count_ham = 0
            count_margin = 0
            for record in records_by_query[query_id]:
                retrieval_file.write(json.dumps(record) + "\n")
                valid = set(item["ground_truth"][str(record["source_index"])])
                kept_sum += float(record.get("kept_in_pool", 0.0))
                if not math.isnan(float(record.get("avg_pred_hamming_kept", float("nan")))):
                    hamming_sum += float(record["avg_pred_hamming_kept"])
                    count_ham += 1
                if not math.isnan(float(record.get("avg_query_margin_kept", float("nan")))):
                    margin_sum += float(record["avg_query_margin_kept"])
                    count_margin += 1
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
                    "avg_kept_in_pool": kept_sum / max(1, source_count),
                    "avg_pred_hamming_kept": hamming_sum / max(1, count_ham),
                    "avg_query_margin_kept": margin_sum / max(1, count_margin),
                    **{name: value / source_count for name, value in totals.items()},
                }
            )
    write_csv(method_dir / "per_query_metrics.csv", per_query)
    metric_names = [name for name in per_query[0] if "@" in name]
    total_sources = sum(row["sources"] for row in per_query)
    summary = {
        "method": method_dir.name,
        "checkpoint": str(checkpoint),
        "probe_checkpoint": str(probe_path),
        "query_entries": len(per_query),
        "source_query_cases": total_sources,
        "avg_kept_in_pool": sum(row["avg_kept_in_pool"] * row["sources"] for row in per_query) / total_sources,
        "avg_pred_hamming_kept": sum(row["avg_pred_hamming_kept"] * row["sources"] for row in per_query) / total_sources,
        "avg_query_margin_kept": sum(row["avg_query_margin_kept"] * row["sources"] for row in per_query) / total_sources,
        **{f"macro_{name}": sum(row[name] for row in per_query) / len(per_query) for name in metric_names},
        **{f"micro_{name}": sum(row[name] * row["sources"] for row in per_query) / total_sources for name in metric_names},
    }
    write_csv(method_dir / "summary.csv", [summary])
    atomic_json_dump(extra_config, method_dir / "config.json")
    (method_dir / "COMPLETE").write_text("complete\n", encoding="utf-8")


def evaluate_calibrated(
    args: argparse.Namespace,
    output_root: Path,
    progress: Path,
    device: torch.device,
    probe_path: Path,
    threshold_by_objective: dict[str, torch.Tensor],
) -> None:
    checkpoint, beta = v1.find_checkpoint_and_beta(args, progress)
    model, checkpoint_data = load_model_checkpoint(checkpoint, device)
    model.eval()
    config = checkpoint_data["config"]
    prompt_cache = load_prompt_embedding_cache(blends.router.resolve_project_path(config.get("prompt_cache_path")))
    text_bank, _ = blends.router.load_text_banks(device)

    probe, probe_attributes, probe_checkpoint = probe_loader.load_embedding_probe(probe_path, device)
    attributes, _, _ = read_attribute_table()
    if attributes != probe_attributes:
        raise RuntimeError("Probe attributes do not match CelebA attributes")
    attribute_to_index = {name: idx for idx, name in enumerate(attributes)}
    annotations_all = load_evaluation()
    query_ids = list(range(len(annotations_all))) if args.query_ids is None else args.query_ids

    gallery_cache = v1.load_split_embeddings("test")
    gallery = F.normalize(gallery_cache["embeddings"].float(), dim=-1).to(device)
    probe_probs_path = output_root / "probe" / "test_probe_probs.pt"
    if probe_probs_path.exists() and not args.force:
        gallery_probs = load_torch(probe_probs_path)["probs"].float()
        progress_line(progress, f"REUSE test probe probs {probe_probs_path}")
    else:
        progress_line(progress, "PROBE predicting test attributes")
        gallery_probs = probe_loader.predict_embedding_probe_probs(
            probe,
            gallery_cache["embeddings"],
            device,
            int(args.probe_batch_size),
        )
        atomic_torch_save({"model_id": MODEL_ID, "attributes": attributes, "probs": gallery_probs}, probe_probs_path)
    gallery_probs_device = gallery_probs.to(device)
    methods = calibrated_methods(list(threshold_by_objective))
    records = {method.name: {} for method in methods}
    started = time.monotonic()

    atomic_json_dump(
        {
            "checkpoint": str(checkpoint),
            "probe_checkpoint": str(probe_path),
            "probe_valid_metrics": probe_checkpoint.get("valid_metrics", {}),
            "model_id": MODEL_ID,
            "beta": beta,
            "top_pool": args.top_pool,
            "top_k": args.top_k,
            "methods": [method.__dict__ for method in methods],
            "threshold_objectives": list(threshold_by_objective),
            "notes": "Calibrated probe reranking. JSON is used only for final metric computation.",
        },
        output_root / "experiment_config.json",
    )

    with torch.inference_mode():
        for query_id in query_ids:
            item = annotations_all[query_id]
            conditions = parse_query(item["query"])
            attr_indices_one, signs_one = v1.query_tensors(conditions, attribute_to_index, device)
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
                query = blends.vector_delta(q_model, q_generic, source, beta)
                scores = query @ gallery.T
                rows = torch.arange(len(batch_indices), device=device)
                scores[rows, idx] = -torch.inf
                top_scores, top_indices = scores.topk(int(args.top_pool), dim=1)
                source_probs = gallery_probs_device[idx]

                for method in methods:
                    thresholds = threshold_by_objective.get(method.objective)
                    if thresholds is None:
                        thresholds = torch.full((len(attributes),), 0.5)
                    rankings, kept_counts, avg_hamming, avg_margin = rerank_calibrated(
                        method,
                        top_indices,
                        top_scores,
                        source,
                        gallery,
                        source_probs,
                        gallery_probs_device,
                        attr_indices_one[0],
                        signs_one[0],
                        thresholds,
                        int(args.top_k),
                    )
                    records[method.name].setdefault(query_id, []).extend(
                        {
                            "method": method.name,
                            "query_id": query_id,
                            "query": item["query"],
                            "source_index": source_index,
                            "top10": ranking,
                            "kept_in_pool": kept_count,
                            "avg_pred_hamming_kept": ham,
                            "avg_query_margin_kept": margin,
                        }
                        for source_index, ranking, kept_count, ham, margin in zip(
                            batch_indices, rankings, kept_counts, avg_hamming, avg_margin
                        )
                    )

                progress_line(progress, f"EVAL query={query_id + 1}/{len(annotations_all)} sources={end}/{len(source_indices)}")
                if STOP_REQUESTED or (
                    args.time_budget_seconds and time.monotonic() - started >= int(args.time_budget_seconds)
                ):
                    raise SystemExit("Time budget reached during calibrated reranker evaluation")

    for method in methods:
        method_dir = output_root / method.name
        method_dir.mkdir(parents=True, exist_ok=True)
        thresholds = threshold_by_objective.get(method.objective)
        finalize_method(
            annotations_all,
            records[method.name],
            method_dir,
            checkpoint,
            probe_path,
            {
                "method": method.__dict__,
                "checkpoint": str(checkpoint),
                "probe_checkpoint": str(probe_path),
                "beta": beta,
                "top_pool": args.top_pool,
                "top_k": args.top_k,
                "threshold_objective": method.objective,
                "thresholds": thresholds.tolist() if thresholds is not None else None,
            },
        )
    v1.plot_comparison(output_root)
    progress_line(progress, f"COMPLETE probe reranker v2 calibrated output={output_root}")


def main() -> int:
    args = parse_args()
    signal.signal(signal.SIGTERM, request_stop)
    if hasattr(signal, "SIGUSR1"):
        signal.signal(signal.SIGUSR1, request_stop)
    device = choose_device(args.device)
    output_root = args.output_root or (
        ARTIFACTS_DIR / "results" / "probe_reranker_v2_calibrated" / f"probe_reranker_v2_calibrated_{timestamp()}"
    )
    output_root.mkdir(parents=True, exist_ok=True)
    progress = output_root / "progress.txt"
    progress_line(progress, f"START probe_reranker_v2_calibrated output={output_root}")
    progress_line(progress, f"args={json.dumps(vars(args), default=str, sort_keys=True)}")
    probe_path = v1.train_probe(args, output_root, progress, device)
    threshold_by_objective = calibrate_probe_thresholds(args, output_root, progress, device, probe_path)
    evaluate_calibrated(args, output_root, progress, device, probe_path, threshold_by_objective)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
