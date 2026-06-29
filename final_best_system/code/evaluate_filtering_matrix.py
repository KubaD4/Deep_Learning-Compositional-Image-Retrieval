#!/usr/bin/env python3
"""Oracle/probe filtering matrix for the packaged final system.

Runs the current best hybrid system once, retrieves a top-500 pool, then tests
strict filtering on top-500/top-200/top-100 candidates:

- no filter: current best q_hybrid ranking;
- oracle query only;
- oracle non-query Hamming only;
- oracle query + non-query Hamming;
- probe query only;
- probe non-query Hamming only;
- probe query + non-query Hamming.

The "accuracy@K" reported here is the assignment-style Recall@K / Hit@K:
1 if at least one official target is present in the top-K, otherwise 0.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = PACKAGE_ROOT / "code"
SCRIPTS = CODE_ROOT / "scripts"
ORCH = CODE_ROOT / "orchestrator"
PROBE = CODE_ROOT / "probe_filtering"

os.environ["DL_PROJECT_ROOT"] = str(PACKAGE_ROOT)
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(ORCH))
sys.path.insert(0, str(PROBE))

import evaluate_sum_model_blends as blends  # noqa: E402
import probe_reranker_v1 as probe_v1  # noqa: E402
import probe_reranker_v2_calibrated as probe_v2  # noqa: E402
from learned_gate_core import load_model_checkpoint, load_prompt_embedding_cache  # noqa: E402
from project_core import (  # noqa: E402
    EMBEDDING_DIR,
    MODEL_ID,
    TOP_KS,
    choose_device,
    load_evaluation,
    load_torch,
    parse_query,
    read_attribute_table,
    retrieval_metrics,
)


FILTERS = (
    ("best_no_filter", "none", "Current q_hybrid top-k, no filtering."),
    ("oracle_query_only", "oracle_query", "Oracle keeps candidates satisfying query attributes."),
    ("oracle_hamming_only", "oracle_hamming", "Oracle keeps candidates with non-query Hamming <= 2."),
    ("oracle_query_and_hamming", "oracle_both", "Oracle keeps candidates satisfying both constraints."),
    ("probe_query_only", "probe_query", "Probe keeps candidates satisfying query attributes."),
    ("probe_hamming_only", "probe_hamming", "Probe keeps candidates with predicted non-query Hamming <= 2."),
    ("probe_query_and_hamming", "probe_both", "Probe keeps candidates satisfying both predicted constraints."),
    (
        "probe_query_and_hamming_fill",
        "probe_both_fill",
        "Current best style: promote probe query+Hamming candidates, then fill with q_hybrid ranking.",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=PACKAGE_ROOT / "weights" / "best_val_official_like_at10.pt")
    parser.add_argument(
        "--probe-results",
        type=Path,
        default=PACKAGE_ROOT / "results" / "probe_reranker_v2_calibrated",
    )
    parser.add_argument(
        "--probe-checkpoint",
        type=Path,
        default=PACKAGE_ROOT / "results" / "probe_reranker_v2_calibrated" / "probe" / "best_probe.pt",
    )
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--source-batch-size", type=int, default=256)
    parser.add_argument("--top-pools", nargs="+", type=int, default=[500, 200, 100])
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--beta", type=float, default=1.25)
    parser.add_argument("--max-hamming", type=int, default=2)
    parser.add_argument("--threshold-objective", default="accuracy")
    parser.add_argument("--query-ids", nargs="*", type=int, default=None)
    parser.add_argument("--max-sources-per-query", type=int, default=0)
    return parser.parse_args()


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_gallery_attribute_table(full_filenames: list[str], full_attrs: torch.Tensor, gallery_filenames: list[str]) -> torch.Tensor:
    full_index = {filename: idx for idx, filename in enumerate(full_filenames)}
    missing = [filename for filename in gallery_filenames if filename not in full_index]
    if missing:
        raise RuntimeError(f"Missing {len(missing)} gallery filenames in CelebA attributes")
    return full_attrs[[full_index[filename] for filename in gallery_filenames]]


def query_tensors(conditions: list[tuple[int, str]], attribute_to_index: dict[str, int], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    attrs = torch.tensor([attribute_to_index[attr] for _, attr in conditions], dtype=torch.long, device=device)
    signs = torch.tensor([sign for sign, _ in conditions], dtype=torch.int8, device=device)
    return attrs, signs


def oracle_masks(
    top_indices: torch.Tensor,
    source_indices: torch.Tensor,
    attr_indices: torch.Tensor,
    signs: torch.Tensor,
    gallery_attrs: torch.Tensor,
    max_hamming: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    attrs_device = gallery_attrs.to(device)
    candidate_attrs = attrs_device[top_indices]
    source_attrs = attrs_device[source_indices]

    query_ok = torch.ones(top_indices.shape, dtype=torch.bool, device=device)
    for attr, sign in zip(attr_indices.tolist(), signs.tolist()):
        query_ok &= candidate_attrs[:, :, int(attr)] == int(sign)

    nonquery = torch.ones(gallery_attrs.shape[1], dtype=torch.bool, device=device)
    nonquery[attr_indices.long()] = False
    hamming = (candidate_attrs[:, :, nonquery] != source_attrs[:, None, nonquery]).sum(dim=-1)
    hamming_ok = hamming <= int(max_hamming)
    return query_ok, hamming_ok


def probe_masks(
    top_indices: torch.Tensor,
    source_indices: torch.Tensor,
    attr_indices: torch.Tensor,
    signs: torch.Tensor,
    gallery_probs: torch.Tensor,
    thresholds: torch.Tensor,
    max_hamming: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    probs = gallery_probs.to(device)
    source_probs = probs[source_indices]
    candidate_probs = probs[top_indices]
    thresholds = thresholds.to(device)
    query_ok = probe_v2.calibrated_query_ok(candidate_probs, attr_indices, signs, thresholds)
    hard_ham, _, _ = probe_v2.calibrated_hamming(source_probs, candidate_probs, attr_indices, thresholds)
    return query_ok, hard_ham <= float(max_hamming)


def filtered_rankings(
    top_indices: torch.Tensor,
    mask: torch.Tensor | None,
    top_k: int,
    fill_to_k: bool = False,
) -> tuple[list[list[int]], list[int]]:
    rankings: list[list[int]] = []
    kept_counts: list[int] = []
    for row in range(top_indices.shape[0]):
        if mask is None:
            selected = top_indices[row, :top_k].tolist()
            kept = top_k
        else:
            local = torch.nonzero(mask[row], as_tuple=False).flatten()
            selected = top_indices[row, local[:top_k]].tolist()
            kept = int(local.numel())
            if fill_to_k and len(selected) < top_k:
                selected_set = set(selected)
                for idx in top_indices[row].tolist():
                    if int(idx) not in selected_set:
                        selected.append(int(idx))
                        selected_set.add(int(idx))
                    if len(selected) >= top_k:
                        break
        rankings.append([int(x) for x in selected])
        kept_counts.append(kept)
    return rankings, kept_counts


def update_stats(
    stats: dict[str, Any],
    rankings: list[list[int]],
    kept_counts: list[int],
    top_pool: list[list[int]],
    source_indices: list[int],
    item: dict[str, Any],
) -> None:
    for ranking, kept, raw_pool, source_idx in zip(rankings, kept_counts, top_pool, source_indices):
        valid = set(item["ground_truth"][str(source_idx)])
        stats["sources"] += 1
        stats["kept_sum"] += kept
        stats["pool_hit_sum"] += int(bool(set(raw_pool).intersection(valid)))
        stats["valid_in_kept_sum"] += len(set(ranking).intersection(valid))
        for k in TOP_KS:
            hit, precision = retrieval_metrics(ranking, valid, k)
            stats[f"acc@{k}"] += hit
            stats[f"precision@{k}"] += precision


def make_summary_rows(raw: dict[tuple[int, str], dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for (pool, method), stats in sorted(raw.items(), key=lambda x: (x[0][0], x[0][1])):
        n = max(1, stats["sources"])
        row = {
            "top_pool": pool,
            "method": method,
            "source_query_cases": stats["sources"],
            "avg_kept": stats["kept_sum"] / n,
            "raw_pool_hit_rate": stats["pool_hit_sum"] / n,
            "valid_in_returned_top10_avg": stats["valid_in_kept_sum"] / n,
        }
        for k in TOP_KS:
            row[f"Accuracy@{k}"] = stats[f"acc@{k}"] / n
            row[f"Recall@{k}"] = stats[f"acc@{k}"] / n
            row[f"Precision@{k}"] = stats[f"precision@{k}"] / n
        rows.append(row)
    return rows


def print_nested(rows: list[dict[str, Any]]) -> None:
    by_pool: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_pool[int(row["top_pool"])].append(row)

    order = [
        "best_no_filter",
        "oracle_query_only",
        "oracle_hamming_only",
        "oracle_query_and_hamming",
        "probe_query_only",
        "probe_hamming_only",
        "probe_query_and_hamming",
        "probe_query_and_hamming_fill",
    ]
    labels = {
        "best_no_filter": "best system, no filtering",
        "oracle_query_only": "oracle / solo query",
        "oracle_hamming_only": "oracle / solo hamming",
        "oracle_query_and_hamming": "oracle / query + hamming",
        "probe_query_only": "learned probe / solo query",
        "probe_hamming_only": "learned probe / solo hamming",
        "probe_query_and_hamming": "learned probe / query + hamming",
        "probe_query_and_hamming_fill": "learned probe / query + hamming + fill",
    }

    for pool in sorted(by_pool):
        rows_by_method = {row["method"]: row for row in by_pool[pool]}
        print(f"\nfiltering top {pool}:")
        for method in order:
            row = rows_by_method[method]
            print(f"  - {labels[method]}:")
            print(
                "      "
                f"Acc/Recall@1={row['Accuracy@1']:.4f} "
                f"@5={row['Accuracy@5']:.4f} "
                f"@10={row['Accuracy@10']:.4f} | "
                f"Precision@1={row['Precision@1']:.4f} "
                f"@5={row['Precision@5']:.4f} "
                f"@10={row['Precision@10']:.4f} | "
                f"avg_kept={row['avg_kept']:.1f}"
            )


def main() -> int:
    args = parse_args()
    output_root = args.output_root or (PACKAGE_ROOT / "results" / "filtering_matrix" / f"filtering_matrix_{timestamp()}")
    output_root.mkdir(parents=True, exist_ok=True)
    max_pool = max(args.top_pools)
    if max_pool < args.top_k:
        raise ValueError("max(top_pools) must be >= top_k")

    device = choose_device(args.device)
    print(f"device={device}")
    print(f"output={output_root}")

    model, checkpoint_data = load_model_checkpoint(args.checkpoint, device)
    model.eval()
    config = checkpoint_data["config"]
    prompt_cache = load_prompt_embedding_cache(blends.router.resolve_project_path(config.get("prompt_cache_path")))
    text_bank, _ = blends.router.load_text_banks(device)

    attributes, full_filenames, full_attrs = read_attribute_table()
    attribute_to_index = {name: idx for idx, name in enumerate(attributes)}
    annotations = load_evaluation()

    gallery_cache = load_torch(EMBEDDING_DIR / "test_image_embeddings.pt")
    if gallery_cache.get("model_id") != MODEL_ID:
        raise RuntimeError(f"Unexpected embedding model: {gallery_cache.get('model_id')}")
    gallery = F.normalize(gallery_cache["embeddings"].float(), dim=-1).to(device)
    gallery_attrs = build_gallery_attribute_table(full_filenames, full_attrs, gallery_cache["filenames"])

    probe_probs_path = args.probe_results / "probe" / "test_probe_probs.pt"
    threshold_path = args.probe_results / "probe" / "calibrated_thresholds.pt"
    if not probe_probs_path.exists():
        raise FileNotFoundError(f"Missing probe probabilities: {probe_probs_path}")
    if not threshold_path.exists():
        raise FileNotFoundError(f"Missing calibrated thresholds: {threshold_path}")
    gallery_probs = load_torch(probe_probs_path)["probs"].float()
    thresholds = load_torch(threshold_path)["thresholds"][args.threshold_objective].float()

    query_ids = list(range(len(annotations))) if args.query_ids is None else args.query_ids
    raw_stats: dict[tuple[int, str], dict[str, Any]] = defaultdict(lambda: defaultdict(float))
    started = time.monotonic()

    with torch.inference_mode():
        for query_counter, query_id in enumerate(query_ids, start=1):
            item = annotations[query_id]
            conditions = parse_query(item["query"])
            attr_indices, signs = query_tensors(conditions, attribute_to_index, device)
            source_indices_all = [int(idx) for idx in item["ground_truth"]]
            if args.max_sources_per_query:
                source_indices_all = source_indices_all[: args.max_sources_per_query]

            for start in range(0, len(source_indices_all), args.source_batch_size):
                end = min(start + args.source_batch_size, len(source_indices_all))
                batch_sources = source_indices_all[start:end]
                idx = torch.tensor(batch_sources, dtype=torch.long, device=device)
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
                q_sum = blends.generic_sum_query(source, conditions, attribute_to_index, text_bank)
                q_final = blends.vector_delta(q_model, q_sum, source, args.beta)
                scores = q_final @ gallery.T
                rows = torch.arange(len(batch_sources), device=device)
                scores[rows, idx] = -torch.inf
                _, top_indices_500 = scores.topk(max_pool, dim=1)

                oracle_query_500, oracle_hamming_500 = oracle_masks(
                    top_indices_500,
                    idx,
                    attr_indices,
                    signs,
                    gallery_attrs,
                    args.max_hamming,
                    device,
                )
                probe_query_500, probe_hamming_500 = probe_masks(
                    top_indices_500,
                    idx,
                    attr_indices,
                    signs,
                    gallery_probs,
                    thresholds,
                    args.max_hamming,
                    device,
                )

                for pool in args.top_pools:
                    top_indices = top_indices_500[:, :pool]
                    raw_pool_lists = top_indices.cpu().tolist()
                    masks = {
                        "best_no_filter": None,
                        "oracle_query_only": oracle_query_500[:, :pool],
                        "oracle_hamming_only": oracle_hamming_500[:, :pool],
                        "oracle_query_and_hamming": oracle_query_500[:, :pool] & oracle_hamming_500[:, :pool],
                        "probe_query_only": probe_query_500[:, :pool],
                        "probe_hamming_only": probe_hamming_500[:, :pool],
                        "probe_query_and_hamming": probe_query_500[:, :pool] & probe_hamming_500[:, :pool],
                        "probe_query_and_hamming_fill": probe_query_500[:, :pool] & probe_hamming_500[:, :pool],
                    }
                    for method, mask in masks.items():
                        rankings, kept = filtered_rankings(
                            top_indices,
                            mask,
                            args.top_k,
                            fill_to_k=method.endswith("_fill"),
                        )
                        update_stats(raw_stats[(pool, method)], rankings, kept, raw_pool_lists, batch_sources, item)

                print(
                    f"query {query_counter}/{len(query_ids)} source {end}/{len(source_indices_all)} "
                    f"elapsed={time.monotonic() - started:.1f}s",
                    flush=True,
                )

    rows = make_summary_rows(raw_stats)
    write_csv(output_root / "summary.csv", rows)
    (output_root / "config.json").write_text(json.dumps(vars(args), indent=2, default=str) + "\n", encoding="utf-8")
    print_nested(rows)
    print(f"\nSaved: {output_root / 'summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
