#!/usr/bin/env python3
"""Oracle ablation: query constraint vs non-query Hamming constraint.

Diagnostic only. This uses CelebA ground-truth attributes at inference time.

Goal:
Given the same q_final top-pool, test which oracle filtering component is most
useful:

1. query-only: keep candidates satisfying requested +/- attributes;
2. hamming-only: keep candidates preserving non-query attributes with Hamming<=2;
3. query+hamming: official-style oracle filter;
4. no filter: raw q_final baseline.

This helps decide whether a learned/probe filter should focus more on query
satisfaction, preservation/Hamming, or both.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[2]
PROJECT_DATA_ROOT = ROOT / "final_best_system"
CODE_ROOT = ROOT / "final_best_system" / "code"
SCRIPTS = CODE_ROOT / "scripts"
ORCH = CODE_ROOT / "orchestrator"

os.environ["DL_PROJECT_ROOT"] = str(PROJECT_DATA_ROOT)
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(ORCH))

import evaluate_sum_model_blends as blends  # noqa: E402
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


DEFAULT_CHECKPOINT = ROOT / "final_best_system" / "weights" / "best_val_official_like_at10.pt"
DEFAULT_PROMPT_CACHE = ROOT / "final_best_system" / "embeddings" / "signed_attribute_prompt_embeddings_v2_photo_templates.pt"
DEFAULT_OUTPUT_DIR = ROOT / "final_best_system" / "results" / "oracle_filter_component_ablation"


METHODS = {
    "baseline_q_final": "No oracle filtering; use q_final top-k directly.",
    "oracle_query_only": "Keep candidates satisfying requested query attributes only.",
    "oracle_hamming_only": "Keep candidates with non-query Hamming<=max_hamming only.",
    "oracle_query_and_hamming": "Keep candidates satisfying both query and non-query Hamming constraints.",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--prompt-cache", type=Path, default=DEFAULT_PROMPT_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--source-batch-size", type=int, default=128)
    parser.add_argument("--top-pool", type=int, default=500)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--beta", type=float, default=1.25)
    parser.add_argument("--max-hamming", type=int, default=2)
    parser.add_argument("--query-ids", nargs="*", type=int, default=None)
    parser.add_argument("--max-sources-per-query", type=int, default=0)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_gallery_attribute_table(
    full_filenames: list[str],
    full_attrs: torch.Tensor,
    gallery_filenames: list[str],
) -> torch.Tensor:
    full_index = {filename: idx for idx, filename in enumerate(full_filenames)}
    missing = [filename for filename in gallery_filenames if filename not in full_index]
    if missing:
        raise RuntimeError(f"Missing {len(missing)} gallery filenames in attribute table")
    return full_attrs[[full_index[filename] for filename in gallery_filenames]]


def filter_pool_components(
    pool_indices: list[int],
    source_index: int,
    conditions: list[tuple[int, str]],
    attribute_to_index: dict[str, int],
    gallery_attrs: torch.Tensor,
    max_hamming: int,
) -> dict[str, list[int]]:
    if not pool_indices:
        return {method: [] for method in METHODS}

    candidate_tensor = torch.tensor(pool_indices, dtype=torch.long)
    candidate_attrs = gallery_attrs[candidate_tensor]
    source_attrs = gallery_attrs[source_index]

    query_attr_indices: list[int] = []
    query_ok = torch.ones(len(pool_indices), dtype=torch.bool)
    for sign, attribute in conditions:
        attr_idx = attribute_to_index[attribute]
        query_attr_indices.append(attr_idx)
        query_ok &= candidate_attrs[:, attr_idx] == int(sign)

    nonquery_mask = torch.ones(gallery_attrs.shape[1], dtype=torch.bool)
    nonquery_mask[query_attr_indices] = False
    hamming = (candidate_attrs[:, nonquery_mask] != source_attrs[nonquery_mask]).sum(dim=1)
    hamming_ok = hamming <= int(max_hamming)

    masks = {
        "baseline_q_final": torch.ones(len(pool_indices), dtype=torch.bool),
        "oracle_query_only": query_ok,
        "oracle_hamming_only": hamming_ok,
        "oracle_query_and_hamming": query_ok & hamming_ok,
    }
    return {
        method: [int(idx) for idx, keep in zip(pool_indices, mask.tolist()) if keep]
        for method, mask in masks.items()
    }


def summarize_method(
    annotations: list[dict[str, Any]],
    records_by_query: dict[int, list[dict[str, Any]]],
    method: str,
    checkpoint: Path,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    per_query: list[dict[str, Any]] = []
    for query_id, records in sorted(records_by_query.items()):
        if not records:
            continue
        item = annotations[query_id]
        totals = {f"Recall@{k}": 0.0 for k in TOP_KS}
        totals.update({f"Precision@{k}": 0.0 for k in TOP_KS})
        filtered_sum = 0
        valid_in_filtered_sum = 0
        filtered_hit_sum = 0
        raw_pool_hit_sum = 0

        for record in records:
            valid_targets = set(item["ground_truth"][str(record["source_index"])])
            filtered = record["filtered_pool"]
            topk = filtered[: args.top_k]
            raw_pool = record["top_pool"]
            for k in TOP_KS:
                recall, precision = retrieval_metrics(topk, valid_targets, k)
                totals[f"Recall@{k}"] += recall
                totals[f"Precision@{k}"] += precision
            filtered_sum += len(filtered)
            valid_in_filtered_sum += len(set(filtered).intersection(valid_targets))
            filtered_hit_sum += int(bool(set(filtered).intersection(valid_targets)))
            raw_pool_hit_sum += int(bool(set(raw_pool).intersection(valid_targets)))

        source_count = len(records)
        per_query.append(
            {
                "method": method,
                "query_id": query_id,
                "query": item["query"],
                "sources": source_count,
                "avg_filtered_candidates_in_pool": filtered_sum / source_count,
                "avg_official_valid_candidates_in_filtered_pool": valid_in_filtered_sum / source_count,
                "filtered_pool_hit_rate": filtered_hit_sum / source_count,
                "raw_pool_hit_rate": raw_pool_hit_sum / source_count,
                **{name: value / source_count for name, value in totals.items()},
            }
        )

    metric_names = [name for name in per_query[0] if "@" in name]
    total_sources = sum(row["sources"] for row in per_query)
    summary = {
        "method": method,
        "description": METHODS[method],
        "checkpoint": str(checkpoint),
        "model_id": MODEL_ID,
        "top_pool": args.top_pool,
        "top_k": args.top_k,
        "beta": args.beta,
        "max_hamming": args.max_hamming,
        "query_entries": len(per_query),
        "source_query_cases": total_sources,
        "macro_avg_filtered_candidates_in_pool": sum(row["avg_filtered_candidates_in_pool"] for row in per_query) / len(per_query),
        "macro_avg_official_valid_candidates_in_filtered_pool": sum(
            row["avg_official_valid_candidates_in_filtered_pool"] for row in per_query
        )
        / len(per_query),
        "macro_filtered_pool_hit_rate": sum(row["filtered_pool_hit_rate"] for row in per_query) / len(per_query),
        "macro_raw_pool_hit_rate": sum(row["raw_pool_hit_rate"] for row in per_query) / len(per_query),
        **{f"macro_{name}": sum(row[name] for row in per_query) / len(per_query) for name in metric_names},
        **{f"micro_{name}": sum(row[name] * row["sources"] for row in per_query) / total_sources for name in metric_names},
    }
    return summary, per_query


def write_outputs(
    output_dir: Path,
    annotations: list[dict[str, Any]],
    all_records: dict[str, dict[int, list[dict[str, Any]]]],
    checkpoint: Path,
    args: argparse.Namespace,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    combined_summary = []
    combined_per_query = []
    for method, records_by_query in all_records.items():
        method_dir = output_dir / method
        method_dir.mkdir(parents=True, exist_ok=True)
        summary, per_query = summarize_method(annotations, records_by_query, method, checkpoint, args)
        combined_summary.append(summary)
        combined_per_query.extend(per_query)
        write_csv(method_dir / "summary.csv", [summary])
        write_csv(method_dir / "per_query_metrics.csv", per_query)
        with (method_dir / "retrievals.jsonl").open("w", encoding="utf-8") as handle:
            for records in records_by_query.values():
                for record in records:
                    output_record = dict(record)
                    output_record["top10"] = output_record["filtered_pool"][: args.top_k]
                    handle.write(json.dumps(output_record) + "\n")

    combined_summary.sort(key=lambda row: row["macro_Recall@10"], reverse=True)
    write_csv(output_dir / "combined_summary.csv", combined_summary)
    write_csv(output_dir / "combined_per_query_metrics.csv", combined_per_query)
    (output_dir / "config.json").write_text(json.dumps(vars(args), indent=2, default=str) + "\n", encoding="utf-8")
    (output_dir / "README.md").write_text(
        "\n".join(
            [
                "# Oracle Filter Component Ablation",
                "",
                "Diagnostic only. This uses CelebA ground-truth attributes at inference time.",
                "",
                "Methods:",
                *[f"- `{name}`: {desc}" for name, desc in METHODS.items()],
                "",
                f"top_pool: {args.top_pool}",
                f"beta: {args.beta}",
                f"max_hamming: {args.max_hamming}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Saved oracle component ablation in {output_dir}")
    print(json.dumps(combined_summary, indent=2))


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir / f"top{args.top_pool}_beta{str(args.beta).replace('.', 'p')}_h{args.max_hamming}"
    device = choose_device(args.device)

    model, checkpoint_data = load_model_checkpoint(args.checkpoint, device)
    model.eval()
    config = checkpoint_data["config"]

    prompt_cache = load_prompt_embedding_cache(args.prompt_cache)
    text_bank, _ = blends.router.load_text_banks(device)
    attributes, full_filenames, full_attrs = read_attribute_table()
    attribute_to_index = {name: idx for idx, name in enumerate(attributes)}
    annotations = load_evaluation()

    gallery_cache = load_torch(EMBEDDING_DIR / "test_image_embeddings.pt")
    gallery = blends.normalize(gallery_cache["embeddings"]).to(device)
    gallery_filenames = list(gallery_cache["filenames"])
    gallery_attrs = build_gallery_attribute_table(full_filenames, full_attrs, gallery_filenames)

    query_ids = list(range(len(annotations))) if args.query_ids is None else args.query_ids
    all_records: dict[str, dict[int, list[dict[str, Any]]]] = {method: {} for method in METHODS}

    with torch.inference_mode():
        for query_id in query_ids:
            item = annotations[query_id]
            conditions = parse_query(item["query"])
            source_indices = [int(index) for index in item["ground_truth"]]
            if args.max_sources_per_query:
                source_indices = source_indices[: args.max_sources_per_query]
            for start in range(0, len(source_indices), args.source_batch_size):
                end = min(start + args.source_batch_size, len(source_indices))
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
                q_final = blends.vector_delta(q_model, q_generic, source, args.beta)

                scores = q_final @ gallery.T
                rows = torch.arange(len(batch_indices), device=device)
                scores[rows, idx] = -torch.inf
                pools = scores.topk(args.top_pool, dim=1).indices.cpu().tolist()

                for source_index, pool_indices in zip(batch_indices, pools):
                    filtered_by_method = filter_pool_components(
                        pool_indices,
                        source_index,
                        conditions,
                        attribute_to_index,
                        gallery_attrs,
                        args.max_hamming,
                    )
                    for method, filtered in filtered_by_method.items():
                        records = all_records[method].setdefault(query_id, [])
                        records.append(
                            {
                                "method": method,
                                "query_id": query_id,
                                "query": item["query"],
                                "source_index": source_index,
                                "top_pool_size": args.top_pool,
                                "top_pool": pool_indices,
                                "filtered_count_in_pool": len(filtered),
                                "filtered_pool": filtered,
                            }
                        )
                print(f"query {query_id + 1}/{len(annotations)} sources {end}/{len(source_indices)}")

    write_outputs(output_dir, annotations, all_records, args.checkpoint, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
