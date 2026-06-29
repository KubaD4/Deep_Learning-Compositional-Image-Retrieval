#!/usr/bin/env python3
"""Oracle top-pool reranking experiment for the final system.

This is intentionally NOT the final fair retrieval system.

It tests the hypothesis:

    q_final is good enough to retrieve a useful broad candidate pool, but the
    top-10 is noisy because cosine neighbours do not perfectly match the
    official CelebA attribute/Hamming constraints.

For each official JSON source/query:

1. compute the final learned query vector:
       q_final = normalize(q_model + beta * (q_sum - source))
2. retrieve top-N candidates by cosine similarity, e.g. top 500;
3. keep only candidates that satisfy the official-style constraints:
       requested query attributes match;
       non-query attribute Hamming distance from source <= max_hamming;
4. report Recall@K/Precision@K against the official JSON.

Because step 3 uses ground-truth CelebA attributes at inference time, this is an
analysis upper bound / diagnostic, not a deployable general method.
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
DEFAULT_PROMPT_CACHE = (
    ROOT
    / "final_best_system"
    / "embeddings"
    / "signed_attribute_prompt_embeddings_v2_photo_templates.pt"
)
DEFAULT_OUTPUT_DIR = (
    ROOT
    / "final_best_system"
    / "results"
    / "oracle_top500_hamming_filter"
    / "model_plus_generic_delta_beta_1p50"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--prompt-cache", type=Path, default=DEFAULT_PROMPT_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--source-batch-size", type=int, default=128)
    parser.add_argument("--top-pool", type=int, default=500)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--beta", type=float, default=1.5)
    parser.add_argument("--max-hamming", type=int, default=2)
    parser.add_argument(
        "--query-ids",
        nargs="*",
        type=int,
        default=None,
        help="Optional subset of official query ids. Default: all queries.",
    )
    parser.add_argument(
        "--source-index",
        type=int,
        default=None,
        help="Optional single test source index. Useful for debugging one JSON case.",
    )
    parser.add_argument(
        "--max-sources-per-query",
        type=int,
        default=0,
        help="Optional smoke-test cap. 0 means evaluate all sources.",
    )
    parser.add_argument(
        "--fill-to-k",
        action="store_true",
        help=(
            "After oracle filtering, fill missing slots with original cosine candidates. "
            "Default keeps only filtered candidates, so missing slots count as precision errors."
        ),
    )
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
    rows = [full_index[filename] for filename in gallery_filenames]
    return full_attrs[rows]


def official_like_filter(
    candidate_indices: list[int],
    source_index: int,
    conditions: list[tuple[int, str]],
    attribute_to_index: dict[str, int],
    gallery_attrs: torch.Tensor,
    max_hamming: int,
) -> tuple[list[int], list[int]]:
    """Return candidates satisfying query attrs and non-query Hamming <= threshold."""
    if not candidate_indices:
        return [], []

    candidate_tensor = torch.tensor(candidate_indices, dtype=torch.long)
    candidate_attrs = gallery_attrs[candidate_tensor]
    source_attrs = gallery_attrs[source_index]

    valid = torch.ones(len(candidate_indices), dtype=torch.bool)
    query_attr_indices: list[int] = []
    for sign, attribute in conditions:
        attr_idx = attribute_to_index[attribute]
        query_attr_indices.append(attr_idx)
        valid &= candidate_attrs[:, attr_idx] == int(sign)

    nonquery_mask = torch.ones(gallery_attrs.shape[1], dtype=torch.bool)
    nonquery_mask[query_attr_indices] = False
    hamming = (candidate_attrs[:, nonquery_mask] != source_attrs[nonquery_mask]).sum(dim=1)
    valid &= hamming <= int(max_hamming)

    filtered: list[int] = []
    filtered_hamming: list[int] = []
    for idx, is_valid, distance in zip(candidate_indices, valid.tolist(), hamming.tolist()):
        if is_valid:
            filtered.append(int(idx))
            filtered_hamming.append(int(distance))
    return filtered, filtered_hamming


def finalize(
    annotations: list[dict],
    records_by_query: dict[int, list[dict[str, Any]]],
    output_dir: Path,
    checkpoint: Path,
    args: argparse.Namespace,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    retrieval_path = output_dir / "retrievals.jsonl"
    per_query: list[dict[str, Any]] = []

    with retrieval_path.open("w", encoding="utf-8") as retrieval_file:
        for query_id, records in sorted(records_by_query.items()):
            if not records:
                continue
            item = annotations[query_id]
            totals = {f"Recall@{k}": 0.0 for k in TOP_KS}
            totals.update({f"Precision@{k}": 0.0 for k in TOP_KS})
            total_filtered_in_pool = 0
            total_pool_hits = 0

            for record in records:
                retrieval_file.write(json.dumps(record) + "\n")
                valid_targets = set(item["ground_truth"][str(record["source_index"])])
                for k in TOP_KS:
                    recall, precision = retrieval_metrics(record["top10"], valid_targets, k)
                    totals[f"Recall@{k}"] += recall
                    totals[f"Precision@{k}"] += precision
                total_filtered_in_pool += int(record["filtered_count_in_pool"])
                total_pool_hits += int(bool(set(record["top_pool"]).intersection(valid_targets)))

            source_count = len(records)
            per_query.append(
                {
                    "query_id": query_id,
                    "query": item["query"],
                    "sources": source_count,
                    "avg_filtered_candidates_in_pool": total_filtered_in_pool / source_count,
                    "pool_hit_rate": total_pool_hits / source_count,
                    **{name: value / source_count for name, value in totals.items()},
                }
            )

    write_csv(output_dir / "per_query_metrics.csv", per_query)
    metric_names = [name for name in per_query[0] if "@" in name]
    total_sources = sum(row["sources"] for row in per_query)
    summary = {
        "method": "oracle_top_pool_hamming_filter",
        "checkpoint": str(checkpoint),
        "model_id": MODEL_ID,
        "top_pool": args.top_pool,
        "top_k": args.top_k,
        "beta": args.beta,
        "max_hamming": args.max_hamming,
        "fill_to_k": args.fill_to_k,
        "query_entries": len(per_query),
        "source_query_cases": total_sources,
        "macro_pool_hit_rate": sum(row["pool_hit_rate"] for row in per_query) / len(per_query),
        "macro_avg_filtered_candidates_in_pool": sum(
            row["avg_filtered_candidates_in_pool"] for row in per_query
        )
        / len(per_query),
        **{
            f"macro_{name}": sum(row[name] for row in per_query) / len(per_query)
            for name in metric_names
        },
        **{
            f"micro_{name}": sum(row[name] * row["sources"] for row in per_query)
            / total_sources
            for name in metric_names
        },
    }
    write_csv(output_dir / "summary.csv", [summary])
    (output_dir / "config.json").write_text(json.dumps(vars(args), indent=2, default=str))
    (output_dir / "README.md").write_text(
        "\n".join(
            [
                "# Oracle Top-Pool Hamming Filter",
                "",
                "Diagnostic only. This uses CelebA ground-truth attributes at inference time.",
                "It is an upper-bound test for candidate-pool quality, not the final fair system.",
                "",
                f"- top_pool: {args.top_pool}",
                f"- beta: {args.beta}",
                f"- max_hamming: {args.max_hamming}",
                f"- fill_to_k: {args.fill_to_k}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Saved oracle rerank results in {output_dir}")
    print(json.dumps(summary, indent=2))


def main() -> int:
    args = parse_args()
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
    records_by_query: dict[int, list[dict[str, Any]]] = {}

    with torch.inference_mode():
        for query_id in query_ids:
            item = annotations[query_id]
            conditions = parse_query(item["query"])
            source_indices = [int(index) for index in item["ground_truth"]]
            if args.source_index is not None:
                source_indices = [idx for idx in source_indices if idx == args.source_index]
                if not source_indices:
                    print(f"Skipping query {query_id}: source_index={args.source_index} not present")
                    continue
            if args.max_sources_per_query:
                source_indices = source_indices[: args.max_sources_per_query]

            rows_for_query: list[dict[str, Any]] = records_by_query.setdefault(query_id, [])
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
                pool = scores.topk(args.top_pool, dim=1).indices.cpu().tolist()

                for source_index, pool_indices in zip(batch_indices, pool):
                    filtered, filtered_hamming = official_like_filter(
                        pool_indices,
                        source_index,
                        conditions,
                        attribute_to_index,
                        gallery_attrs,
                        args.max_hamming,
                    )
                    top_filtered = filtered[: args.top_k]
                    if args.fill_to_k and len(top_filtered) < args.top_k:
                        already = set(top_filtered)
                        top_filtered.extend(
                            idx
                            for idx in pool_indices
                            if idx not in already
                        )
                        top_filtered = top_filtered[: args.top_k]

                    rows_for_query.append(
                        {
                            "method": "oracle_top_pool_hamming_filter",
                            "query_id": query_id,
                            "query": item["query"],
                            "source_index": source_index,
                            "top_pool_size": args.top_pool,
                            "filtered_count_in_pool": len(filtered),
                            "top_pool": pool_indices,
                            "top10_before_filter": pool_indices[: args.top_k],
                            "top10": top_filtered,
                            "top10_filtered_hamming": filtered_hamming[: args.top_k],
                        }
                    )
                print(f"query {query_id + 1}/{len(annotations)} sources {end}/{len(source_indices)}")

    finalize(annotations, records_by_query, args.output_dir, args.checkpoint, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
