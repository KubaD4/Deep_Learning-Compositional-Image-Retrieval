#!/usr/bin/env python3
"""Build train/valid official-like multi-positive sets without JSON leakage.

This v6 builder intentionally does not read ``celeba_evaluation.json``. It
uses only a CelebA split, its attributes, identities, and frozen CLIP image
embeddings. For each source image and signed query, it finds multiple target
images in the same split that satisfy the official-style rule:

    query attributes match the requested signs
    non-query Hamming distance from source <= threshold

The output is a row-wise pair index compatible with the existing trainers, plus
a ``group_ids`` tensor. Rows with the same group id share the same
source/query and represent multiple valid positives.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))
os.environ.setdefault("DL_PROJECT_ROOT", str(ROOT))

from learned_gate_core import load_image_embedding_cache  # noqa: E402
from project_core import (  # noqa: E402
    ARTIFACTS_DIR,
    atomic_json_dump,
    atomic_torch_save,
    choose_device,
    read_attribute_table,
    read_identity_map,
)


OFFICIAL_STYLE_QUERIES = [
    "+Smiling",
    "+Eyeglasses",
    "-Heavy_Makeup",
    "+Male",
    "-Young",
    "+Blond_Hair",
    "+Mustache",
    "+Eyeglasses,+Smiling",
    "+Black_Hair,-Wavy_Hair",
    "-Male,-Mustache",
    "+Chubby,-Young",
    "-Smiling,+Eyeglasses,+Wearing_Hat",
    "+Wearing_Lipstick,-Heavy_Makeup,+Smiling",
]

WEAK_GLOBAL_QUERIES = [
    "+Male",
    "-Male",
    "+Young",
    "-Young",
    "+Chubby",
    "-Chubby",
    "+Male,+Chubby",
    "+Male,-Young",
    "+Chubby,-Young",
    "-Male,-Mustache",
]


def default_output(split: str, preset: str, max_hamming: int, top_targets: int) -> Path:
    return (
        ARTIFACTS_DIR
        / "training_pairs"
        / f"official_like_v6_{preset}_{split}_h{max_hamming}_top{top_targets}.pt"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("train", "valid"), required=True)
    parser.add_argument("--preset", choices=("official", "weak", "both"), default="official")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--queries", nargs="+")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--max-query-len", type=int, default=3)
    parser.add_argument("--max-hamming-other", type=int, default=2)
    parser.add_argument("--top-targets-per-source-query", type=int, default=4)
    parser.add_argument("--max-sources-per-query", type=int, default=20000)
    parser.add_argument("--source-chunk-size", type=int, default=128)
    parser.add_argument("--candidate-chunk-size", type=int, default=4096)
    parser.add_argument("--hamming-penalty", type=float, default=0.02)
    parser.add_argument("--exclude-same-identity", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--source-must-need-edit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.output is None:
        args.output = default_output(
            args.split,
            args.preset,
            args.max_hamming_other,
            args.top_targets_per_source_query,
        )
    return args


def query_list(args: argparse.Namespace) -> list[str]:
    if args.queries:
        return args.queries
    if args.preset == "official":
        return OFFICIAL_STYLE_QUERIES
    if args.preset == "weak":
        return WEAK_GLOBAL_QUERIES
    merged = list(dict.fromkeys(OFFICIAL_STYLE_QUERIES + WEAK_GLOBAL_QUERIES))
    return merged


def parse_query(query: str, attr_to_index: dict[str, int]) -> tuple[list[int], list[int]]:
    attrs: list[int] = []
    signs: list[int] = []
    for raw in query.split(","):
        token = raw.strip()
        if not token or token[0] not in "+-":
            raise ValueError(f"Bad query token: {raw!r}")
        attr = token[1:].strip()
        if attr not in attr_to_index:
            raise ValueError(f"Unknown attribute {attr!r} in query {query!r}")
        attrs.append(attr_to_index[attr])
        signs.append(1 if token[0] == "+" else -1)
    return attrs, signs


def load_split_metadata(split: str) -> tuple[list[str], torch.Tensor, torch.Tensor, list[str]]:
    attributes, all_filenames, all_attrs = read_attribute_table()
    attr_by_file = {filename: all_attrs[index] for index, filename in enumerate(all_filenames)}
    identities = read_identity_map()
    cache = load_image_embedding_cache(split)
    filenames = list(cache["filenames"])
    attrs = torch.stack([attr_by_file[filename] for filename in filenames]).to(torch.int8)
    identity_tensor = torch.tensor([identities[filename] for filename in filenames], dtype=torch.long)
    return filenames, attrs, identity_tensor, attributes


def topk_merge(
    old_scores: torch.Tensor,
    old_indices: torch.Tensor,
    old_hamming: torch.Tensor,
    old_cosine: torch.Tensor,
    new_scores: torch.Tensor,
    new_indices: torch.Tensor,
    new_hamming: torch.Tensor,
    new_cosine: torch.Tensor,
    keep: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    scores = torch.cat([old_scores, new_scores], dim=1)
    indices = torch.cat([old_indices, new_indices], dim=1)
    hamming = torch.cat([old_hamming, new_hamming], dim=1)
    cosine = torch.cat([old_cosine, new_cosine], dim=1)
    values, positions = scores.topk(keep, dim=1)
    return (
        values,
        indices.gather(1, positions),
        hamming.gather(1, positions),
        cosine.gather(1, positions),
    )


def build_for_query(
    query_text: str,
    query_attrs: list[int],
    query_signs: list[int],
    split_attrs: torch.Tensor,
    identities: torch.Tensor,
    embeddings: torch.Tensor,
    next_group_id: int,
    args: argparse.Namespace,
) -> tuple[dict[str, list], int]:
    device = embeddings.device
    attr_ids = torch.tensor(query_attrs, dtype=torch.long)
    sign_tensor = torch.tensor(query_signs, dtype=torch.int8)

    query_attr_values = split_attrs[:, attr_ids]
    candidate_mask = (query_attr_values == sign_tensor).all(dim=1)
    if args.source_must_need_edit:
        source_mask = (query_attr_values != sign_tensor).any(dim=1)
    else:
        source_mask = torch.ones(len(split_attrs), dtype=torch.bool)

    candidate_indices = torch.nonzero(candidate_mask).flatten()
    source_indices = torch.nonzero(source_mask).flatten()
    if len(candidate_indices) == 0 or len(source_indices) == 0:
        return empty_result(), next_group_id

    if args.max_sources_per_query and len(source_indices) > args.max_sources_per_query:
        generator = torch.Generator().manual_seed(args.seed + sum(query_attrs) + len(query_attrs) * 997)
        order = torch.randperm(len(source_indices), generator=generator)[: args.max_sources_per_query]
        source_indices = source_indices[order]

    nonquery_mask = torch.ones(split_attrs.shape[1], dtype=torch.bool)
    nonquery_mask[attr_ids] = False
    padded_attr = torch.full((args.max_query_len,), -1, dtype=torch.long)
    padded_sign = torch.zeros((args.max_query_len,), dtype=torch.int8)
    padded_attr[: len(query_attrs)] = attr_ids
    padded_sign[: len(query_signs)] = sign_tensor

    result = empty_result()
    keep = int(args.top_targets_per_source_query)
    split_attrs_device = split_attrs.to(device)
    nonquery_mask_device = nonquery_mask.to(device)
    identities_device = identities.to(device)
    candidate_indices_device = candidate_indices.to(device)

    for start in range(0, len(source_indices), args.source_chunk_size):
        src_cpu = source_indices[start : start + args.source_chunk_size]
        src = src_cpu.to(device)
        src_emb = embeddings[src]
        src_attr = split_attrs_device[src]
        src_identity = identities_device[src]

        best_scores = torch.full((len(src), keep), -torch.inf, device=device)
        best_indices = torch.full((len(src), keep), -1, dtype=torch.long, device=device)
        best_hamming = torch.full((len(src), keep), 10_000, dtype=torch.long, device=device)
        best_cosine = torch.full((len(src), keep), -torch.inf, device=device)

        for cand_start in range(0, len(candidate_indices), args.candidate_chunk_size):
            cand = candidate_indices_device[cand_start : cand_start + args.candidate_chunk_size]
            if len(cand) == 0:
                continue
            cand_emb = embeddings[cand]
            cand_attr = split_attrs_device[cand]
            cosine = src_emb @ cand_emb.T
            hamming = (
                src_attr[:, None, nonquery_mask_device]
                != cand_attr[None, :, nonquery_mask_device]
            ).sum(dim=2)
            valid = hamming <= int(args.max_hamming_other)
            valid &= src[:, None] != cand[None, :]
            if args.exclude_same_identity:
                valid &= src_identity[:, None] != identities_device[cand][None, :]

            scores = cosine - float(args.hamming_penalty) * hamming.float()
            scores = scores.masked_fill(~valid, -torch.inf)
            local_k = min(keep, scores.shape[1])
            local_scores, local_pos = scores.topk(local_k, dim=1)
            local_indices = cand[local_pos]
            local_hamming = hamming.gather(1, local_pos)
            local_cosine = cosine.gather(1, local_pos)
            if local_k < keep:
                pad = keep - local_k
                local_scores = torch.cat([local_scores, torch.full((len(src), pad), -torch.inf, device=device)], dim=1)
                local_indices = torch.cat([local_indices, torch.full((len(src), pad), -1, dtype=torch.long, device=device)], dim=1)
                local_hamming = torch.cat([local_hamming, torch.full((len(src), pad), 10_000, dtype=torch.long, device=device)], dim=1)
                local_cosine = torch.cat([local_cosine, torch.full((len(src), pad), -torch.inf, device=device)], dim=1)

            best_scores, best_indices, best_hamming, best_cosine = topk_merge(
                best_scores,
                best_indices,
                best_hamming,
                best_cosine,
                local_scores,
                local_indices,
                local_hamming,
                local_cosine,
                keep,
            )

        src_list = src_cpu.tolist()
        scores_cpu = best_scores.cpu()
        indices_cpu = best_indices.cpu()
        hamming_cpu = best_hamming.cpu()
        cosine_cpu = best_cosine.cpu()
        for row, source_index in enumerate(src_list):
            finite_cols = [
                col
                for col in range(keep)
                if torch.isfinite(scores_cpu[row, col]) and int(indices_cpu[row, col]) >= 0
            ]
            if not finite_cols:
                continue
            group_id = next_group_id
            next_group_id += 1
            for rank, col in enumerate(finite_cols):
                result["source_indices"].append(int(source_index))
                result["target_indices"].append(int(indices_cpu[row, col]))
                result["attr_indices"].append(padded_attr.clone())
                result["signs"].append(padded_sign.clone())
                result["query_lengths"].append(len(query_attrs))
                result["group_ids"].append(group_id)
                result["positive_rank"].append(rank)
                result["nonquery_hamming"].append(int(hamming_cpu[row, col]))
                result["target_cosine"].append(float(cosine_cpu[row, col]))
                result["query"].append(query_text)

        print(
            f"{query_text}: sources {min(start + args.source_chunk_size, len(source_indices))}/"
            f"{len(source_indices)} rows={len(result['source_indices'])} groups={next_group_id}",
            flush=True,
        )

    return result, next_group_id


def empty_result() -> dict[str, list]:
    return {
        "source_indices": [],
        "target_indices": [],
        "attr_indices": [],
        "signs": [],
        "query_lengths": [],
        "group_ids": [],
        "positive_rank": [],
        "nonquery_hamming": [],
        "target_cosine": [],
        "query": [],
    }


def extend_result(total: dict[str, list], partial: dict[str, list]) -> None:
    for key, values in partial.items():
        total[key].extend(values)


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.output.exists() and not args.force:
        print(f"Reusing existing official-like set index: {args.output}")
        return 0

    device = choose_device(args.device)
    filenames, attrs, identities, attributes = load_split_metadata(args.split)
    cache = load_image_embedding_cache(args.split)
    embeddings = F.normalize(cache["embeddings"].float(), dim=-1).to(device)
    attr_to_index = {name: index for index, name in enumerate(attributes)}

    built = empty_result()
    query_summaries = []
    next_group_id = 0
    for query_text in query_list(args):
        query_attrs, query_signs = parse_query(query_text, attr_to_index)
        if len(query_attrs) > args.max_query_len:
            raise ValueError(f"Query {query_text!r} exceeds max_query_len={args.max_query_len}")
        partial, next_group_id = build_for_query(
            query_text,
            query_attrs,
            query_signs,
            attrs,
            identities,
            embeddings,
            next_group_id,
            args,
        )
        extend_result(built, partial)
        query_summaries.append(
            {
                "query": query_text,
                "rows": len(partial["source_indices"]),
                "groups": len(set(partial["group_ids"])),
                "attrs": query_attrs,
                "signs": query_signs,
            }
        )

    if not built["source_indices"]:
        raise RuntimeError("No official-like positive rows were created.")

    source_tensor = torch.tensor(built["source_indices"], dtype=torch.long)
    target_tensor = torch.tensor(built["target_indices"], dtype=torch.long)
    group_tensor = torch.tensor(built["group_ids"], dtype=torch.long)
    index = {
        "split": args.split,
        "pair_source": f"official_like_v6_{args.preset}",
        "attributes": attributes,
        "filenames": filenames,
        "attrs": attrs,
        "source_indices": source_tensor,
        "target_indices": target_tensor,
        "attr_indices": torch.stack(built["attr_indices"]),
        "signs": torch.stack(built["signs"]),
        "query_lengths": torch.tensor(built["query_lengths"], dtype=torch.long),
        "group_ids": group_tensor,
        "positive_rank": torch.tensor(built["positive_rank"], dtype=torch.long),
        "identities": identities[source_tensor],
        "target_identities": identities[target_tensor],
        "nonquery_hamming": torch.tensor(built["nonquery_hamming"], dtype=torch.long),
        "target_cosine": torch.tensor(built["target_cosine"], dtype=torch.float32),
        "queries": built["query"],
        "max_query_len": args.max_query_len,
        "counts_by_query_len": dict(sorted(Counter(built["query_lengths"]).items())),
        "groups": int(group_tensor.max().item() + 1) if len(group_tensor) else 0,
        "rows_per_group_mean": float(len(group_tensor) / max(1, len(set(built["group_ids"])))),
        "build_config": vars(args),
    }
    atomic_torch_save(index, args.output)
    summary = {
        "output": str(args.output),
        "rows": len(index["source_indices"]),
        "groups": index["groups"],
        "rows_per_group_mean": index["rows_per_group_mean"],
        "counts_by_query_len": {str(k): int(v) for k, v in index["counts_by_query_len"].items()},
        "query_summaries": query_summaries,
        "no_json_used": True,
    }
    atomic_json_dump(summary, args.output.with_suffix(".summary.json"))
    print(json.dumps(summary, indent=2))
    print(f"Saved official-like set index: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
