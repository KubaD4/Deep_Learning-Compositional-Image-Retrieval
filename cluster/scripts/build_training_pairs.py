#!/usr/bin/env python3
"""Build same-identity edit tuples for the learned gate model."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import torch

from learned_gate_core import signed_condition_text
from project_core import (
    ARTIFACTS_DIR,
    atomic_json_dump,
    atomic_torch_save,
    load_torch,
    read_attribute_table,
    read_identity_map,
)


DEFAULT_OUTPUT_DIR = ARTIFACTS_DIR / "training_pairs"
SPLITS = {"train": 0, "valid": 1}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", nargs="+", choices=tuple(SPLITS), default=["train", "valid"])
    parser.add_argument("--min-query-len", type=int, default=1)
    parser.add_argument("--max-query-len", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-pairs-per-length", type=int, default=0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--write-jsonl", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def embedding_filenames(split: str) -> list[str]:
    from learned_gate_core import load_image_embedding_cache

    cache = load_image_embedding_cache(split)
    return list(cache["filenames"])


def collect_pairs_for_split(
    split: str,
    attributes: list[str],
    attr_by_file: dict[str, torch.Tensor],
    identities: dict[str, int],
    min_query_len: int,
    max_query_len: int,
) -> dict:
    filenames = embedding_filenames(split)
    split_attrs = torch.stack([attr_by_file[filename] for filename in filenames])
    split_identities = [identities[filename] for filename in filenames]

    by_identity: dict[int, list[int]] = defaultdict(list)
    for index, identity in enumerate(split_identities):
        by_identity[identity].append(index)

    source_indices = []
    target_indices = []
    padded_attrs = []
    padded_signs = []
    query_lengths = []
    pair_identities = []
    counts = Counter()

    for identity, indices in by_identity.items():
        if len(indices) < 2:
            continue
        for source_index in indices:
            source_attr = split_attrs[source_index]
            for target_index in indices:
                if source_index == target_index:
                    continue
                target_attr = split_attrs[target_index]
                diff_positions = torch.nonzero(source_attr != target_attr).flatten()
                query_len = int(diff_positions.numel())
                if not min_query_len <= query_len <= max_query_len:
                    continue
                signs = target_attr[diff_positions].to(torch.int8)
                padded_attr = torch.full((max_query_len,), -1, dtype=torch.long)
                padded_sign = torch.zeros((max_query_len,), dtype=torch.int8)
                padded_attr[:query_len] = diff_positions.long()
                padded_sign[:query_len] = signs

                source_indices.append(source_index)
                target_indices.append(target_index)
                padded_attrs.append(padded_attr)
                padded_signs.append(padded_sign)
                query_lengths.append(query_len)
                pair_identities.append(identity)
                counts[query_len] += 1

    if not source_indices:
        raise RuntimeError(f"No pairs found for split={split}")

    return {
        "split": split,
        "attributes": attributes,
        "filenames": filenames,
        "attrs": split_attrs,
        "source_indices": torch.tensor(source_indices, dtype=torch.long),
        "target_indices": torch.tensor(target_indices, dtype=torch.long),
        "attr_indices": torch.stack(padded_attrs),
        "signs": torch.stack(padded_signs),
        "query_lengths": torch.tensor(query_lengths, dtype=torch.long),
        "identities": torch.tensor(pair_identities, dtype=torch.long),
        "max_query_len": max_query_len,
        "counts_by_query_len": dict(sorted(counts.items())),
    }


def maybe_subsample_by_length(index: dict, max_pairs_per_length: int, seed: int) -> dict:
    if max_pairs_per_length <= 0:
        return index
    generator = torch.Generator().manual_seed(seed)
    keep_parts = []
    lengths = index["query_lengths"]
    for query_len in sorted(set(lengths.tolist())):
        positions = torch.nonzero(lengths == query_len).flatten()
        if len(positions) > max_pairs_per_length:
            order = torch.randperm(len(positions), generator=generator)
            positions = positions[order[:max_pairs_per_length]]
        keep_parts.append(positions)
    keep = torch.cat(keep_parts)
    keep = keep[torch.randperm(len(keep), generator=generator)]

    output = dict(index)
    for key in (
        "source_indices",
        "target_indices",
        "attr_indices",
        "signs",
        "query_lengths",
        "identities",
    ):
        output[key] = index[key][keep]
    output["subsampled"] = True
    output["max_pairs_per_length"] = max_pairs_per_length
    output["counts_by_query_len"] = dict(
        sorted(Counter(output["query_lengths"].tolist()).items())
    )
    return output


def write_jsonl(path: Path, index: dict, limit: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    total = len(index["source_indices"])
    n = total if limit is None else min(total, limit)
    with path.open("w", encoding="utf-8") as handle:
        for row in range(n):
            length = int(index["query_lengths"][row])
            source_index = int(index["source_indices"][row])
            target_index = int(index["target_indices"][row])
            record = {
                "source_index": source_index,
                "target_index": target_index,
                "source_filename": index["filenames"][source_index],
                "target_filename": index["filenames"][target_index],
                "identity": int(index["identities"][row]),
                "query_length": length,
                "query": signed_condition_text(
                    index["attr_indices"][row],
                    index["signs"][row],
                    length,
                    index["attributes"],
                ),
            }
            handle.write(json.dumps(record) + "\n")


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    attributes, filenames, all_attrs = read_attribute_table()
    identities = read_identity_map()
    attr_by_file = {
        filename: all_attrs[index] for index, filename in enumerate(filenames)
    }

    summaries = []
    for split in args.splits:
        output_path = (
            args.output_dir
            / f"{split}_pairs_len{args.min_query_len}_{args.max_query_len}.pt"
        )
        if output_path.exists() and not args.force:
            index = load_torch(output_path)
            print(f"Reusing pair index: {output_path}")
        else:
            print(f"Building {split} pairs...")
            index = collect_pairs_for_split(
                split,
                attributes,
                attr_by_file,
                identities,
                args.min_query_len,
                args.max_query_len,
            )
            index = maybe_subsample_by_length(
                index,
                args.max_pairs_per_length,
                args.seed + SPLITS[split],
            )
            atomic_torch_save(index, output_path)
            print(f"Saved pair index: {output_path}")

        summary = {
            "split": split,
            "pairs": int(len(index["source_indices"])),
            "counts_by_query_len": {
                str(key): int(value)
                for key, value in index["counts_by_query_len"].items()
            },
            "path": str(output_path),
        }
        summaries.append(summary)
        print(json.dumps(summary, indent=2))
        if args.write_jsonl:
            write_jsonl(output_path.with_suffix(".jsonl"), index)

    atomic_json_dump(
        {
            "min_query_len": args.min_query_len,
            "max_query_len": args.max_query_len,
            "splits": summaries,
        },
        args.output_dir / "summary.json",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
