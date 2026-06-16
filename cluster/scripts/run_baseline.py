#!/usr/bin/env python3
"""Run one resumable official zero-shot baseline."""

from __future__ import annotations

import argparse
import csv
import json
import signal
import time
from pathlib import Path

import torch

from project_core import (
    CHECKPOINT_DIR,
    EMBEDDING_DIR,
    METHOD_FOLDERS,
    MODEL_ID,
    RESULTS_DIR,
    TOP_KS,
    atomic_json_dump,
    attribute_names,
    choose_device,
    compose_queries,
    load_celeba,
    load_evaluation,
    load_torch,
    parse_query,
    retrieval_metrics,
)


STOP_REQUESTED = False


def request_stop(signum, frame) -> None:
    del signum, frame
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print("Stop requested: finishing the current source batch.")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=tuple(METHOD_FOLDERS), required=True)
    parser.add_argument("--source-batch-size", type=int, default=256)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--time-budget-seconds", type=int, default=480)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def finalize(method, annotations, chunks_dir, result_dir) -> None:
    per_query = []
    retrieval_path = result_dir / "retrievals.jsonl"
    retrieval_tmp = retrieval_path.with_suffix(".jsonl.tmp")
    with retrieval_tmp.open("w", encoding="utf-8") as retrieval_file:
        for query_id, item in enumerate(annotations):
            source_count = len(item["ground_truth"])
            totals = {f"Recall@{k}": 0.0 for k in TOP_KS}
            totals.update({f"Precision@{k}": 0.0 for k in TOP_KS})
            seen = 0
            for path in sorted(chunks_dir.glob(f"q{query_id:02d}_*.json")):
                chunk = json.loads(path.read_text(encoding="utf-8"))
                for record in chunk["records"]:
                    retrieval_file.write(json.dumps(record) + "\n")
                    valid = set(item["ground_truth"][str(record["source_index"])])
                    for k in TOP_KS:
                        recall, precision = retrieval_metrics(record["top10"], valid, k)
                        totals[f"Recall@{k}"] += recall
                        totals[f"Precision@{k}"] += precision
                    seen += 1
            if seen != source_count:
                raise RuntimeError(
                    f"Query {query_id} incomplete: {seen}/{source_count} source cases"
                )
            per_query.append(
                {
                    "query_id": query_id,
                    "query": item["query"],
                    "sources": source_count,
                    **{name: value / source_count for name, value in totals.items()},
                }
            )
    retrieval_tmp.replace(retrieval_path)
    write_csv(result_dir / "per_query_metrics.csv", per_query)
    metric_names = [name for name in per_query[0] if "@" in name]
    total_sources = sum(row["sources"] for row in per_query)
    summary = {
        "method": method,
        "query_entries": len(per_query),
        "source_query_cases": total_sources,
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
    write_csv(result_dir / "summary.csv", [summary])
    atomic_json_dump(
        {"model_id": MODEL_ID, "method": method, "top_ks": TOP_KS},
        result_dir / "config.json",
    )
    (result_dir / "COMPLETE").write_text("complete\n", encoding="utf-8")
    print(f"Final results saved in {result_dir}")


def main() -> int:
    args = parse_args()
    started = time.monotonic()
    signal.signal(signal.SIGTERM, request_stop)
    if hasattr(signal, "SIGUSR1"):
        signal.signal(signal.SIGUSR1, request_stop)

    result_dir = RESULTS_DIR / METHOD_FOLDERS[args.method]
    chunks_dir = CHECKPOINT_DIR / "baselines" / args.method
    result_dir.mkdir(parents=True, exist_ok=True)
    chunks_dir.mkdir(parents=True, exist_ok=True)
    complete_marker = result_dir / "COMPLETE"
    if complete_marker.exists() and not args.force:
        print(f"Baseline already complete: {result_dir}")
        return 0
    if args.force:
        for path in chunks_dir.glob("*.json"):
            path.unlink()
        for path in result_dir.iterdir():
            if path.is_file():
                path.unlink()

    checkpoint_config_path = chunks_dir / "checkpoint_config.json"
    checkpoint_config = {
        "model_id": MODEL_ID,
        "method": args.method,
        "source_batch_size": args.source_batch_size,
    }
    if checkpoint_config_path.exists():
        existing_config = json.loads(
            checkpoint_config_path.read_text(encoding="utf-8")
        )
        if existing_config != checkpoint_config:
            raise RuntimeError(
                "Checkpoint configuration changed. Reuse the original batch size "
                "or restart with --force."
            )
    else:
        atomic_json_dump(checkpoint_config, checkpoint_config_path)

    gallery_cache = load_torch(EMBEDDING_DIR / "test_image_embeddings.pt")
    text_cache = load_torch(EMBEDDING_DIR / "attribute_text_embeddings.pt")
    if gallery_cache["model_id"] != MODEL_ID or text_cache["model_id"] != MODEL_ID:
        raise RuntimeError("Embedding cache model IDs do not match the required CLIP model")

    dataset = load_celeba("test")
    names = attribute_names(dataset)
    if text_cache["attributes"] != names:
        raise RuntimeError("Text cache attributes do not match CelebA")
    attribute_to_index = {name: index for index, name in enumerate(names)}
    annotations = load_evaluation()
    device = choose_device(args.device)
    gallery = gallery_cache["embeddings"].float().to(device)
    positive = text_cache["positive"].float().to(device)
    directions = text_cache["directions"].float().to(device)

    with torch.inference_mode():
        for query_id, item in enumerate(annotations):
            conditions = parse_query(item["query"])
            source_indices = [int(index) for index in item["ground_truth"]]
            for start in range(0, len(source_indices), args.source_batch_size):
                end = min(start + args.source_batch_size, len(source_indices))
                chunk_path = chunks_dir / f"q{query_id:02d}_{start:06d}_{end:06d}.json"
                if chunk_path.exists():
                    continue
                batch_indices = source_indices[start:end]
                queries = compose_queries(
                    gallery[batch_indices],
                    conditions,
                    args.method,
                    attribute_to_index,
                    positive,
                    directions,
                )
                scores = queries @ gallery.T
                rows = torch.arange(len(batch_indices), device=device)
                scores[rows, torch.tensor(batch_indices, device=device)] = -torch.inf
                rankings = scores.topk(max(TOP_KS), dim=1).indices.cpu().tolist()
                atomic_json_dump(
                    {
                        "method": args.method,
                        "query_id": query_id,
                        "query": item["query"],
                        "start": start,
                        "end": end,
                        "records": [
                            {
                                "query_id": query_id,
                                "query": item["query"],
                                "source_index": source_index,
                                "top10": ranking,
                            }
                            for source_index, ranking in zip(batch_indices, rankings)
                        ],
                    },
                    chunk_path,
                )
                print(
                    f"{args.method}: query {query_id + 1}/{len(annotations)}, "
                    f"sources {end}/{len(source_indices)}"
                )
                elapsed = time.monotonic() - started
                if STOP_REQUESTED or elapsed >= args.time_budget_seconds:
                    print("Checkpoint saved; submit the same job again.")
                    return 3

    finalize(args.method, annotations, chunks_dir, result_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
