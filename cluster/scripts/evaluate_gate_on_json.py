#!/usr/bin/env python3
"""Evaluate a trained gate checkpoint on celeba_evaluation.json."""

from __future__ import annotations

import argparse
import csv
import json
import signal
import time
from pathlib import Path

import torch

from learned_gate_core import (
    TRAINING_RUN_DIRNAME,
    condition_embeddings,
    load_image_embedding_cache,
    load_model_checkpoint,
    load_prompt_embedding_cache,
)
from project_core import (
    ARTIFACTS_DIR,
    CHECKPOINT_DIR,
    MODEL_ID,
    TOP_KS,
    atomic_json_dump,
    choose_device,
    load_evaluation,
    parse_query,
    read_attribute_table,
    retrieval_metrics,
)


STOP_REQUESTED = False


def request_stop(signum, frame) -> None:
    del signum, frame
    global STOP_REQUESTED
    STOP_REQUESTED = True


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help=(
            "Gate checkpoint to evaluate. If omitted, the script selects the "
            "checkpoint with the best val_official_like@10 under artifacts/training_runs."
        ),
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--source-batch-size", type=int, default=256)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--time-budget-seconds", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def find_best_checkpoint() -> Path:
    runs_root = ARTIFACTS_DIR / TRAINING_RUN_DIRNAME
    best_score = float("-inf")
    best_checkpoint = None
    for metrics_path in sorted(runs_root.glob("**/metrics.csv")):
        run_dir = metrics_path.parent
        checkpoint = run_dir / "checkpoints" / "best_val_official_like_at10.pt"
        if not checkpoint.exists():
            continue
        with metrics_path.open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            try:
                score = float(row.get("val_official_like@10", 0.0))
            except ValueError:
                continue
            if score > best_score:
                best_score = score
                best_checkpoint = checkpoint
    if best_checkpoint is None:
        raise FileNotFoundError(
            f"No best_val_official_like_at10.pt checkpoint found under {runs_root}"
        )
    print(
        f"Selected best checkpoint: {best_checkpoint} "
        f"(val_official_like@10={best_score:.4f})"
    )
    return best_checkpoint


def query_tensors(query: str, attribute_to_index: dict[str, int], device):
    parsed = parse_query(query)
    attr_indices = torch.tensor(
        [[attribute_to_index[attribute] for _, attribute in parsed]],
        dtype=torch.long,
        device=device,
    )
    signs = torch.tensor([[sign for sign, _ in parsed]], dtype=torch.int8, device=device)
    return attr_indices, signs


def finalize(annotations, chunks_dir: Path, result_dir: Path, checkpoint: Path) -> None:
    per_query = []
    retrieval_path = result_dir / "retrievals.jsonl"
    with retrieval_path.open("w", encoding="utf-8") as retrieval_file:
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

    write_csv(result_dir / "per_query_metrics.csv", per_query)
    metric_names = [name for name in per_query[0] if "@" in name]
    total_sources = sum(row["sources"] for row in per_query)
    summary = {
        "method": "learned_gate_v1",
        "checkpoint": str(checkpoint),
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
        {
            "model_id": MODEL_ID,
            "method": "learned_gate_v1",
            "checkpoint": str(checkpoint),
            "top_ks": TOP_KS,
        },
        result_dir / "config.json",
    )
    (result_dir / "COMPLETE").write_text("complete\n", encoding="utf-8")
    print(f"Final learned gate JSON results saved in {result_dir}")


def main() -> int:
    args = parse_args()
    started = time.monotonic()
    signal.signal(signal.SIGTERM, request_stop)
    if hasattr(signal, "SIGUSR1"):
        signal.signal(signal.SIGUSR1, request_stop)

    device = choose_device(args.device)
    checkpoint = args.checkpoint or find_best_checkpoint()
    model, checkpoint_data = load_model_checkpoint(checkpoint, device)
    model.eval()
    config = checkpoint_data["config"]
    run_name = checkpoint.parents[1].name
    result_dir = args.output_dir or (ARTIFACTS_DIR / "results" / "gate_model" / run_name)
    chunks_dir = CHECKPOINT_DIR / "gate_json" / run_name
    result_dir.mkdir(parents=True, exist_ok=True)
    chunks_dir.mkdir(parents=True, exist_ok=True)
    if (result_dir / "COMPLETE").exists() and not args.force:
        print(f"Evaluation already complete: {result_dir}")
        return 0
    if args.force:
        for path in chunks_dir.glob("*.json"):
            path.unlink()
        for path in result_dir.glob("*"):
            if path.is_file():
                path.unlink()

    attributes, _, _ = read_attribute_table()
    attribute_to_index = {name: index for index, name in enumerate(attributes)}
    annotations = load_evaluation()
    gallery_cache = load_image_embedding_cache("test")
    prompt_cache = load_prompt_embedding_cache()
    gallery = gallery_cache["embeddings"].float().to(device)

    with torch.inference_mode():
        for query_id, item in enumerate(annotations):
            attr_indices_one, signs_one = query_tensors(item["query"], attribute_to_index, device)
            source_indices = [int(index) for index in item["ground_truth"]]
            for start in range(0, len(source_indices), args.source_batch_size):
                end = min(start + args.source_batch_size, len(source_indices))
                chunk_path = chunks_dir / f"q{query_id:02d}_{start:06d}_{end:06d}.json"
                if chunk_path.exists():
                    continue
                batch_indices = source_indices[start:end]
                source = gallery[batch_indices]
                attr_indices = attr_indices_one.expand(len(batch_indices), -1)
                signs = signs_one.expand(len(batch_indices), -1)
                conditions, mask = condition_embeddings(
                    prompt_cache,
                    attr_indices,
                    signs,
                    device,
                    str(config.get("condition_mode", "signed_prompt")),
                )
                queries, _, _ = model(source, conditions, mask)
                scores = queries @ gallery.T
                rows = torch.arange(len(batch_indices), device=device)
                scores[rows, torch.tensor(batch_indices, device=device)] = -torch.inf
                rankings = scores.topk(max(TOP_KS), dim=1).indices.cpu().tolist()
                atomic_json_dump(
                    {
                        "method": "learned_gate_v1",
                        "checkpoint": str(checkpoint),
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
                    f"learned_gate: query {query_id + 1}/{len(annotations)}, "
                    f"sources {end}/{len(source_indices)}"
                )
                if STOP_REQUESTED or (
                    args.time_budget_seconds
                    and time.monotonic() - started >= args.time_budget_seconds
                ):
                    print("Checkpoint saved; submit the same job again.")
                    return 3

    finalize(annotations, chunks_dir, result_dir, checkpoint)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
