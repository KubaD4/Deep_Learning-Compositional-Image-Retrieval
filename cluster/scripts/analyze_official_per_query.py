#!/usr/bin/env python3
"""Compare official JSON per-query metrics across baselines and learned gates."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


BASELINE_DIRS = {
    "01_direct_sum": "Direct sum",
    "02_direct_sequential": "Direct sequential",
    "03_contrastive_sum": "Contrastive sum",
    "04_contrastive_sequential": "Contrastive sequential",
    "05_adaptive_tangent_sequential": "Adaptive tangent sequential",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, default=Path("artifacts/results"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/results/official_comparison"))
    parser.add_argument("--metric", default="Recall@10")
    parser.add_argument("--focus", default="Male,Young,Chubby,Hair,Hat")
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def compact_gate_name(path: Path) -> str:
    name = path.parent.name
    match = re.search(
        r"(hybmp_l\d+(?:_[A-Za-z0-9]+)*|offmp_l\d+(?:_[A-Za-z0-9]+)*|seqp2_ov\d+|seqp2_l\d+|seq_l\d+|seq_s\d+|add_l\d+|add_s\d+|ov\d+|cfg\d+|manual)",
        name,
    )
    if match:
        return match.group(1)
    if "gate_v3" in name:
        return "gate_v3"
    if "gate_v2" in name:
        return "gate_v2"
    return name


def load_methods(results_root: Path) -> dict[str, list[dict[str, str]]]:
    methods: dict[str, list[dict[str, str]]] = {}
    for folder, label in BASELINE_DIRS.items():
        path = results_root / "baselines" / folder / "per_query_metrics.csv"
        if path.exists():
            methods[label] = read_rows(path)
    for path in sorted((results_root / "gate_model").glob("*/per_query_metrics.csv")):
        methods[compact_gate_name(path)] = read_rows(path)
    return methods


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    methods = load_methods(args.results_root)
    if not methods:
        raise FileNotFoundError(f"No per-query results under {args.results_root}")

    query_ids = sorted(
        {int(row["query_id"]) for rows in methods.values() for row in rows}
    )
    metric = args.metric
    focus_terms = [term.strip().lower() for term in args.focus.split(",") if term.strip()]

    rows_out = []
    for query_id in query_ids:
        query = ""
        source_count = ""
        method_scores = {}
        for method, rows in methods.items():
            row = next((item for item in rows if int(item["query_id"]) == query_id), None)
            if row is None:
                continue
            query = row["query"]
            source_count = row.get("sources", "")
            method_scores[method] = float(row[metric])

        baseline_scores = {
            method: score
            for method, score in method_scores.items()
            if method in BASELINE_DIRS.values()
        }
        gate_scores = {
            method: score
            for method, score in method_scores.items()
            if method not in BASELINE_DIRS.values()
        }
        if not baseline_scores or not gate_scores:
            continue

        best_baseline, best_baseline_score = max(
            baseline_scores.items(), key=lambda item: item[1]
        )
        best_gate, best_gate_score = max(gate_scores.items(), key=lambda item: item[1])
        row_out = {
            "query_id": query_id,
            "query": query,
            "sources": source_count,
            "metric": metric,
            "best_baseline": best_baseline,
            "best_baseline_score": best_baseline_score,
            "best_gate": best_gate,
            "best_gate_score": best_gate_score,
            "gate_delta_vs_best_baseline": best_gate_score - best_baseline_score,
        }
        for method, score in sorted(method_scores.items()):
            row_out[method] = score
        rows_out.append(row_out)

    output_path = args.output_dir / f"per_query_delta_{metric.replace('@', 'at')}.csv"
    fieldnames = sorted({key for row in rows_out for key in row.keys()})
    preferred = [
        "query_id",
        "query",
        "sources",
        "metric",
        "best_baseline",
        "best_baseline_score",
        "best_gate",
        "best_gate_score",
        "gate_delta_vs_best_baseline",
    ]
    fieldnames = preferred + [key for key in fieldnames if key not in preferred]
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"Saved {output_path}")
    print()
    print("Worst learned-gate deltas vs best baseline:")
    for row in sorted(rows_out, key=lambda item: item["gate_delta_vs_best_baseline"])[:6]:
        print(
            f"q{row['query_id']:02d} {row['query']}: "
            f"gate={row['best_gate_score']:.4f} ({row['best_gate']}) vs "
            f"baseline={row['best_baseline_score']:.4f} ({row['best_baseline']}), "
            f"delta={row['gate_delta_vs_best_baseline']:+.4f}"
        )
    print()
    print("Best learned-gate gains vs best baseline:")
    for row in sorted(rows_out, key=lambda item: item["gate_delta_vs_best_baseline"], reverse=True)[:6]:
        print(
            f"q{row['query_id']:02d} {row['query']}: "
            f"gate={row['best_gate_score']:.4f} ({row['best_gate']}) vs "
            f"baseline={row['best_baseline_score']:.4f} ({row['best_baseline']}), "
            f"delta={row['gate_delta_vs_best_baseline']:+.4f}"
        )
    print()
    print("Focus queries:")
    for row in rows_out:
        if any(term in str(row["query"]).lower() for term in focus_terms):
            print(
                f"q{row['query_id']:02d} {row['query']}: "
                f"gate={row['best_gate_score']:.4f}, "
                f"baseline={row['best_baseline_score']:.4f}, "
                f"delta={row['gate_delta_vs_best_baseline']:+.4f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
