#!/usr/bin/env python3
"""Plot per-query comparisons across the available baseline methods."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


METHOD_FOLDERS = {
    "01_direct_sum": "Direct sum",
    "02_direct_sequential": "Direct sequential",
    "03_contrastive_sum": "Contrastive sum",
    "04_contrastive_sequential": "Contrastive sequential",
    "05_adaptive_tangent_sequential": "Adaptive tangent sequential",
}
METHOD_ORDER = list(METHOD_FOLDERS.values())

METRICS = [
    "Recall@1",
    "Recall@5",
    "Recall@10",
    "Precision@1",
    "Precision@5",
    "Precision@10",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("/Users/kuba/deep_learning/cluster/artifacts/results/baselines"),
        help="Folder containing the baseline result directories.",
    )
    parser.add_argument(
        "--query-id",
        type=int,
        default=None,
        help="Plot only one query id. By default all queries are plotted.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to save the PNG plots. Defaults to <results-root>/plots.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open the matplotlib windows in addition to saving the PNGs.",
    )
    return parser.parse_args()


def load_method_frame(results_root: Path, folder: str, method_name: str) -> pd.DataFrame:
    csv_path = results_root / folder / "per_query_metrics.csv"
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    frame = pd.read_csv(csv_path)
    frame["method"] = method_name
    return frame


def build_long_frame(results_root: Path) -> pd.DataFrame:
    frames = []
    for folder, method_name in METHOD_FOLDERS.items():
        if (results_root / folder / "per_query_metrics.csv").exists():
            frames.append(load_method_frame(results_root, folder, method_name))
    if not frames:
        raise FileNotFoundError(f"No per_query_metrics.csv files under {results_root}")
    return pd.concat(frames, ignore_index=True)


def load_summaries(results_root: Path) -> pd.DataFrame:
    frames = []
    for folder, method_name in METHOD_FOLDERS.items():
        csv_path = results_root / folder / "summary.csv"
        if not csv_path.exists():
            continue
        frame = pd.read_csv(csv_path)
        frame["method"] = method_name
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No summary.csv files under {results_root}")
    return pd.concat(frames, ignore_index=True)


def sanitize_title(value: str) -> str:
    cleaned = "".join(char if char.isalnum() else "_" for char in value.strip())
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_") or "query"


def plot_query(query_frame: pd.DataFrame, output_dir: Path, show: bool) -> Path:
    query_id = int(query_frame["query_id"].iloc[0])
    query_text = str(query_frame["query"].iloc[0])
    sources = int(query_frame["sources"].iloc[0])

    methods = [method for method in METHOD_ORDER if method in set(query_frame["method"])]
    width = 0.8 / len(methods)
    x_positions = list(range(len(METRICS)))
    center = (len(methods) - 1) / 2
    offsets = [(index - center) * width for index in range(len(methods))]

    figure, axis = plt.subplots(figsize=(13, 6))
    all_values = []
    for method, offset in zip(methods, offsets):
        row = query_frame[query_frame["method"] == method].iloc[0]
        values = [float(row[metric]) for metric in METRICS]
        all_values.extend(values)
        shifted = [position + offset for position in x_positions]
        bars = axis.bar(shifted, values, width=width, label=method)
        axis.bar_label(bars, fmt="%.3f", fontsize=7, rotation=90, padding=2)

    upper_limit = min(1.0, max(all_values) * 1.3 + 0.01)
    axis.set_xticks(x_positions, METRICS, rotation=20)
    axis.set_ylim(0.0, upper_limit)
    axis.set_ylabel("Score")
    axis.set_title(f"Query {query_id} | {query_text} | {sources} source images")
    axis.legend()
    figure.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"query_{query_id:02d}_{sanitize_title(query_text)[:80]}.png"
    output_path = output_dir / filename
    figure.savefig(output_path, dpi=160)
    if show:
        plt.show()
    plt.close()
    return output_path


def write_numeric_comparisons(
    long_frame: pd.DataFrame, summaries: pd.DataFrame, output_dir: Path
) -> None:
    comparison_rows = []
    for aggregation in ("macro", "micro"):
        for metric in METRICS:
            column = f"{aggregation}_{metric}"
            scores = summaries.set_index("method")[column]
            baseline = float(scores["Direct sum"])
            ranks = scores.rank(method="min", ascending=False)
            for method in (method for method in METHOD_ORDER if method in scores.index):
                score = float(scores[method])
                delta = score - baseline
                relative = (delta / baseline * 100.0) if baseline else float("nan")
                comparison_rows.append(
                    {
                        "aggregation": aggregation,
                        "metric": metric,
                        "method": method,
                        "score": score,
                        "rank": int(ranks[method]),
                        "absolute_delta_vs_direct_sum": delta,
                        "relative_change_vs_direct_sum_pct": relative,
                    }
                )
    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(output_dir / "method_comparison.csv", index=False)

    winner_rows = []
    for query_id, query_frame in long_frame.groupby("query_id"):
        for metric in METRICS:
            ordered = query_frame.sort_values(metric, ascending=False).reset_index(drop=True)
            best = ordered.iloc[0]
            second = ordered.iloc[1]
            margin = float(best[metric] - second[metric])
            relative_margin = (
                margin / float(second[metric]) * 100.0
                if float(second[metric]) > 0
                else float("nan")
            )
            winner_rows.append(
                {
                    "query_id": int(query_id),
                    "query": best["query"],
                    "sources": int(best["sources"]),
                    "metric": metric,
                    "best_method": best["method"],
                    "best_score": float(best[metric]),
                    "second_method": second["method"],
                    "second_score": float(second[metric]),
                    "absolute_margin": margin,
                    "relative_margin_pct": relative_margin,
                }
            )
    pd.DataFrame(winner_rows).to_csv(output_dir / "per_query_winners.csv", index=False)

    recall10 = comparison[
        (comparison["aggregation"] == "macro")
        & (comparison["metric"] == "Recall@10")
    ].sort_values("rank")
    print("\nMacro Recall@10 ranking:")
    print(
        recall10[
            [
                "rank",
                "method",
                "score",
                "absolute_delta_vs_direct_sum",
                "relative_change_vs_direct_sum_pct",
            ]
        ].to_string(index=False, float_format=lambda value: f"{value:.4f}")
    )


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or (args.results_root / "plots")
    long_frame = build_long_frame(args.results_root)
    summaries = load_summaries(args.results_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_numeric_comparisons(long_frame, summaries, output_dir)

    query_ids = sorted(long_frame["query_id"].unique())
    if args.query_id is not None:
        if args.query_id not in query_ids:
            raise ValueError(f"query_id {args.query_id} not found. Available: {query_ids}")
        query_ids = [args.query_id]

    saved_paths = []
    for query_id in query_ids:
        query_frame = (
            long_frame[long_frame["query_id"] == query_id]
            .assign(method=lambda frame: pd.Categorical(
                frame["method"], categories=METHOD_ORDER, ordered=True
            ))
            .sort_values("method")
            .reset_index(drop=True)
        )
        saved_paths.append(plot_query(query_frame, output_dir, args.show))

    print(f"Saved {len(saved_paths)} plot(s) in {output_dir}")
    print(output_dir / "method_comparison.csv")
    print(output_dir / "per_query_winners.csv")
    for path in saved_paths:
        print(path)


if __name__ == "__main__":
    main()
