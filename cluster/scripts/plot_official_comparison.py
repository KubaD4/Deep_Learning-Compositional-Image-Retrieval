#!/usr/bin/env python3
"""Plot official JSON evaluation comparisons for baselines and learned gates."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from project_core import ARTIFACTS_DIR


BASELINE_LABELS = {
    "01_direct_sum": "Direct sum",
    "02_direct_sequential": "Direct sequential",
    "03_contrastive_sum": "Contrastive sum",
    "04_contrastive_sequential": "Contrastive sequential",
    "05_adaptive_tangent_sequential": "Adaptive tangent sequential",
}
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
        "--baselines-root",
        type=Path,
        default=ARTIFACTS_DIR / "results" / "baselines",
    )
    parser.add_argument(
        "--gate-root",
        type=Path,
        default=ARTIFACTS_DIR / "results" / "gate_model",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ARTIFACTS_DIR / "results" / "official_comparison",
    )
    parser.add_argument(
        "--include-all-gates",
        action="store_true",
        help="Plot all learned gate evaluations instead of only the best one.",
    )
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def load_baselines(root: Path) -> tuple[list[pd.DataFrame], list[pd.DataFrame]]:
    per_query_frames = []
    summary_frames = []
    for folder, label in BASELINE_LABELS.items():
        method_dir = root / folder
        per_query_path = method_dir / "per_query_metrics.csv"
        summary_path = method_dir / "summary.csv"
        if not per_query_path.exists() or not summary_path.exists():
            continue
        per_query = read_csv(per_query_path)
        summary = read_csv(summary_path)
        per_query["method"] = label
        summary["method"] = label
        per_query["result_dir"] = str(method_dir)
        summary["result_dir"] = str(method_dir)
        per_query_frames.append(per_query)
        summary_frames.append(summary)
    return per_query_frames, summary_frames


def load_gate_results(root: Path, include_all: bool) -> tuple[list[pd.DataFrame], list[pd.DataFrame]]:
    candidates = []
    for summary_path in sorted(root.glob("*/summary.csv")):
        result_dir = summary_path.parent
        per_query_path = result_dir / "per_query_metrics.csv"
        if not per_query_path.exists():
            continue
        summary = read_csv(summary_path)
        score = float(summary.iloc[0].get("macro_Recall@10", 0.0))
        candidates.append((score, result_dir, per_query_path, summary_path))
    if not candidates:
        return [], []
    if include_all:
        selected = sorted(candidates, reverse=True)
    else:
        selected = [max(candidates)]

    per_query_frames = []
    summary_frames = []
    for score, result_dir, per_query_path, summary_path in selected:
        label = "Learned gate" if not include_all else f"Learned gate {result_dir.name[:18]}"
        per_query = read_csv(per_query_path)
        summary = read_csv(summary_path)
        per_query["method"] = label
        summary["method"] = label
        per_query["result_dir"] = str(result_dir)
        summary["result_dir"] = str(result_dir)
        per_query_frames.append(per_query)
        summary_frames.append(summary)
    return per_query_frames, summary_frames


def sanitize(value: str) -> str:
    cleaned = "".join(char if char.isalnum() else "_" for char in value)
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_") or "query"


def method_order(methods: list[str]) -> list[str]:
    preferred = list(BASELINE_LABELS.values()) + ["Learned gate"]
    ordered = [method for method in preferred if method in methods]
    ordered.extend(method for method in methods if method not in ordered)
    return ordered


def write_numeric_tables(summaries: pd.DataFrame, per_query: pd.DataFrame, output_dir: Path) -> None:
    rows = []
    methods = method_order(sorted(summaries["method"].unique()))
    best_baseline = {}
    for aggregation in ("macro", "micro"):
        for metric in METRICS:
            column = f"{aggregation}_{metric}"
            if column not in summaries:
                continue
            baseline_rows = summaries[~summaries["method"].str.startswith("Learned gate")]
            best_baseline_score = float(baseline_rows[column].max())
            best_baseline_method = str(
                baseline_rows.sort_values(column, ascending=False).iloc[0]["method"]
            )
            direct_sum_row = summaries[summaries["method"] == "Direct sum"]
            direct_sum_score = (
                float(direct_sum_row.iloc[0][column]) if not direct_sum_row.empty else 0.0
            )
            scores = summaries.set_index("method")[column]
            ranks = scores.rank(method="min", ascending=False)
            for method in methods:
                if method not in scores:
                    continue
                score = float(scores[method])
                rows.append(
                    {
                        "aggregation": aggregation,
                        "metric": metric,
                        "method": method,
                        "score": score,
                        "rank": int(ranks[method]),
                        "best_baseline_method": best_baseline_method,
                        "absolute_delta_vs_best_baseline": score - best_baseline_score,
                        "absolute_delta_vs_direct_sum": score - direct_sum_score,
                    }
                )
    pd.DataFrame(rows).to_csv(output_dir / "method_comparison.csv", index=False)

    winner_rows = []
    for query_id, query_frame in per_query.groupby("query_id"):
        for metric in METRICS:
            ordered = query_frame.sort_values(metric, ascending=False).reset_index(drop=True)
            best = ordered.iloc[0]
            second = ordered.iloc[1] if len(ordered) > 1 else ordered.iloc[0]
            winner_rows.append(
                {
                    "query_id": int(query_id),
                    "query": best["query"],
                    "metric": metric,
                    "best_method": best["method"],
                    "best_score": float(best[metric]),
                    "second_method": second["method"],
                    "second_score": float(second[metric]),
                    "absolute_margin": float(best[metric] - second[metric]),
                }
            )
    pd.DataFrame(winner_rows).to_csv(output_dir / "per_query_winners.csv", index=False)


def plot_overview(summaries: pd.DataFrame, output_dir: Path, show: bool) -> Path:
    methods = method_order(sorted(summaries["method"].unique()))
    frame = summaries.set_index("method").loc[methods]
    overview_metrics = [f"macro_{metric}" for metric in METRICS]
    values = frame[overview_metrics]
    values.columns = METRICS

    figure, axis = plt.subplots(figsize=(15, 7))
    values.plot(kind="bar", ax=axis)
    axis.set_title("Official JSON Evaluation: Method Comparison")
    axis.set_ylabel("Score")
    axis.set_ylim(0.0, min(1.0, float(values.max().max()) * 1.25 + 0.02))
    axis.set_xlabel("")
    axis.tick_params(axis="x", rotation=25)
    axis.legend(title="Macro metric", ncol=3)
    figure.tight_layout()

    output_path = output_dir / "official_comparison_overview.png"
    figure.savefig(output_path, dpi=170)
    if show:
        plt.show()
    plt.close()
    return output_path


def plot_queries(per_query: pd.DataFrame, output_dir: Path, show: bool) -> list[Path]:
    saved = []
    methods = method_order(sorted(per_query["method"].unique()))
    for query_id, query_frame in per_query.groupby("query_id"):
        query_frame = (
            query_frame.assign(
                method=lambda frame: pd.Categorical(
                    frame["method"], categories=methods, ordered=True
                )
            )
            .sort_values("method")
            .reset_index(drop=True)
        )
        query_text = str(query_frame["query"].iloc[0])
        sources = int(query_frame["sources"].iloc[0])
        width = 0.8 / len(methods)
        x_positions = list(range(len(METRICS)))
        center = (len(methods) - 1) / 2

        figure, axis = plt.subplots(figsize=(14, 6))
        all_values = []
        for index, method in enumerate(methods):
            row = query_frame[query_frame["method"] == method]
            if row.empty:
                continue
            values = [float(row.iloc[0][metric]) for metric in METRICS]
            all_values.extend(values)
            offset = (index - center) * width
            bars = axis.bar(
                [position + offset for position in x_positions],
                values,
                width=width,
                label=method,
            )
            axis.bar_label(bars, fmt="%.3f", fontsize=7, rotation=90, padding=2)

        axis.set_title(f"Query {int(query_id)} | {query_text} | {sources} source images")
        axis.set_xticks(x_positions, METRICS, rotation=20)
        axis.set_ylabel("Score")
        axis.set_ylim(0.0, min(1.0, max(all_values) * 1.25 + 0.02))
        axis.legend(fontsize=8)
        figure.tight_layout()

        output_path = output_dir / f"query_{int(query_id):02d}_{sanitize(query_text)[:80]}.png"
        figure.savefig(output_path, dpi=170)
        if show:
            plt.show()
        plt.close()
        saved.append(output_path)
    return saved


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    per_query_frames, summary_frames = load_baselines(args.baselines_root)
    gate_per_query, gate_summaries = load_gate_results(args.gate_root, args.include_all_gates)
    per_query_frames.extend(gate_per_query)
    summary_frames.extend(gate_summaries)
    if not per_query_frames or not summary_frames:
        raise FileNotFoundError("No evaluation results found to plot")

    per_query = pd.concat(per_query_frames, ignore_index=True)
    summaries = pd.concat(summary_frames, ignore_index=True)
    per_query.to_csv(args.output_dir / "combined_per_query_metrics.csv", index=False)
    summaries.to_csv(args.output_dir / "combined_summary.csv", index=False)
    write_numeric_tables(summaries, per_query, args.output_dir)

    overview = plot_overview(summaries, args.output_dir, args.show)
    query_plots = plot_queries(per_query, args.output_dir, args.show)

    print(f"Saved overview: {overview}")
    print(f"Saved {len(query_plots)} per-query plots in {args.output_dir}")
    print(args.output_dir / "combined_summary.csv")
    print(args.output_dir / "method_comparison.csv")
    print(args.output_dir / "per_query_winners.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
