#!/usr/bin/env python3
"""Create clean JSON-evaluation plots for the final v6 correction ablation.

The report comparison intentionally includes only:

1. assignment baseline: direct_sum;
2. learned model without correction: model_only;
3. generic arithmetic only: generic_sum_only;
4. learned model with correction: q_final = normalize(q_model + 1.5 * (q_sum - source)).
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS = ROOT / "final_best_system" / "results"
DEFAULT_OUTPUT = ROOT / "final_best_system" / "results" / "correction_ablation"


SYSTEMS = [
    {
        "label": "Assignment baseline",
        "short": "Baseline",
        "folder": "assignment_baseline_direct_sum",
        "color": "#7f7f7f",
    },
    {
        "label": "Learned gate only",
        "short": "No correction",
        "folder": "final_best_model_only",
        "color": "#2f6fb0",
    },
    {
        "label": "Generic sum only",
        "short": "Sum only",
        "folder": "final_best_generic_sum_only",
        "color": "#3a8c45",
    },
    {
        "label": "Learned gate + corrective sum",
        "short": "With correction",
        "folder": "final_best_model_plus_generic_delta_beta_1p50",
        "color": "#d9791f",
    },
]


def read_one_row_csv(path: Path) -> dict[str, str]:
    with path.open(encoding="utf-8") as handle:
        return next(csv.DictReader(handle))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def system_dir(system: dict, results_dir: Path) -> Path:
    return results_dir / system["folder"]


def load_summary_rows(results_dir: Path) -> list[dict]:
    rows = []
    for system in SYSTEMS:
        folder = system_dir(system, results_dir)
        summary = read_one_row_csv(folder / "summary.csv")
        row = {
            "system": system["label"],
            "short": system["short"],
            "source": str(folder),
        }
        for averaging in ("macro", "micro"):
            for metric in ("Recall", "Precision"):
                for k in (1, 5, 10):
                    key = f"{averaging}_{metric}@{k}"
                    row[key] = float(summary[key])
        rows.append(row)
    return rows


def load_per_query_rows(results_dir: Path) -> list[dict]:
    rows = []
    for system in SYSTEMS:
        folder = system_dir(system, results_dir)
        for row in read_csv(folder / "per_query_metrics.csv"):
            rows.append(
                {
                    "system": system["label"],
                    "short": system["short"],
                    "query_id": int(row["query_id"]),
                    "query": row["query"],
                    "sources": int(row["sources"]),
                    "Recall@1": float(row["Recall@1"]),
                    "Recall@5": float(row["Recall@5"]),
                    "Recall@10": float(row["Recall@10"]),
                    "Precision@1": float(row["Precision@1"]),
                    "Precision@5": float(row["Precision@5"]),
                    "Precision@10": float(row["Precision@10"]),
                }
            )
    return rows


def plot_overall(summary_rows: list[dict], output_dir: Path) -> None:
    metrics = [
        ("macro_Recall@1", "Macro R@1"),
        ("macro_Recall@5", "Macro R@5"),
        ("macro_Recall@10", "Macro R@10"),
        ("macro_Precision@1", "Macro P@1"),
        ("macro_Precision@5", "Macro P@5"),
        ("macro_Precision@10", "Macro P@10"),
        ("micro_Recall@10", "Micro R@10"),
        ("micro_Precision@10", "Micro P@10"),
    ]
    x = range(len(metrics))
    width = 0.19
    offsets = [-1.5 * width, -0.5 * width, 0.5 * width, 1.5 * width]

    plt.figure(figsize=(14, 6))
    for offset, system in zip(offsets, SYSTEMS):
        row = next(item for item in summary_rows if item["short"] == system["short"])
        values = [row[key] for key, _ in metrics]
        plt.bar([pos + offset for pos in x], values, width=width, label=system["short"], color=system["color"])
        for pos, value in zip(x, values):
            plt.text(pos + offset, value + 0.005, f"{value:.3f}", ha="center", va="bottom", fontsize=8, rotation=90)

    plt.xticks(list(x), [label for _, label in metrics], rotation=25, ha="right")
    plt.ylabel("Official JSON score")
    plt.title("Official JSON comparison: baseline vs learned gate vs corrective sum")
    plt.ylim(0, max(row["macro_Recall@10"] for row in summary_rows) * 1.35)
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "overall_metrics_baseline_no_sum_with_sum.png", dpi=180)
    plt.close()


def plot_per_query(per_query_rows: list[dict], output_dir: Path) -> None:
    queries = sorted({row["query_id"] for row in per_query_rows})
    query_labels = {
        row["query_id"]: row["query"]
        for row in per_query_rows
        if row["short"] == "Baseline"
    }
    width = 0.19
    offsets = [-1.5 * width, -0.5 * width, 0.5 * width, 1.5 * width]

    plt.figure(figsize=(14, 8))
    for offset, system in zip(offsets, SYSTEMS):
        values = []
        for qid in queries:
            row = next(
                item for item in per_query_rows
                if item["query_id"] == qid and item["short"] == system["short"]
            )
            values.append(row["Recall@10"])
        plt.bar(
            [qid + offset for qid in queries],
            values,
            width=width,
            label=system["short"],
            color=system["color"],
        )

    plt.xticks(queries, [query_labels[qid] for qid in queries], rotation=45, ha="right")
    plt.ylabel("Recall@10")
    plt.title("Per-query Recall@10 on official JSON")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "per_query_recall10_baseline_no_sum_with_sum.png", dpi=180)
    plt.close()


def plot_gain(per_query_rows: list[dict], output_dir: Path) -> None:
    queries = sorted({row["query_id"] for row in per_query_rows})
    labels = []
    no_sum_gain = []
    with_sum_gain = []
    sum_only_gain = []
    correction_gain = []

    for qid in queries:
        rows = {
            row["short"]: row
            for row in per_query_rows
            if row["query_id"] == qid
        }
        labels.append(rows["Baseline"]["query"])
        baseline = rows["Baseline"]["Recall@10"]
        no_sum = rows["No correction"]["Recall@10"]
        sum_only = rows["Sum only"]["Recall@10"]
        with_sum = rows["With correction"]["Recall@10"]
        no_sum_gain.append(no_sum - baseline)
        sum_only_gain.append(sum_only - baseline)
        with_sum_gain.append(with_sum - baseline)
        correction_gain.append(with_sum - no_sum)

    x = range(len(queries))
    plt.figure(figsize=(14, 7))
    plt.axhline(0, color="#222222", linewidth=1)
    plt.bar([i - 0.28 for i in x], no_sum_gain, width=0.26, label="No correction - baseline", color="#2f6fb0")
    plt.bar([i for i in x], sum_only_gain, width=0.26, label="Sum only - baseline", color="#3a8c45")
    plt.bar([i + 0.28 for i in x], with_sum_gain, width=0.26, label="With correction - baseline", color="#d9791f")
    for i, value in enumerate(correction_gain):
        plt.text(i, max(no_sum_gain[i], with_sum_gain[i]) + 0.01, f"+{value:.2f}", ha="center", fontsize=8)
    plt.xticks(list(x), labels, rotation=45, ha="right")
    plt.ylabel("Recall@10 delta")
    plt.title("Per-query improvement over baseline; labels show correction gain over gate-only")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "per_query_recall10_gain_from_correction.png", dpi=180)
    plt.close()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = load_summary_rows(args.results_dir)
    per_query_rows = load_per_query_rows(args.results_dir)

    write_csv(args.output_dir / "overall_metrics_baseline_no_sum_with_sum.csv", summary_rows)
    write_csv(args.output_dir / "per_query_metrics_baseline_no_sum_with_sum.csv", per_query_rows)
    plot_overall(summary_rows, args.output_dir)
    plot_per_query(per_query_rows, args.output_dir)
    plot_gain(per_query_rows, args.output_dir)

    best = next(row for row in summary_rows if row["short"] == "With correction")
    no_sum = next(row for row in summary_rows if row["short"] == "No correction")
    baseline = next(row for row in summary_rows if row["short"] == "Baseline")
    readme = args.output_dir / "README.md"
    readme.write_text(
        "\n".join(
            [
                "# V6 Correction Ablation",
                "",
                "Official JSON comparison with exactly four systems:",
                "",
                "1. Assignment baseline (`direct_sum`).",
                "2. Learned gate only (`model_only`).",
                "3. Generic CLIP arithmetic only (`generic_sum_only`).",
                "4. Learned gate + generic corrective sum (`model_plus_generic_delta_beta_1p50`).",
                "",
                "## Macro Recall@10",
                "",
                f"- Baseline: {baseline['macro_Recall@10']:.6f}",
                f"- Learned gate only: {no_sum['macro_Recall@10']:.6f}",
                f"- Generic sum only: {next(row for row in summary_rows if row['short'] == 'Sum only')['macro_Recall@10']:.6f}",
                f"- Learned gate + correction: {best['macro_Recall@10']:.6f}",
                "",
                "## Formula",
                "",
                "```text",
                "q_model = learned_gate(source, query)",
                "q_sum = normalize(source + sum signed generic CLIP directions)",
                "q_final = normalize(q_model + 1.5 * (q_sum - source))",
                "```",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"Wrote ablation report to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
