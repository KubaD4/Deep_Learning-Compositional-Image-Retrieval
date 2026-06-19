#!/usr/bin/env python3
"""Summarize learned gate training and hpsearch runs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from project_core import ARTIFACTS_DIR


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=ARTIFACTS_DIR / "training_runs",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ARTIFACTS_DIR / "results" / "gate_model" / "all_gate_runs_summary.csv",
    )
    parser.add_argument("--top", type=int, default=20)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def best_from_metrics(run_dir: Path) -> dict | None:
    rows = read_csv(run_dir / "metrics.csv")
    if not rows:
        return None
    best = max(rows, key=lambda row: float(row.get("val_official_like@10", 0.0)))
    config = {}
    config_path = run_dir / "config.json"
    if config_path.exists():
        import json

        config = json.loads(config_path.read_text(encoding="utf-8"))
    return {
        "source": "metrics",
        "config_id": config.get("config_id", run_dir.name),
        "run_dir": str(run_dir),
        "status": "complete" if (run_dir / "COMPLETE").exists() else "running_or_partial",
        "sampler_mode": config.get("sampler_mode", ""),
        "composer_type": config.get("composer_type", ""),
        "condition_mode": config.get("condition_mode", ""),
        "gate_state": config.get("gate_state", ""),
        "learning_rate": config.get("learning_rate", ""),
        "lambda_source": config.get("lambda_source", ""),
        "residual_scale": config.get("residual_scale", ""),
        "edit_scale": config.get("edit_scale", ""),
        "gate_max": config.get("gate_max", ""),
        "dropout": config.get("dropout", ""),
        "batch_size": config.get("batch_size", ""),
        "temperature": config.get("temperature", ""),
        "best_val_official_like@10": best.get("val_official_like@10", ""),
        "best_val_exact_R@10": best.get("val_exact_R@10", ""),
        "best_val_attr_success@10": best.get("val_attr_success@10", ""),
        "best_epoch": best.get("epoch", ""),
        "best_step": best.get("step", ""),
        "val_mean_rank_B": best.get("val_mean_rank_B", ""),
    }


def hpsearch_rows(summary_path: Path) -> list[dict]:
    rows = []
    for row in read_csv(summary_path):
        output = {
            "source": "hpsearch",
            "config_id": row.get("config_id", ""),
            "run_dir": row.get("run_dir", ""),
            "status": row.get("status", ""),
            "sampler_mode": row.get("sampler_mode", ""),
            "composer_type": row.get("composer_type", ""),
            "condition_mode": row.get("condition_mode", ""),
            "gate_state": row.get("gate_state", ""),
            "learning_rate": row.get("learning_rate", ""),
            "lambda_source": row.get("lambda_source", ""),
            "residual_scale": row.get("residual_scale", ""),
            "edit_scale": row.get("edit_scale", ""),
            "gate_max": row.get("gate_max", ""),
            "dropout": row.get("dropout", ""),
            "batch_size": row.get("batch_size", ""),
            "temperature": row.get("temperature", ""),
            "best_val_official_like@10": row.get("best_val_official_like@10", ""),
            "best_val_exact_R@10": row.get("best_val_exact_R@10", ""),
            "best_val_attr_success@10": row.get("best_val_attr_success@10", ""),
            "best_epoch": row.get("best_epoch", ""),
            "best_step": row.get("best_step", ""),
            "val_mean_rank_B": row.get("val_mean_rank_B", ""),
        }
        rows.append(output)
    return rows


def score(row: dict) -> float:
    try:
        return float(row.get("best_val_official_like@10") or 0.0)
    except ValueError:
        return 0.0


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    rows = []
    seen_run_dirs = set()

    for summary_path in sorted(args.runs_root.glob("hpsearch_*/summary.csv")):
        for row in hpsearch_rows(summary_path):
            rows.append(row)
            if row.get("run_dir"):
                seen_run_dirs.add(row["run_dir"])

    for metrics_path in sorted(args.runs_root.glob("**/metrics.csv")):
        run_dir = metrics_path.parent
        if str(run_dir) in seen_run_dirs:
            continue
        row = best_from_metrics(run_dir)
        if row:
            rows.append(row)

    rows = sorted(rows, key=score, reverse=True)
    write_csv(args.output, rows)

    print(f"Wrote: {args.output}")
    print()
    print("rank,config_id,status,official@10,exact@10,attr@10,epoch,step,run_dir")
    for rank, row in enumerate(rows[: args.top], start=1):
        print(
            f"{rank},{row.get('config_id','')},{row.get('status','')},"
            f"{row.get('best_val_official_like@10','')},"
            f"{row.get('best_val_exact_R@10','')},"
            f"{row.get('best_val_attr_success@10','')},"
            f"{row.get('best_epoch','')},{row.get('best_step','')},"
            f"{row.get('run_dir','')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
