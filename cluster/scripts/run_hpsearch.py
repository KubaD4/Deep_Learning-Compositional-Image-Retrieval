#!/usr/bin/env python3
"""Run a small sequential hyperparameter search for the gate model."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from learned_gate_core import TRAINING_RUN_DIRNAME, progress_line
from project_core import ARTIFACTS_DIR, atomic_json_dump


DEFAULT_CONFIGS = Path(__file__).resolve().parents[1] / "configs" / "gate_hpsearch_configs.json"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("short", "long"), default="short")
    parser.add_argument("--configs", type=Path, default=DEFAULT_CONFIGS)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--time-budget-seconds-per-run", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def read_best_metrics(run_dir: Path) -> dict:
    metrics_path = run_dir / "metrics.csv"
    if not metrics_path.exists():
        return {}
    with metrics_path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return {}
    best = max(rows, key=lambda row: float(row.get("val_official_like@10", 0.0)))
    return {
        "best_val_official_like@10": best.get("val_official_like@10", ""),
        "best_val_exact_R@10": best.get("val_exact_R@10", ""),
        "best_val_attr_success@10": best.get("val_attr_success@10", ""),
        "best_epoch": best.get("epoch", ""),
        "best_step": best.get("step", ""),
        "val_mean_rank_B": best.get("val_mean_rank_B", ""),
    }


def append_summary(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main() -> int:
    args = parse_args()
    with args.configs.open(encoding="utf-8") as handle:
        configs = json.load(handle)
    if args.limit:
        configs = configs[: args.limit]

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    family = "gate_v2" if any(
        config.get("composer_type") == "additive_gate" for config in configs
    ) else "gate_v1"
    search_dir = ARTIFACTS_DIR / TRAINING_RUN_DIRNAME / f"hpsearch_{family}_{stamp}_{args.profile}"
    search_dir.mkdir(parents=True, exist_ok=True)
    summary_path = search_dir / "summary.csv"
    progress = search_dir / "progress.txt"
    progress_line(progress, f"HPSEARCH start profile={args.profile} configs={len(configs)}")

    for index, config in enumerate(configs, start=1):
        config = dict(config)
        config["profile"] = args.profile
        config_path = search_dir / f"{config['config_id']}.json"
        atomic_json_dump(config, config_path)
        batch_size = config.get("batch_size", 128 if args.profile == "short" else 256)
        composer = str(config.get("composer_type", "residual_only")).replace("_", "")
        run_dir = search_dir / (
            f"{family}_{composer}_{config['config_id']}_{config['sampler_mode']}"
            f"_b{batch_size}_lr{config['learning_rate']}"
            f"_src{config['lambda_source']}_scale{config['residual_scale']}_{args.profile}"
        )
        command = [
            sys.executable,
            "scripts/train_gate_model.py",
            "--profile",
            args.profile,
            "--config",
            str(config_path),
            "--run-dir",
            str(run_dir),
            "--device",
            args.device,
        ]
        if args.time_budget_seconds_per_run:
            command.extend(["--time-budget-seconds", str(args.time_budget_seconds_per_run)])
        if args.force:
            command.append("--force")

        progress_line(progress, f"START config={config['config_id']} {index}/{len(configs)}")
        status = "unknown"
        try:
            completed = subprocess.run(command, check=False)
            status = "complete" if completed.returncode == 0 else f"exit_{completed.returncode}"
        except Exception as exc:
            status = f"error_{type(exc).__name__}"
            progress_line(progress, f"ERROR config={config['config_id']} {exc}")

        best = read_best_metrics(run_dir)
        row = {
            "config_id": config["config_id"],
            "run_dir": str(run_dir),
            "status": status,
            "sampler_mode": config.get("sampler_mode", ""),
            "composer_type": config.get("composer_type", ""),
            "condition_mode": config.get("condition_mode", ""),
            "learning_rate": config.get("learning_rate", ""),
            "lambda_source": config.get("lambda_source", ""),
            "residual_scale": config.get("residual_scale", ""),
            "edit_scale": config.get("edit_scale", ""),
            "gate_max": config.get("gate_max", ""),
            "dropout": config.get("dropout", ""),
            "batch_size": batch_size,
            "temperature": config.get("temperature", ""),
            **best,
        }
        append_summary(summary_path, row)
        progress_line(progress, f"END config={config['config_id']} status={status} best={best}")

    progress_line(progress, f"HPSEARCH complete summary={summary_path}")
    print(f"Hyperparameter search summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
