#!/usr/bin/env python3
"""HP-search runner for v6 official-like multi-positive training."""

from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))
os.environ.setdefault("DL_PROJECT_ROOT", str(ROOT))

from learned_gate_core import TRAINING_RUN_DIRNAME, progress_line  # noqa: E402
from project_core import ARTIFACTS_DIR, atomic_json_dump  # noqa: E402


DEFAULT_CONFIGS = ROOT / "configs" / "gate_v6_official_mix_3h_configs.json"
STOP_REQUESTED = False


def request_stop(signum, frame) -> None:
    del signum, frame
    global STOP_REQUESTED
    STOP_REQUESTED = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("short", "long"), default="long")
    parser.add_argument("--configs", type=Path, default=DEFAULT_CONFIGS)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--time-budget-seconds-per-run", type=int, default=0)
    parser.add_argument("--job-time-budget-seconds", type=int, default=0)
    parser.add_argument("--finalize-window-seconds", type=int, default=300)
    parser.add_argument("--min-seconds-for-new-run", type=int, default=900)
    parser.add_argument("--beta-eval-time-budget-seconds", type=int, default=0)
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def compact(value) -> str:
    text = str(value)
    return text.replace(".", "p").replace("-", "m")


def remaining_seconds(started: float, budget: int) -> float | None:
    if not budget:
        return None
    return budget - (time.monotonic() - started)


def append_summary(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


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


def checkpoint_for_run(run_dir: Path) -> Path | None:
    checkpoint = run_dir / "checkpoints" / "best_val_official_like_at10.pt"
    return checkpoint if checkpoint.exists() else None


def evaluate_beta_sweep(
    checkpoint: Path,
    output_root: Path,
    progress: Path,
    device: str,
    force: bool,
    time_budget_seconds: int,
) -> dict:
    command = [
        sys.executable,
        "orchestrator/evaluate_beta_sweep_blends.py",
        "--checkpoint",
        str(checkpoint),
        "--output-root",
        str(output_root),
        "--device",
        device,
        "--source-batch-size",
        "256",
    ]
    if time_budget_seconds:
        command.extend(["--time-budget-seconds", str(time_budget_seconds)])
    if force:
        command.append("--force")
    progress_line(progress, f"START beta_sweep checkpoint={checkpoint} output={output_root}")
    completed = subprocess.run(command, check=False)
    progress_line(progress, f"END beta_sweep status=exit_{completed.returncode} output={output_root}")

    best_path = output_root / "comparison" / "BEST_OVERALL_BETA.txt"
    result = {"beta_eval_status": f"exit_{completed.returncode}"}
    if best_path.exists():
        for line in best_path.read_text(encoding="utf-8").splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                result[f"json_{key}"] = value
    return result


def aggregate_evaluations(results_root: Path, progress: Path) -> None:
    rows = []
    for best_path in sorted(results_root.glob("*/comparison/BEST_OVERALL_BETA.txt")):
        row = {"evaluation_id": best_path.parents[1].name}
        for line in best_path.read_text(encoding="utf-8").splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                row[key] = value
        rows.append(row)
    if not rows:
        progress_line(progress, "SKIP aggregate: no beta sweep outputs yet")
        return
    rows = sorted(rows, key=lambda row: float(row.get("macro_Recall@10", 0.0)), reverse=True)
    aggregate_dir = results_root / "_aggregate"
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    output = aggregate_dir / "best_beta_sweeps.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = sorted({key for row in rows for key in row})
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    best = rows[0]
    (aggregate_dir / "BEST_OFFICIAL_MIX_METHOD.txt").write_text(
        "\n".join(f"{key}={value}" for key, value in best.items()) + "\n",
        encoding="utf-8",
    )
    progress_line(
        progress,
        "AGGREGATE best_json "
        f"evaluation_id={best.get('evaluation_id')} "
        f"method={best.get('method')} "
        f"macro_R@10={best.get('macro_Recall@10')} "
        f"micro_R@10={best.get('micro_Recall@10')} "
        f"summary={output}",
    )


def main() -> int:
    args = parse_args()
    started = time.monotonic()
    signal.signal(signal.SIGTERM, request_stop)
    if hasattr(signal, "SIGUSR1"):
        signal.signal(signal.SIGUSR1, request_stop)

    with args.configs.open(encoding="utf-8") as handle:
        configs = json.load(handle)
    if args.limit:
        configs = configs[: args.limit]

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    search_dir = ARTIFACTS_DIR / TRAINING_RUN_DIRNAME / f"hpsearch_official_mix_v6_{stamp}_{args.profile}"
    search_dir.mkdir(parents=True, exist_ok=True)
    summary_path = search_dir / "summary.csv"
    progress = search_dir / "progress.txt"
    results_root = ARTIFACTS_DIR / "results" / "official_mix_v6" / search_dir.name
    progress_line(progress, f"HPSEARCH_OFFICIAL_MIX_V6 start profile={args.profile} configs={len(configs)}")
    progress_line(progress, "objective=official_like_multi_positive + same_identity + weak_global")
    progress_line(progress, f"results_root={results_root}")

    trainer = Path("experimental") / "train_official_mix_v6.py"
    for index, config in enumerate(configs, start=1):
        remaining = remaining_seconds(started, args.job_time_budget_seconds)
        if remaining is not None and remaining <= args.min_seconds_for_new_run:
            progress_line(progress, f"STOP before next config remaining_seconds={remaining:.0f}")
            break
        if STOP_REQUESTED:
            progress_line(progress, "STOP before next config due to signal")
            break

        config = dict(config)
        config["profile"] = args.profile
        config_path = search_dir / f"{config['config_id']}.json"
        atomic_json_dump(config, config_path)
        run_name = (
            f"official_mix_v6_{config['config_id']}"
            f"_off{compact(config.get('official_pair_fraction', 0.0))}"
            f"_same{compact(config.get('same_pair_fraction', 0.0))}"
            f"_weak{compact(config.get('weak_pair_fraction', 0.0))}"
            f"_lr{compact(config.get('learning_rate'))}"
            f"_beta{compact(config.get('blend_beta'))}"
            f"_{args.profile}"
        )
        run_dir = search_dir / run_name
        command = [
            sys.executable,
            str(trainer),
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
        completed = subprocess.run(command, check=False)
        status = "complete" if completed.returncode == 0 else f"exit_{completed.returncode}"
        best = read_best_metrics(run_dir)
        beta_eval = {}
        checkpoint = checkpoint_for_run(run_dir)
        if status == "complete" and checkpoint is not None and not args.skip_eval:
            beta_eval = evaluate_beta_sweep(
                checkpoint,
                results_root / config["config_id"],
                progress,
                args.device,
                args.force,
                args.beta_eval_time_budget_seconds,
            )
            aggregate_evaluations(results_root, progress)

        row = {
            "config_id": config["config_id"],
            "run_dir": str(run_dir),
            "status": status,
            "official_pair_fraction": config.get("official_pair_fraction", ""),
            "same_pair_fraction": config.get("same_pair_fraction", ""),
            "weak_pair_fraction": config.get("weak_pair_fraction", ""),
            "official_positives_per_group": config.get("official_positives_per_group", ""),
            "learning_rate": config.get("learning_rate", ""),
            "blend_beta": config.get("blend_beta", ""),
            "lambda_triplet": config.get("lambda_triplet", ""),
            "lambda_source": config.get("lambda_source", ""),
            "multipositive_weight": config.get("multipositive_weight", ""),
            "max_steps": config.get("max_steps", ""),
            "batch_size": config.get("batch_size", ""),
            **best,
            **beta_eval,
        }
        append_summary(summary_path, row)
        progress_line(progress, f"END config={config['config_id']} status={status} best={best} beta_eval={beta_eval}")

        remaining = remaining_seconds(started, args.job_time_budget_seconds)
        if remaining is not None and remaining <= args.finalize_window_seconds:
            progress_line(progress, f"FINALIZE window reached remaining_seconds={remaining:.0f}")
            break

    aggregate_evaluations(results_root, progress)
    progress_line(progress, f"HPSEARCH_OFFICIAL_MIX_V6 complete summary={summary_path} results={results_root}")
    print(f"Official mix v6 hpsearch summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
