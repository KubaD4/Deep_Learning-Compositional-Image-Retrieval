#!/usr/bin/env python3
"""Train v6 with official-like multi-positive sets plus regularizers.

This trainer keeps the current final-system architecture intact:

    q_final = normalize(q_model + beta * (q_sum - source))

The change is the training mixture:

- official-like multi-positive rows from the train split only;
- same-identity rows from the original train-pair index;
- weak/global official-like oversampling rows from the train split only.

No official JSON target list is used here. The JSON is reserved for the
hpsearch runner's post-training evaluation.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import signal
import sys
import time
from collections import Counter
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
EXPERIMENTAL = ROOT / "experimental"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(EXPERIMENTAL))
os.environ.setdefault("DL_PROJECT_ROOT", str(ROOT))

from learned_gate_core import (  # noqa: E402
    create_model_from_config,
    dump_config,
    load_image_embedding_cache,
    load_prompt_embedding_cache,
    progress_line,
    save_checkpoint,
    write_csv_rows,
)
from project_core import choose_device, load_torch  # noqa: E402
from train_gate_model import (  # noqa: E402
    load_pair_index,
    make_batch,
    plot_metric_curves,
    positions_by_length,
    sample_positions,
)
from train_blend_finetune_v4 import (  # noqa: E402
    compute_losses,
    load_config,
    load_start_weights,
    load_text_bank,
    resolve_project_path,
    validate_blend,
)


STOP_REQUESTED = False


def request_stop(signum, frame) -> None:
    del signum, frame
    global STOP_REQUESTED
    STOP_REQUESTED = True


def find_default_start_checkpoint_v6() -> Path:
    candidates = [
        ROOT / "final_best_system" / "weights" / "best_val_official_like_at10.pt",
        ROOT.parent / "final_best_system" / "weights" / "best_val_official_like_at10.pt",
    ]
    candidates.extend(
        sorted(
            (ROOT / "artifacts" / "training_runs").glob(
                "hpsearch_mixed_weak_v5_*_long/"
                "mixed_v5_mw85_012*/"
                "checkpoints/best_val_official_like_at10.pt"
            ),
            reverse=True,
        )
    )
    candidates.extend(
        sorted(
            (ROOT / "artifacts" / "training_runs").glob(
                "hpsearch_gate_v3_*_long/"
                "gate_v3_sequentialgate_hybmp_l002*/"
                "checkpoints/best_val_official_like_at10.pt"
            ),
            reverse=True,
        )
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Could not find a v5/v3 start checkpoint for v6 training")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("short", "long"), default="long")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--time-budget-seconds", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def load_pair_index_from_path(path: str | os.PathLike | None, default_split: str) -> dict:
    if path:
        resolved = resolve_project_path(path)
        if resolved is None or not resolved.exists():
            raise FileNotFoundError(f"Missing pair index: {resolved}")
        return load_torch(resolved)
    return load_pair_index(default_split)


def batch_from_positions(index: dict, positions: torch.Tensor) -> dict:
    batch = make_batch(index, positions)
    if "group_ids" in index:
        batch["group_ids"] = index["group_ids"][positions]
    return batch


def sample_from(index: dict, by_length: dict[int, torch.Tensor], count: int, config: dict) -> dict:
    if count <= 0:
        return {}
    local_config = dict(config)
    local_config["batch_size"] = count
    positions = sample_positions(index, by_length, local_config)
    return batch_from_positions(index, positions)


def positions_by_group(index: dict) -> list[torch.Tensor]:
    group_ids = index.get("group_ids")
    if group_ids is None:
        raise RuntimeError("Official-like v6 index must contain group_ids")
    order = torch.argsort(group_ids)
    ordered_groups = group_ids[order]
    _, counts = torch.unique_consecutive(ordered_groups, return_counts=True)
    return list(torch.split(order, counts.tolist()))


def sample_official_groups(index: dict, group_positions: list[torch.Tensor], count: int, config: dict) -> dict:
    if count <= 0:
        return {}
    positives_per_group = max(1, int(config.get("official_positives_per_group", 2)))
    group_count = max(1, (count + positives_per_group - 1) // positives_per_group)
    selected_groups = torch.randint(0, len(group_positions), (group_count,))
    positions: list[torch.Tensor] = []
    for group_id in selected_groups.tolist():
        pool = group_positions[group_id]
        if len(pool) >= positives_per_group:
            choice = pool[torch.randperm(len(pool))[:positives_per_group]]
        else:
            repeats = torch.randint(0, len(pool), (positives_per_group,))
            choice = pool[repeats]
        positions.append(choice)
    sampled = torch.cat(positions)
    if len(sampled) > count:
        sampled = sampled[:count]
    elif len(sampled) < count:
        extra = torch.randint(0, len(index["source_indices"]), (count - len(sampled),))
        sampled = torch.cat([sampled, extra])
    return batch_from_positions(index, sampled)


def concat_batches(parts: list[dict]) -> dict:
    parts = [part for part in parts if part]
    if len(parts) == 1:
        return parts[0]
    keys = set().union(*(part.keys() for part in parts))
    result = {}
    for key in keys:
        values = [part[key] for part in parts if key in part]
        if len(values) == len(parts):
            result[key] = torch.cat(values, dim=0)
    return result


def source_counts(config: dict, has_official: bool, has_weak: bool) -> tuple[int, int, int]:
    batch_size = int(config["batch_size"])
    official_fraction = float(config.get("official_pair_fraction", 0.0)) if has_official else 0.0
    same_fraction = float(config.get("same_pair_fraction", 1.0))
    weak_fraction = float(config.get("weak_pair_fraction", 0.0)) if has_weak else 0.0
    total = official_fraction + same_fraction + weak_fraction
    if total <= 0:
        raise ValueError("At least one pair fraction must be positive")
    official_count = int(round(batch_size * official_fraction / total))
    weak_count = int(round(batch_size * weak_fraction / total))
    official_count = max(0, min(batch_size, official_count))
    weak_count = max(0, min(batch_size - official_count, weak_count))
    same_count = batch_size - official_count - weak_count
    return official_count, same_count, weak_count


def mixed_batch(
    official_index: dict | None,
    same_index: dict,
    weak_index: dict | None,
    official_groups: list[torch.Tensor] | None,
    weak_groups: list[torch.Tensor] | None,
    same_by_length: dict[int, torch.Tensor],
    weak_by_length: dict[int, torch.Tensor] | None,
    config: dict,
) -> tuple[dict, dict[str, int]]:
    official_count, same_count, weak_count = source_counts(
        config,
        has_official=official_index is not None,
        has_weak=weak_index is not None,
    )
    parts = []
    if official_count and official_index is not None and official_groups is not None:
        parts.append(sample_official_groups(official_index, official_groups, official_count, config))
    if same_count:
        parts.append(sample_from(same_index, same_by_length, same_count, config))
    if weak_count and weak_index is not None and weak_by_length is not None:
        if weak_groups is not None:
            parts.append(sample_official_groups(weak_index, weak_groups, weak_count, config))
        else:
            parts.append(sample_from(weak_index, weak_by_length, weak_count, config))
    batch = concat_batches(parts)
    order = torch.randperm(len(batch["source_indices"]))
    batch = {key: value[order] for key, value in batch.items()}
    return batch, {"official": official_count, "same": same_count, "weak": weak_count}


def ensure_attrs_match(reference: dict, *indices: dict | None) -> None:
    for index in indices:
        if index is None:
            continue
        if reference["attributes"] != index["attributes"]:
            raise RuntimeError("Pair index attributes do not match")


def main() -> int:
    args = parse_args()
    started = time.monotonic()
    signal.signal(signal.SIGTERM, request_stop)
    if hasattr(signal, "SIGUSR1"):
        signal.signal(signal.SIGUSR1, request_stop)

    config = load_config(args.config, args.profile)
    config.setdefault("official_pair_fraction", 0.6)
    config.setdefault("same_pair_fraction", 0.3)
    config.setdefault("weak_pair_fraction", 0.1)
    config.setdefault("official_positives_per_group", 2)
    random.seed(int(config["seed"]))
    torch.manual_seed(int(config["seed"]))
    device = choose_device(args.device)

    run_dir = args.run_dir
    if run_dir.exists() and any(run_dir.iterdir()) and not args.force:
        raise RuntimeError(f"Run dir already exists and is not empty: {run_dir}")
    for child in ("checkpoints", "plots", "samples", "logs"):
        (run_dir / child).mkdir(parents=True, exist_ok=True)
    dump_config(config, run_dir / "config.json")

    progress = run_dir / "progress.txt"
    progress_line(progress, f"RUN_DIR {run_dir}")
    progress_line(progress, f"config {json.dumps(config, sort_keys=True)}")
    progress_line(
        progress,
        "TRAINING_OBJECTIVE v6 official_like_multi_positive + same_identity + weak_global "
        "q_final=normalize(q_model + beta*(q_sum-source)) "
        f"fractions={config.get('official_pair_fraction')}/"
        f"{config.get('same_pair_fraction')}/{config.get('weak_pair_fraction')}",
    )

    train_cache = load_image_embedding_cache("train")
    valid_cache = load_image_embedding_cache("valid")
    prompt_cache = load_prompt_embedding_cache(resolve_project_path(config.get("prompt_cache_path")))
    text_bank = load_text_bank()

    same_index = load_pair_index_from_path(config.get("same_pair_index_path"), "train")
    official_index = None
    if float(config.get("official_pair_fraction", 0.0)) > 0:
        official_index = load_pair_index_from_path(config.get("official_pair_index_path"), "train")
    weak_index = None
    if float(config.get("weak_pair_fraction", 0.0)) > 0:
        weak_index = load_pair_index_from_path(config.get("weak_pair_index_path"), "train")
    valid_index = load_pair_index_from_path(config.get("valid_pair_index_path"), "valid")

    ensure_attrs_match(same_index, official_index, weak_index, valid_index)
    if prompt_cache["attributes"] != same_index["attributes"]:
        raise RuntimeError("Prompt cache attributes do not match pair index")
    if text_bank["attributes"] != same_index["attributes"]:
        raise RuntimeError("Text bank attributes do not match pair index")

    train_embeddings = train_cache["embeddings"].float()
    valid_embeddings = valid_cache["embeddings"].float()
    train_attrs = same_index["attrs"]
    same_by_length = positions_by_length(same_index)
    official_groups = positions_by_group(official_index) if official_index is not None else None
    weak_by_length = positions_by_length(weak_index) if weak_index is not None else None
    weak_groups = positions_by_group(weak_index) if weak_index is not None and "group_ids" in weak_index else None

    progress_line(
        progress,
        "pair_counts "
        f"official_rows={len(official_index['source_indices']) if official_index is not None else 0} "
        f"official_groups={len(official_groups) if official_groups is not None else 0} "
        f"same={len(same_index['source_indices'])} "
        f"weak_rows={len(weak_index['source_indices']) if weak_index is not None else 0} "
        f"same_lengths={same_index.get('counts_by_query_len')} "
        f"official_lengths={official_index.get('counts_by_query_len') if official_index is not None else None} "
        f"weak_lengths={weak_index.get('counts_by_query_len') if weak_index is not None else None}",
    )

    model = create_model_from_config(config).to(device)
    checkpoint_path = resolve_project_path(config.get("start_checkpoint_path")) or find_default_start_checkpoint_v6()
    load_start_weights(model, checkpoint_path, progress)

    teacher_model = None
    if float(config.get("lambda_distill", 0.0)) > 0:
        teacher_model = create_model_from_config(config).to(device)
        load_start_weights(teacher_model, checkpoint_path, progress)
        teacher_model.eval()
        for parameter in teacher_model.parameters():
            parameter.requires_grad_(False)
        progress_line(progress, f"teacher model enabled lambda_distill={config['lambda_distill']}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )

    latest_path = run_dir / "checkpoints" / "latest.pt"
    best = {"val_official_like@10": -1.0, "val_exact_R@10": -1.0}
    start_epoch = 1
    global_step = 0
    if latest_path.exists():
        checkpoint = load_torch(latest_path)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(checkpoint["step"])
        best = checkpoint.get("best", best)
        progress_line(progress, f"resumed latest.pt epoch={start_epoch} step={global_step}")

    metrics_path = run_dir / "metrics.csv"
    train_accumulator = Counter()
    train_count = 0
    seen = Counter()
    max_steps = int(config.get("max_steps", 0))
    model.train()

    for epoch in range(start_epoch, int(config["epochs"]) + 1):
        for _ in range(int(config["steps_per_epoch"])):
            batch, counts = mixed_batch(
                official_index,
                same_index,
                weak_index,
                official_groups,
                weak_groups,
                same_by_length,
                weak_by_length,
                config,
            )
            optimizer.zero_grad(set_to_none=True)
            loss, loss_parts = compute_losses(
                model,
                teacher_model,
                batch,
                train_embeddings,
                train_attrs,
                prompt_cache,
                text_bank,
                config,
                device,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.get("grad_clip", 1.0)))
            optimizer.step()

            global_step += 1
            train_count += 1
            seen.update(counts)
            for key, value in loss_parts.items():
                train_accumulator[key] += value

            if global_step % 25 == 0:
                progress_line(
                    progress,
                    "step "
                    f"epoch={epoch} step={global_step} "
                    f"loss={loss_parts['loss']:.4f} "
                    f"info_nce={loss_parts['info_nce']:.4f} "
                    f"triplet={loss_parts['triplet_loss']:.4f} "
                    f"cos_final_target={loss_parts['cos_final_target']:.4f} "
                    f"official_seen={seen['official']} same_seen={seen['same']} weak_seen={seen['weak']}",
                )

            should_validate = global_step % int(config["validate_every_steps"]) == 0
            should_stop = STOP_REQUESTED
            if args.time_budget_seconds:
                should_stop |= time.monotonic() - started >= args.time_budget_seconds
            if max_steps:
                should_stop |= global_step >= max_steps

            if should_validate or should_stop:
                train_means = {
                    f"train_{key}": value / max(1, train_count)
                    for key, value in train_accumulator.items()
                }
                train_means["train_official_rows_seen"] = seen["official"]
                train_means["train_same_rows_seen"] = seen["same"]
                train_means["train_weak_rows_seen"] = seen["weak"]
                val_metrics = validate_blend(
                    model,
                    valid_index,
                    valid_embeddings,
                    prompt_cache,
                    text_bank,
                    config,
                    device,
                    epoch,
                    run_dir,
                )
                row = {"epoch": epoch, "step": global_step, **train_means, **val_metrics}
                write_csv_rows(metrics_path, [row], append=True)
                plot_metric_curves(metrics_path, run_dir)
                save_checkpoint(latest_path, model, optimizer, config, epoch, global_step, best)
                progress_line(
                    progress,
                    "validation "
                    f"epoch={epoch} step={global_step} "
                    f"exact@10={val_metrics['val_exact_R@10']:.4f} "
                    f"attr@10={val_metrics['val_attr_success@10']:.4f} "
                    f"official_like@10={val_metrics['val_official_like@10']:.4f} "
                    f"mean_rank_B={val_metrics['val_mean_rank_B']:.1f}",
                )

                if val_metrics["val_official_like@10"] > best["val_official_like@10"]:
                    best["val_official_like@10"] = val_metrics["val_official_like@10"]
                    save_checkpoint(
                        run_dir / "checkpoints" / "best_val_official_like_at10.pt",
                        model,
                        optimizer,
                        config,
                        epoch,
                        global_step,
                        best,
                    )
                    progress_line(
                        progress,
                        "BEST_OFFICIAL_LIKE "
                        f"epoch={epoch} step={global_step} "
                        f"val_official_like@10={best['val_official_like@10']:.4f}",
                    )

                if val_metrics["val_exact_R@10"] > best["val_exact_R@10"]:
                    best["val_exact_R@10"] = val_metrics["val_exact_R@10"]
                    save_checkpoint(
                        run_dir / "checkpoints" / "best_val_exact_at10.pt",
                        model,
                        optimizer,
                        config,
                        epoch,
                        global_step,
                        best,
                    )
                    progress_line(
                        progress,
                        "BEST_EXACT "
                        f"epoch={epoch} step={global_step} "
                        f"val_exact_R@10={best['val_exact_R@10']:.4f}",
                    )

                train_accumulator = Counter()
                train_count = 0
                seen = Counter()

            if should_stop:
                if max_steps and global_step >= max_steps and not STOP_REQUESTED:
                    (run_dir / "COMPLETE").write_text("complete\n", encoding="utf-8")
                    progress_line(progress, f"COMPLETE max_steps epoch={epoch} step={global_step}")
                    return 0
                progress_line(progress, f"checkpoint stop epoch={epoch} step={global_step}")
                return 3

    save_checkpoint(latest_path, model, optimizer, config, int(config["epochs"]), global_step, best)
    (run_dir / "COMPLETE").write_text("complete\n", encoding="utf-8")
    progress_line(progress, f"COMPLETE step={global_step}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
