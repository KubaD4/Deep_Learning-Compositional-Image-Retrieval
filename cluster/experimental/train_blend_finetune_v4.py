#!/usr/bin/env python3
"""Fine-tune the best learned gate on the final model+sum blended query.

This is an experimental v4 trainer. It intentionally lives outside
``cluster/scripts`` so the current final system remains untouched.

The current best inference system is:

    q_final = normalize(q_model + beta * (q_sum - source))

Earlier training optimized ``q_model`` only, and the arithmetic correction was
added afterwards. This trainer optimizes the blended ``q_final`` directly and
can add a hard-negative triplet-style loss on top of the existing exact and
multi-positive InfoNCE losses.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import signal
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))
os.environ.setdefault("DL_PROJECT_ROOT", str(ROOT))

from learned_gate_core import (  # noqa: E402
    TRAINING_PAIR_DIRNAME,
    TRAINING_RUN_DIRNAME,
    create_model_from_config,
    dump_config,
    load_image_embedding_cache,
    load_prompt_embedding_cache,
    progress_line,
    save_checkpoint,
    signed_condition_text,
    write_csv_rows,
)
from project_core import ARTIFACTS_DIR, EMBEDDING_DIR, TOP_KS, choose_device, load_torch  # noqa: E402
from train_gate_model import (  # noqa: E402
    false_negative_mask,
    load_pair_index,
    make_batch,
    multi_positive_contrastive_loss,
    official_like_mask,
    plot_metric_curves,
    plot_validation_artifacts,
    positions_by_length,
    sample_positions,
    source_similarity_filter,
    topk_attribute_success,
)


STOP_REQUESTED = False


def request_stop(signum, frame) -> None:
    del signum, frame
    global STOP_REQUESTED
    STOP_REQUESTED = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("short", "long"), default="long")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--time-budget-seconds", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def resolve_project_path(path_value: str | os.PathLike | None) -> Path | None:
    if not path_value:
        return None
    path = Path(path_value)
    return path if path.is_absolute() else ROOT / path


def compact_float(value: float) -> str:
    text = f"{value:.0e}" if value < 0.001 else f"{value:g}"
    return text.replace("-", "m").replace("+", "")


def find_default_start_checkpoint() -> Path:
    candidates = [
        ROOT / "final_best_system" / "weights" / "best_val_official_like_at10.pt",
    ]
    candidates.extend(
        sorted(
            (ARTIFACTS_DIR / TRAINING_RUN_DIRNAME).glob(
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
    raise FileNotFoundError(
        "Could not find the current best checkpoint. Expected either "
        "final_best_system/weights/best_val_official_like_at10.pt or the "
        "hybmp_l002 training artifact under artifacts/training_runs."
    )


def load_config(path: Path, profile: str) -> dict:
    with path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    config["profile"] = profile
    config.setdefault("composer_type", "sequential_gate")
    config.setdefault("condition_mode", "signed_direction")
    config.setdefault(
        "prompt_cache_path",
        "data/celeba/embeddings/openai_clip_vit_b32/"
        "signed_attribute_prompt_embeddings_v2_photo_templates.pt",
    )
    config.setdefault("sampler_mode", "balanced_length")
    config.setdefault("batch_size", 256)
    config.setdefault("epochs", 40)
    config.setdefault("steps_per_epoch", 2000)
    config.setdefault("max_steps", 16000)
    config.setdefault("validate_every_steps", 4000)
    config.setdefault("val_max_queries", 4096)
    config.setdefault("val_batch_size", 256)
    config.setdefault("learning_rate", 5e-5)
    config.setdefault("weight_decay", 1e-4)
    config.setdefault("temperature", 0.02)
    config.setdefault("lambda_target", 0.1)
    config.setdefault("lambda_source", 0.02)
    config.setdefault("false_negative_hamming", 2)
    config.setdefault("residual_scale", 0.02)
    config.setdefault("edit_scale", 1.0)
    config.setdefault("gate_max", 1.5)
    config.setdefault("dropout", 0.1)
    config.setdefault("seed", 123)
    config.setdefault("blend_beta", 1.0)
    config.setdefault("sum_alpha", 1.0)
    config.setdefault("sum_source_weight", 1.0)
    config.setdefault("lambda_triplet", 0.0)
    config.setdefault("triplet_margin", 0.05)
    config.setdefault("lambda_model_aux", 0.0)
    config.setdefault("lambda_distill", 0.0)
    config.setdefault("use_multipositive_loss", True)
    config.setdefault("multipositive_weight", 0.5)
    config.setdefault("multipositive_hamming", 2)
    config.setdefault("multipositive_top_fraction", 0.15)
    return config


def load_text_bank() -> dict:
    path = EMBEDDING_DIR / "attribute_text_embeddings.pt"
    if not path.exists():
        raise FileNotFoundError(f"Missing CLIP text direction cache at {path}")
    bank = load_torch(path)
    if "directions" not in bank:
        raise RuntimeError(f"Text cache at {path} has no 'directions' tensor")
    return bank


def generic_sum_query(
    source: torch.Tensor,
    attr_indices: torch.Tensor,
    signs: torch.Tensor,
    text_bank: dict,
    alpha: float = 1.0,
    source_weight: float = 1.0,
) -> torch.Tensor:
    """Build the generic CLIP arithmetic query used by the best final system."""
    source = F.normalize(source.float(), dim=-1)
    directions = F.normalize(text_bank["directions"].float().to(source.device), dim=-1)
    mask = attr_indices >= 0
    safe_attrs = attr_indices.clamp_min(0)
    signed = directions[safe_attrs] * signs.float().unsqueeze(-1)
    edit = (signed * mask.unsqueeze(-1)).sum(dim=1)
    return F.normalize(float(source_weight) * source + float(alpha) * edit, dim=-1)


def blended_query(
    model_query: torch.Tensor,
    sum_query: torch.Tensor,
    source: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    """Apply the model+arithmetic vector-delta correction."""
    return F.normalize(
        F.normalize(model_query.float(), dim=-1)
        + float(beta) * (F.normalize(sum_query.float(), dim=-1) - F.normalize(source.float(), dim=-1)),
        dim=-1,
    )


def hard_triplet_loss(
    query: torch.Tensor,
    target: torch.Tensor,
    invalid_negatives: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    """Batch-hard cosine triplet loss with official-like false negatives masked."""
    scores = F.normalize(query, dim=-1) @ F.normalize(target, dim=-1).T
    positive = scores.diagonal()
    negative_scores = scores.masked_fill(invalid_negatives, -torch.inf)
    hard_negative = negative_scores.max(dim=1).values
    valid_rows = torch.isfinite(hard_negative)
    if not bool(valid_rows.any()):
        return torch.zeros((), device=query.device)
    return F.relu(float(margin) + hard_negative[valid_rows] - positive[valid_rows]).mean()


def compute_losses(model, teacher_model, batch, embeddings, attrs, prompt_cache, text_bank, config, device):
    src_idx_cpu = batch["source_indices"].long()
    tgt_idx_cpu = batch["target_indices"].long()
    attr_indices = batch["attr_indices"].to(device)
    signs = batch["signs"].to(device)

    source = embeddings[src_idx_cpu].float().to(device)
    target = embeddings[tgt_idx_cpu].float().to(device)
    source_attrs = attrs[src_idx_cpu].to(device)
    target_attrs = attrs[tgt_idx_cpu].to(device)

    from learned_gate_core import condition_embeddings

    conditions, mask = condition_embeddings(
        prompt_cache,
        attr_indices,
        signs,
        device,
        str(config.get("condition_mode", "signed_direction")),
    )

    q_model, alpha, _ = model(source, conditions, mask)
    q_sum = generic_sum_query(
        source,
        attr_indices,
        signs,
        text_bank,
        alpha=float(config.get("sum_alpha", 1.0)),
        source_weight=float(config.get("sum_source_weight", 1.0)),
    )
    q_final = blended_query(q_model, q_sum, source, float(config.get("blend_beta", 1.0)))

    labels = torch.arange(len(src_idx_cpu), device=device)
    eye = torch.eye(len(src_idx_cpu), dtype=torch.bool, device=device)

    false_negatives = false_negative_mask(
        source_attrs,
        target_attrs,
        attr_indices,
        signs,
        int(config["false_negative_hamming"]),
    )
    official_like = official_like_mask(
        source_attrs,
        target_attrs,
        attr_indices,
        signs,
        int(config.get("multipositive_hamming", config["false_negative_hamming"])),
    )
    source_like = source_similarity_filter(source, target, config)
    positive_mask = eye | (official_like & source_like)
    neutral_mask = official_like & ~positive_mask

    scores = (q_final @ F.normalize(target, dim=-1).T) / float(config["temperature"])
    exact_scores = scores.masked_fill(false_negatives, -torch.inf)
    exact_info_nce = F.cross_entropy(exact_scores, labels)

    if bool(config.get("use_multipositive_loss", True)):
        multipositive_info_nce = multi_positive_contrastive_loss(scores, positive_mask, neutral_mask)
        multipositive_weight = float(config.get("multipositive_weight", 0.5))
        info_nce = (
            (1.0 - multipositive_weight) * exact_info_nce
            + multipositive_weight * multipositive_info_nce
        )
        positive_count = positive_mask.float().sum(dim=1).mean()
    else:
        multipositive_info_nce = torch.zeros((), device=device)
        multipositive_weight = 0.0
        info_nce = exact_info_nce
        positive_count = torch.ones((), device=device)

    invalid_negatives = eye | false_negatives | official_like
    triplet = hard_triplet_loss(
        q_final,
        target,
        invalid_negatives=invalid_negatives,
        margin=float(config.get("triplet_margin", 0.05)),
    )

    model_aux = torch.zeros((), device=device)
    if float(config.get("lambda_model_aux", 0.0)) > 0:
        model_scores = (F.normalize(q_model, dim=-1) @ F.normalize(target, dim=-1).T) / float(config["temperature"])
        model_aux = F.cross_entropy(model_scores.masked_fill(false_negatives, -torch.inf), labels)

    distill = torch.zeros((), device=device)
    if teacher_model is not None and float(config.get("lambda_distill", 0.0)) > 0:
        # Use no_grad rather than inference_mode: q_teacher participates in a
        # loss with q_final, and inference tensors can break backward.
        with torch.no_grad():
            teacher_q_model, _, _ = teacher_model(source, conditions, mask)
            teacher_q_final = blended_query(
                teacher_q_model,
                q_sum,
                source,
                float(config.get("blend_beta", 1.0)),
            )
        distill = 1.0 - (q_final * teacher_q_final).sum(dim=-1).mean()

    target_cos = 1.0 - (q_final * F.normalize(target, dim=-1)).sum(dim=-1).mean()
    source_cos = 1.0 - (q_final * F.normalize(source, dim=-1)).sum(dim=-1).mean()

    loss = (
        info_nce
        + float(config["lambda_target"]) * target_cos
        + float(config["lambda_source"]) * source_cos
        + float(config.get("lambda_triplet", 0.0)) * triplet
        + float(config.get("lambda_model_aux", 0.0)) * model_aux
        + float(config.get("lambda_distill", 0.0)) * distill
    )

    return loss, {
        "loss": float(loss.detach().cpu()),
        "info_nce": float(info_nce.detach().cpu()),
        "exact_info_nce": float(exact_info_nce.detach().cpu()),
        "multipositive_info_nce": float(multipositive_info_nce.detach().cpu()),
        "multipositive_weight": float(multipositive_weight),
        "triplet_loss": float(triplet.detach().cpu()),
        "model_aux_loss": float(model_aux.detach().cpu()),
        "distill_loss": float(distill.detach().cpu()),
        "target_cosine_loss": float(target_cos.detach().cpu()),
        "source_preservation_loss": float(source_cos.detach().cpu()),
        "cos_final_target": float((q_final * F.normalize(target, dim=-1)).sum(dim=-1).mean().detach().cpu()),
        "cos_final_source": float((q_final * F.normalize(source, dim=-1)).sum(dim=-1).mean().detach().cpu()),
        "cos_model_target": float((F.normalize(q_model, dim=-1) * F.normalize(target, dim=-1)).sum(dim=-1).mean().detach().cpu()),
        "cos_sum_target": float((F.normalize(q_sum, dim=-1) * F.normalize(target, dim=-1)).sum(dim=-1).mean().detach().cpu()),
        "gate_mean": float(alpha[mask].mean().detach().cpu()) if bool(mask.any()) else 0.0,
        "positive_count": float(positive_count.detach().cpu()),
    }


def validate_blend(model, valid_index, valid_embeddings, prompt_cache, text_bank, config, device, epoch, run_dir):
    model.eval()
    gallery = F.normalize(valid_embeddings.float(), dim=-1).to(device)
    gallery_attrs = valid_index["attrs"].to(device)
    total_pairs = len(valid_index["source_indices"])
    max_queries = int(config["val_max_queries"])
    if max_queries and max_queries < total_pairs:
        generator = torch.Generator().manual_seed(int(config["seed"]) + epoch)
        positions = torch.randperm(total_pairs, generator=generator)[:max_queries]
    else:
        positions = torch.arange(total_pairs)

    totals = {f"val_exact_R@{k}": 0.0 for k in TOP_KS}
    totals.update({f"val_attr_success@{k}": 0.0 for k in TOP_KS})
    totals.update({f"val_official_like@{k}": 0.0 for k in TOP_KS})
    length_totals = defaultdict(lambda: Counter())
    attr_totals = defaultdict(lambda: Counter())
    rank_sum = 0.0
    gate_sums = torch.zeros((2, len(valid_index["attributes"])), dtype=torch.float64)
    gate_counts = torch.zeros((2, len(valid_index["attributes"])), dtype=torch.float64)
    examples = []

    from learned_gate_core import condition_embeddings

    with torch.inference_mode():
        for start in range(0, len(positions), int(config["val_batch_size"])):
            batch_pos = positions[start : start + int(config["val_batch_size"])]
            src_idx = valid_index["source_indices"][batch_pos].to(device)
            tgt_idx = valid_index["target_indices"][batch_pos].to(device)
            attr_indices = valid_index["attr_indices"][batch_pos].to(device)
            signs = valid_index["signs"][batch_pos].to(device)
            lengths = valid_index["query_lengths"][batch_pos]
            source = gallery[src_idx]
            source_attrs = gallery_attrs[src_idx]
            conditions, mask = condition_embeddings(
                prompt_cache,
                attr_indices,
                signs,
                device,
                str(config.get("condition_mode", "signed_direction")),
            )
            q_model, alpha, _ = model(source, conditions, mask)
            q_sum = generic_sum_query(
                source,
                attr_indices,
                signs,
                text_bank,
                alpha=float(config.get("sum_alpha", 1.0)),
                source_weight=float(config.get("sum_source_weight", 1.0)),
            )
            query = blended_query(q_model, q_sum, source, float(config.get("blend_beta", 1.0)))

            scores = query @ gallery.T
            scores[torch.arange(len(src_idx), device=device), src_idx] = -torch.inf
            target_scores = scores[torch.arange(len(src_idx), device=device), tgt_idx]
            rank_sum += float((scores > target_scores[:, None]).sum(dim=1).float().add(1).sum().cpu())
            top_indices = scores.topk(max(TOP_KS), dim=1).indices
            query_ok, official_like = topk_attribute_success(
                top_indices,
                source_attrs,
                gallery_attrs,
                attr_indices,
                signs,
            )
            exact_matrix = top_indices == tgt_idx[:, None]
            for k in TOP_KS:
                exact_hits = exact_matrix[:, :k].any(dim=1).float()
                attr_hits = query_ok[:, :k].any(dim=1).float()
                official_hits = official_like[:, :k].any(dim=1).float()
                totals[f"val_exact_R@{k}"] += float(exact_hits.sum().cpu())
                totals[f"val_attr_success@{k}"] += float(attr_hits.sum().cpu())
                totals[f"val_official_like@{k}"] += float(official_hits.sum().cpu())
                for row, query_len in enumerate(lengths.tolist()):
                    length_totals[int(query_len)][f"exact@{k}"] += float(exact_hits[row].cpu())
                    length_totals[int(query_len)][f"attr@{k}"] += float(attr_hits[row].cpu())
                    length_totals[int(query_len)][f"official@{k}"] += float(official_hits[row].cpu())
                    length_totals[int(query_len)]["count"] += 1 if k == TOP_KS[0] else 0

            for row in range(len(batch_pos)):
                length = int(lengths[row])
                for pos in range(length):
                    attr = int(attr_indices[row, pos])
                    sign_slot = 1 if int(signs[row, pos]) > 0 else 0
                    gate_sums[sign_slot, attr] += float(alpha[row, pos].detach().cpu())
                    gate_counts[sign_slot, attr] += 1
                    attr_totals[attr]["count"] += 1
                    attr_totals[attr]["official@10"] += float(official_like[row, :10].any().cpu())

            if len(examples) < 8:
                top_cpu = top_indices[: 8 - len(examples)].cpu().tolist()
                for local_row, ranking in enumerate(top_cpu):
                    row = local_row
                    length = int(lengths[row])
                    examples.append(
                        {
                            "source_index": int(src_idx[row].cpu()),
                            "target_index": int(tgt_idx[row].cpu()),
                            "query": signed_condition_text(
                                attr_indices[row].cpu(),
                                signs[row].cpu(),
                                length,
                                valid_index["attributes"],
                            ),
                            "top10": ranking,
                        }
                    )

    n = len(positions)
    metrics = {name: value / n for name, value in totals.items()}
    metrics["val_mean_rank_B"] = rank_sum / n
    metrics["val_queries"] = n
    metrics["epoch"] = epoch
    plot_validation_artifacts(
        run_dir,
        metrics,
        length_totals,
        attr_totals,
        gate_sums,
        gate_counts,
        valid_index,
        examples,
        epoch,
    )
    model.train()
    return metrics


def load_start_weights(model: torch.nn.Module, checkpoint_path: Path, progress: Path) -> None:
    checkpoint = load_torch(checkpoint_path)
    missing, unexpected = model.load_state_dict(checkpoint["model_state"], strict=False)
    if missing or unexpected:
        progress_line(
            progress,
            "start checkpoint loaded non-strict "
            f"missing={len(missing)} unexpected={len(unexpected)}",
        )
    else:
        progress_line(progress, f"start checkpoint loaded strict-compatible {checkpoint_path}")


def main() -> int:
    args = parse_args()
    started = time.monotonic()
    signal.signal(signal.SIGTERM, request_stop)
    if hasattr(signal, "SIGUSR1"):
        signal.signal(signal.SIGUSR1, request_stop)

    config = load_config(args.config, args.profile)
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
    progress_line(progress, "TRAINING_OBJECTIVE q_final = normalize(q_model + beta * (q_sum - source))")

    train_cache = load_image_embedding_cache("train")
    valid_cache = load_image_embedding_cache("valid")
    prompt_cache = load_prompt_embedding_cache(resolve_project_path(config.get("prompt_cache_path")))
    text_bank = load_text_bank()
    train_index = load_pair_index("train")
    valid_index = load_pair_index("valid")
    if prompt_cache["attributes"] != train_index["attributes"]:
        raise RuntimeError("Prompt cache attributes do not match pair index")
    if text_bank["attributes"] != train_index["attributes"]:
        raise RuntimeError("Text bank attributes do not match pair index")

    train_embeddings = train_cache["embeddings"].float()
    valid_embeddings = valid_cache["embeddings"].float()
    train_attrs = train_index["attrs"]
    by_length = positions_by_length(train_index)

    model = create_model_from_config(config).to(device)
    checkpoint_path = resolve_project_path(config.get("start_checkpoint_path")) or find_default_start_checkpoint()
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
    best = {
        "val_official_like@10": -1.0,
        "val_exact_R@10": -1.0,
    }
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
    max_steps = int(config.get("max_steps", 0))
    model.train()

    for epoch in range(start_epoch, int(config["epochs"]) + 1):
        for _ in range(int(config["steps_per_epoch"])):
            positions = sample_positions(train_index, by_length, config)
            batch = make_batch(train_index, positions)
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            global_step += 1
            train_count += 1
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
                    f"cos_sum_target={loss_parts['cos_sum_target']:.4f} "
                    f"pos_count={loss_parts['positive_count']:.2f}",
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
                row = {
                    "epoch": epoch,
                    "step": global_step,
                    **train_means,
                    **val_metrics,
                }
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
