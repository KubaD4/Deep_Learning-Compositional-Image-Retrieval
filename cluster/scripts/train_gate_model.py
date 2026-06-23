#!/usr/bin/env python3
"""Train the learned gated residual CLIP retrieval model."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import signal
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from learned_gate_core import (
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
from project_core import ARTIFACTS_DIR, CELEBA_DIR, ROOT, TOP_KS, choose_device, load_torch


STOP_REQUESTED = False


def request_stop(signum, frame) -> None:
    del signum, frame
    global STOP_REQUESTED
    STOP_REQUESTED = True


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("short", "long"), default="short")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--run-root", type=Path, default=ARTIFACTS_DIR / TRAINING_RUN_DIRNAME)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--time-budget-seconds", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def default_config(profile: str) -> dict:
    if profile == "short":
        return {
            "config_id": "manual",
            "profile": "short",
            "batch_size": 128,
            "epochs": 40,
            "steps_per_epoch": 25,
            "max_steps": 300,
            "validate_every_steps": 100,
            "val_max_queries": 512,
            "val_batch_size": 128,
            "sampler_mode": "balanced_length",
            "learning_rate": 1e-4,
            "weight_decay": 1e-4,
            "temperature": 0.07,
            "lambda_target": 0.1,
            "lambda_source": 0.02,
            "false_negative_hamming": 2,
            "composer_type": "residual_only",
            "condition_mode": "signed_prompt",
            "dropout": 0.1,
            "residual_scale": 0.1,
            "seed": 123,
        }
    return {
        "config_id": "manual",
        "profile": "long",
        "batch_size": 256,
        "epochs": 30,
        "steps_per_epoch": 2000,
        "max_steps": 0,
        "validate_every_steps": 2000,
        "val_max_queries": 4096,
        "val_batch_size": 256,
        "sampler_mode": "balanced_length",
        "learning_rate": 1e-4,
        "weight_decay": 1e-4,
        "temperature": 0.07,
        "lambda_target": 0.1,
        "lambda_source": 0.02,
        "false_negative_hamming": 2,
        "composer_type": "residual_only",
        "condition_mode": "signed_prompt",
        "dropout": 0.1,
        "residual_scale": 0.1,
        "seed": 123,
    }


def load_config(args) -> dict:
    config = default_config(args.profile)
    if args.config:
        with args.config.open(encoding="utf-8") as handle:
            loaded = json.load(handle)
        config.update(loaded)
    config["profile"] = args.profile
    return config


def compact_float(value: float) -> str:
    text = f"{value:.0e}" if value < 0.001 else f"{value:g}"
    return text.replace("-", "m").replace("+", "")


def make_run_dir(config: dict, run_root: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    composer = str(config.get("composer_type", "residual_only")).replace("_", "")
    name = (
        f"gate_v2_{composer}_{config['config_id']}_{config['sampler_mode']}"
        f"_b{config['batch_size']}_lr{compact_float(config['learning_rate'])}"
        f"_src{config['lambda_source']}_scale{config['residual_scale']}"
        f"_{stamp}_{config['profile']}"
    )
    return run_root / name


def pair_index_path(split: str) -> Path:
    return ARTIFACTS_DIR / TRAINING_PAIR_DIRNAME / f"{split}_pairs_len1_3.pt"


def load_pair_index(split: str) -> dict:
    path = pair_index_path(split)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing pair index {path}. Run scripts/build_training_pairs.py first."
        )
    return load_torch(path)


def resolve_project_path(path_value: str | os.PathLike | None) -> Path | None:
    if not path_value:
        return None
    path = Path(path_value)
    return path if path.is_absolute() else ROOT / path


def positions_by_length(index: dict) -> dict[int, torch.Tensor]:
    result = {}
    lengths = index["query_lengths"]
    for query_len in sorted(set(lengths.tolist())):
        result[int(query_len)] = torch.nonzero(lengths == query_len).flatten()
    return result


def sample_positions(index: dict, by_length: dict[int, torch.Tensor], config: dict) -> torch.Tensor:
    batch_size = int(config["batch_size"])
    mode = config["sampler_mode"]
    if mode == "natural_length":
        return torch.randint(0, len(index["source_indices"]), (batch_size,))
    if mode != "balanced_length":
        raise ValueError(f"Unknown sampler_mode: {mode}")
    lengths = sorted(by_length)
    counts = [batch_size // len(lengths)] * len(lengths)
    for i in range(batch_size - sum(counts)):
        counts[i % len(counts)] += 1
    chunks = []
    for query_len, count in zip(lengths, counts):
        pool = by_length[query_len]
        choices = torch.randint(0, len(pool), (count,))
        chunks.append(pool[choices])
    sampled = torch.cat(chunks)
    return sampled[torch.randperm(len(sampled))]


def condition_embeddings(prompt_cache, attr_indices, signs, device, mode: str = "signed_prompt"):
    positive = prompt_cache["positive"].float().to(device)
    negative = prompt_cache["negative"].float().to(device)
    directions = prompt_cache.get("directions")
    if directions is not None:
        directions = directions.float().to(device)
    attrs = attr_indices.to(device)
    sign_tensor = signs.to(device)
    mask = attrs >= 0
    safe_attrs = attrs.clamp_min(0)
    if mode == "signed_prompt":
        pos = positive[safe_attrs]
        neg = negative[safe_attrs]
        conditions = torch.where((sign_tensor > 0).unsqueeze(-1), pos, neg)
    elif mode == "signed_direction":
        if directions is None:
            raise RuntimeError("Prompt cache does not contain contrastive directions")
        base = directions[safe_attrs]
        conditions = torch.where((sign_tensor > 0).unsqueeze(-1), base, -base)
    else:
        raise ValueError(f"Unknown condition embedding mode: {mode}")
    conditions = conditions * mask.unsqueeze(-1)
    return conditions, mask


def false_negative_mask(source_attrs, candidate_attrs, attr_indices, signs, hamming_threshold):
    batch = source_attrs.shape[0]
    device = source_attrs.device
    valid = torch.ones((batch, batch), dtype=torch.bool, device=device)
    query_attr_mask = torch.zeros((batch, source_attrs.shape[1]), dtype=torch.bool, device=device)

    for position in range(attr_indices.shape[1]):
        active = attr_indices[:, position] >= 0
        if not bool(active.any()):
            continue
        attrs = attr_indices[:, position].clamp_min(0)
        desired = signs[:, position]
        candidate_values = candidate_attrs[:, attrs].T
        valid &= (~active[:, None]) | (candidate_values == desired[:, None])
        query_attr_mask[torch.arange(batch, device=device), attrs] |= active

    nonquery_diff = candidate_attrs.unsqueeze(0) != source_attrs.unsqueeze(1)
    nonquery_diff &= ~query_attr_mask[:, None, :]
    valid &= nonquery_diff.sum(dim=-1) <= hamming_threshold
    eye = torch.eye(batch, dtype=torch.bool, device=device)
    return valid & ~eye


def official_like_mask(source_attrs, candidate_attrs, attr_indices, signs, hamming_threshold):
    batch = source_attrs.shape[0]
    device = source_attrs.device
    valid = torch.ones((batch, batch), dtype=torch.bool, device=device)
    query_attr_mask = torch.zeros((batch, source_attrs.shape[1]), dtype=torch.bool, device=device)

    for position in range(attr_indices.shape[1]):
        active = attr_indices[:, position] >= 0
        if not bool(active.any()):
            continue
        attrs = attr_indices[:, position].clamp_min(0)
        desired = signs[:, position]
        candidate_values = candidate_attrs[:, attrs].T
        valid &= (~active[:, None]) | (candidate_values == desired[:, None])
        query_attr_mask[torch.arange(batch, device=device), attrs] |= active

    nonquery_diff = candidate_attrs.unsqueeze(0) != source_attrs.unsqueeze(1)
    nonquery_diff &= ~query_attr_mask[:, None, :]
    valid &= nonquery_diff.sum(dim=-1) <= hamming_threshold
    return valid


def source_similarity_filter(source, target, config):
    batch = source.shape[0]
    device = source.device
    similarity = F.normalize(source, dim=-1) @ F.normalize(target, dim=-1).T
    keep = torch.ones((batch, batch), dtype=torch.bool, device=device)
    min_cos = config.get("multipositive_min_source_cos")
    if min_cos is not None:
        keep &= similarity >= float(min_cos)
    top_fraction = float(config.get("multipositive_top_fraction", 0.0))
    if top_fraction > 0.0:
        count = max(1, min(batch, math.ceil(batch * top_fraction)))
        top_indices = similarity.topk(count, dim=1).indices
        top_mask = torch.zeros_like(keep)
        top_mask.scatter_(1, top_indices, True)
        keep &= top_mask
    return keep


def multi_positive_contrastive_loss(scores, positive_mask, neutral_mask):
    scores = scores.masked_fill(neutral_mask & ~positive_mask, -torch.inf)
    positive_scores = scores.masked_fill(~positive_mask, -torch.inf)
    numerator = torch.logsumexp(positive_scores, dim=1)
    denominator = torch.logsumexp(scores, dim=1)
    return -(numerator - denominator).mean()


def compute_losses(model, batch, embeddings, attrs, prompt_cache, config, device):
    src_idx_cpu = batch["source_indices"].long()
    tgt_idx_cpu = batch["target_indices"].long()
    attr_indices = batch["attr_indices"].to(device)
    signs = batch["signs"].to(device)

    source = embeddings[src_idx_cpu].float().to(device)
    target = embeddings[tgt_idx_cpu].float().to(device)
    source_attrs = attrs[src_idx_cpu].to(device)
    target_attrs = attrs[tgt_idx_cpu].to(device)
    conditions, mask = condition_embeddings(
        prompt_cache,
        attr_indices,
        signs,
        device,
        str(config.get("condition_mode", "signed_prompt")),
    )

    query, alpha, _ = model(source, conditions, mask)
    scores = (query @ target.T) / float(config["temperature"])
    labels = torch.arange(len(src_idx_cpu), device=device)
    eye = torch.eye(len(src_idx_cpu), dtype=torch.bool, device=device)

    false_negatives = false_negative_mask(
        source_attrs,
        target_attrs,
        attr_indices,
        signs,
        int(config["false_negative_hamming"]),
    )
    exact_scores = scores.masked_fill(false_negatives, -torch.inf)
    exact_info_nce = F.cross_entropy(exact_scores, labels)

    if bool(config.get("use_multipositive_loss", False)):
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
        multipositive_info_nce = multi_positive_contrastive_loss(scores, positive_mask, neutral_mask)
        multipositive_weight = float(config.get("multipositive_weight", 1.0))
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
    target_cos = 1.0 - (query * F.normalize(target, dim=-1)).sum(dim=-1).mean()
    source_cos = 1.0 - (query * F.normalize(source, dim=-1)).sum(dim=-1).mean()
    attr_probe_loss = torch.zeros((), device=device)
    attr_logits = getattr(model, "last_attr_logits", None)
    if attr_logits is not None and float(config.get("lambda_attr_probe", 0.0)) > 0:
        attr_targets = (source_attrs.float() > 0).float()
        attr_probe_loss = F.binary_cross_entropy_with_logits(attr_logits, attr_targets)
    loss = (
        info_nce
        + float(config["lambda_target"]) * target_cos
        + float(config["lambda_source"]) * source_cos
        + float(config.get("lambda_attr_probe", 0.0)) * attr_probe_loss
    )
    return loss, {
        "loss": float(loss.detach().cpu()),
        "info_nce": float(info_nce.detach().cpu()),
        "exact_info_nce": float(exact_info_nce.detach().cpu()),
        "multipositive_info_nce": float(multipositive_info_nce.detach().cpu()),
        "multipositive_weight": float(multipositive_weight),
        "target_cosine_loss": float(target_cos.detach().cpu()),
        "source_preservation_loss": float(source_cos.detach().cpu()),
        "attr_probe_loss": float(attr_probe_loss.detach().cpu()),
        "cos_q_target": float((query * F.normalize(target, dim=-1)).sum(dim=-1).mean().detach().cpu()),
        "cos_q_source": float((query * F.normalize(source, dim=-1)).sum(dim=-1).mean().detach().cpu()),
        "gate_mean": float(alpha[mask].mean().detach().cpu()) if bool(mask.any()) else 0.0,
        "positive_count": float(positive_count.detach().cpu()),
    }


def topk_attribute_success(top_indices, source_attrs, gallery_attrs, attr_indices, signs):
    batch, top_k = top_indices.shape
    device = top_indices.device
    candidate_attrs = gallery_attrs[top_indices]
    query_ok = torch.ones((batch, top_k), dtype=torch.bool, device=device)
    query_attr_mask = torch.zeros((batch, source_attrs.shape[1]), dtype=torch.bool, device=device)

    for position in range(attr_indices.shape[1]):
        active = attr_indices[:, position] >= 0
        if not bool(active.any()):
            continue
        attrs = attr_indices[:, position].clamp_min(0)
        desired = signs[:, position]
        gathered = torch.gather(
            candidate_attrs,
            2,
            attrs[:, None, None].expand(batch, top_k, 1),
        ).squeeze(-1)
        query_ok &= (~active[:, None]) | (gathered == desired[:, None])
        query_attr_mask[torch.arange(batch, device=device), attrs] |= active

    nonquery_diff = candidate_attrs != source_attrs[:, None, :]
    nonquery_diff &= ~query_attr_mask[:, None, :]
    official_like = query_ok & (nonquery_diff.sum(dim=-1) <= 2)
    return query_ok, official_like


def validate(model, valid_index, valid_embeddings, prompt_cache, config, device, epoch, run_dir):
    model.eval()
    gallery = valid_embeddings.float().to(device)
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
                str(config.get("condition_mode", "signed_prompt")),
            )
            query, alpha, _ = model(source, conditions, mask)
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


def plot_validation_artifacts(
    run_dir,
    metrics,
    length_totals,
    attr_totals,
    gate_sums,
    gate_counts,
    valid_index,
    examples,
    epoch,
):
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = run_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    with (samples_dir / f"val_examples_epoch_{epoch:03d}.jsonl").open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example) + "\n")

    # Query length breakdown.
    plt.figure(figsize=(7, 4))
    lengths = sorted(length_totals)
    values = [
        length_totals[length]["official@10"] / max(1, length_totals[length]["count"])
        for length in lengths
    ]
    plt.bar([str(length) for length in lengths], values, color="#2f6f73")
    plt.ylim(0, 1)
    plt.xlabel("Query length")
    plt.ylabel("official_like@10")
    plt.title("Validation by query length")
    plt.tight_layout()
    plt.savefig(plots_dir / "query_length_breakdown.png", dpi=160)
    plt.close()

    # Attribute breakdown.
    names = valid_index["attributes"]
    attr_rows = [
        (names[attr], totals["official@10"] / max(1, totals["count"]))
        for attr, totals in attr_totals.items()
        if totals["count"] > 0
    ]
    attr_rows = sorted(attr_rows, key=lambda row: row[1])[:20]
    if attr_rows:
        plt.figure(figsize=(9, 6))
        labels, vals = zip(*attr_rows)
        plt.barh(labels, vals, color="#8d5a2b")
        plt.xlim(0, 1)
        plt.xlabel("official_like@10")
        plt.title("Hardest edited attributes in validation")
        plt.tight_layout()
        plt.savefig(plots_dir / "attribute_breakdown.png", dpi=160)
        plt.close()

    # Gate heatmap.
    mean_gate = gate_sums / gate_counts.clamp_min(1)
    plt.figure(figsize=(12, 3.8))
    plt.imshow(mean_gate.numpy(), aspect="auto", cmap="viridis", vmin=0, vmax=1)
    plt.yticks([0, 1], ["negative edit", "positive edit"])
    plt.xticks(range(len(names)), names, rotation=90, fontsize=6)
    plt.colorbar(label="mean gate weight")
    plt.title("Gate weights by signed attribute")
    plt.tight_layout()
    plt.savefig(plots_dir / "gate_weights_heatmap.png", dpi=180)
    plt.close()

    plot_retrieval_examples(run_dir, valid_index, examples, epoch)


def plot_retrieval_examples(run_dir, valid_index, examples, epoch):
    image_dir = CELEBA_DIR / "img_align_celeba"
    if not image_dir.is_dir() or not examples:
        return
    try:
        from PIL import Image

        rows = len(examples)
        cols = 5
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.25))
        if rows == 1:
            axes = axes[None, :]
        for row, example in enumerate(examples):
            indices = [
                example["source_index"],
                example["target_index"],
                *example["top10"][:3],
            ]
            titles = ["source", "target", "top1", "top2", "top3"]
            for col, (index, title) in enumerate(zip(indices, titles)):
                path = image_dir / valid_index["filenames"][index]
                axes[row, col].imshow(Image.open(path).convert("RGB"))
                axes[row, col].axis("off")
                axes[row, col].set_title(title, fontsize=8)
            axes[row, 0].set_ylabel(example["query"], fontsize=7)
        plt.tight_layout()
        plt.savefig(run_dir / "plots" / f"retrieval_examples_epoch_{epoch:03d}.png", dpi=160)
        plt.close()
    except Exception as exc:  # Plotting must never kill training.
        progress_line(run_dir / "progress.txt", f"retrieval plot skipped: {exc}")


def plot_metric_curves(metrics_path: Path, run_dir: Path) -> None:
    if not metrics_path.exists():
        return
    with metrics_path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    epochs = [float(row["epoch"]) for row in rows]

    plt.figure(figsize=(8, 4))
    for name in ("train_loss", "train_info_nce", "train_target_cosine_loss", "train_source_preservation_loss"):
        if name in rows[0]:
            plt.plot(epochs, [float(row[name]) for row in rows], label=name)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training curves")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(plots_dir / "training_curves.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 4))
    for name in ("val_exact_R@10", "val_attr_success@10", "val_official_like@10"):
        if name in rows[0]:
            plt.plot(epochs, [float(row[name]) for row in rows], label=name)
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.title("Validation metrics")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(plots_dir / "validation_metrics.png", dpi=160)
    plt.close()


def make_batch(index: dict, positions: torch.Tensor) -> dict:
    return {
        "source_indices": index["source_indices"][positions],
        "target_indices": index["target_indices"][positions],
        "attr_indices": index["attr_indices"][positions],
        "signs": index["signs"][positions],
    }


def main() -> int:
    args = parse_args()
    started = time.monotonic()
    signal.signal(signal.SIGTERM, request_stop)
    if hasattr(signal, "SIGUSR1"):
        signal.signal(signal.SIGUSR1, request_stop)

    config = load_config(args)
    random.seed(int(config["seed"]))
    torch.manual_seed(int(config["seed"]))
    device = choose_device(args.device)

    run_dir = args.run_dir or make_run_dir(config, args.run_root)
    if run_dir.exists() and args.force:
        raise RuntimeError(f"Refusing to delete existing run dir: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    for child in ("checkpoints", "plots", "samples", "logs"):
        (run_dir / child).mkdir(exist_ok=True)
    dump_config(config, run_dir / "config.json")
    progress = run_dir / "progress.txt"
    progress_line(progress, f"RUN_DIR {run_dir}")
    progress_line(progress, f"config {json.dumps(config, sort_keys=True)}")

    train_cache = load_image_embedding_cache("train")
    valid_cache = load_image_embedding_cache("valid")
    prompt_cache_path = config.get("prompt_cache_path")
    prompt_cache = load_prompt_embedding_cache(resolve_project_path(prompt_cache_path))
    train_index = load_pair_index("train")
    valid_index = load_pair_index("valid")
    if prompt_cache["attributes"] != train_index["attributes"]:
        raise RuntimeError("Prompt cache attributes do not match pair index")

    train_embeddings = train_cache["embeddings"].float()
    valid_embeddings = valid_cache["embeddings"].float()
    train_attrs = train_index["attrs"]
    by_length = positions_by_length(train_index)

    model = create_model_from_config(config).to(device)
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
                batch,
                train_embeddings,
                train_attrs,
                prompt_cache,
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
                    f"cos_q_source={loss_parts['cos_q_source']:.4f} "
                    f"cos_q_target={loss_parts['cos_q_target']:.4f} "
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
                val_metrics = validate(
                    model,
                    valid_index,
                    valid_embeddings,
                    prompt_cache,
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
