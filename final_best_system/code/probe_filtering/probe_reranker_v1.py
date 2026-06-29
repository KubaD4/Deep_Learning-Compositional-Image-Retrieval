#!/usr/bin/env python3
"""Train a CelebA attribute probe and evaluate probe-based reranking.

This script is intentionally experimental and does not modify the current final
system. It tests a fair approximation of the previous oracle Hamming filter:

1. Use the frozen learned system to retrieve a top-N candidate pool.
2. Predict CelebA attributes with a probe trained only on train/valid labels.
3. Rerank/filter candidates using predicted query satisfaction and predicted
   non-query Hamming preservation.

The official JSON target lists are used only for final evaluation metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import signal
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import ImageOps
from torch.utils.data import DataLoader, Dataset, Subset


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = ROOT.parent
SCRIPTS = ROOT / "scripts"
ORCH = ROOT / "orchestrator"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(ORCH))
os.environ.setdefault("DL_PROJECT_ROOT", str(PACKAGE_ROOT))

import evaluate_sum_model_blends as blends  # noqa: E402
from learned_gate_core import TRAINING_RUN_DIRNAME, load_model_checkpoint, load_prompt_embedding_cache, progress_line  # noqa: E402
from project_core import (  # noqa: E402
    ARTIFACTS_DIR,
    CHECKPOINT_DIR,
    EMBEDDING_DIR,
    MODEL_ID,
    TOP_KS,
    atomic_json_dump,
    atomic_torch_save,
    choose_device,
    load_celeba,
    load_evaluation,
    load_torch,
    parse_query,
    read_attribute_table,
    retrieval_metrics,
    unwrap_features,
)


STOP_REQUESTED = False


def request_stop(signum, frame) -> None:
    del signum, frame
    global STOP_REQUESTED
    STOP_REQUESTED = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--probe-checkpoint", type=Path, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--source-batch-size", type=int, default=256)
    parser.add_argument("--probe-batch-size", type=int, default=2048)
    parser.add_argument("--probe-epochs", type=int, default=8)
    parser.add_argument("--probe-max-steps", type=int, default=0)
    parser.add_argument("--probe-lr", type=float, default=3e-4)
    parser.add_argument("--probe-weight-decay", type=float, default=1e-4)
    parser.add_argument("--probe-hidden-dims", type=str, default="512,256")
    parser.add_argument("--probe-dropout", type=float, default=0.1)
    parser.add_argument("--probe-hpsearch", action="store_true")
    parser.add_argument("--probe-use-flip", action="store_true")
    parser.add_argument("--max-probe-configs", type=int, default=0)
    parser.add_argument("--flip-batch-size", type=int, default=256)
    parser.add_argument("--flip-workers", type=int, default=8)
    parser.add_argument("--random-probe-smoke", action="store_true")
    parser.add_argument("--top-pool", type=int, default=500)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--query-ids", nargs="*", type=int, default=None)
    parser.add_argument("--max-sources-per-query", type=int, default=0)
    parser.add_argument("--time-budget-seconds", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-probe-training", action="store_true")
    return parser.parse_args()


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


class AttributeProbe(nn.Module):
    def __init__(self, input_dim: int = 512, hidden_dims: tuple[int, ...] = (512, 256), output_dim: int = 40, dropout: float = 0.1):
        super().__init__()
        layers: list[nn.Module] = [nn.LayerNorm(input_dim)]
        current = input_dim
        for hidden in hidden_dims:
            layers.extend([nn.Linear(current, hidden), nn.GELU(), nn.Dropout(dropout)])
            current = hidden
        layers.append(nn.Linear(current, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(F.normalize(x.float(), dim=-1))


class FlippedIndexedImages(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        image, _ = self.dataset[index]
        return ImageOps.mirror(image), index


class ClipImageCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, batch):
        images, indices = zip(*batch)
        pixels = self.processor(images=list(images), return_tensors="pt")["pixel_values"]
        return pixels, torch.tensor(indices)


def parse_hidden_dims(value: str) -> tuple[int, ...]:
    dims = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not dims:
        raise ValueError("--probe-hidden-dims must contain at least one integer")
    return dims


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def append_csv(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def load_split_embeddings(split: str) -> dict:
    path = EMBEDDING_DIR / f"{split}_image_embeddings.pt"
    if not path.exists():
        raise FileNotFoundError(f"Missing embedding cache: {path}")
    cache = load_torch(path)
    if cache.get("model_id") != MODEL_ID:
        raise RuntimeError(f"Unexpected model_id in {path}: {cache.get('model_id')}")
    return cache


def create_flipped_embedding_cache(args: argparse.Namespace, progress: Path, device: torch.device) -> Path:
    """Create train CLIP embeddings for horizontally flipped images.

    The labels are unchanged because horizontal flip preserves CelebA attributes
    such as hair colour, glasses, smile, age, gender annotation, etc. The cache
    is only used to augment the attribute probe, not the retrieval gallery.
    """
    final_path = EMBEDDING_DIR / "train_image_embeddings_flipped.pt"
    if final_path.exists() and not args.force:
        cache = load_torch(final_path)
        if cache.get("model_id") == MODEL_ID and cache.get("split") == "train_flipped":
            progress_line(progress, f"REUSE flipped train embeddings {final_path}")
            return final_path

    from transformers import CLIPModel, CLIPProcessor

    dataset = load_celeba("train")
    chunks_dir = CHECKPOINT_DIR / "embeddings" / "train_flipped"
    if args.force and final_path.exists():
        final_path.unlink()
    if args.force and chunks_dir.exists():
        shutil.rmtree(chunks_dir)
    chunks_dir.mkdir(parents=True, exist_ok=True)

    existing = sorted(chunks_dir.glob("chunk_*.pt"))
    start_index = 0
    chunks = []
    for path in existing:
        chunk = load_torch(path)
        indices = chunk["indices"]
        if int(indices[0]) != start_index:
            break
        start_index = int(indices[-1]) + 1
        chunks.append(chunk["embeddings"])
    progress_line(progress, f"FLIP embeddings start_index={start_index}/{len(dataset)} cache={final_path}")

    if start_index < len(dataset):
        processor = CLIPProcessor.from_pretrained(MODEL_ID)
        model = CLIPModel.from_pretrained(MODEL_ID).to(device).eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        remaining = Subset(FlippedIndexedImages(dataset), range(start_index, len(dataset)))
        loader = DataLoader(
            remaining,
            batch_size=int(args.flip_batch_size),
            shuffle=False,
            num_workers=int(args.flip_workers),
            pin_memory=device.type == "cuda",
            persistent_workers=int(args.flip_workers) > 0,
            collate_fn=ClipImageCollator(processor),
        )
        with torch.inference_mode():
            for pixels, indices in loader:
                features = unwrap_features(model.get_image_features(pixel_values=pixels.to(device)))
                features = F.normalize(features.float(), dim=-1).cpu().half()
                first = int(indices[0])
                end = int(indices[-1]) + 1
                atomic_torch_save(
                    {
                        "model_id": MODEL_ID,
                        "split": "train_flipped",
                        "indices": indices,
                        "filenames": [dataset.filename[index] for index in indices.tolist()],
                        "embeddings": features,
                    },
                    chunks_dir / f"chunk_{first:08d}_{end:08d}.pt",
                )
                if end % 20000 == 0 or end == len(dataset):
                    progress_line(progress, f"FLIP embeddings encoded={end}/{len(dataset)}")
                if STOP_REQUESTED:
                    raise SystemExit("Stop requested during flipped embedding creation")

    chunks = []
    expected = 0
    for path in sorted(chunks_dir.glob("chunk_*.pt")):
        chunk = load_torch(path)
        indices = chunk["indices"]
        if int(indices[0]) != expected:
            raise RuntimeError(f"Non-contiguous flipped chunks at {expected}")
        expected = int(indices[-1]) + 1
        chunks.append(chunk["embeddings"])
    if expected != len(dataset):
        raise RuntimeError(f"Cannot finalize flipped embeddings: {expected}/{len(dataset)}")
    atomic_torch_save(
        {
            "model_id": MODEL_ID,
            "split": "train_flipped",
            "embeddings": torch.cat(chunks),
            "filenames": list(dataset.filename),
            "augmentation": "horizontal_flip",
            "label_policy": "same CelebA attributes as original image",
        },
        final_path,
    )
    progress_line(progress, f"FLIP embeddings saved {final_path}")
    return final_path


def attrs_for_filenames(filenames: list[str]) -> tuple[list[str], torch.Tensor]:
    attributes, all_filenames, all_attrs = read_attribute_table()
    index = {filename: row for filename, row in zip(all_filenames, all_attrs)}
    rows = []
    missing = []
    for filename in filenames:
        row = index.get(filename)
        if row is None:
            missing.append(filename)
        else:
            rows.append(row)
    if missing:
        raise RuntimeError(f"Missing attributes for {len(missing)} filenames, first={missing[:3]}")
    labels = (torch.stack(rows).float() > 0).float()
    return attributes, labels


def class_pos_weight(labels: torch.Tensor) -> torch.Tensor:
    positives = labels.sum(dim=0).clamp_min(1.0)
    negatives = (labels.shape[0] - labels.sum(dim=0)).clamp_min(1.0)
    # Clamp avoids exploding weights for extremely rare attributes.
    return (negatives / positives).clamp(0.25, 20.0)


def probe_eval_metrics(logits: torch.Tensor, labels: torch.Tensor) -> tuple[dict, list[dict]]:
    probs = torch.sigmoid(logits)
    preds = probs >= 0.5
    truth = labels.bool()
    eps = 1e-8
    tp = (preds & truth).sum(dim=0).float()
    fp = (preds & ~truth).sum(dim=0).float()
    fn = (~preds & truth).sum(dim=0).float()
    tn = (~preds & ~truth).sum(dim=0).float()
    acc = (tp + tn) / (tp + tn + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    bce = F.binary_cross_entropy_with_logits(logits, labels).item()
    macro = {
        "bce": bce,
        "macro_accuracy": float(acc.mean()),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "micro_accuracy": float((preds == truth).float().mean()),
    }
    attributes, _, _ = read_attribute_table()
    per_attr = []
    for idx, name in enumerate(attributes):
        per_attr.append(
            {
                "attribute": name,
                "accuracy": float(acc[idx]),
                "precision": float(precision[idx]),
                "recall": float(recall[idx]),
                "f1": float(f1[idx]),
                "positive_rate": float(labels[:, idx].mean()),
            }
        )
    return macro, per_attr


@dataclass(frozen=True)
class ProbeConfig:
    config_id: str
    hidden_dims: tuple[int, ...]
    dropout: float
    lr: float
    weight_decay: float
    use_flip: bool


def probe_config_rows(args: argparse.Namespace) -> list[ProbeConfig]:
    base = ProbeConfig(
        config_id="probe_manual",
        hidden_dims=parse_hidden_dims(args.probe_hidden_dims),
        dropout=float(args.probe_dropout),
        lr=float(args.probe_lr),
        weight_decay=float(args.probe_weight_decay),
        use_flip=bool(args.probe_use_flip),
    )
    if not args.probe_hpsearch:
        return [base]
    configs = [
        ProbeConfig("p001_base_noflip", (512, 256), 0.10, 3e-4, 1e-4, False),
        ProbeConfig("p002_wide_noflip", (1024, 512), 0.10, 3e-4, 1e-4, False),
        ProbeConfig("p003_compact_noflip", (256, 128), 0.05, 5e-4, 1e-4, False),
        ProbeConfig("p004_base_flip", (512, 256), 0.10, 3e-4, 1e-4, True),
        ProbeConfig("p005_wide_flip", (1024, 512), 0.10, 3e-4, 1e-4, True),
        ProbeConfig("p006_compact_flip", (256, 128), 0.05, 5e-4, 1e-4, True),
        ProbeConfig("p007_base_low_lr_flip", (512, 256), 0.10, 1e-4, 1e-4, True),
        ProbeConfig("p008_wide_low_dropout_flip", (1024, 512), 0.05, 1e-4, 1e-4, True),
    ]
    if args.max_probe_configs:
        configs = configs[: int(args.max_probe_configs)]
    return configs


def save_random_probe(args: argparse.Namespace, output_root: Path, progress: Path) -> Path:
    probe_dir = output_root / "probe"
    probe_dir.mkdir(parents=True, exist_ok=True)
    attributes, _, _ = read_attribute_table()
    hidden_dims = parse_hidden_dims(args.probe_hidden_dims)
    probe = AttributeProbe(hidden_dims=hidden_dims, output_dim=len(attributes), dropout=float(args.probe_dropout))
    path = probe_dir / "random_probe_smoke.pt"
    atomic_torch_save(
        {
            "model_id": MODEL_ID,
            "model_state": probe.state_dict(),
            "attributes": attributes,
            "hidden_dims": hidden_dims,
            "dropout": float(args.probe_dropout),
            "valid_metrics": {"note": "random weights smoke test, metrics intentionally meaningless"},
            "epoch": 0,
            "step": 0,
            "config_id": "random_probe_smoke",
        },
        path,
    )
    progress_line(progress, f"RANDOM_PROBE_SMOKE checkpoint={path}")
    return path


def train_one_probe_config(
    args: argparse.Namespace,
    output_root: Path,
    progress: Path,
    device: torch.device,
    config: ProbeConfig,
    train_embeddings_base: torch.Tensor,
    train_labels_base: torch.Tensor,
    train_embeddings_flip: torch.Tensor | None,
    train_labels_flip: torch.Tensor | None,
    valid_embeddings: torch.Tensor,
    valid_labels: torch.Tensor,
    attributes: list[str],
) -> tuple[Path, dict]:
    probe_dir = output_root / "probe" / config.config_id
    probe_dir.mkdir(parents=True, exist_ok=True)
    best_path = probe_dir / "best_probe.pt"

    if config.use_flip:
        if train_embeddings_flip is None or train_labels_flip is None:
            raise RuntimeError(f"{config.config_id} requested flip augmentation but flipped cache is unavailable")
        train_embeddings = torch.cat([train_embeddings_base, train_embeddings_flip], dim=0)
        train_labels = torch.cat([train_labels_base, train_labels_flip], dim=0)
    else:
        train_embeddings = train_embeddings_base
        train_labels = train_labels_base

    probe = AttributeProbe(hidden_dims=config.hidden_dims, output_dim=len(attributes), dropout=config.dropout).to(device)
    optimizer = torch.optim.AdamW(
        probe.parameters(),
        lr=float(config.lr),
        weight_decay=float(config.weight_decay),
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=class_pos_weight(train_labels).to(device))
    generator = torch.Generator().manual_seed(20260626)
    best_bce = float("inf")
    best_row: dict = {}
    global_step = 0
    metrics_path = probe_dir / "probe_metrics.csv"

    progress_line(
        progress,
        "PROBE train start "
        f"config={config.config_id} hidden={config.hidden_dims} dropout={config.dropout} "
        f"lr={config.lr} flip={config.use_flip} "
        f"train={len(train_embeddings)} valid={len(valid_embeddings)} "
        f"epochs={args.probe_epochs} batch={args.probe_batch_size}",
    )

    for epoch in range(1, int(args.probe_epochs) + 1):
        order = torch.randperm(len(train_embeddings), generator=generator)
        epoch_loss = 0.0
        steps = 0
        for start in range(0, len(order), int(args.probe_batch_size)):
            if STOP_REQUESTED:
                break
            positions = order[start : start + int(args.probe_batch_size)]
            x = train_embeddings[positions].to(device)
            y = train_labels[positions].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = probe(x)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(probe.parameters(), 1.0)
            optimizer.step()
            epoch_loss += float(loss.detach().cpu())
            steps += 1
            global_step += 1
            if args.probe_max_steps and global_step >= int(args.probe_max_steps):
                break

        valid_logits = []
        probe.eval()
        with torch.inference_mode():
            for start in range(0, len(valid_embeddings), int(args.probe_batch_size)):
                x = valid_embeddings[start : start + int(args.probe_batch_size)].to(device)
                valid_logits.append(probe(x).cpu())
        probe.train()
        valid_logits_tensor = torch.cat(valid_logits, dim=0)
        macro, per_attr = probe_eval_metrics(valid_logits_tensor, valid_labels)
        row = {"epoch": epoch, "step": global_step, "train_loss": epoch_loss / max(1, steps), **macro}
        append_csv(metrics_path, row)
        write_csv(probe_dir / "per_attribute_metrics.csv", per_attr)
        progress_line(
            progress,
            "PROBE epoch "
            f"config={config.config_id} epoch={epoch} step={global_step} train_loss={row['train_loss']:.4f} "
            f"valid_bce={macro['bce']:.4f} valid_f1={macro['macro_f1']:.4f} "
            f"valid_acc={macro['macro_accuracy']:.4f}",
        )
        if macro["bce"] < best_bce:
            best_bce = macro["bce"]
            best_row = row
            atomic_torch_save(
                {
                    "model_id": MODEL_ID,
                    "model_state": probe.state_dict(),
                    "attributes": attributes,
                    "hidden_dims": config.hidden_dims,
                    "dropout": config.dropout,
                    "config_id": config.config_id,
                    "use_flip": config.use_flip,
                    "valid_metrics": macro,
                    "epoch": epoch,
                    "step": global_step,
                },
                best_path,
            )
            progress_line(progress, f"PROBE BEST config={config.config_id} epoch={epoch} valid_bce={best_bce:.4f} checkpoint={best_path}")
        if args.probe_max_steps and global_step >= int(args.probe_max_steps):
            break
        if STOP_REQUESTED:
            break

    return best_path, {
        "config_id": config.config_id,
        "checkpoint": str(best_path),
        "hidden_dims": "-".join(str(value) for value in config.hidden_dims),
        "dropout": config.dropout,
        "lr": config.lr,
        "weight_decay": config.weight_decay,
        "use_flip": config.use_flip,
        **best_row,
    }


def train_probe(args: argparse.Namespace, output_root: Path, progress: Path, device: torch.device) -> Path:
    probe_dir = output_root / "probe"
    probe_dir.mkdir(parents=True, exist_ok=True)
    stable_best_path = args.probe_checkpoint or (probe_dir / "best_probe.pt")
    if args.random_probe_smoke:
        return save_random_probe(args, output_root, progress)
    if args.skip_probe_training and stable_best_path.exists():
        progress_line(progress, f"SKIP probe training; using {stable_best_path}")
        return stable_best_path
    if stable_best_path.exists() and not args.force:
        progress_line(progress, f"REUSE existing probe checkpoint {stable_best_path}")
        return stable_best_path

    configs = probe_config_rows(args)
    train_cache = load_split_embeddings("train")
    valid_cache = load_split_embeddings("valid")
    attributes, train_labels = attrs_for_filenames(train_cache["filenames"])
    _, valid_labels = attrs_for_filenames(valid_cache["filenames"])
    train_embeddings = F.normalize(train_cache["embeddings"].float(), dim=-1)
    valid_embeddings = F.normalize(valid_cache["embeddings"].float(), dim=-1)
    train_embeddings_flip = None
    train_labels_flip = None
    if any(config.use_flip for config in configs):
        flip_path = create_flipped_embedding_cache(args, progress, device)
        flip_cache = load_torch(flip_path)
        _, train_labels_flip = attrs_for_filenames(flip_cache["filenames"])
        train_embeddings_flip = F.normalize(flip_cache["embeddings"].float(), dim=-1)

    results = []
    for config in configs:
        checkpoint, row = train_one_probe_config(
            args,
            output_root,
            progress,
            device,
            config,
            train_embeddings,
            train_labels,
            train_embeddings_flip,
            train_labels_flip,
            valid_embeddings,
            valid_labels,
            attributes,
        )
        results.append(row)
        write_csv(probe_dir / "hpsearch_summary.csv", results)
        if STOP_REQUESTED:
            break

    if not results:
        raise RuntimeError("No probe configs completed")
    best = min(results, key=lambda row: float(row.get("bce", "inf")))
    best_source = Path(best["checkpoint"])
    checkpoint = load_torch(best_source)
    atomic_torch_save(checkpoint, stable_best_path)
    progress_line(
        progress,
        f"PROBE HPSEARCH BEST config={best['config_id']} bce={best.get('bce')} "
        f"f1={best.get('macro_f1')} checkpoint={stable_best_path}",
    )
    return stable_best_path


def load_probe(path: Path, device: torch.device) -> tuple[AttributeProbe, list[str], dict]:
    checkpoint = load_torch(path)
    attributes = checkpoint["attributes"]
    probe = AttributeProbe(
        hidden_dims=tuple(checkpoint.get("hidden_dims", (512, 256))),
        output_dim=len(attributes),
        dropout=float(checkpoint.get("dropout", 0.1)),
    ).to(device)
    probe.load_state_dict(checkpoint["model_state"])
    probe.eval()
    return probe, attributes, checkpoint


def predict_probe_probs(probe: AttributeProbe, embeddings: torch.Tensor, device: torch.device, batch_size: int) -> torch.Tensor:
    probs = []
    with torch.inference_mode():
        for start in range(0, len(embeddings), batch_size):
            x = F.normalize(embeddings[start : start + batch_size].float(), dim=-1).to(device)
            probs.append(torch.sigmoid(probe(x)).cpu())
    return torch.cat(probs, dim=0)


def scan_best_v7_checkpoint() -> tuple[Path | None, float | None, float | None]:
    runs_root = ARTIFACTS_DIR / TRAINING_RUN_DIRNAME
    best = None
    best_macro = float("-inf")
    best_beta = None
    for summary_path in sorted(runs_root.glob("hpsearch_hamming_weighted_v7_*_long/summary.csv")):
        with summary_path.open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            if row.get("status") != "complete":
                continue
            try:
                score = float(row.get("json_macro_Recall@10", "nan"))
            except ValueError:
                continue
            run_dir = Path(row["run_dir"])
            checkpoint = run_dir / "checkpoints" / "best_val_official_like_at10.pt"
            if checkpoint.exists() and score > best_macro:
                best = checkpoint
                best_macro = score
                try:
                    best_beta = float(row.get("json_beta", "nan"))
                except ValueError:
                    best_beta = None
    if best is None:
        return None, None, None
    return best, best_macro, best_beta


def find_checkpoint_and_beta(args: argparse.Namespace, progress: Path) -> tuple[Path, float]:
    if args.checkpoint is not None:
        beta = float(args.beta if args.beta is not None else 1.25)
        progress_line(progress, f"CHECKPOINT explicit {args.checkpoint} beta={beta}")
        return args.checkpoint, beta
    checkpoint, score, beta = scan_best_v7_checkpoint()
    if checkpoint is not None:
        chosen_beta = float(args.beta if args.beta is not None else beta if beta is not None and not math.isnan(beta) else 1.25)
        progress_line(progress, f"CHECKPOINT best_v7 {checkpoint} json_macro_R10={score} beta={chosen_beta}")
        return checkpoint, chosen_beta
    fallback = ROOT / "final_best_system" / "weights" / "best_val_official_like_at10.pt"
    if not fallback.exists():
        fallback = ROOT.parent / "final_best_system" / "weights" / "best_val_official_like_at10.pt"
    if not fallback.exists():
        raise FileNotFoundError("No checkpoint found: pass --checkpoint explicitly")
    beta = float(args.beta if args.beta is not None else 1.5)
    progress_line(progress, f"CHECKPOINT fallback_final_system {fallback} beta={beta}")
    return fallback, beta


def query_tensors(conditions: list[tuple[int, str]], attribute_to_index: dict[str, int], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    attrs = torch.tensor([[attribute_to_index[attr] for _, attr in conditions]], dtype=torch.long, device=device)
    signs = torch.tensor([[sign for sign, _ in conditions]], dtype=torch.int8, device=device)
    return attrs, signs


def query_scores(candidate_probs: torch.Tensor, attr_indices: torch.Tensor, signs: torch.Tensor) -> torch.Tensor:
    parts = []
    for pos in range(attr_indices.numel()):
        attr = int(attr_indices[pos])
        sign = int(signs[pos])
        prob = candidate_probs[:, :, attr]
        parts.append(prob if sign > 0 else 1.0 - prob)
    return torch.stack(parts, dim=-1).mean(dim=-1)


def query_ok(candidate_probs: torch.Tensor, attr_indices: torch.Tensor, signs: torch.Tensor, threshold: float) -> torch.Tensor:
    ok = torch.ones(candidate_probs.shape[:2], dtype=torch.bool, device=candidate_probs.device)
    for pos in range(attr_indices.numel()):
        attr = int(attr_indices[pos])
        sign = int(signs[pos])
        prob = candidate_probs[:, :, attr]
        if sign > 0:
            ok &= prob >= threshold
        else:
            ok &= prob <= (1.0 - threshold)
    return ok


def expected_hamming(source_probs: torch.Tensor, candidate_probs: torch.Tensor, attr_indices: torch.Tensor) -> torch.Tensor:
    nonquery = torch.ones(source_probs.shape[-1], dtype=torch.bool, device=source_probs.device)
    nonquery[attr_indices.long()] = False
    ps = source_probs[:, None, nonquery]
    pc = candidate_probs[:, :, nonquery]
    diff = ps * (1.0 - pc) + (1.0 - ps) * pc
    return diff.sum(dim=-1), diff.mean(dim=-1)


@dataclass(frozen=True)
class RerankMethod:
    name: str
    kind: str
    query_threshold: float = 0.5
    lambda_query: float = 0.0
    lambda_hamming: float = 0.0
    lambda_source: float = 0.0


def default_methods() -> list[RerankMethod]:
    return [
        RerankMethod("baseline_q_final_top10", "baseline"),
        RerankMethod("A_hard_filter_t050", "hard", query_threshold=0.50),
        RerankMethod("A_hard_filter_t060", "hard", query_threshold=0.60),
        RerankMethod("B_soft_lq005_lh005_ls000", "soft", lambda_query=0.05, lambda_hamming=0.05, lambda_source=0.00),
        RerankMethod("B_soft_lq010_lh005_ls000", "soft", lambda_query=0.10, lambda_hamming=0.05, lambda_source=0.00),
        RerankMethod("B_soft_lq010_lh010_ls005", "soft", lambda_query=0.10, lambda_hamming=0.10, lambda_source=0.05),
        RerankMethod("B_soft_lq020_lh010_ls005", "soft", lambda_query=0.20, lambda_hamming=0.10, lambda_source=0.05),
        RerankMethod("C_hybrid_t050_lh005_ls005", "hybrid", query_threshold=0.50, lambda_hamming=0.05, lambda_source=0.05),
        RerankMethod("C_hybrid_t050_lh010_ls005", "hybrid", query_threshold=0.50, lambda_hamming=0.10, lambda_source=0.05),
        RerankMethod("C_hybrid_t060_lh005_ls005", "hybrid", query_threshold=0.60, lambda_hamming=0.05, lambda_source=0.05),
        RerankMethod("C_hybrid_t060_lh010_ls005", "hybrid", query_threshold=0.60, lambda_hamming=0.10, lambda_source=0.05),
    ]


def rerank_top_pool(
    method: RerankMethod,
    top_indices: torch.Tensor,
    top_scores: torch.Tensor,
    source: torch.Tensor,
    gallery: torch.Tensor,
    source_probs: torch.Tensor,
    gallery_probs: torch.Tensor,
    attr_indices: torch.Tensor,
    signs: torch.Tensor,
    top_k: int,
) -> tuple[list[list[int]], list[float]]:
    if method.kind == "baseline":
        return top_indices[:, :top_k].cpu().tolist(), [float(top_k)] * int(top_indices.shape[0])

    candidate_probs = gallery_probs[top_indices].to(top_scores.device)
    source_probs = source_probs.to(top_scores.device)
    attr_indices_local = attr_indices.detach().cpu()
    signs_local = signs.detach().cpu()
    q_score = query_scores(candidate_probs, attr_indices_local, signs_local)
    hard_query = query_ok(candidate_probs, attr_indices_local, signs_local, method.query_threshold)
    ham_raw, ham_norm = expected_hamming(source_probs, candidate_probs, attr_indices_local.to(top_scores.device))
    candidate_embeddings = gallery[top_indices]
    source_sim = (source[:, None, :] * candidate_embeddings).sum(dim=-1)

    if method.kind == "hard":
        hamming_ok = ham_raw <= 2.0
        keep = hard_query & hamming_ok
        score = top_scores.clone()
        score = score.masked_fill(~keep, -torch.inf)
    elif method.kind == "soft":
        score = (
            top_scores
            + float(method.lambda_query) * q_score
            - float(method.lambda_hamming) * ham_norm
            + float(method.lambda_source) * source_sim
        )
        keep = torch.ones_like(score, dtype=torch.bool)
    elif method.kind == "hybrid":
        score = (
            top_scores
            - float(method.lambda_hamming) * ham_norm
            + float(method.lambda_source) * source_sim
        )
        keep = hard_query
        score = score.masked_fill(~keep, -torch.inf)
    else:
        raise ValueError(method.kind)

    rankings = []
    kept_counts = []
    for row in range(score.shape[0]):
        row_score = score[row]
        finite = torch.isfinite(row_score)
        kept_counts.append(float(finite.sum().detach().cpu()))
        if bool(finite.any()):
            ordered_local = row_score.topk(min(top_k, int(finite.sum().item()))).indices
            selected = top_indices[row, ordered_local].tolist()
        else:
            selected = []
        # Return a full top-k list for assignment-style evaluation while still
        # exposing kept_counts. Failed candidates are appended after passing
        # candidates, so the filter can help without producing short lists.
        if len(selected) < top_k:
            selected_set = set(selected)
            for idx in top_indices[row].tolist():
                if idx not in selected_set:
                    selected.append(idx)
                    selected_set.add(idx)
                if len(selected) >= top_k:
                    break
        rankings.append(selected[:top_k])
    return rankings, kept_counts


def finalize_method(annotations: list[dict], records_by_query: dict[int, list[dict]], method_dir: Path, checkpoint: Path, probe_path: Path, extra_config: dict) -> None:
    per_query = []
    retrieval_path = method_dir / "retrievals.jsonl"
    with retrieval_path.open("w", encoding="utf-8") as retrieval_file:
        for query_id, item in enumerate(annotations):
            if query_id not in records_by_query:
                continue
            totals = {f"Recall@{k}": 0.0 for k in TOP_KS}
            totals.update({f"Precision@{k}": 0.0 for k in TOP_KS})
            kept_sum = 0.0
            for record in records_by_query[query_id]:
                retrieval_file.write(json.dumps(record) + "\n")
                valid = set(item["ground_truth"][str(record["source_index"])])
                kept_sum += float(record.get("kept_in_pool", 0.0))
                for k in TOP_KS:
                    recall, precision = retrieval_metrics(record["top10"], valid, k)
                    totals[f"Recall@{k}"] += recall
                    totals[f"Precision@{k}"] += precision
            source_count = len(records_by_query[query_id])
            per_query.append(
                {
                    "query_id": query_id,
                    "query": item["query"],
                    "sources": source_count,
                    "avg_kept_in_pool": kept_sum / max(1, source_count),
                    **{name: value / source_count for name, value in totals.items()},
                }
            )
    write_csv(method_dir / "per_query_metrics.csv", per_query)
    metric_names = [name for name in per_query[0] if "@" in name]
    total_sources = sum(row["sources"] for row in per_query)
    summary = {
        "method": method_dir.name,
        "checkpoint": str(checkpoint),
        "probe_checkpoint": str(probe_path),
        "query_entries": len(per_query),
        "source_query_cases": total_sources,
        "avg_kept_in_pool": sum(row["avg_kept_in_pool"] * row["sources"] for row in per_query) / total_sources,
        **{
            f"macro_{name}": sum(row[name] for row in per_query) / len(per_query)
            for name in metric_names
        },
        **{
            f"micro_{name}": sum(row[name] * row["sources"] for row in per_query) / total_sources
            for name in metric_names
        },
    }
    write_csv(method_dir / "summary.csv", [summary])
    atomic_json_dump(extra_config, method_dir / "config.json")
    (method_dir / "COMPLETE").write_text("complete\n", encoding="utf-8")


def plot_comparison(output_root: Path) -> None:
    rows = []
    perq = []
    for method_dir in sorted(output_root.glob("*")):
        if not method_dir.is_dir() or method_dir.name in {"probe", "comparison"}:
            continue
        summary = method_dir / "summary.csv"
        per_query = method_dir / "per_query_metrics.csv"
        if summary.exists() and per_query.exists():
            s = pd.read_csv(summary)
            p = pd.read_csv(per_query)
            s["method"] = method_dir.name
            p["method"] = method_dir.name
            rows.append(s)
            perq.append(p)
    if not rows:
        return
    summary_df = pd.concat(rows, ignore_index=True)
    perq_df = pd.concat(perq, ignore_index=True)
    comp = output_root / "comparison"
    comp.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(comp / "combined_summary.csv", index=False)
    perq_df.to_csv(comp / "combined_per_query_metrics.csv", index=False)
    best = summary_df.sort_values("macro_Recall@10", ascending=False).iloc[0]
    (comp / "BEST_PROBE_RERANKER_METHOD.txt").write_text(
        "\n".join(
            [
                f"method={best['method']}",
                f"macro_Recall@10={best['macro_Recall@10']}",
                f"micro_Recall@10={best['micro_Recall@10']}",
                f"macro_Precision@10={best['macro_Precision@10']}",
                f"avg_kept_in_pool={best['avg_kept_in_pool']}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    top = summary_df.sort_values("macro_Recall@10", ascending=False).iloc[::-1]
    plt.figure(figsize=(10, max(5, 0.35 * len(top))))
    plt.barh(top["method"], top["macro_Recall@10"], color="#2f6f73")
    plt.xlabel("Macro Recall@10")
    plt.title("Probe reranker methods on official JSON")
    plt.tight_layout()
    plt.savefig(comp / "macro_recall10_methods.png", dpi=180)
    plt.close()

    top_p = summary_df.sort_values("macro_Precision@10", ascending=False).iloc[::-1]
    plt.figure(figsize=(10, max(5, 0.35 * len(top_p))))
    plt.barh(top_p["method"], top_p["macro_Precision@10"], color="#b76e22")
    plt.xlabel("Macro Precision@10")
    plt.title("Probe reranker methods on official JSON")
    plt.tight_layout()
    plt.savefig(comp / "macro_precision10_methods.png", dpi=180)
    plt.close()

    focus_methods = summary_df.sort_values("macro_Recall@10", ascending=False)["method"].head(6).tolist()
    pivot = perq_df[perq_df["method"].isin(focus_methods)].pivot_table(
        index="query",
        columns="method",
        values="Recall@10",
        aggfunc="first",
    )
    pivot.to_csv(comp / "per_query_recall10_top_methods.csv")
    pivot.plot(kind="barh", figsize=(12, 8))
    plt.xlabel("Recall@10")
    plt.title("Per-query Recall@10 for top probe reranker methods")
    plt.tight_layout()
    plt.savefig(comp / "per_query_recall10_top_methods.png", dpi=180)
    plt.close()


def evaluate_rerankers(args: argparse.Namespace, output_root: Path, progress: Path, device: torch.device, probe_path: Path) -> None:
    checkpoint, beta = find_checkpoint_and_beta(args, progress)
    model, checkpoint_data = load_model_checkpoint(checkpoint, device)
    model.eval()
    config = checkpoint_data["config"]
    prompt_cache = load_prompt_embedding_cache(blends.router.resolve_project_path(config.get("prompt_cache_path")))
    text_bank, _ = blends.router.load_text_banks(device)

    probe, probe_attributes, probe_checkpoint = load_probe(probe_path, device)
    attributes, _, _ = read_attribute_table()
    if attributes != probe_attributes:
        raise RuntimeError("Probe attributes do not match CelebA attributes")
    attribute_to_index = {name: idx for idx, name in enumerate(attributes)}
    annotations_all = load_evaluation()
    query_ids = list(range(len(annotations_all))) if args.query_ids is None else args.query_ids

    gallery_cache = load_split_embeddings("test")
    gallery = F.normalize(gallery_cache["embeddings"].float(), dim=-1).to(device)
    probe_probs_path = output_root / "probe" / "test_probe_probs.pt"
    if probe_probs_path.exists() and not args.force:
        gallery_probs = load_torch(probe_probs_path)["probs"].float()
        progress_line(progress, f"REUSE test probe probs {probe_probs_path}")
    else:
        progress_line(progress, "PROBE predicting test attributes")
        gallery_probs = predict_probe_probs(probe, gallery_cache["embeddings"], device, int(args.probe_batch_size))
        atomic_torch_save({"model_id": MODEL_ID, "attributes": attributes, "probs": gallery_probs}, probe_probs_path)
    gallery_probs_device = gallery_probs.to(device)
    methods = default_methods()
    records = {method.name: {} for method in methods}
    started = time.monotonic()

    atomic_json_dump(
        {
            "checkpoint": str(checkpoint),
            "probe_checkpoint": str(probe_path),
            "probe_valid_metrics": probe_checkpoint.get("valid_metrics", {}),
            "model_id": MODEL_ID,
            "beta": beta,
            "top_pool": args.top_pool,
            "top_k": args.top_k,
            "methods": [method.__dict__ for method in methods],
            "notes": "Probe-based reranking. JSON is used only for final metric computation.",
        },
        output_root / "experiment_config.json",
    )

    with torch.inference_mode():
        for query_id in query_ids:
            item = annotations_all[query_id]
            conditions = parse_query(item["query"])
            attr_indices_one, signs_one = query_tensors(conditions, attribute_to_index, device)
            source_indices = [int(index) for index in item["ground_truth"]]
            if args.max_sources_per_query:
                source_indices = source_indices[: int(args.max_sources_per_query)]
            for start in range(0, len(source_indices), int(args.source_batch_size)):
                end = min(start + int(args.source_batch_size), len(source_indices))
                batch_indices = source_indices[start:end]
                idx = torch.tensor(batch_indices, device=device)
                source = gallery[idx]
                q_model = blends.model_query(
                    model,
                    source,
                    conditions,
                    attribute_to_index,
                    prompt_cache,
                    str(config.get("condition_mode", "signed_prompt")),
                    device,
                )
                q_generic = blends.generic_sum_query(source, conditions, attribute_to_index, text_bank)
                query = blends.vector_delta(q_model, q_generic, source, beta)
                scores = query @ gallery.T
                rows = torch.arange(len(batch_indices), device=device)
                scores[rows, idx] = -torch.inf
                top_scores, top_indices = scores.topk(int(args.top_pool), dim=1)
                source_probs = gallery_probs_device[idx]

                for method in methods:
                    rankings, kept_counts = rerank_top_pool(
                        method,
                        top_indices,
                        top_scores,
                        source,
                        gallery,
                        source_probs,
                        gallery_probs_device,
                        attr_indices_one[0],
                        signs_one[0],
                        int(args.top_k),
                    )
                    records[method.name].setdefault(query_id, []).extend(
                        {
                            "method": method.name,
                            "query_id": query_id,
                            "query": item["query"],
                            "source_index": source_index,
                            "top10": ranking,
                            "kept_in_pool": kept_count,
                        }
                        for source_index, ranking, kept_count in zip(batch_indices, rankings, kept_counts)
                    )

                progress_line(progress, f"EVAL query={query_id + 1}/{len(annotations_all)} sources={end}/{len(source_indices)}")
                if STOP_REQUESTED or (
                    args.time_budget_seconds and time.monotonic() - started >= int(args.time_budget_seconds)
                ):
                    raise SystemExit("Time budget reached during reranker evaluation")

    for method in methods:
        method_dir = output_root / method.name
        method_dir.mkdir(parents=True, exist_ok=True)
        finalize_method(
            annotations_all,
            records[method.name],
            method_dir,
            checkpoint,
            probe_path,
            {
                "method": method.__dict__,
                "checkpoint": str(checkpoint),
                "probe_checkpoint": str(probe_path),
                "beta": beta,
                "top_pool": args.top_pool,
                "top_k": args.top_k,
            },
        )
    plot_comparison(output_root)
    progress_line(progress, f"COMPLETE probe reranker output={output_root}")


def main() -> int:
    args = parse_args()
    signal.signal(signal.SIGTERM, request_stop)
    if hasattr(signal, "SIGUSR1"):
        signal.signal(signal.SIGUSR1, request_stop)
    device = choose_device(args.device)
    output_root = args.output_root or (ARTIFACTS_DIR / "results" / "probe_reranker_v1" / f"probe_reranker_v1_{timestamp()}")
    output_root.mkdir(parents=True, exist_ok=True)
    progress = output_root / "progress.txt"
    progress_line(progress, f"START probe_reranker_v1 output={output_root}")
    progress_line(progress, f"args={json.dumps(vars(args), default=str, sort_keys=True)}")
    probe_path = train_probe(args, output_root, progress, device)
    evaluate_rerankers(args, output_root, progress, device, probe_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
