#!/usr/bin/env python3
"""Probe architecture sweep for CelebA attribute filtering.

This experiment keeps the current best compositional retrieval system frozen.
For each probe configuration it:

1. trains a CelebA multi-label attribute probe on CLIP image embeddings;
2. calibrates one threshold per attribute on the validation split;
3. predicts attributes for the official test gallery;
4. evaluates probe-based filtering/reranking on the official JSON;
5. saves per-config summaries and a global ranking.

The official JSON is used only for evaluation after each probe is trained.

Academic-integrity note
-----------------------
The architectures below are implemented from scratch with standard PyTorch
building blocks. They are inspired by high-level ideas from multi-label
classification literature, but no third-party project/repository code is copied:

- ASL: "Asymmetric Loss For Multi-Label Classification", arXiv:2009.14119.
- C-Tran-style label dependency modelling: "General Multi-label Image
  Classification with Transformers", arXiv:2011.14027.
- ML-GCN-style label correlation modelling: "Multi-Label Image Recognition
  with Graph Convolutional Networks", arXiv:1904.03582.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import signal
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - plotting is optional on minimal envs.
    plt = None


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTAL = ROOT / "experimental"
SCRIPTS = ROOT / "scripts"
ORCH = ROOT / "orchestrator"
sys.path.insert(0, str(EXPERIMENTAL))
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(ORCH))
os.environ.setdefault("DL_PROJECT_ROOT", str(ROOT))

import evaluate_sum_model_blends as blends  # noqa: E402
import probe_reranker_v1 as v1  # noqa: E402
import probe_reranker_v2_calibrated as v2  # noqa: E402
from learned_gate_core import load_model_checkpoint, load_prompt_embedding_cache, progress_line  # noqa: E402
from project_core import (  # noqa: E402
    ARTIFACTS_DIR,
    MODEL_ID,
    TOP_KS,
    atomic_json_dump,
    atomic_torch_save,
    choose_device,
    load_evaluation,
    load_torch,
    parse_query,
    read_attribute_table,
    retrieval_metrics,
)


STOP_REQUESTED = False


def request_stop(signum, frame) -> None:
    del signum, frame
    global STOP_REQUESTED
    STOP_REQUESTED = True
    v1.STOP_REQUESTED = True


@dataclass(frozen=True)
class ProbeArchConfig:
    config_id: str
    arch: str
    loss: str = "bce"
    epochs: int = 10
    max_steps: int = 0
    batch_size: int = 2048
    lr: float = 3e-4
    weight_decay: float = 1e-4
    dropout: float = 0.10
    hidden_dims: tuple[int, ...] = (1024, 512)
    hidden_dim: int = 1024
    blocks: int = 2
    label_dim: int = 256
    layers: int = 2
    heads: int = 4
    use_pos_weight: bool = True
    noise_std: float = 0.0
    asym_gamma_neg: float = 4.0
    asym_gamma_pos: float = 0.0
    focal_gamma: float = 2.0
    notes: str = ""


class LinearProbe(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, **_: Any):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, output_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(F.normalize(x.float(), dim=-1))


class MLPProbe(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: tuple[int, ...], dropout: float, **_: Any):
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


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class ResidualMLPProbe(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, blocks: int, dropout: float, **_: Any):
        super().__init__()
        self.input = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden_dim), nn.GELU())
        self.blocks = nn.Sequential(*[ResidualBlock(hidden_dim, dropout) for _ in range(blocks)])
        self.head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, output_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input(F.normalize(x.float(), dim=-1))
        return self.head(self.blocks(h))


class LabelWiseProbe(nn.Module):
    """Shared per-label head with learned label embeddings.

    This is still lightweight, but unlike a plain MLP it gives each attribute a
    learned token and lets the classifier share statistical strength across
    labels.
    """

    def __init__(self, input_dim: int, output_dim: int, label_dim: int, dropout: float, **_: Any):
        super().__init__()
        self.label_tokens = nn.Parameter(torch.randn(output_dim, label_dim) * 0.02)
        self.image = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, label_dim), nn.GELU())
        self.head = nn.Sequential(
            nn.LayerNorm(label_dim * 3),
            nn.Linear(label_dim * 3, label_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(label_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        image = self.image(F.normalize(x.float(), dim=-1))
        labels = self.label_tokens.unsqueeze(0).expand(x.shape[0], -1, -1)
        image_tokens = image.unsqueeze(1).expand_as(labels)
        pair = torch.cat([image_tokens, labels, image_tokens * labels], dim=-1)
        return self.head(pair).squeeze(-1)


class LabelTransformerProbe(nn.Module):
    """Transformer over one image token plus 40 attribute tokens."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        label_dim: int,
        layers: int,
        heads: int,
        dropout: float,
        **_: Any,
    ):
        super().__init__()
        self.image = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, label_dim), nn.GELU())
        self.label_tokens = nn.Parameter(torch.randn(output_dim, label_dim) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=label_dim,
            nhead=heads,
            dim_feedforward=label_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.head = nn.Sequential(nn.LayerNorm(label_dim), nn.Linear(label_dim, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        image = self.image(F.normalize(x.float(), dim=-1)).unsqueeze(1)
        labels = self.label_tokens.unsqueeze(0).expand(x.shape[0], -1, -1)
        tokens = torch.cat([image, labels], dim=1)
        encoded = self.encoder(tokens)
        return self.head(encoded[:, 1:, :]).squeeze(-1)


class GraphLabelProbe(nn.Module):
    """Label graph classifier using CelebA train-label co-occurrence.

    The graph produces correlated label classifiers; logits are dot products
    between projected image embeddings and graph-smoothed label embeddings.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        label_dim: int,
        dropout: float,
        adjacency: torch.Tensor,
        **_: Any,
    ):
        super().__init__()
        self.label_tokens = nn.Parameter(torch.randn(output_dim, label_dim) * 0.02)
        self.register_buffer("adjacency", adjacency.float())
        self.g1 = nn.Linear(label_dim, label_dim)
        self.g2 = nn.Linear(label_dim, label_dim)
        self.image = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, label_dim), nn.GELU(), nn.Dropout(dropout))
        self.bias = nn.Parameter(torch.zeros(output_dim))
        self.scale = math.sqrt(float(label_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        labels = self.label_tokens
        labels = F.gelu(self.g1(self.adjacency @ labels))
        labels = self.g2(self.adjacency @ labels)
        image = self.image(F.normalize(x.float(), dim=-1))
        return image @ labels.T / self.scale + self.bias


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("short", "long"), default="long")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--time-budget-seconds", type=int, default=0)
    parser.add_argument("--finalize-window-seconds", type=int, default=300)
    parser.add_argument("--min-seconds-for-new-run", type=int, default=900)
    parser.add_argument("--top-pool", type=int, default=500)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--eval-source-batch-size", type=int, default=256)
    parser.add_argument("--probe-batch-size", type=int, default=2048)
    parser.add_argument("--query-ids", nargs="*", type=int, default=None)
    parser.add_argument("--max-sources-per-query", type=int, default=0)
    parser.add_argument("--threshold-objectives", nargs="+", default=["accuracy", "f1", "balanced_accuracy"])
    parser.add_argument("--threshold-min", type=float, default=0.05)
    parser.add_argument("--threshold-max", type=float, default=0.95)
    parser.add_argument("--threshold-step", type=float, default=0.025)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def append_csv(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def default_configs(profile: str) -> list[ProbeArchConfig]:
    if profile == "short":
        return [
            ProbeArchConfig(
                "smoke_residual_asl",
                "residual_mlp",
                loss="asl",
                epochs=1,
                max_steps=4,
                batch_size=512,
                hidden_dim=512,
                blocks=1,
                lr=3e-4,
                notes="End-to-end smoke only.",
            )
        ]

    return [
        ProbeArchConfig("p01_linear_bce", "linear", loss="bce", epochs=10, lr=5e-4, dropout=0.0, notes="Sanity linear probe."),
        ProbeArchConfig("p02_mlp_current_bce", "mlp", loss="bce", epochs=10, hidden_dims=(1024, 512), lr=3e-4, dropout=0.10),
        ProbeArchConfig("p03_mlp_current_asl", "mlp", loss="asl", epochs=10, hidden_dims=(1024, 512), lr=3e-4, dropout=0.10),
        ProbeArchConfig("p04_mlp_deep_bce", "mlp", loss="bce", epochs=12, hidden_dims=(1536, 1024, 512), lr=2e-4, dropout=0.10),
        ProbeArchConfig("p05_mlp_deep_asl", "mlp", loss="asl", epochs=12, hidden_dims=(1536, 1024, 512), lr=2e-4, dropout=0.10),
        ProbeArchConfig("p06_mlp_lowdrop_asl", "mlp", loss="asl", epochs=12, hidden_dims=(1024, 1024, 512), lr=2e-4, dropout=0.05),
        ProbeArchConfig("p07_residual_1024_bce", "residual_mlp", loss="bce", epochs=12, hidden_dim=1024, blocks=2, lr=3e-4, dropout=0.10),
        ProbeArchConfig("p08_residual_1024_asl", "residual_mlp", loss="asl", epochs=12, hidden_dim=1024, blocks=2, lr=3e-4, dropout=0.10),
        ProbeArchConfig("p09_residual_1536_asl", "residual_mlp", loss="asl", epochs=12, hidden_dim=1536, blocks=3, lr=2e-4, dropout=0.10),
        ProbeArchConfig("p10_residual_noise_asl", "residual_mlp", loss="asl", epochs=12, hidden_dim=1024, blocks=3, lr=2e-4, dropout=0.05, noise_std=0.01),
        ProbeArchConfig("p11_labelwise_256_bce", "labelwise", loss="bce", epochs=12, label_dim=256, lr=3e-4, dropout=0.10),
        ProbeArchConfig("p12_labelwise_512_asl", "labelwise", loss="asl", epochs=12, label_dim=512, lr=2e-4, dropout=0.10),
        ProbeArchConfig("p13_labelwise_512_focal", "labelwise", loss="focal_bce", epochs=12, label_dim=512, lr=2e-4, dropout=0.10),
        ProbeArchConfig("p14_transformer_128_l2_asl", "label_transformer", loss="asl", epochs=12, label_dim=128, layers=2, heads=4, lr=2e-4, dropout=0.10),
        ProbeArchConfig("p15_transformer_192_l3_asl", "label_transformer", loss="asl", epochs=12, label_dim=192, layers=3, heads=4, lr=2e-4, dropout=0.10),
        ProbeArchConfig("p16_transformer_256_l4_asl", "label_transformer", loss="asl", epochs=12, label_dim=256, layers=4, heads=4, lr=1e-4, dropout=0.10),
        ProbeArchConfig("p17_graph_256_bce", "graph_label", loss="bce", epochs=12, label_dim=256, lr=3e-4, dropout=0.10),
        ProbeArchConfig("p18_graph_512_asl", "graph_label", loss="asl", epochs=12, label_dim=512, lr=2e-4, dropout=0.10),
    ]


def build_adjacency(labels: torch.Tensor) -> torch.Tensor:
    y = labels.float()
    co = (y.T @ y) / y.shape[0]
    freq = y.mean(dim=0).clamp_min(1e-4)
    norm = co / torch.sqrt(freq[:, None] * freq[None, :])
    norm.fill_diagonal_(1.0)
    norm = norm.clamp(0.0, 1.0)
    norm = norm + torch.eye(norm.shape[0]) * 0.25
    return norm / norm.sum(dim=-1, keepdim=True).clamp_min(1e-8)


def build_probe(config: ProbeArchConfig, output_dim: int, adjacency: torch.Tensor | None) -> nn.Module:
    common = {
        "input_dim": 512,
        "output_dim": output_dim,
        "hidden_dims": config.hidden_dims,
        "hidden_dim": config.hidden_dim,
        "blocks": config.blocks,
        "label_dim": config.label_dim,
        "layers": config.layers,
        "heads": config.heads,
        "dropout": config.dropout,
        "adjacency": adjacency,
    }
    if config.arch == "linear":
        return LinearProbe(**common)
    if config.arch == "mlp":
        return MLPProbe(**common)
    if config.arch == "residual_mlp":
        return ResidualMLPProbe(**common)
    if config.arch == "labelwise":
        return LabelWiseProbe(**common)
    if config.arch == "label_transformer":
        return LabelTransformerProbe(**common)
    if config.arch == "graph_label":
        if adjacency is None:
            raise RuntimeError("GraphLabelProbe requires adjacency")
        return GraphLabelProbe(**common)
    raise ValueError(f"Unknown probe architecture: {config.arch}")


def class_pos_weight(labels: torch.Tensor) -> torch.Tensor:
    positives = labels.sum(dim=0).clamp_min(1.0)
    negatives = (labels.shape[0] - labels.sum(dim=0)).clamp_min(1.0)
    return (negatives / positives).clamp(0.25, 20.0)


def asymmetric_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma_neg: float = 4.0,
    gamma_pos: float = 0.0,
    clip: float = 0.05,
    pos_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    pos = probs
    neg = 1.0 - probs
    if clip > 0:
        neg = (neg + clip).clamp(max=1.0)
    log_pos = torch.log(pos.clamp_min(1e-8))
    log_neg = torch.log(neg.clamp_min(1e-8))
    loss_pos = targets * log_pos
    loss_neg = (1.0 - targets) * log_neg
    pt = pos * targets + neg * (1.0 - targets)
    gamma = gamma_pos * targets + gamma_neg * (1.0 - targets)
    weight = (1.0 - pt).pow(gamma)
    if pos_weight is not None:
        weight = weight * (1.0 + targets * (pos_weight.view(1, -1) - 1.0))
    return -(weight * (loss_pos + loss_neg)).mean()


def focal_bce_loss(logits: torch.Tensor, targets: torch.Tensor, gamma: float, pos_weight: torch.Tensor | None) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight, reduction="none")
    pt = torch.exp(-bce)
    return (((1.0 - pt) ** gamma) * bce).mean()


def loss_fn(config: ProbeArchConfig, logits: torch.Tensor, labels: torch.Tensor, pos_weight: torch.Tensor) -> torch.Tensor:
    weight = pos_weight if config.use_pos_weight else None
    if config.loss == "bce":
        return F.binary_cross_entropy_with_logits(logits, labels, pos_weight=weight)
    if config.loss == "asl":
        return asymmetric_loss(logits, labels, config.asym_gamma_neg, config.asym_gamma_pos, pos_weight=weight)
    if config.loss == "focal_bce":
        return focal_bce_loss(logits, labels, config.focal_gamma, weight)
    raise ValueError(f"Unknown loss: {config.loss}")


def binary_metrics(preds: torch.Tensor, labels: torch.Tensor) -> dict[str, torch.Tensor]:
    truth = labels.bool()
    pred = preds.bool()
    eps = 1e-8
    tp = (pred & truth).sum(dim=0).float()
    fp = (pred & ~truth).sum(dim=0).float()
    fn = (~pred & truth).sum(dim=0).float()
    tn = (~pred & ~truth).sum(dim=0).float()
    accuracy = (tp + tn) / (tp + tn + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    specificity = tn / (tn + fp + eps)
    balanced_accuracy = 0.5 * (recall + specificity)
    f1 = 2 * precision * recall / (precision + recall + eps)
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "balanced_accuracy": balanced_accuracy,
        "f1": f1,
    }


def probe_prediction_stats(probs: torch.Tensor, labels: torch.Tensor, thresholds: torch.Tensor) -> dict[str, float]:
    preds = probs >= thresholds[None, :]
    metrics = binary_metrics(preds, labels)
    errors = (preds != labels.bool()).sum(dim=1).float()
    return {
        "macro_accuracy": float(metrics["accuracy"].mean()),
        "macro_precision": float(metrics["precision"].mean()),
        "macro_recall": float(metrics["recall"].mean()),
        "macro_f1": float(metrics["f1"].mean()),
        "macro_balanced_accuracy": float(metrics["balanced_accuracy"].mean()),
        "micro_accuracy": float((preds == labels.bool()).float().mean()),
        "mean_hamming_errors": float(errors.mean()),
        "pct_exact_40": float((errors == 0).float().mean()),
        "pct_hamming_le1": float((errors <= 1).float().mean()),
        "pct_hamming_le2": float((errors <= 2).float().mean()),
        "pct_hamming_le3": float((errors <= 3).float().mean()),
        "pct_hamming_le5": float((errors <= 5).float().mean()),
    }


def predict_probs(model: nn.Module, embeddings: torch.Tensor, device: torch.device, batch_size: int) -> torch.Tensor:
    model.eval()
    chunks = []
    with torch.inference_mode():
        for start in range(0, len(embeddings), batch_size):
            x = embeddings[start : start + batch_size].to(device)
            chunks.append(torch.sigmoid(model(x)).cpu())
    return torch.cat(chunks)


def calibrate_thresholds(
    probs: torch.Tensor,
    labels: torch.Tensor,
    attributes: list[str],
    objectives: list[str],
    threshold_min: float,
    threshold_max: float,
    threshold_step: float,
) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]], list[dict[str, Any]]]:
    thresholds = torch.arange(threshold_min, threshold_max + 0.5 * threshold_step, threshold_step).clamp(0.0, 1.0)
    per_attr_rows = []
    summary_rows = []
    threshold_by_objective: dict[str, torch.Tensor] = {}
    for objective in objectives:
        chosen = []
        for attr_idx, attr_name in enumerate(attributes):
            best_score = -1.0
            best_threshold = 0.5
            best_metrics = None
            attr_labels = labels[:, attr_idx : attr_idx + 1]
            attr_probs = probs[:, attr_idx : attr_idx + 1]
            for threshold in thresholds:
                metrics = binary_metrics(attr_probs >= threshold, attr_labels)
                if objective == "precision_recall_mid":
                    value = torch.minimum(metrics["precision"], metrics["recall"])[0]
                else:
                    value = metrics[objective][0]
                if float(value) > best_score:
                    best_score = float(value)
                    best_threshold = float(threshold)
                    best_metrics = metrics
            assert best_metrics is not None
            chosen.append(best_threshold)
            per_attr_rows.append(
                {
                    "objective": objective,
                    "attribute": attr_name,
                    "threshold": best_threshold,
                    "score": best_score,
                    "accuracy": float(best_metrics["accuracy"][0]),
                    "precision": float(best_metrics["precision"][0]),
                    "recall": float(best_metrics["recall"][0]),
                    "balanced_accuracy": float(best_metrics["balanced_accuracy"][0]),
                    "f1": float(best_metrics["f1"][0]),
                    "positive_rate": float(attr_labels.mean()),
                }
            )
        threshold_by_objective[objective] = torch.tensor(chosen, dtype=torch.float32)
        summary_rows.append({"objective": objective, **probe_prediction_stats(probs, labels, threshold_by_objective[objective])})
    return threshold_by_objective, per_attr_rows, summary_rows


def load_split_labels(split: str) -> tuple[dict[str, Any], list[str], torch.Tensor]:
    cache = v1.load_split_embeddings(split)
    attributes, labels = v1.attrs_for_filenames(cache["filenames"])
    return cache, attributes, labels


def train_one_probe(
    args: argparse.Namespace,
    config: ProbeArchConfig,
    output_root: Path,
    progress: Path,
    device: torch.device,
    train_cache: dict[str, Any],
    train_labels: torch.Tensor,
    valid_cache: dict[str, Any],
    valid_labels: torch.Tensor,
    attributes: list[str],
    adjacency: torch.Tensor,
) -> tuple[Path, dict[str, torch.Tensor], dict[str, float]]:
    run_dir = output_root / "probes" / config.config_id
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "train_metrics.csv"
    best_path = run_dir / "best_probe.pt"
    atomic_json_dump(asdict(config), run_dir / "config.json")

    model = build_probe(config, len(attributes), adjacency.to(device)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    pos_weight = class_pos_weight(train_labels).to(device)
    train_x = train_cache["embeddings"].float()
    train_y = train_labels.float()
    valid_x = valid_cache["embeddings"].float()
    valid_y = valid_labels.float()
    batch_size = int(args.probe_batch_size or config.batch_size)
    best_score = float("-inf")
    best_stats: dict[str, float] = {}
    global_step = 0

    progress_line(progress, f"TRAIN_PROBE start config={config.config_id} arch={config.arch} loss={config.loss}")
    for epoch in range(1, config.epochs + 1):
        model.train()
        order = torch.randperm(len(train_x))
        losses = []
        for start in range(0, len(order), batch_size):
            positions = order[start : start + batch_size]
            x = train_x[positions].to(device)
            y = train_y[positions].to(device)
            if config.noise_std > 0:
                x = F.normalize(x + torch.randn_like(x) * float(config.noise_std), dim=-1)
            logits = model(x)
            loss = loss_fn(config, logits, y, pos_weight)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            global_step += 1
            if config.max_steps and global_step >= config.max_steps:
                break
            if STOP_REQUESTED:
                raise SystemExit("Stop requested during probe training")

        valid_probs = predict_probs(model, valid_x, device, batch_size)
        threshold_by_objective, _, summary_rows = calibrate_thresholds(
            valid_probs,
            valid_y,
            attributes,
            ["accuracy", "f1", "balanced_accuracy"],
            float(args.threshold_min),
            float(args.threshold_max),
            float(args.threshold_step),
        )
        accuracy_stats = probe_prediction_stats(valid_probs, valid_y, threshold_by_objective["accuracy"])
        f1_stats = probe_prediction_stats(valid_probs, valid_y, threshold_by_objective["f1"])
        # Score balances per-attribute F1 with low whole-image Hamming error,
        # because filtering needs many labels to be simultaneously right.
        score = f1_stats["macro_f1"] + 0.20 * accuracy_stats["pct_hamming_le2"]
        row = {
            "epoch": epoch,
            "step": global_step,
            "train_loss": sum(losses) / max(1, len(losses)),
            "selection_score": score,
            "acc_macro_f1": accuracy_stats["macro_f1"],
            "acc_macro_accuracy": accuracy_stats["macro_accuracy"],
            "acc_pct_hamming_le2": accuracy_stats["pct_hamming_le2"],
            "f1_macro_f1": f1_stats["macro_f1"],
            "f1_macro_recall": f1_stats["macro_recall"],
            "f1_pct_hamming_le2": f1_stats["pct_hamming_le2"],
        }
        append_csv(metrics_path, row)
        progress_line(
            progress,
            "TRAIN_PROBE "
            f"config={config.config_id} epoch={epoch} step={global_step} "
            f"loss={row['train_loss']:.4f} score={score:.4f} "
            f"acc_ham_le2={accuracy_stats['pct_hamming_le2']:.4f} f1={f1_stats['macro_f1']:.4f}",
        )
        if score > best_score:
            best_score = score
            best_stats = row
            atomic_torch_save(
                {
                    "model_id": MODEL_ID,
                    "attributes": attributes,
                    "config": asdict(config),
                    "model_state": model.state_dict(),
                    "valid_selection_score": best_score,
                    "valid_epoch": epoch,
                    "valid_stats": row,
                },
                best_path,
            )
            atomic_torch_save({"thresholds": threshold_by_objective, "attributes": attributes}, run_dir / "best_thresholds.pt")
            write_csv(run_dir / "best_threshold_summary.csv", summary_rows)

        if config.max_steps and global_step >= config.max_steps:
            break

    checkpoint = load_torch(best_path)
    best_model = build_probe(config, len(attributes), adjacency.to(device)).to(device)
    best_model.load_state_dict(checkpoint["model_state"])
    best_model.eval()
    valid_probs = predict_probs(best_model, valid_x, device, batch_size)
    threshold_by_objective, per_attr_rows, summary_rows = calibrate_thresholds(
        valid_probs,
        valid_y,
        attributes,
        list(args.threshold_objectives),
        float(args.threshold_min),
        float(args.threshold_max),
        float(args.threshold_step),
    )
    atomic_torch_save({"thresholds": threshold_by_objective, "attributes": attributes}, run_dir / "calibrated_thresholds.pt")
    atomic_torch_save({"model_id": MODEL_ID, "attributes": attributes, "probs": valid_probs}, run_dir / "valid_probe_probs.pt")
    write_csv(run_dir / "calibrated_thresholds_per_attribute.csv", per_attr_rows)
    write_csv(run_dir / "calibrated_threshold_summary.csv", summary_rows)
    return best_path, threshold_by_objective, best_stats


def load_probe_checkpoint(path: Path, device: torch.device, adjacency: torch.Tensor) -> tuple[nn.Module, list[str], ProbeArchConfig]:
    checkpoint = load_torch(path)
    config = ProbeArchConfig(**checkpoint["config"])
    attributes = list(checkpoint["attributes"])
    model = build_probe(config, len(attributes), adjacency.to(device)).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, attributes, config


def query_mask_and_hamming(
    source_probs: torch.Tensor,
    candidate_probs: torch.Tensor,
    attr_indices: torch.Tensor,
    signs: torch.Tensor,
    thresholds: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    query_ok = v2.calibrated_query_ok(candidate_probs, attr_indices, signs, thresholds)
    hard_ham, _, _ = v2.calibrated_hamming(source_probs, candidate_probs, attr_indices, thresholds)
    hamming_ok = hard_ham <= 2.0
    return query_ok, hamming_ok


def select_with_mask(top_indices: torch.Tensor, mask: torch.Tensor | None, top_k: int, fill: bool) -> tuple[list[list[int]], list[float]]:
    rankings: list[list[int]] = []
    kept_counts: list[float] = []
    for row in range(top_indices.shape[0]):
        if mask is None:
            selected = top_indices[row, :top_k].tolist()
            kept = float(top_k)
        else:
            local = torch.nonzero(mask[row], as_tuple=False).flatten()
            selected = top_indices[row, local[:top_k]].tolist()
            kept = float(local.numel())
            if fill and len(selected) < top_k:
                selected_set = set(selected)
                for idx in top_indices[row].tolist():
                    if int(idx) not in selected_set:
                        selected.append(int(idx))
                        selected_set.add(int(idx))
                    if len(selected) >= top_k:
                        break
        rankings.append([int(x) for x in selected[:top_k]])
        kept_counts.append(kept)
    return rankings, kept_counts


def update_records(
    records_by_method: dict[str, dict[int, list[dict[str, Any]]]],
    method: str,
    query_id: int,
    query: str,
    source_indices: list[int],
    rankings: list[list[int]],
    kept_counts: list[float],
) -> None:
    bucket = records_by_method[method].setdefault(query_id, [])
    for source_index, ranking, kept in zip(source_indices, rankings, kept_counts):
        bucket.append(
            {
                "method": method,
                "query_id": query_id,
                "query": query,
                "source_index": int(source_index),
                "top10": [int(x) for x in ranking],
                "kept_in_pool": float(kept),
            }
        )


def summarize_records(
    annotations: list[dict[str, Any]],
    records_by_query: dict[int, list[dict[str, Any]]],
    method: str,
    checkpoint: Path,
    probe_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    per_query = []
    for query_id, item in enumerate(annotations):
        records = records_by_query.get(query_id, [])
        if not records:
            continue
        totals = {f"Recall@{k}": 0.0 for k in TOP_KS}
        totals.update({f"Precision@{k}": 0.0 for k in TOP_KS})
        kept_sum = 0.0
        for record in records:
            valid = set(item["ground_truth"][str(record["source_index"])])
            kept_sum += float(record.get("kept_in_pool", 0.0))
            for k in TOP_KS:
                recall, precision = retrieval_metrics(record["top10"], valid, k)
                totals[f"Recall@{k}"] += recall
                totals[f"Precision@{k}"] += precision
        count = len(records)
        per_query.append(
            {
                "query_id": query_id,
                "query": item["query"],
                "sources": count,
                "avg_kept_in_pool": kept_sum / max(1, count),
                **{name: value / count for name, value in totals.items()},
            }
        )
    metric_names = [name for name in per_query[0] if "@" in name]
    total_sources = sum(row["sources"] for row in per_query)
    summary = {
        "method": method,
        "checkpoint": str(checkpoint),
        "probe_checkpoint": str(probe_path),
        "query_entries": len(per_query),
        "source_query_cases": total_sources,
        "avg_kept_in_pool": sum(row["avg_kept_in_pool"] * row["sources"] for row in per_query) / max(1, total_sources),
        **{f"macro_{name}": sum(row[name] for row in per_query) / len(per_query) for name in metric_names},
        **{f"micro_{name}": sum(row[name] * row["sources"] for row in per_query) / max(1, total_sources) for name in metric_names},
    }
    return summary, per_query


def evaluate_probe_on_json(
    args: argparse.Namespace,
    config: ProbeArchConfig,
    output_root: Path,
    progress: Path,
    device: torch.device,
    probe_path: Path,
    thresholds_by_objective: dict[str, torch.Tensor],
    test_cache: dict[str, Any],
    test_probs: torch.Tensor,
    adjacency: torch.Tensor,
) -> list[dict[str, Any]]:
    checkpoint, beta = v1.find_checkpoint_and_beta(args, progress)
    model, checkpoint_data = load_model_checkpoint(checkpoint, device)
    model.eval()
    gate_config = checkpoint_data["config"]
    prompt_cache = load_prompt_embedding_cache(blends.router.resolve_project_path(gate_config.get("prompt_cache_path")))
    text_bank, _ = blends.router.load_text_banks(device)
    del adjacency

    annotations = load_evaluation()
    attributes, _, _ = read_attribute_table()
    attribute_to_index = {name: idx for idx, name in enumerate(attributes)}
    gallery = F.normalize(test_cache["embeddings"].float(), dim=-1).to(device)
    gallery_probs = test_probs.float().to(device)

    query_ids = list(range(len(annotations))) if args.query_ids is None else args.query_ids
    method_names = ["baseline_q_final"]
    for objective in thresholds_by_objective:
        suffix = objective.replace("_", "")
        method_names.extend(
            [
                f"query_only_fill_{suffix}",
                f"hamming_only_fill_{suffix}",
                f"query_hamming_strict_{suffix}",
                f"query_hamming_fill_{suffix}",
            ]
        )
    records: dict[str, dict[int, list[dict[str, Any]]]] = {method: {} for method in method_names}
    started = time.monotonic()

    with torch.inference_mode():
        for query_counter, query_id in enumerate(query_ids, start=1):
            item = annotations[query_id]
            conditions = parse_query(item["query"])
            attr_indices = torch.tensor([attribute_to_index[attr] for _, attr in conditions], dtype=torch.long, device=device)
            signs = torch.tensor([sign for sign, _ in conditions], dtype=torch.int8, device=device)
            source_indices_all = [int(index) for index in item["ground_truth"]]
            if args.max_sources_per_query:
                source_indices_all = source_indices_all[: int(args.max_sources_per_query)]

            for start in range(0, len(source_indices_all), int(args.eval_source_batch_size)):
                end = min(start + int(args.eval_source_batch_size), len(source_indices_all))
                source_indices = source_indices_all[start:end]
                idx = torch.tensor(source_indices, dtype=torch.long, device=device)
                source = gallery[idx]
                q_model = blends.model_query(
                    model,
                    source,
                    conditions,
                    attribute_to_index,
                    prompt_cache,
                    str(gate_config.get("condition_mode", "signed_prompt")),
                    device,
                )
                q_sum = blends.generic_sum_query(source, conditions, attribute_to_index, text_bank)
                q_final = blends.vector_delta(q_model, q_sum, source, float(beta))
                scores = q_final @ gallery.T
                rows = torch.arange(len(source_indices), device=device)
                scores[rows, idx] = -torch.inf
                _, top_indices = scores.topk(int(args.top_pool), dim=1)

                baseline_rankings, baseline_kept = select_with_mask(top_indices, None, int(args.top_k), True)
                update_records(records, "baseline_q_final", query_id, item["query"], source_indices, baseline_rankings, baseline_kept)

                source_probs = gallery_probs[idx]
                candidate_probs = gallery_probs[top_indices]
                for objective, thresholds in thresholds_by_objective.items():
                    suffix = objective.replace("_", "")
                    thresholds_device = thresholds.to(device)
                    query_ok, hamming_ok = query_mask_and_hamming(source_probs, candidate_probs, attr_indices, signs, thresholds_device)
                    method_masks = {
                        f"query_only_fill_{suffix}": (query_ok, True),
                        f"hamming_only_fill_{suffix}": (hamming_ok, True),
                        f"query_hamming_strict_{suffix}": (query_ok & hamming_ok, False),
                        f"query_hamming_fill_{suffix}": (query_ok & hamming_ok, True),
                    }
                    for method, (mask, fill) in method_masks.items():
                        rankings, kept = select_with_mask(top_indices, mask, int(args.top_k), fill)
                        update_records(records, method, query_id, item["query"], source_indices, rankings, kept)

                progress_line(
                    progress,
                    f"EVAL config={config.config_id} query={query_counter}/{len(query_ids)} "
                    f"sources={end}/{len(source_indices_all)} elapsed={time.monotonic() - started:.1f}s",
                )

    eval_root = output_root / "evaluations" / config.config_id
    eval_root.mkdir(parents=True, exist_ok=True)
    summaries = []
    for method, by_query in records.items():
        method_dir = eval_root / method
        method_dir.mkdir(parents=True, exist_ok=True)
        summary, per_query = summarize_records(annotations, by_query, method, checkpoint, probe_path)
        summary.update({"config_id": config.config_id, "arch": config.arch, "loss": config.loss})
        write_csv(method_dir / "summary.csv", [summary])
        write_csv(method_dir / "per_query_metrics.csv", per_query)
        summaries.append(summary)
    write_csv(eval_root / "combined_summary.csv", summaries)
    best = sorted(summaries, key=lambda row: float(row["macro_Recall@10"]), reverse=True)[0]
    (eval_root / "BEST_METHOD.txt").write_text(
        "\n".join(
            [
                f"config_id={config.config_id}",
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
    return summaries


def plot_global_summary(output_root: Path, rows: list[dict[str, Any]]) -> None:
    write_csv(output_root / "aggregate_summary.csv", rows)
    best = sorted(rows, key=lambda row: float(row["macro_Recall@10"]), reverse=True)[0]
    (output_root / "BEST_PROBE_ARCHITECTURE.txt").write_text(
        "\n".join(
            [
                f"config_id={best['config_id']}",
                f"arch={best['arch']}",
                f"loss={best['loss']}",
                f"method={best['method']}",
                f"macro_Recall@10={best['macro_Recall@10']}",
                f"micro_Recall@10={best['micro_Recall@10']}",
                f"macro_Precision@10={best['macro_Precision@10']}",
                f"probe_checkpoint={best['probe_checkpoint']}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    if plt is None:
        return
    top = sorted(rows, key=lambda row: float(row["macro_Recall@10"]), reverse=True)[:20]
    labels = [f"{row['config_id']}\\n{row['method']}" for row in top][::-1]
    values = [float(row["macro_Recall@10"]) for row in top][::-1]
    plt.figure(figsize=(12, max(6, 0.45 * len(top))))
    plt.barh(labels, values, color="#2f6f73")
    plt.xlabel("Macro Recall@10")
    plt.title("Probe architecture sweep: best JSON methods")
    plt.tight_layout()
    plt.savefig(output_root / "top_macro_recall10.png", dpi=160)
    plt.close()


def main() -> int:
    signal.signal(signal.SIGUSR1, request_stop)
    args = parse_args()
    device = choose_device(args.device)
    output_root = args.output_root or (
        ARTIFACTS_DIR / "results" / "probe_arch_sweep_v3" / f"probe_arch_sweep_v3_{timestamp()}_{args.profile}"
    )
    output_root.mkdir(parents=True, exist_ok=True)
    progress = output_root / "progress.txt"
    progress_line(progress, f"START probe_arch_sweep_v3 profile={args.profile} output={output_root}")
    progress_line(progress, f"args={json.dumps(vars(args), default=str, sort_keys=True)}")

    train_cache, attributes, train_labels = load_split_labels("train")
    valid_cache, valid_attributes, valid_labels = load_split_labels("valid")
    test_cache, test_attributes, _ = load_split_labels("test")
    if attributes != valid_attributes or attributes != test_attributes:
        raise RuntimeError("Attribute names differ across splits")
    adjacency = build_adjacency(train_labels)

    configs = default_configs(args.profile)
    if args.limit:
        configs = configs[: int(args.limit)]
    atomic_json_dump([asdict(config) for config in configs], output_root / "planned_configs.json")
    progress_line(progress, f"CONFIGS {len(configs)}")

    started = time.monotonic()
    all_eval_rows: list[dict[str, Any]] = []
    for config_index, config in enumerate(configs, start=1):
        elapsed = time.monotonic() - started
        remaining = float(args.time_budget_seconds) - elapsed if args.time_budget_seconds else float("inf")
        if remaining < float(args.min_seconds_for_new_run):
            progress_line(progress, f"STOP before config={config.config_id}; remaining={remaining:.1f}s")
            break
        progress_line(progress, f"START_CONFIG {config.config_id} {config_index}/{len(configs)}")
        try:
            probe_path, thresholds_by_objective, best_stats = train_one_probe(
                args,
                config,
                output_root,
                progress,
                device,
                train_cache,
                train_labels,
                valid_cache,
                valid_labels,
                attributes,
                adjacency,
            )
            probe, probe_attributes, _ = load_probe_checkpoint(probe_path, device, adjacency)
            if probe_attributes != attributes:
                raise RuntimeError("Loaded probe attributes mismatch")
            test_probs = predict_probs(probe, test_cache["embeddings"].float(), device, int(args.probe_batch_size))
            probe_dir = output_root / "probes" / config.config_id
            atomic_torch_save({"model_id": MODEL_ID, "attributes": attributes, "probs": test_probs}, probe_dir / "test_probe_probs.pt")
            eval_rows = evaluate_probe_on_json(
                args,
                config,
                output_root,
                progress,
                device,
                probe_path,
                thresholds_by_objective,
                test_cache,
                test_probs,
                adjacency,
            )
            for row in eval_rows:
                row.update({f"valid_{k}": v for k, v in best_stats.items() if isinstance(v, (int, float, str))})
            all_eval_rows.extend(eval_rows)
            plot_global_summary(output_root, all_eval_rows)
            progress_line(progress, f"END_CONFIG {config.config_id} status=complete")
        except SystemExit:
            raise
        except Exception as exc:  # Keep the sweep moving if one architecture fails.
            progress_line(progress, f"END_CONFIG {config.config_id} status=failed error={type(exc).__name__}: {exc}")
            append_csv(
                output_root / "failed_configs.csv",
                {
                    "config_id": config.config_id,
                    "arch": config.arch,
                    "loss": config.loss,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
        if STOP_REQUESTED:
            progress_line(progress, "STOP requested after config completion")
            break

    if all_eval_rows:
        plot_global_summary(output_root, all_eval_rows)
    progress_line(progress, f"COMPLETE probe_arch_sweep_v3 output={output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
