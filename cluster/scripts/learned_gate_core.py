"""Shared utilities for the learned gated residual retrieval model."""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from project_core import EMBEDDING_DIR, MODEL_ID, atomic_json_dump, load_torch


TRAINING_PAIR_DIRNAME = "training_pairs"
TRAINING_RUN_DIRNAME = "training_runs"


def timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def progress_line(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"[{timestamp()}] {message}\n")
        handle.flush()


def write_csv_rows(path: Path, rows: list[dict], append: bool = False) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    file_exists = path.exists()
    mode = "a" if append else "w"
    with path.open(mode, newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not append or not file_exists:
            writer.writeheader()
        writer.writerows(rows)


def signed_condition_text(
    attr_indices: torch.Tensor,
    signs: torch.Tensor,
    length: int,
    attributes: list[str],
) -> str:
    parts = []
    for position in range(length):
        sign = "+" if int(signs[position]) > 0 else "-"
        parts.append(f"{sign}{attributes[int(attr_indices[position])]}")
    return ", ".join(parts)


def load_image_embedding_cache(split: str) -> dict:
    path = EMBEDDING_DIR / f"{split}_image_embeddings.pt"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {split} embeddings at {path}. "
            f"Submit the embedding job for split={split} first."
        )
    cache = load_torch(path)
    if cache.get("model_id") != MODEL_ID:
        raise RuntimeError(f"Unexpected model_id in {path}: {cache.get('model_id')}")
    return cache


def load_prompt_embedding_cache(path: Path | None = None) -> dict:
    prompt_path = path or (EMBEDDING_DIR / "signed_attribute_prompt_embeddings.pt")
    if not prompt_path.exists():
        raise FileNotFoundError(
            f"Missing signed prompt embeddings at {prompt_path}. "
            "Run scripts/create_prompt_embeddings.py first."
        )
    cache = load_torch(prompt_path)
    if cache.get("model_id") != MODEL_ID:
        raise RuntimeError(
            f"Unexpected model_id in {prompt_path}: {cache.get('model_id')}"
        )
    return cache


def condition_embeddings(prompt_cache, attr_indices, signs, device, mode: str = "signed_prompt"):
    """Map padded signed attribute indices to prompt or direction embeddings."""
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


def build_mlp(input_dim: int, hidden_dims: list[int], output_dim: int, dropout: float):
    layers: list[nn.Module] = [nn.LayerNorm(input_dim)]
    current = input_dim
    for hidden in hidden_dims:
        layers.extend([nn.Linear(current, hidden), nn.GELU()])
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        current = hidden
    layers.append(nn.Linear(current, output_dim))
    return nn.Sequential(*layers)


class GateResidualComposer(nn.Module):
    """Source-conditioned edit weighting plus residual CLIP-space movement."""

    def __init__(
        self,
        clip_dim: int = 512,
        gate_hidden: tuple[int, int] = (512, 128),
        residual_hidden: tuple[int, int] = (1024, 512),
        dropout: float = 0.1,
        residual_scale: float = 0.1,
    ):
        super().__init__()
        self.clip_dim = clip_dim
        self.residual_scale = float(residual_scale)
        combined_dim = clip_dim * 4
        self.gate = build_mlp(combined_dim, list(gate_hidden), 1, dropout)
        self.residual = build_mlp(combined_dim, list(residual_hidden), clip_dim, dropout)

    def forward(
        self,
        source: torch.Tensor,
        conditions: torch.Tensor,
        condition_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        source = F.normalize(source.float(), dim=-1)
        conditions = F.normalize(conditions.float(), dim=-1)
        mask = condition_mask.float()

        expanded_source = source.unsqueeze(1).expand_as(conditions)
        gate_input = torch.cat(
            [
                expanded_source,
                conditions,
                expanded_source * conditions,
                (expanded_source - conditions).abs(),
            ],
            dim=-1,
        )
        alpha = torch.sigmoid(self.gate(gate_input).squeeze(-1)) * mask
        aggregated = (alpha.unsqueeze(-1) * conditions).sum(dim=1)

        residual_input = torch.cat(
            [
                source,
                aggregated,
                source * aggregated,
                (source - aggregated).abs(),
            ],
            dim=-1,
        )
        delta = self.residual(residual_input)
        query = F.normalize(source + self.residual_scale * delta, dim=-1)
        return query, alpha, delta


class GateAdditiveComposer(nn.Module):
    """Learn weights over CLIP edit directions, then optionally add a residual."""

    def __init__(
        self,
        clip_dim: int = 512,
        gate_hidden: tuple[int, ...] = (512, 128),
        residual_hidden: tuple[int, ...] = (1024, 512),
        dropout: float = 0.1,
        edit_scale: float = 1.0,
        gate_max: float = 1.5,
        residual_scale: float = 0.02,
    ):
        super().__init__()
        self.clip_dim = clip_dim
        self.edit_scale = float(edit_scale)
        self.gate_max = float(gate_max)
        self.residual_scale = float(residual_scale)
        combined_dim = clip_dim * 4
        self.gate = build_mlp(combined_dim, list(gate_hidden), 1, dropout)
        self.residual = build_mlp(combined_dim, list(residual_hidden), clip_dim, dropout)

    def forward(
        self,
        source: torch.Tensor,
        conditions: torch.Tensor,
        condition_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        source = F.normalize(source.float(), dim=-1)
        conditions = F.normalize(conditions.float(), dim=-1)
        mask = condition_mask.float()

        expanded_source = source.unsqueeze(1).expand_as(conditions)
        gate_input = torch.cat(
            [
                expanded_source,
                conditions,
                expanded_source * conditions,
                (expanded_source - conditions).abs(),
            ],
            dim=-1,
        )
        alpha = torch.sigmoid(self.gate(gate_input).squeeze(-1)) * self.gate_max * mask
        aggregated = (alpha.unsqueeze(-1) * conditions).sum(dim=1)

        residual_input = torch.cat(
            [
                source,
                aggregated,
                source * aggregated,
                (source - aggregated).abs(),
            ],
            dim=-1,
        )
        delta = self.residual(residual_input)
        query = F.normalize(
            source + self.edit_scale * aggregated + self.residual_scale * delta,
            dim=-1,
        )
        return query, alpha, delta


class GateSequentialComposer(nn.Module):
    """Learned version of the contrastive sequential CLIP arithmetic baseline.

    Instead of summing all edited directions and normalizing once at the end,
    this composer applies one signed CLIP direction at a time:

        q_j = normalize(q_{j-1} + alpha_j * d_j)

    The gate can look either at the current query state or at the original
    source embedding. The default uses the current state because that is the
    closest learned analogue of the sequential baseline.
    """

    def __init__(
        self,
        clip_dim: int = 512,
        gate_hidden: tuple[int, ...] = (512, 128),
        residual_hidden: tuple[int, ...] = (1024, 512),
        dropout: float = 0.1,
        edit_scale: float = 1.0,
        gate_max: float = 1.5,
        residual_scale: float = 0.02,
        gate_uses_current: bool = True,
    ):
        super().__init__()
        self.clip_dim = clip_dim
        self.edit_scale = float(edit_scale)
        self.gate_max = float(gate_max)
        self.residual_scale = float(residual_scale)
        self.gate_uses_current = bool(gate_uses_current)
        combined_dim = clip_dim * 4
        self.gate = build_mlp(combined_dim, list(gate_hidden), 1, dropout)
        self.residual = build_mlp(combined_dim, list(residual_hidden), clip_dim, dropout)

    def forward(
        self,
        source: torch.Tensor,
        conditions: torch.Tensor,
        condition_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        source = F.normalize(source.float(), dim=-1)
        conditions = F.normalize(conditions.float(), dim=-1)
        active_mask = condition_mask.bool()

        query = source
        weighted_steps = []
        alpha_values = []

        for position in range(conditions.shape[1]):
            condition = conditions[:, position, :]
            active = active_mask[:, position]
            gate_source = query if self.gate_uses_current else source
            gate_input = torch.cat(
                [
                    gate_source,
                    condition,
                    gate_source * condition,
                    (gate_source - condition).abs(),
                ],
                dim=-1,
            )
            alpha = torch.sigmoid(self.gate(gate_input).squeeze(-1)) * self.gate_max
            alpha = alpha * active.float()
            step = alpha.unsqueeze(-1) * condition
            stepped_query = F.normalize(query + self.edit_scale * step, dim=-1)
            query = torch.where(active.unsqueeze(-1), stepped_query, query)
            weighted_steps.append(step)
            alpha_values.append(alpha)

        if weighted_steps:
            aggregated = torch.stack(weighted_steps, dim=1).sum(dim=1)
            alpha_out = torch.stack(alpha_values, dim=1)
        else:
            aggregated = torch.zeros_like(source)
            alpha_out = torch.zeros_like(condition_mask.float())

        residual_input = torch.cat(
            [
                source,
                aggregated,
                source * aggregated,
                (source - aggregated).abs(),
            ],
            dim=-1,
        )
        delta = self.residual(residual_input)
        query = F.normalize(query + self.residual_scale * delta, dim=-1)
        return query, alpha_out, delta


def create_model_from_config(config: dict, clip_dim: int = 512) -> nn.Module:
    composer_type = config.get("composer_type", "residual_only")
    common = {
        "clip_dim": clip_dim,
        "gate_hidden": tuple(config.get("gate_hidden", [512, 128])),
        "residual_hidden": tuple(config.get("residual_hidden", [1024, 512])),
        "dropout": float(config.get("dropout", 0.1)),
        "residual_scale": float(config.get("residual_scale", 0.1)),
    }
    if composer_type == "residual_only":
        return GateResidualComposer(**common)
    if composer_type == "additive_gate":
        return GateAdditiveComposer(
            **common,
            edit_scale=float(config.get("edit_scale", 1.0)),
            gate_max=float(config.get("gate_max", 1.5)),
        )
    if composer_type == "sequential_gate":
        return GateSequentialComposer(
            **common,
            edit_scale=float(config.get("edit_scale", 1.0)),
            gate_max=float(config.get("gate_max", 1.5)),
            gate_uses_current=str(config.get("gate_state", "current")) == "current",
        )
    raise ValueError(f"Unknown composer_type: {composer_type}")


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    config: dict,
    epoch: int,
    step: int,
    best: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_id": MODEL_ID,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": config,
            "epoch": epoch,
            "step": step,
            "best": best,
        },
        path,
    )


def load_model_checkpoint(path: Path, device: torch.device):
    checkpoint = load_torch(path)
    config = checkpoint["config"]
    model = create_model_from_config(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    return model, checkpoint


def dump_config(config: dict, path: Path) -> None:
    serializable = json.loads(json.dumps(config))
    atomic_json_dump(serializable, path)
