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
        use_attr_probe: bool = False,
        attr_count: int = 40,
    ):
        super().__init__()
        self.clip_dim = clip_dim
        self.edit_scale = float(edit_scale)
        self.gate_max = float(gate_max)
        self.residual_scale = float(residual_scale)
        self.gate_uses_current = bool(gate_uses_current)
        self.use_attr_probe = bool(use_attr_probe)
        self.attr_count = int(attr_count)
        self.attr_probe = build_mlp(clip_dim, [512, 128], self.attr_count, dropout) if self.use_attr_probe else None
        probe_dim = self.attr_count if self.use_attr_probe else 0
        combined_dim = clip_dim * 4 + probe_dim
        self.gate = build_mlp(combined_dim, list(gate_hidden), 1, dropout)
        self.residual = build_mlp(combined_dim, list(residual_hidden), clip_dim, dropout)
        self.last_attr_logits: torch.Tensor | None = None

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
        attr_probs = None
        if self.attr_probe is not None:
            self.last_attr_logits = self.attr_probe(source)
            attr_probs = torch.sigmoid(self.last_attr_logits)
        else:
            self.last_attr_logits = None
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
            if attr_probs is not None:
                gate_input = torch.cat([gate_input, attr_probs], dim=-1)
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
        if attr_probs is not None:
            residual_input = torch.cat([residual_input, attr_probs], dim=-1)
        delta = self.residual(residual_input)
        query = F.normalize(query + self.residual_scale * delta, dim=-1)
        return query, alpha_out, delta


class GateSESequentialComposer(nn.Module):
    """SENet-inspired sequential gate over CLIP edit directions.

    SENet squeezes a feature map into a compact context vector and excites the
    useful channels.  Here the "channels" are CLIP dimensions: the model looks
    at the source and requested edit directions, predicts a per-dimension gate,
    and applies each edit sequentially after suppressing dimensions that look
    unhelpful for this source/query pair.
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
        se_reduction: int = 8,
    ):
        super().__init__()
        self.edit_scale = float(edit_scale)
        self.gate_max = float(gate_max)
        self.residual_scale = float(residual_scale)
        squeeze_dim = max(32, clip_dim // max(1, int(se_reduction)))
        self.channel_gate = nn.Sequential(
            nn.LayerNorm(clip_dim * 4),
            nn.Linear(clip_dim * 4, squeeze_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(squeeze_dim, clip_dim),
            nn.Sigmoid(),
        )
        self.gate = build_mlp(clip_dim * 4, list(gate_hidden), 1, dropout)
        self.residual = build_mlp(clip_dim * 4, list(residual_hidden), clip_dim, dropout)

    def forward(
        self,
        source: torch.Tensor,
        conditions: torch.Tensor,
        condition_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        source = F.normalize(source.float(), dim=-1)
        conditions = F.normalize(conditions.float(), dim=-1)
        mask = condition_mask.float()
        active_mask = condition_mask.bool()

        denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        coarse_edit = (conditions * mask.unsqueeze(-1)).sum(dim=1) / denom
        se_input = torch.cat([source, coarse_edit, source * coarse_edit, (source - coarse_edit).abs()], dim=-1)
        channel = self.channel_gate(se_input)
        modulated_conditions = F.normalize(conditions * channel.unsqueeze(1), dim=-1)

        query = source
        alpha_values = []
        weighted_steps = []
        for position in range(conditions.shape[1]):
            condition = modulated_conditions[:, position, :]
            active = active_mask[:, position]
            gate_input = torch.cat([query, condition, query * condition, (query - condition).abs()], dim=-1)
            alpha = torch.sigmoid(self.gate(gate_input).squeeze(-1)) * self.gate_max
            alpha = alpha * active.float()
            step = alpha.unsqueeze(-1) * condition
            stepped_query = F.normalize(query + self.edit_scale * step, dim=-1)
            query = torch.where(active.unsqueeze(-1), stepped_query, query)
            alpha_values.append(alpha)
            weighted_steps.append(step)

        aggregated = torch.stack(weighted_steps, dim=1).sum(dim=1) if weighted_steps else torch.zeros_like(source)
        alpha_out = torch.stack(alpha_values, dim=1) if alpha_values else torch.zeros_like(condition_mask.float())
        residual_input = torch.cat([source, aggregated, source * aggregated, (source - aggregated).abs()], dim=-1)
        delta = self.residual(residual_input)
        query = F.normalize(query + self.residual_scale * delta, dim=-1)
        return query, alpha_out, delta


class DenseMLP(nn.Module):
    """DenseNet-style MLP: every layer reuses all previous hidden states."""

    def __init__(self, input_dim: int, growth_dim: int, layers: int, output_dim: int, dropout: float):
        super().__init__()
        self.blocks = nn.ModuleList()
        current = input_dim
        for _ in range(int(layers)):
            self.blocks.append(
                nn.Sequential(
                    nn.LayerNorm(current),
                    nn.Linear(current, growth_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
            )
            current += growth_dim
        self.head = nn.Sequential(nn.LayerNorm(current), nn.Linear(current, output_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = x
        for block in self.blocks:
            new_features = block(features)
            features = torch.cat([features, new_features], dim=-1)
        return self.head(features)


class GateDenseSequentialComposer(nn.Module):
    """DenseNet-inspired gate with dense feature reuse for alpha and residual."""

    def __init__(
        self,
        clip_dim: int = 512,
        gate_hidden: tuple[int, ...] = (512, 128),
        residual_hidden: tuple[int, ...] = (1024, 512),
        dropout: float = 0.1,
        edit_scale: float = 1.0,
        gate_max: float = 1.5,
        residual_scale: float = 0.02,
        dense_growth: int = 256,
        dense_layers: int = 3,
    ):
        super().__init__()
        del gate_hidden, residual_hidden
        self.edit_scale = float(edit_scale)
        self.gate_max = float(gate_max)
        self.residual_scale = float(residual_scale)
        self.gate = DenseMLP(clip_dim * 4, int(dense_growth), int(dense_layers), 1, dropout)
        self.residual = DenseMLP(clip_dim * 4, int(dense_growth), int(dense_layers), clip_dim, dropout)

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
        alpha_values = []
        weighted_steps = []
        for position in range(conditions.shape[1]):
            condition = conditions[:, position, :]
            active = active_mask[:, position]
            gate_input = torch.cat([query, condition, query * condition, (query - condition).abs()], dim=-1)
            alpha = torch.sigmoid(self.gate(gate_input).squeeze(-1)) * self.gate_max
            alpha = alpha * active.float()
            step = alpha.unsqueeze(-1) * condition
            stepped_query = F.normalize(query + self.edit_scale * step, dim=-1)
            query = torch.where(active.unsqueeze(-1), stepped_query, query)
            alpha_values.append(alpha)
            weighted_steps.append(step)

        aggregated = torch.stack(weighted_steps, dim=1).sum(dim=1) if weighted_steps else torch.zeros_like(source)
        alpha_out = torch.stack(alpha_values, dim=1) if alpha_values else torch.zeros_like(condition_mask.float())
        residual_input = torch.cat([source, aggregated, source * aggregated, (source - aggregated).abs()], dim=-1)
        delta = self.residual(residual_input)
        query = F.normalize(query + self.residual_scale * delta, dim=-1)
        return query, alpha_out, delta


class GateRNNSequentialComposer(nn.Module):
    """GRU/LSTM-inspired composer treating the query edits as a sequence."""

    def __init__(
        self,
        clip_dim: int = 512,
        gate_hidden: tuple[int, ...] = (512, 128),
        residual_hidden: tuple[int, ...] = (1024, 512),
        dropout: float = 0.1,
        edit_scale: float = 1.0,
        gate_max: float = 1.5,
        residual_scale: float = 0.02,
        rnn_hidden: int = 512,
        rnn_type: str = "gru",
    ):
        super().__init__()
        del gate_hidden
        self.edit_scale = float(edit_scale)
        self.gate_max = float(gate_max)
        self.residual_scale = float(residual_scale)
        self.rnn_type = str(rnn_type).lower()
        self.source_init = nn.Sequential(nn.LayerNorm(clip_dim), nn.Linear(clip_dim, int(rnn_hidden)), nn.Tanh())
        if self.rnn_type == "lstm":
            self.rnn_cell = nn.LSTMCell(clip_dim, int(rnn_hidden))
            self.cell_init = nn.Sequential(nn.LayerNorm(clip_dim), nn.Linear(clip_dim, int(rnn_hidden)), nn.Tanh())
        else:
            self.rnn_cell = nn.GRUCell(clip_dim, int(rnn_hidden))
            self.cell_init = None
        self.alpha_head = nn.Sequential(
            nn.LayerNorm(int(rnn_hidden) + clip_dim * 2),
            nn.Linear(int(rnn_hidden) + clip_dim * 2, int(rnn_hidden)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(rnn_hidden), 1),
        )
        self.residual = build_mlp(clip_dim * 4, list(residual_hidden), clip_dim, dropout)

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
        hidden = self.source_init(source)
        cell = self.cell_init(source) if self.cell_init is not None else None
        alpha_values = []
        weighted_steps = []
        for position in range(conditions.shape[1]):
            condition = conditions[:, position, :]
            active = active_mask[:, position]
            if self.rnn_type == "lstm":
                hidden_next, cell_next = self.rnn_cell(condition, (hidden, cell))
                hidden = torch.where(active.unsqueeze(-1), hidden_next, hidden)
                cell = torch.where(active.unsqueeze(-1), cell_next, cell)
            else:
                hidden_next = self.rnn_cell(condition, hidden)
                hidden = torch.where(active.unsqueeze(-1), hidden_next, hidden)
            alpha_input = torch.cat([hidden, query, condition], dim=-1)
            alpha = torch.sigmoid(self.alpha_head(alpha_input).squeeze(-1)) * self.gate_max
            alpha = alpha * active.float()
            step = alpha.unsqueeze(-1) * condition
            stepped_query = F.normalize(query + self.edit_scale * step, dim=-1)
            query = torch.where(active.unsqueeze(-1), stepped_query, query)
            alpha_values.append(alpha)
            weighted_steps.append(step)

        aggregated = torch.stack(weighted_steps, dim=1).sum(dim=1) if weighted_steps else torch.zeros_like(source)
        alpha_out = torch.stack(alpha_values, dim=1) if alpha_values else torch.zeros_like(condition_mask.float())
        residual_input = torch.cat([source, aggregated, source * aggregated, (source - aggregated).abs()], dim=-1)
        delta = self.residual(residual_input)
        query = F.normalize(query + self.residual_scale * delta, dim=-1)
        return query, alpha_out, delta


class GateConvSequentialComposer(nn.Module):
    """CNN-inspired 1D convolution over the sequence of signed edit tokens."""

    def __init__(
        self,
        clip_dim: int = 512,
        gate_hidden: tuple[int, ...] = (512, 128),
        residual_hidden: tuple[int, ...] = (1024, 512),
        dropout: float = 0.1,
        edit_scale: float = 1.0,
        gate_max: float = 1.5,
        residual_scale: float = 0.02,
        conv_channels: int = 256,
        conv_layers: int = 2,
        conv_kernel: int = 3,
    ):
        super().__init__()
        del gate_hidden
        self.edit_scale = float(edit_scale)
        self.gate_max = float(gate_max)
        self.residual_scale = float(residual_scale)
        self.input_proj = nn.Sequential(nn.LayerNorm(clip_dim), nn.Linear(clip_dim, int(conv_channels)), nn.GELU())
        blocks = []
        for _ in range(int(conv_layers)):
            blocks.extend(
                [
                    nn.Conv1d(int(conv_channels), int(conv_channels), kernel_size=int(conv_kernel), padding=int(conv_kernel) // 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
        self.conv = nn.Sequential(*blocks)
        self.alpha_head = nn.Linear(int(conv_channels) + clip_dim * 2, 1)
        self.channel_head = nn.Sequential(nn.LayerNorm(int(conv_channels)), nn.Linear(int(conv_channels), clip_dim), nn.Sigmoid())
        self.residual = build_mlp(clip_dim * 4, list(residual_hidden), clip_dim, dropout)

    def forward(
        self,
        source: torch.Tensor,
        conditions: torch.Tensor,
        condition_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        source = F.normalize(source.float(), dim=-1)
        conditions = F.normalize(conditions.float(), dim=-1)
        active_mask = condition_mask.bool()
        hidden = self.input_proj(conditions)
        hidden = self.conv(hidden.transpose(1, 2)).transpose(1, 2)
        channel = self.channel_head(hidden)
        modulated_conditions = F.normalize(conditions * channel, dim=-1)

        query = source
        alpha_values = []
        weighted_steps = []
        for position in range(conditions.shape[1]):
            condition = modulated_conditions[:, position, :]
            active = active_mask[:, position]
            alpha_input = torch.cat([hidden[:, position, :], query, condition], dim=-1)
            alpha = torch.sigmoid(self.alpha_head(alpha_input).squeeze(-1)) * self.gate_max
            alpha = alpha * active.float()
            step = alpha.unsqueeze(-1) * condition
            stepped_query = F.normalize(query + self.edit_scale * step, dim=-1)
            query = torch.where(active.unsqueeze(-1), stepped_query, query)
            alpha_values.append(alpha)
            weighted_steps.append(step)

        aggregated = torch.stack(weighted_steps, dim=1).sum(dim=1) if weighted_steps else torch.zeros_like(source)
        alpha_out = torch.stack(alpha_values, dim=1) if alpha_values else torch.zeros_like(condition_mask.float())
        residual_input = torch.cat([source, aggregated, source * aggregated, (source - aggregated).abs()], dim=-1)
        delta = self.residual(residual_input)
        query = F.normalize(query + self.residual_scale * delta, dim=-1)
        return query, alpha_out, delta


class GateAttributeAttentionComposer(nn.Module):
    """Attribute-aware attention gate for source/query interpolation.

    The model first predicts a soft attribute-presence vector from the source
    embedding, then lets signed edit tokens attend to the source token and to
    each other.  The attention context controls both scalar edit weights and
    optional per-dimension edit masks.  This is the most direct test of the
    idea: "if the source already contains an attribute, or if two edits interact,
    learn how much of each CLIP direction should be injected."
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
        attention_dim: int = 256,
        attention_heads: int = 4,
        attention_layers: int = 2,
        attr_count: int = 40,
        use_channel_gate: bool = True,
        source_skip: float = 0.0,
    ):
        super().__init__()
        self.edit_scale = float(edit_scale)
        self.gate_max = float(gate_max)
        self.residual_scale = float(residual_scale)
        self.use_channel_gate = bool(use_channel_gate)
        self.source_skip = float(source_skip)
        self.attr_probe = build_mlp(clip_dim, [512, 128], int(attr_count), dropout)
        self.source_proj = nn.Sequential(nn.LayerNorm(clip_dim), nn.Linear(clip_dim, int(attention_dim)), nn.GELU())
        self.condition_proj = nn.Sequential(nn.LayerNorm(clip_dim), nn.Linear(clip_dim, int(attention_dim)), nn.GELU())
        layer = nn.TransformerEncoderLayer(
            d_model=int(attention_dim),
            nhead=int(attention_heads),
            dim_feedforward=int(attention_dim) * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=int(attention_layers))
        gate_input_dim = clip_dim * 4 + int(attention_dim) + int(attr_count)
        self.gate = build_mlp(gate_input_dim, list(gate_hidden), 1, dropout)
        self.channel_head = (
            nn.Sequential(nn.LayerNorm(int(attention_dim)), nn.Linear(int(attention_dim), clip_dim), nn.Sigmoid())
            if self.use_channel_gate
            else None
        )
        self.residual = build_mlp(clip_dim * 4 + int(attr_count), list(residual_hidden), clip_dim, dropout)
        self.last_attr_logits: torch.Tensor | None = None

    def forward(
        self,
        source: torch.Tensor,
        conditions: torch.Tensor,
        condition_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        source = F.normalize(source.float(), dim=-1)
        conditions = F.normalize(conditions.float(), dim=-1)
        active_mask = condition_mask.bool()

        self.last_attr_logits = self.attr_probe(source)
        attr_probs = torch.sigmoid(self.last_attr_logits)
        source_token = self.source_proj(source).unsqueeze(1)
        condition_tokens = self.condition_proj(conditions)
        tokens = torch.cat([source_token, condition_tokens], dim=1)
        padding = torch.cat(
            [
                torch.zeros((condition_mask.shape[0], 1), dtype=torch.bool, device=condition_mask.device),
                ~active_mask,
            ],
            dim=1,
        )
        encoded = self.encoder(tokens, src_key_padding_mask=padding)
        condition_context = encoded[:, 1:, :]
        if self.channel_head is not None:
            channel = self.channel_head(condition_context)
            edit_conditions = F.normalize(conditions * channel, dim=-1)
        else:
            edit_conditions = conditions

        query = source
        alpha_values = []
        weighted_steps = []
        expanded_attr = attr_probs.unsqueeze(1).expand(-1, conditions.shape[1], -1)
        for position in range(conditions.shape[1]):
            condition = edit_conditions[:, position, :]
            active = active_mask[:, position]
            gate_input = torch.cat(
                [
                    query,
                    condition,
                    query * condition,
                    (query - condition).abs(),
                    condition_context[:, position, :],
                    expanded_attr[:, position, :],
                ],
                dim=-1,
            )
            alpha = torch.sigmoid(self.gate(gate_input).squeeze(-1)) * self.gate_max
            alpha = alpha * active.float()
            step = alpha.unsqueeze(-1) * condition
            stepped_query = F.normalize(query + self.edit_scale * step, dim=-1)
            query = torch.where(active.unsqueeze(-1), stepped_query, query)
            alpha_values.append(alpha)
            weighted_steps.append(step)

        aggregated = torch.stack(weighted_steps, dim=1).sum(dim=1) if weighted_steps else torch.zeros_like(source)
        alpha_out = torch.stack(alpha_values, dim=1) if alpha_values else torch.zeros_like(condition_mask.float())
        residual_input = torch.cat([source, aggregated, source * aggregated, (source - aggregated).abs(), attr_probs], dim=-1)
        delta = self.residual(residual_input)
        if self.source_skip > 0:
            query = F.normalize((1.0 - self.source_skip) * query + self.source_skip * source + self.residual_scale * delta, dim=-1)
        else:
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
            use_attr_probe=bool(config.get("use_attr_probe", False)),
            attr_count=int(config.get("attr_count", 40)),
        )
    if composer_type == "se_sequential_gate":
        return GateSESequentialComposer(
            **common,
            edit_scale=float(config.get("edit_scale", 1.0)),
            gate_max=float(config.get("gate_max", 1.5)),
            se_reduction=int(config.get("se_reduction", 8)),
        )
    if composer_type == "dense_sequential_gate":
        return GateDenseSequentialComposer(
            **common,
            edit_scale=float(config.get("edit_scale", 1.0)),
            gate_max=float(config.get("gate_max", 1.5)),
            dense_growth=int(config.get("dense_growth", 256)),
            dense_layers=int(config.get("dense_layers", 3)),
        )
    if composer_type == "rnn_sequential_gate":
        return GateRNNSequentialComposer(
            **common,
            edit_scale=float(config.get("edit_scale", 1.0)),
            gate_max=float(config.get("gate_max", 1.5)),
            rnn_hidden=int(config.get("rnn_hidden", 512)),
            rnn_type=str(config.get("rnn_type", "gru")),
        )
    if composer_type == "conv_sequential_gate":
        return GateConvSequentialComposer(
            **common,
            edit_scale=float(config.get("edit_scale", 1.0)),
            gate_max=float(config.get("gate_max", 1.5)),
            conv_channels=int(config.get("conv_channels", 256)),
            conv_layers=int(config.get("conv_layers", 2)),
            conv_kernel=int(config.get("conv_kernel", 3)),
        )
    if composer_type == "attribute_attention_gate":
        return GateAttributeAttentionComposer(
            **common,
            edit_scale=float(config.get("edit_scale", 1.0)),
            gate_max=float(config.get("gate_max", 1.5)),
            attention_dim=int(config.get("attention_dim", 256)),
            attention_heads=int(config.get("attention_heads", 4)),
            attention_layers=int(config.get("attention_layers", 2)),
            attr_count=int(config.get("attr_count", 40)),
            use_channel_gate=bool(config.get("use_channel_gate", True)),
            source_skip=float(config.get("source_skip", 0.0)),
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
