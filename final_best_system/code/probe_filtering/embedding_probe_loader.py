#!/usr/bin/env python3
"""Universal embedding-probe loader for the packaged final system.

The first packaged probe (`probe_reranker_v2_calibrated`) used a small
AttributeProbe class. Later sweeps saved richer probe checkpoints with an
architecture config under `checkpoint["config"]`.  This module loads both
formats so the final system can be upgraded without breaking older results.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from project_core import load_torch
except Exception:  # pragma: no cover - fallback for standalone package use.

    def load_torch(path: Path | str) -> Any:
        return torch.load(path, map_location="cpu")


@dataclass(frozen=True)
class ProbeArchConfig:
    config_id: str
    arch: str
    loss: str = "asl"
    hidden_dims: tuple[int, ...] = (1024, 512)
    dropout: float = 0.1
    label_dim: int = 128
    hidden_dim: int = 1024
    transformer_layers: int = 1
    transformer_heads: int = 4
    residual_blocks: int = 2
    layers: int = 2
    heads: int = 4
    blocks: int = 2
    lr: float = 1e-4
    weight_decay: float = 1e-4
    batch_size: int = 512
    epochs: int = 10
    pos_weight_clip: float = 8.0
    gamma_pos: float = 0.0
    gamma_neg: float = 4.0
    noise_std: float = 0.0
    notes: str = ""


class MLPProbe(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: tuple[int, ...], dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.LayerNorm(input_dim)]
        prev = input_dim
        for hidden in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev, hidden),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            prev = hidden
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.net(F.normalize(embeddings.float(), dim=-1))


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float) -> None:
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
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, blocks: int, dropout: float) -> None:
        super().__init__()
        self.input = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden_dim), nn.GELU())
        self.blocks = nn.Sequential(*[ResidualBlock(hidden_dim, dropout) for _ in range(blocks)])
        self.head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, output_dim))

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        hidden = self.input(F.normalize(embeddings.float(), dim=-1))
        return self.head(self.blocks(hidden))


class LabelWiseProbe(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, label_dim: int, dropout: float) -> None:
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

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        image = self.image(F.normalize(embeddings.float(), dim=-1))
        labels = self.label_tokens.unsqueeze(0).expand(embeddings.shape[0], -1, -1)
        image_tokens = image.unsqueeze(1).expand_as(labels)
        pair = torch.cat([image_tokens, labels, image_tokens * labels], dim=-1)
        return self.head(pair).squeeze(-1)


class LabelTransformerProbe(nn.Module):
    """Small scratch-built label-attention probe over CelebA attributes.

    The input CLIP embedding is projected into a token and each attribute owns a
    learnable query token. Self-attention lets attribute decisions share context
    such as Male/No_Beard or Lipstick/Heavy_Makeup correlations.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        label_dim: int,
        layers: int,
        heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.image = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, label_dim), nn.GELU())
        self.label_tokens = nn.Parameter(torch.randn(output_dim, label_dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=label_dim,
            nhead=heads,
            dim_feedforward=label_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.head = nn.Sequential(nn.LayerNorm(label_dim), nn.Linear(label_dim, 1))

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        image = self.image(F.normalize(embeddings.float(), dim=-1)).unsqueeze(1)
        labels = self.label_tokens.unsqueeze(0).expand(embeddings.shape[0], -1, -1)
        tokens = torch.cat([image, labels], dim=1)
        encoded = self.encoder(tokens)
        return self.head(encoded[:, 1:, :]).squeeze(-1)


def _config_from_checkpoint(raw_config: dict[str, Any]) -> ProbeArchConfig:
    valid = {field.name for field in fields(ProbeArchConfig)}
    filtered = {key: value for key, value in raw_config.items() if key in valid}
    if "hidden_dims" in filtered:
        filtered["hidden_dims"] = tuple(int(value) for value in filtered["hidden_dims"])
    return ProbeArchConfig(**filtered)


def build_probe(config: ProbeArchConfig, output_dim: int, input_dim: int = 512) -> nn.Module:
    if config.arch == "mlp":
        return MLPProbe(input_dim, output_dim, config.hidden_dims, config.dropout)
    if config.arch == "residual_mlp":
        return ResidualMLPProbe(input_dim, output_dim, config.hidden_dim, config.blocks, config.dropout)
    if config.arch == "labelwise":
        return LabelWiseProbe(input_dim, output_dim, config.label_dim, config.dropout)
    if config.arch == "label_transformer":
        return LabelTransformerProbe(
            input_dim,
            output_dim,
            config.label_dim,
            config.layers,
            config.heads,
            config.dropout,
        )
    raise ValueError(f"Unsupported packaged probe architecture: {config.arch}")


def load_embedding_probe(path: Path, device: torch.device) -> tuple[nn.Module, list[str], dict[str, Any]]:
    checkpoint = load_torch(path)
    raw_config = checkpoint.get("config") if isinstance(checkpoint, dict) else None

    if isinstance(raw_config, dict) and raw_config.get("arch"):
        attributes = list(checkpoint["attributes"])
        config = _config_from_checkpoint(raw_config)
        model = build_probe(config, output_dim=len(attributes))
        model.load_state_dict(checkpoint["model_state"])
        model.to(device)
        model.eval()
        return model, attributes, checkpoint

    import probe_reranker_v1 as old_probe  # noqa: WPS433

    return old_probe.load_probe(path, device)


def predict_embedding_probe_probs(
    probe: nn.Module,
    embeddings: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    outputs: list[torch.Tensor] = []
    probe.eval()
    with torch.inference_mode():
        for start in range(0, len(embeddings), batch_size):
            batch = embeddings[start : start + batch_size].float().to(device)
            outputs.append(torch.sigmoid(probe(batch)).cpu())
    return torch.cat(outputs, dim=0)
