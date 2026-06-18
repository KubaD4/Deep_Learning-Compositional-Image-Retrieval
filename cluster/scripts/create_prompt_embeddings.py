#!/usr/bin/env python3
"""Create CLIP text embeddings for the manual prompt ensemble.

This is separate from the older attribute_text_embeddings.pt cache used by the
zero-shot baselines. The learned gate model uses this file because each signed
attribute is represented by the normalized mean of 2-3 natural prompts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor

from project_core import (
    EMBEDDING_DIR,
    MODEL_ID,
    ROOT,
    atomic_torch_save,
    choose_device,
    read_attribute_table,
    unwrap_features,
)


DEFAULT_PROMPTS = ROOT / "configs" / "attribute_prompts.json"
DEFAULT_OUTPUT = EMBEDDING_DIR / "signed_attribute_prompt_embeddings.pt"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-config", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def load_prompt_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    return config


def validate_prompt_config(config: dict, attributes: list[str]) -> None:
    missing = [attribute for attribute in attributes if attribute not in config]
    extra = [attribute for attribute in config if attribute not in set(attributes)]
    if missing:
        raise RuntimeError(f"Missing prompt entries: {missing}")
    if extra:
        raise RuntimeError(f"Unknown prompt entries: {extra}")
    for attribute in attributes:
        entry = config[attribute]
        for polarity in ("positive", "negative"):
            prompts = entry.get(polarity)
            if not isinstance(prompts, list) or not 2 <= len(prompts) <= 5:
                raise RuntimeError(
                    f"{attribute}.{polarity} must contain 2-5 prompt strings"
                )
            if not all(isinstance(prompt, str) and prompt.strip() for prompt in prompts):
                raise RuntimeError(f"{attribute}.{polarity} contains invalid prompts")


def embed_prompt_groups(model, processor, device, groups: list[list[str]]) -> torch.Tensor:
    group_embeddings = []
    for prompts in groups:
        inputs = processor(text=prompts, return_tensors="pt", padding=True).to(device)
        with torch.inference_mode():
            features = unwrap_features(model.get_text_features(**inputs)).float()
            features = F.normalize(features, dim=-1)
            group_embedding = F.normalize(features.mean(dim=0, keepdim=True), dim=-1)
        group_embeddings.append(group_embedding.cpu())
    return torch.cat(group_embeddings, dim=0)


def main() -> int:
    args = parse_args()
    attributes, _, _ = read_attribute_table()
    prompt_config = load_prompt_config(args.prompt_config)
    validate_prompt_config(prompt_config, attributes)

    if args.output.exists() and not args.force:
        cache = torch.load(args.output, map_location="cpu", weights_only=False)
        if (
            cache.get("model_id") == MODEL_ID
            and cache.get("attributes") == attributes
            and cache.get("prompt_config_path") == str(args.prompt_config)
        ):
            print(f"Complete prompt cache already exists: {args.output}")
            return 0

    device = choose_device(args.device)
    processor = CLIPProcessor.from_pretrained(MODEL_ID)
    model = CLIPModel.from_pretrained(MODEL_ID).to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    positive_groups = [prompt_config[attribute]["positive"] for attribute in attributes]
    negative_groups = [prompt_config[attribute]["negative"] for attribute in attributes]
    positive = embed_prompt_groups(model, processor, device, positive_groups)
    negative = embed_prompt_groups(model, processor, device, negative_groups)

    atomic_torch_save(
        {
            "model_id": MODEL_ID,
            "attributes": attributes,
            "prompt_config_path": str(args.prompt_config),
            "prompt_config": prompt_config,
            "positive": positive.half(),
            "negative": negative.half(),
            "directions": F.normalize(positive - negative, dim=-1).half(),
            "notes": (
                "positive/negative store normalized mean prompt embeddings. "
                "For a signed edit +A use positive[A]; for -A use negative[A]."
            ),
        },
        args.output,
    )
    print(f"Saved signed prompt cache: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
