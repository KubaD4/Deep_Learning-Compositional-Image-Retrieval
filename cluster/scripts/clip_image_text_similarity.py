#!/usr/bin/env python3
"""Compute CLIP cosine similarity between one image and one text prompt."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from project_core import CELEBA_DIR, DATA_ROOT, MODEL_ID, ROOT, choose_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute CLIP cosine similarity between an image and a phrase."
    )
    parser.add_argument("image", help="Image path or CelebA filename, e.g. 000001.jpg")
    parser.add_argument("text", help='Text prompt, e.g. "a person with orange hair"')
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Inference device. Default: auto.",
    )
    return parser.parse_args()


def resolve_image(value: str) -> Path:
    requested = Path(value).expanduser()
    candidates = [
        requested,
        Path.cwd() / requested,
        CELEBA_DIR / "img_align_celeba" / requested.name,
        DATA_ROOT / "img_align_celeba" / requested.name,
        ROOT / "celeba" / "img_align_celeba" / requested.name,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    searched = "\n".join(f"  - {candidate}" for candidate in candidates)
    raise FileNotFoundError(f"Image {value!r} not found. Searched:\n{searched}")


def main() -> None:
    args = parse_args()
    image_path = resolve_image(args.image)
    device = choose_device(args.device)

    processor = CLIPProcessor.from_pretrained(MODEL_ID)
    model = CLIPModel.from_pretrained(MODEL_ID).to(device).eval()

    with Image.open(image_path) as image:
        inputs = processor(
            text=[args.text],
            images=[image.convert("RGB")],
            return_tensors="pt",
            padding=True,
        ).to(device)

    with torch.inference_mode():
        output = model(**inputs)
        image_embedding = F.normalize(output.image_embeds.float(), dim=-1)
        text_embedding = F.normalize(output.text_embeds.float(), dim=-1)
        similarity = float((image_embedding * text_embedding).sum())

    print(f"Image:      {image_path}")
    print(f"Text:       {args.text}")
    print(f"Model:      {MODEL_ID}")
    print(f"Device:     {device}")
    print(f"Similarity: {similarity:.6f}")


if __name__ == "__main__":
    main()
