#!/usr/bin/env python3
"""Compare a local image with one prompt or a positive/negative prompt pair."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


MODEL_ID = "openai/clip-vit-base-patch32"
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute CLIP cosine similarity between an image and a phrase."
    )
    parser.add_argument("image", help="Image path or CelebA filename")
    parser.add_argument(
        "text", nargs="?", help="Positive text prompt enclosed in quotes"
    )
    parser.add_argument(
        "opposite_text",
        nargs="?",
        help="Optional opposite prompt used for contrastive attribute scoring",
    )
    parser.add_argument(
        "--pair",
        nargs=2,
        action="append",
        metavar=("POSITIVE", "NEGATIVE"),
        help="Positive/negative prompt pair. Repeat --pair to build an ensemble.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "mps"),
        default="auto",
        help="Inference device. Default: auto.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Use only model files already present in the Hugging Face cache.",
    )
    return parser.parse_args()


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_image(value: str) -> Path:
    requested = Path(value).expanduser()
    candidates = [
        requested,
        Path.cwd() / requested,
        PROJECT_ROOT / "celeba" / "img_align_celeba" / requested.name,
        PROJECT_ROOT / "data" / "celeba" / "img_align_celeba" / requested.name,
        PROJECT_ROOT / "cluster" / "data" / "celeba" / "img_align_celeba" / requested.name,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    searched = "\n".join(f"  - {candidate}" for candidate in candidates)
    raise FileNotFoundError(f"Image {value!r} not found. Searched:\n{searched}")


def main() -> None:
    args = parse_args()
    pairs = list(args.pair or [])
    if args.text and args.opposite_text:
        pairs.insert(0, [args.text, args.opposite_text])
    elif args.text:
        single_prompt = args.text
    elif not pairs:
        raise SystemExit("Provide one text prompt or at least one --pair POSITIVE NEGATIVE")

    image_path = resolve_image(args.image)
    device = choose_device(args.device)

    processor = CLIPProcessor.from_pretrained(
        MODEL_ID, local_files_only=args.offline
    )
    model = CLIPModel.from_pretrained(
        MODEL_ID, local_files_only=args.offline
    ).to(device).eval()

    if pairs:
        prompts = [prompt for pair in pairs for prompt in pair]
    else:
        prompts = [single_prompt]

    with Image.open(image_path) as image:
        inputs = processor(
            text=prompts,
            images=[image.convert("RGB")],
            return_tensors="pt",
            padding=True,
        ).to(device)

    with torch.inference_mode():
        output = model(**inputs)
        image_embedding = F.normalize(output.image_embeds.float(), dim=-1)
        text_embeddings = F.normalize(output.text_embeds.float(), dim=-1)
        similarities = (image_embedding @ text_embeddings.T).squeeze(0)

    print(f"Image:      {image_path}")
    print(f"Model:      {MODEL_ID}")
    print(f"Device:     {device}")
    print(f"Execution:  local inference ({'offline cache' if args.offline else 'HF cache'})")

    if not pairs:
        print(f"Text:       {single_prompt}")
        print(f"Similarity: {float(similarities[0]):.6f}")
        return

    logit_scale = float(model.logit_scale.detach().exp().clamp(max=100))
    pair_margins = []
    print()
    print("Per-pair contrastive estimates")
    for index, (positive, negative) in enumerate(pairs):
        positive_embedding = text_embeddings[2 * index]
        negative_embedding = text_embeddings[2 * index + 1]
        positive_similarity = float(similarities[2 * index])
        negative_similarity = float(similarities[2 * index + 1])
        margin = positive_similarity - negative_similarity
        pair_margins.append(margin)

        raw_direction = positive_embedding - negative_embedding
        direction = F.normalize(raw_direction, dim=0)
        direction_similarity = float(image_embedding.squeeze(0) @ direction)
        probabilities = torch.softmax(
            similarities[2 * index : 2 * index + 2] * logit_scale, dim=0
        )

        print(f"\nPair {index + 1}")
        print(f"  Positive:   {positive}")
        print(f"  Negative:   {negative}")
        print(f"  Cos(+):     {positive_similarity:.6f}")
        print(f"  Cos(-):     {negative_similarity:.6f}")
        print(f"  Margin:     {margin:+.6f}")
        print(f"  Dir score:  {direction_similarity:+.6f}")
        print(f"  Pair pref+: {float(probabilities[0]) * 100:.2f}%")

    positive_embeddings = text_embeddings[0::2]
    negative_embeddings = text_embeddings[1::2]
    positive_prototype = F.normalize(positive_embeddings.mean(dim=0), dim=0)
    negative_prototype = F.normalize(negative_embeddings.mean(dim=0), dim=0)
    prototype_similarities = torch.stack(
        [
            image_embedding.squeeze(0) @ positive_prototype,
            image_embedding.squeeze(0) @ negative_prototype,
        ]
    )
    ensemble_margin = float(prototype_similarities[0] - prototype_similarities[1])
    ensemble_direction = F.normalize(positive_prototype - negative_prototype, dim=0)
    ensemble_direction_score = float(image_embedding.squeeze(0) @ ensemble_direction)
    ensemble_probabilities = torch.softmax(prototype_similarities * logit_scale, dim=0)
    margins_tensor = torch.tensor(pair_margins)

    print()
    print(f"Ensemble estimate ({len(pairs)} pairs)")
    print(f"Cos(+ prototype): {float(prototype_similarities[0]):.6f}")
    print(f"Cos(- prototype): {float(prototype_similarities[1]):.6f}")
    print(f"Margin:           {ensemble_margin:+.6f}")
    print(f"Dir score:        {ensemble_direction_score:+.6f}")
    print(f"Mean pair margin: {float(margins_tensor.mean()):+.6f}")
    print(f"Margin std:       {float(margins_tensor.std(unbiased=False)):.6f}")
    print(f"Positive pairs:   {sum(margin > 0 for margin in pair_margins)}/{len(pair_margins)}")
    print(f"Ensemble pref+:   {float(ensemble_probabilities[0]) * 100:.2f}%")
    print("Note: preferences compare only the supplied prompts; they are not calibrated probabilities.")


if __name__ == "__main__":
    main()
