#!/usr/bin/env python3
"""Cache frozen Hugging Face CLIP embeddings used by the project notebook."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CelebA
from tqdm.auto import tqdm
from transformers import CLIPModel, CLIPProcessor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "openai/clip-vit-base-patch32"
MODEL_SLUG = "openai_clip_vit_b32"
OUTPUT_DIR = PROJECT_ROOT / "data" / "celeba" / "embeddings" / MODEL_SLUG


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cache normalized CLIP ViT-B/32 embeddings for CelebA."
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("train", "valid", "test"),
        default=("train", "valid", "test"),
    )
    parser.add_argument(
        "--device", choices=("auto", "cpu", "mps", "cuda"), default="auto"
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def unwrap_features(output: torch.Tensor) -> torch.Tensor:
    """Support both Transformers 4.x tensor and 5.x model-output APIs."""
    return output.pooler_output if hasattr(output, "pooler_output") else output


class CelebAImages(Dataset):
    def __init__(self, dataset: CelebA):
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        image, _ = self.dataset[index]
        return image, index


class ImageCollator:
    def __init__(self, processor: CLIPProcessor):
        self.processor = processor

    def __call__(self, batch):
        images, indices = zip(*batch)
        pixels = self.processor(images=list(images), return_tensors="pt")[
            "pixel_values"
        ]
        return pixels, torch.tensor(indices)


def prompt_pair(attribute: str) -> tuple[str, str]:
    special = {
        "Attractive": ("an attractive face", "an unattractive face"),
        "Bald": ("a bald person", "a person with hair"),
        "Blurry": ("a blurry face photo", "a sharp face photo"),
        "Chubby": ("a chubby face", "a slim face"),
        "Male": ("a male face", "a female face"),
        "Mouth_Slightly_Open": (
            "a face with an open mouth",
            "a face with a closed mouth",
        ),
        "No_Beard": (
            "a clean-shaven face without a beard",
            "a face with a beard",
        ),
        "Smiling": ("a smiling face", "a face that is not smiling"),
        "Young": ("a young face", "an older face"),
    }
    if attribute in special:
        return special[attribute]
    readable = attribute.replace("_", " ").lower()
    return f"a face with {readable}", f"a face without {readable}"


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    text_path = OUTPUT_DIR / "attribute_text_embeddings.pt"

    reference_dataset = CelebA(
        root=PROJECT_ROOT, split="test", target_type="attr", download=False
    )
    processor = CLIPProcessor.from_pretrained(MODEL_ID)
    model = CLIPModel.from_pretrained(MODEL_ID).to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    print(f"Device: {device}")
    print(f"Splits: {', '.join(args.splits)}")
    print(f"Output: {OUTPUT_DIR}")

    attributes = [name for name in reference_dataset.attr_names if name]
    assert len(attributes) == 40
    text_cache_is_valid = False
    if text_path.exists() and not args.force:
        existing_text_cache = torch.load(
            text_path, map_location="cpu", weights_only=False
        )
        text_cache_is_valid = (
            existing_text_cache.get("model_id") == MODEL_ID
            and existing_text_cache.get("attributes") == attributes
        )

    if not text_cache_is_valid:
        prompt_pairs = [prompt_pair(attribute) for attribute in attributes]
        prompts = [prompt for pair in prompt_pairs for prompt in pair]
        inputs = processor(text=prompts, return_tensors="pt", padding=True).to(device)
        with torch.inference_mode():
            features = unwrap_features(model.get_text_features(**inputs)).float()
            features = F.normalize(features, dim=-1).cpu()
        positive = features[0::2]
        negative = features[1::2]
        directions = F.normalize(positive - negative, dim=-1)
        torch.save(
            {
                "model_id": MODEL_ID,
                "attributes": attributes,
                "prompts": prompt_pairs,
                "positive": positive.half(),
                "negative": negative.half(),
                "directions": directions.half(),
            },
            text_path,
        )
        print(f"Saved text embeddings: {text_path}")
    else:
        print(f"Reusing text embeddings: {text_path}")

    for split in args.splits:
        image_path = OUTPUT_DIR / f"{split}_image_embeddings.pt"
        if image_path.exists() and not args.force:
            print(f"Reusing {split} image embeddings: {image_path}")
            continue

        dataset = CelebA(
            root=PROJECT_ROOT, split=split, target_type="attr", download=False
        )
        loader = DataLoader(
            CelebAImages(dataset),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=device.type == "cuda",
            collate_fn=ImageCollator(processor),
        )
        embedding_batches = []
        index_batches = []
        with torch.inference_mode():
            for pixels, indices in tqdm(loader, desc=f"Encoding {split} images"):
                features = unwrap_features(
                    model.get_image_features(pixel_values=pixels.to(device))
                )
                embedding_batches.append(
                    F.normalize(features.float(), dim=-1).cpu()
                )
                index_batches.append(indices)

        embeddings = torch.cat(embedding_batches)
        extracted_indices = torch.cat(index_batches)
        assert torch.equal(extracted_indices, torch.arange(len(dataset)))
        torch.save(
            {
                "model_id": MODEL_ID,
                "split": split,
                "embeddings": embeddings.half(),
                "filenames": list(dataset.filename),
            },
            image_path,
        )
        print(f"Saved {split} image embeddings: {image_path}")


if __name__ == "__main__":
    main()
