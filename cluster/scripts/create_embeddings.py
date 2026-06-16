#!/usr/bin/env python3
"""Create resumable CLIP image and text embedding caches."""

from __future__ import annotations

import argparse
import signal
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm.auto import tqdm
from transformers import CLIPModel, CLIPProcessor

from project_core import (
    CHECKPOINT_DIR,
    EMBEDDING_DIR,
    MODEL_ID,
    atomic_torch_save,
    attribute_names,
    choose_device,
    load_celeba,
    load_torch,
    prompt_pair,
    unwrap_features,
)


STOP_REQUESTED = False


def request_stop(signum, frame) -> None:
    del signum, frame
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print("Stop requested: finishing the current batch before exiting.")


class IndexedImages(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, _ = self.dataset[index]
        return image, index


class ImageCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, batch):
        images, indices = zip(*batch)
        pixels = self.processor(images=list(images), return_tensors="pt")[
            "pixel_values"
        ]
        return pixels, torch.tensor(indices)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("train", "valid", "test"), required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--time-budget-seconds", type=int, default=480)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def contiguous_resume_index(chunks_dir: Path) -> int:
    expected = 0
    for path in sorted(chunks_dir.glob("chunk_*.pt")):
        chunk = load_torch(path)
        start = int(chunk["indices"][0])
        end = int(chunk["indices"][-1]) + 1
        if start != expected:
            break
        expected = end
    return expected


def create_text_cache(model, processor, device, dataset) -> None:
    path = EMBEDDING_DIR / "attribute_text_embeddings.pt"
    names = attribute_names(dataset)
    if path.exists():
        cache = load_torch(path)
        if cache.get("model_id") == MODEL_ID and cache.get("attributes") == names:
            print(f"Reusing text cache: {path}")
            return

    pairs = [prompt_pair(name) for name in names]
    prompts = [prompt for pair in pairs for prompt in pair]
    inputs = processor(text=prompts, return_tensors="pt", padding=True).to(device)
    with torch.inference_mode():
        features = unwrap_features(model.get_text_features(**inputs)).float()
        features = F.normalize(features, dim=-1).cpu()
    positive = features[0::2]
    negative = features[1::2]
    atomic_torch_save(
        {
            "model_id": MODEL_ID,
            "attributes": names,
            "prompts": pairs,
            "positive": positive.half(),
            "negative": negative.half(),
            "directions": F.normalize(positive - negative, dim=-1).half(),
        },
        path,
    )
    print(f"Saved text cache: {path}")


def finalize_split(split, dataset, chunks_dir, final_path) -> None:
    chunks = []
    expected = 0
    for path in sorted(chunks_dir.glob("chunk_*.pt")):
        chunk = load_torch(path)
        indices = chunk["indices"]
        if int(indices[0]) != expected:
            raise RuntimeError(f"Non-contiguous embedding chunks at index {expected}")
        expected = int(indices[-1]) + 1
        chunks.append(chunk["embeddings"])
    if expected != len(dataset):
        raise RuntimeError(f"Cannot finalize {split}: reached {expected}/{len(dataset)}")
    embeddings = torch.cat(chunks)
    atomic_torch_save(
        {
            "model_id": MODEL_ID,
            "split": split,
            "embeddings": embeddings,
            "filenames": list(dataset.filename),
        },
        final_path,
    )
    print(f"Final cache saved: {final_path}")


def main() -> int:
    args = parse_args()
    started = time.monotonic()
    signal.signal(signal.SIGTERM, request_stop)
    if hasattr(signal, "SIGUSR1"):
        signal.signal(signal.SIGUSR1, request_stop)

    dataset = load_celeba(args.split)
    final_path = EMBEDDING_DIR / f"{args.split}_image_embeddings.pt"
    chunks_dir = CHECKPOINT_DIR / "embeddings" / args.split
    chunks_dir.mkdir(parents=True, exist_ok=True)
    EMBEDDING_DIR.mkdir(parents=True, exist_ok=True)

    if final_path.exists() and not args.force:
        cache = load_torch(final_path)
        if len(cache["embeddings"]) == len(dataset) and cache["model_id"] == MODEL_ID:
            print(f"Complete cache already exists: {final_path}")
            return 0

    if args.force:
        for path in chunks_dir.glob("chunk_*.pt"):
            path.unlink()
        final_path.unlink(missing_ok=True)

    start_index = contiguous_resume_index(chunks_dir)
    print(f"Resuming {args.split} at dataset index {start_index}/{len(dataset)}")
    if start_index == len(dataset):
        finalize_split(args.split, dataset, chunks_dir, final_path)
        return 0

    device = choose_device(args.device)
    processor = CLIPProcessor.from_pretrained(MODEL_ID)
    model = CLIPModel.from_pretrained(MODEL_ID).to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    create_text_cache(model, processor, device, dataset)

    remaining = Subset(IndexedImages(dataset), range(start_index, len(dataset)))
    loader = DataLoader(
        remaining,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
        collate_fn=ImageCollator(processor),
    )

    with torch.inference_mode():
        for pixels, indices in tqdm(loader, desc=f"Encoding {args.split}"):
            features = unwrap_features(
                model.get_image_features(pixel_values=pixels.to(device))
            )
            features = F.normalize(features.float(), dim=-1).cpu().half()
            first = int(indices[0])
            end = int(indices[-1]) + 1
            atomic_torch_save(
                {
                    "model_id": MODEL_ID,
                    "split": args.split,
                    "indices": indices,
                    "filenames": [dataset.filename[index] for index in indices.tolist()],
                    "embeddings": features,
                },
                chunks_dir / f"chunk_{first:08d}_{end:08d}.pt",
            )

            elapsed = time.monotonic() - started
            if STOP_REQUESTED or elapsed >= args.time_budget_seconds:
                print(f"Checkpointed through index {end}; submit the job again.")
                return 3

    finalize_split(args.split, dataset, chunks_dir, final_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
