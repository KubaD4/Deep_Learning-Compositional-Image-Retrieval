#!/usr/bin/env python3
"""Verify paths, data, caches, and Python dependencies."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from project_core import (
    DATA_ROOT,
    EMBEDDING_DIR,
    EVALUATION_PATH,
    MODEL_ID,
    ROOT,
    load_torch,
)


def main() -> None:
    print(f"Bundle root: {ROOT}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    archive = DATA_ROOT / "celeba.zip"
    extracted = DATA_ROOT / "celeba" / "img_align_celeba"
    if not archive.exists() and not extracted.exists():
        raise FileNotFoundError("Neither data/celeba.zip nor extracted CelebA was found")
    if not EVALUATION_PATH.exists():
        raise FileNotFoundError(EVALUATION_PATH)
    annotations = json.loads(EVALUATION_PATH.read_text(encoding="utf-8"))
    print(f"Evaluation query entries: {len(annotations)}")

    test_cache = EMBEDDING_DIR / "test_image_embeddings.pt"
    train_cache = EMBEDDING_DIR / "train_image_embeddings.pt"
    valid_cache = EMBEDDING_DIR / "valid_image_embeddings.pt"
    text_cache = EMBEDDING_DIR / "attribute_text_embeddings.pt"
    prompt_cache = EMBEDDING_DIR / "signed_attribute_prompt_embeddings.pt"
    if test_cache.exists():
        cache = load_torch(test_cache)
        assert cache["model_id"] == MODEL_ID
        assert tuple(cache["embeddings"].shape) == (19_962, 512)
        print(f"Test cache ready: {test_cache}")
    else:
        print("Test cache missing; submit jobs/01_embeddings_test.sh")
    if text_cache.exists():
        cache = load_torch(text_cache)
        assert len(cache["attributes"]) == 40
        print(f"Text cache ready: {text_cache}")
    else:
        print("Text cache missing; it will be created by any embedding job")
    if train_cache.exists():
        print(f"Train cache ready: {train_cache}")
    else:
        print("Train cache missing; submit jobs/02_embeddings_train.sh")
    if valid_cache.exists():
        print(f"Validation cache ready: {valid_cache}")
    else:
        print("Validation cache missing; submit jobs/03_embeddings_valid.sh")
    if prompt_cache.exists():
        cache = load_torch(prompt_cache)
        assert len(cache["attributes"]) == 40
        print(f"Prompt ensemble cache ready: {prompt_cache}")
    else:
        print("Prompt ensemble cache missing; submit jobs/30_create_prompt_embeddings.sh")
    pair_dir = ROOT / "artifacts" / "training_pairs"
    train_pairs = pair_dir / "train_pairs_len1_3.pt"
    valid_pairs = pair_dir / "valid_pairs_len1_3.pt"
    if train_pairs.exists() and valid_pairs.exists():
        print(f"Training pair indices ready: {pair_dir}")
    else:
        print("Training pair indices missing; submit jobs/31_build_training_pairs.sh")
    print("Bundle verification complete.")


if __name__ == "__main__":
    main()
