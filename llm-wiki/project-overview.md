# Project Overview

## Plain-English Task

You need to build a deep learning system for compositional image retrieval.

Input:

- A reference image from the CelebA test split.
- One or more text conditions, each positive or negative.

Example conditions:

- `+Smiling`: retrieve similar faces that are smiling.
- `+Eyeglasses`: retrieve similar faces wearing glasses.
- `+Black_Hair, -Wavy_Hair`: retrieve similar faces with black hair but not wavy hair.
- `+Wearing_Lipstick, -Heavy_Makeup, +Smiling`: satisfy all three constraints together.

Output:

- A ranked list of retrieved CelebA test images.
- The top K results should contain images that satisfy the requested attribute changes while preserving the source image's other visual/attribute identity as much as possible.

## Why This Is Hard

Standard CLIP can embed images and text into a shared space, but composing several attribute edits is tricky. A naive approach like:

```text
query = image_embedding + text_embedding("+eyeglasses") - text_embedding("wavy hair")
```

is easy to implement, but may not balance identity preservation, positive constraints, and negative constraints well.

CLAY improves conditional similarity by modulating the similarity space, but the assignment notes a limitation: for multiple conditions, CLAY relies on rigid stacking or concatenation before SVD. The project asks you to replace or improve that fusion step.

## Assignment Objective

Develop, implement, and evaluate a dynamic similarity metric or fusion module `Phi` that combines:

- the reference image embedding,
- positive text constraints,
- negative text constraints,
- optionally learned weights, gates, attention, adapters, projection heads, or other fusion logic.

The composite query embedding or scoring function should retrieve valid target images from CelebA.

## Current Learned-Method Proposal

The group currently plans to train a lightweight residual composition network on frozen CLIP embeddings. Training examples are generated from CelebA image pairs and their signed attribute differences. The model predicts how the source image embedding should move under one or more positive/negative textual conditions.

See [Proposed Training Strategy](training-strategy.md) for the formal definition and experimental plan.

## Required Dataset

Use CelebA, especially its official test split via `torchvision.datasets.CelebA`. The local workspace already contains:

- `celeba/img_align_celeba/`: 202599 aligned face images.
- `celeba/list_attr_celeba.txt`: 40 binary attributes per image.
- `celeba/list_eval_partition.txt`: train/validation/test partition metadata.
- `celeba_evaluation.json`: official query-specific source images and acceptable target indices.

See [CelebA Dataset](dataset-celeba.md).

## Required Model Backbone

The assignment requests CLIP ViT-B/32 from HuggingFace:

```text
openai/clip-vit-base-patch32
```

Other models may be tested, but CLIP ViT-B/32 should be included for comparability.

## Required Final Deliverable

A single self-contained Google Colab notebook containing:

- all code,
- clear modular cells,
- Markdown report sections,
- methodology and math description,
- experimental setup,
- Recall@K results,
- qualitative retrieval examples,
- discussion of successes and failures.
