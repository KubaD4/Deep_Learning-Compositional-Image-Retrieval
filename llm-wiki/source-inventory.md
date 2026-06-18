# Source Inventory

This page records the project files inspected through 2026-06-11.

## Repository Shape

- Project root: `/Users/kuba/deep_learning`
- Total size after setup: about 3.9 GB.
- `.venv/`: about 374 MB, project-local MarkItDown environment.
- `celeba/`: about 1.7 GB extracted CelebA data.
- `celeba.zip.download/`: about 1.4 GB original zip download.
- `llm-wiki/`: generated project wiki and source Markdown derivatives.

The entire `llm-wiki/` directory is intentionally local-only and ignored by git. It must not be published to GitHub unless the user explicitly changes that policy.

## Repository Documentation And Tools

| Path | Meaning |
| --- | --- |
| `PROJECT_STATE_AND_SOLUTION.md` | Current authoritative snapshot: task, completed work, selected proposal, risks, and next milestones. |
| `PROJECT_GUIDE.md` | Detailed onboarding, code architecture, development plan, and final checklist. |
| `README.md` | Repository title, course, authors, and documentation links. |
| `AGENTS.md` | Required reading and local wiki-maintenance contract for future coding-agent sessions. |
| `ground_truth_dashboard/` | Implemented local benchmark explorer with a Python server, browser UI, and four unit tests. |
| `scripts/count_identity_pairs.py` | Read-only CelebA identity-pair counter; reports per-person and global unordered/directional pair counts. |
| `requirements.txt` | Reproducible MarkItDown dependency declaration. |
| `tools/markitdown/` | Local clone of Microsoft MarkItDown; nested tool repository, not project model code. |

## Two Dataset Layouts

The machine currently has two related layouts:

- `celeba/`: ignored local full dataset including all 202,599 images and metadata.
- `data/celeba/annotations/`: committed shareable copy of the six annotation files.
- `data/celeba/images/.gitkeep`: placeholder; raw images are intentionally not committed.

Code intended for Colab should use the unzipped `celeba/` dataset structure expected by `torchvision.datasets.CelebA`. Repository utilities may use either metadata copy when explicitly configured.

## Assignment and Course PDFs

| Raw source | Markdown derivative | Notes |
| --- | --- | --- |
| `Project assignment - V1.2.pdf` | [source-markdown/Project assignment - V1.2.md](source-markdown/Project%20assignment%20-%20V1.2.md) | Canonical assignment. Defines task, dataset, evaluation, roadmap, deliverable. |
| `Project assignment - V1.2-2.pdf` (historical; raw copy removed) | [source-markdown/Project assignment - V1.2-2.md](source-markdown/Project%20assignment%20-%20V1.2-2.md) | Was a byte-identical duplicate of V1.2. Only its local Markdown derivative remains. |
| `Project - Introduction-2.pdf` | [source-markdown/Project - Introduction-2.md](source-markdown/Project%20-%20Introduction-2.md) | Slide deck introducing background, CLAY limitation, task, evaluation, deliverables. |
| `Lab 2 - CNNs.pdf` | [source-markdown/Lab 2 - CNNs.md](source-markdown/Lab%202%20-%20CNNs.md) | Lab slides. Mostly CNN/normalization, with project skeleton/evaluation mention. |
| `VM usage.pdf` | [source-markdown/VM usage.md](source-markdown/VM%20usage.md) | Azure GPU VM access and setup instructions. |

## Research and Background PDFs

| Raw source | Markdown derivative | Role |
| --- | --- | --- |
| `CLIP.pdf` | [source-markdown/CLIP.md](source-markdown/CLIP.md) | Foundational CLIP paper. Use for zero-shot image/text embedding baseline. |
| `CLIP Compositionality - Davide Berasi.pdf` | [source-markdown/CLIP Compositionality - Davide Berasi.md](source-markdown/CLIP%20Compositionality%20-%20Davide%20Berasi.md) | Paper on compositionality in visual representations. Assignment cites this line of work. |
| `CLAY.pdf` | [source-markdown/CLAY.md](source-markdown/CLAY.md) | Conditional visual similarity modulation. Assignment asks to improve the multi-condition fusion bottleneck. |
| `Compositionality.pdf` | [source-markdown/Compositionality.md](source-markdown/Compositionality.md) | Lecture slides on compositionality and CLIP geometry. |

## Notebook and Evaluation Files

| File | Notes |
| --- | --- |
| `Project Skeleton.ipynb` | Starter Colab notebook. Shows dataset setup, loading CelebA test split, evaluation helper, JSON loading, and correct dataset-index access. Converted to [source-markdown/Project Skeleton.md](source-markdown/Project%20Skeleton.md). |
| `celeba_evaluation.json` | Official benchmark: 14 textual queries. Each query maps source dataset indices to acceptable target dataset indices. See [Evaluation Protocol](evaluation-protocol.md). |

## CelebA Data Files

| File | Meaning |
| --- | --- |
| `celeba/img_align_celeba/` | 202599 aligned face images, filenames `000001.jpg` through `202599.jpg`. |
| `celeba/list_attr_celeba.txt` | 40 binary attributes per image, values `1` and `-1`. |
| `celeba/list_bbox_celeba.txt` | Face bounding boxes. |
| `celeba/list_eval_partition.txt` | Split assignment: train, validation, test. |
| `celeba/list_landmarks_align_celeba.txt` | Landmark coordinates for aligned images. |
| `celeba/list_landmarks_celeba.txt` | Landmark coordinates for original images. |
| `celeba/identity_CelebA.txt` | Identity labels for images. |
| `celeba.zip.download/celeba.zip` | Original downloaded CelebA zip. |

## Derived Supervision Notes

The most useful join for training is:

- `identity_CelebA.txt` joined with `list_attr_celeba.txt` on image filename,
- optionally joined with `list_eval_partition.txt` to respect train/validation/test boundaries.

That merged table is enough to derive automatic same-identity attribute-edit pairs such as:

```text
(source image without eyeglasses, "add eyeglasses", target image with eyeglasses)
```

See [CelebA Dataset](dataset-celeba.md) for the pairing logic and caveats.

## Ignored/Generated Files

- `.DS_Store` files are macOS metadata and not project content.
- `.llm-wiki-work/` contains prior intermediate PDF text extraction.
- `.venv/` contains the MarkItDown virtual environment.
- `llm-wiki/source-markdown/` contains generated Markdown derivatives that are useful for LLM reading and wiki linking.
- `.obsidian/` contains local Obsidian workspace state and is not project knowledge.
- `tools/markitdown/` is a local nested clone and is ignored; `requirements.txt` is the reproducible installation contract.
