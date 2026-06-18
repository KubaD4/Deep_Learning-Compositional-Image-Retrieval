# Dynamic and Hybrid Conditioning for Compositional Image Retrieval

Complete project guide for the Deep Learning Assignment 2026.

This is the document every group member should read before implementing anything. It explains the task, supplied files, data structures, starter notebook, evaluation, development sequence, training options, recommended code architecture, and final deliverable.

For the latest distinction between completed work, proposed work, repository state, and the selected learned solution, read [PROJECT_STATE_AND_SOLUTION.md](PROJECT_STATE_AND_SOLUTION.md) first.

## 1. Five-Minute Summary

You are building an **image retrieval system**, not an image generator and not a face classifier.

The system receives:

1. A reference face image from the CelebA test set.
2. One or more signed text conditions such as `+Smiling`, `-Young`, or `+Black_Hair, -Wavy_Hair`.

The system returns:

1. A similarity score for every candidate image in the CelebA test set.
2. A ranked list of candidate dataset indices, from most to least suitable.

Example:

```text
Input reference: celeba_test[13]
Input condition: +Smiling

Desired output:
faces that retain most non-requested properties of source 13,
but have Smiling = true.

Model output:
[579, 325, 456, ...]  # ranked CelebA test indices
```

The official JSON determines which returned indices count as correct. You evaluate the ranked list using Recall@1/5/10 and Precision@1/5/10.

Your main research contribution is a better method for combining the reference image with multiple positive and negative text conditions. The assignment specifically wants something more flexible than CLAY's rigid pre-SVD stacking.

## 2. The Problem, Precisely

### 2.1 Inputs

For one retrieval case:

```text
reference image: v_ref
positive conditions: T+ = {t1+, t2+, ...}
negative conditions: T- = {t1-, t2-, ...}
candidate database: every image in the CelebA test split
```

Examples:

```text
v_ref + {+Smiling}
v_ref + {+Eyeglasses, +Smiling}
v_ref + {+Black_Hair} + {-Wavy_Hair}
v_ref + {+Wearing_Lipstick, +Smiling} + {-Heavy_Makeup}
```

A `+` condition means the retrieved target must have that attribute. A `-` condition means the target must not have it.

### 2.2 Output

Your method must produce a ranked list:

```python
retrieved_indices = [579, 325, 456, 981, 1363, ...]
```

Each integer is an index in the PyTorch `CelebA(split="test")` object. It is not a physical filename.

Your model will normally also produce scores:

```python
retrieved_scores = [0.83, 0.81, 0.79, 0.76, 0.75, ...]
```

Higher score means your model believes that candidate better preserves the reference while satisfying the conditions.

### 2.3 The System Does Not Create A New Face

This is retrieval:

```text
reference + requested edits -> search existing test images -> ranked existing images
```

It is not generation:

```text
reference + requested edits -> synthesize a new edited image
```

## 3. What Does "Similar" Mean?

There are two separate meanings that must not be confused.

### 3.1 Similarity Used By Your Model

Your method needs a numerical score to rank candidates. The simplest baseline uses cosine similarity in CLIP space:

```text
score(candidate) = cosine(candidate_embedding, fused_query_embedding)
```

The fused query can initially be:

```text
q = reference_embedding
    + alpha * sum(positive_text_embeddings)
    - beta  * sum(negative_text_embeddings)
```

Your improved method should replace this naive equal-weight arithmetic with dynamic fusion, gating, attention, learned projections, or another justified mechanism.

### 3.2 Similarity Used For Official Validation

The assignment does not grade your cosine score directly. It checks whether your returned indices belong to the valid target list in `celeba_evaluation.json`.

A target was placed in that list when:

1. It strictly satisfies every requested positive and negative attribute.
2. Its non-queried CelebA attributes differ from the source by at most two bits.

This is a relaxed Hamming-distance definition over the 40 binary CelebA attributes.

Important interpretation: the word "identity" in the assignment does not require the same real celebrity/person identity. The official benchmark preserves the **non-requested attribute profile**, not necessarily `identity_CelebA.txt` identity.

### 3.3 Concrete Verified Example

For query `+Smiling`:

```text
source dataset index: 13
source physical file: 182651.jpg
source Smiling: false
```

One valid target is:

```text
target dataset index: 579
target physical file: 183217.jpg
target Smiling: true
```

Compared with the source, target `579` differs in:

```text
Smiling          # requested change, not counted in non-query Hamming distance
High_Cheekbones  # non-query difference 1
```

Therefore it satisfies the query and has only one non-query difference, which is within the allowed maximum of two.

## 4. What You Already Have

Project root:

```text
/Users/kuba/deep_learning/
```

### 4.1 Essential Files

| Path | What it is | What you use it for |
| --- | --- | --- |
| `Project assignment - V1.2.pdf` | Canonical assignment specification | Final authority for task, metrics, model, and deliverable |
| `Project assignment - V1.2-2.pdf` | Historical byte-identical duplicate, no longer present in the root | Ignore; use the canonical file above. Its converted Markdown remains in the local wiki. |
| `Project Skeleton.ipynb` | Starter Colab notebook | Dataset loading, JSON loading, evaluation helper, indexing examples |
| `celeba_evaluation.json` | Official benchmark | Supplies queries, valid source indices, and correct target indices |
| `celeba.zip.download/celeba.zip` | Original data archive | Upload to Drive for the Colab workflow |
| `celeba/` | Extracted local dataset | Local inspection and development |

### 4.2 CelebA Structure

```text
celeba/
├── img_align_celeba/                 # 202,599 aligned JPG images
├── list_attr_celeba.txt              # 40 binary attributes per image
├── list_eval_partition.txt           # train/validation/test membership
├── identity_CelebA.txt               # person identity metadata; not official retrieval GT
├── list_bbox_celeba.txt              # face bounding boxes
├── list_landmarks_align_celeba.txt   # landmarks in aligned images
└── list_landmarks_celeba.txt         # landmarks in original coordinates
```

Official split sizes:

| Split | Partition value | Images |
| --- | ---: | ---: |
| Train | 0 | 162,770 |
| Validation | 1 | 19,867 |
| Test | 2 | 19,962 |

The required benchmark searches the 19,962-image test split.

### 4.3 Background Material

| File | Why it matters |
| --- | --- |
| `CLIP.pdf` | Explains the pretrained image/text embedding model used as the required backbone |
| `CLAY.pdf` | Main architectural inspiration and the multi-condition fusion limitation to improve |
| `CLIP Compositionality - Davide Berasi.pdf` | Background on compositional structures in visual embeddings |
| `Compositionality.pdf` | Course-level explanation of compositionality and CLIP geometry |
| `Project - Introduction-2.pdf` | Instructor slides summarizing task and deliverables |
| `Lab 2 - CNNs.pdf` | Lab material and starter/evaluation context |
| `VM usage.pdf` | Instructions for the provided Azure GPU VM |

Readable Markdown conversions are in `llm-wiki/source-markdown/`.

### 4.4 Project Knowledge Files

```text
AGENTS.md                  # rules for future coding-agent sessions
llm-wiki/index.md          # wiki entry point
PROJECT_GUIDE.md           # this complete human onboarding guide
```

## 5. Critical Dataset Indexing Rule

The JSON uses **test dataset indices**, not filenames.

Correct:

```python
source_idx = int(source_key)
source_image, source_attributes = celeba_test[source_idx]
```

Incorrect:

```python
Image.open("celeba/img_align_celeba/000013.jpg")
```

For example:

```text
dataset index 13 -> physical filename 182651.jpg
```

The test dataset is a subset of all files. `torchvision.datasets.CelebA` performs the mapping for you.

## 6. The Evaluation JSON

Simplified structure:

```python
annotations = [
    {
        "query": "+Smiling",
        "ground_truth": {
            "13": [325, 456, 579, 685, ...],
            "14": [...],
            "...": [...],
        }
    },
    ...
]
```

Interpretation:

```text
annotations[0]["query"]
    -> "+Smiling"

annotations[0]["ground_truth"]["13"]
    -> all acceptable test targets for source test index 13
```

The local JSON contains 14 entries and 33,052 source-query evaluation cases. `-Young` appears twice with identical ground truth. The assignment explicitly says the JSON is authoritative, so keep both unless the instructors issue a corrected file.

Exact local query list:

| JSON index | Query | Valid sources |
| ---: | --- | ---: |
| 0 | `+Smiling` | 4,786 |
| 1 | `+Eyeglasses` | 2,196 |
| 2 | `-Heavy_Makeup` | 4,087 |
| 3 | `+Male` | 1,595 |
| 4 | `-Young` | 5,355 |
| 5 | `+Blond_Hair` | 5,469 |
| 6 | `+Mustache` | 301 |
| 7 | `-Young` | 5,355 |
| 8 | `+Eyeglasses, +Smiling` | 612 |
| 9 | `+Black_Hair, -Wavy_Hair` | 2,572 |
| 10 | `-Male, -Mustache` | 27 |
| 11 | `+Chubby, -Young` | 584 |
| 12 | `-Smiling, +Eyeglasses, +Wearing_Hat` | 79 |
| 13 | `+Wearing_Lipstick, -Heavy_Makeup, +Smiling` | 34 |

The PDF's printed list and the actual JSON do not match perfectly. Use the JSON strings and source mappings in code, and mention the duplicate/mismatch transparently in the report if it remains in the final instructor-provided file.

Do not use this test ground truth to train your model. That would leak the answers. Use it only for final evaluation and pipeline debugging with a tiny number of examples.

## 7. Metrics

For a single source/query pair, let:

```text
R_K = your top-K retrieved indices
G   = official valid target indices from JSON
```

### Recall@K, Primary Metric

```text
Recall@K = 1 if R_K intersects G, otherwise 0
```

This assignment uses hit-rate Recall, not the traditional fraction of all ground-truth targets recovered.

### Precision@K, Secondary Metric

```text
Precision@K = number of indices in both R_K and G / K
```

Example:

```text
top 5 = [579, 100, 325, 200, 300]
valid = [325, 456, 579, 685, ...]

hits = {579, 325}
Recall@5 = 1
Precision@5 = 2/5 = 0.4
```

Compute K = 1, 5, and 10. Average each metric over all valid sources for each query. Report per-query results and a macro average across queries.

Always remove the source index itself from retrieval results by setting its score to negative infinity.

## 8. What The Starter Notebook Does

The supplied notebook contains 26 cells. It provides infrastructure, not a solution.

### 8.1 Dataset Setup Cells

```python
from google.colab import drive
drive.mount("/content/drive")
```

Mounts persistent Google Drive storage inside Colab.

```bash
!mkdir /content/datasets
!unzip -q /content/drive/MyDrive/datasets/celeba.zip -d /content/datasets/
```

Copies/unzips the large dataset from slow persistent Drive into the Colab runtime's faster temporary disk. The runtime copy disappears when Colab disconnects.

### 8.2 Data Loading Cells

```python
from pathlib import Path
from torchvision.datasets import CelebA

data_root = Path("/content/datasets")
celeba = CelebA(root=data_root, split="test", download=False)
assert len(celeba) == 19_962
```

`data_root` must be the parent directory containing `celeba/`. Do not set it to `/content/datasets/celeba`.

### 8.3 Metric Function

```python
def evaluate_retrieval(retrieved_indices, ground_truth_indices, k):
    top_k = retrieved_indices[:k]
    hits = set(top_k).intersection(ground_truth_indices)
    return {
        f"Recall@{k}": int(len(hits) > 0),
        f"Precision@{k}": len(hits) / k,
    }
```

This evaluates only one source/query case. You still need an outer loop that aggregates results over every query and every source.

### 8.4 JSON Loading

```python
annotations_path = Path(
    "/content/drive/MyDrive/datasets/celeba_evaluation.json"
)

with open(annotations_path) as f:
    annotations = json.load(f)
```

This loads the benchmark separately from the image archive.

### 8.5 Visualization Cells

The notebook displays:

- source `celeba[13]`,
- several official valid targets for `+Smiling`,
- proof that the benchmark targets visually and semantically resemble the source.

### 8.6 Attribute Mapping

```python
idx2attribute = {
    idx: name for idx, name in enumerate(celeba.attr_names)
}
attribute2idx = {
    name: idx for idx, name in enumerate(celeba.attr_names)
}
```

These dictionaries let you translate between attribute tensor columns and names such as `Smiling` or `Wearing_Hat`.

## 9. What The Starter Notebook Does Not Do

You must implement all of these:

- install/load HuggingFace CLIP ViT-B/32,
- preprocess images for CLIP,
- extract and cache image embeddings,
- encode query text,
- parse positive and negative conditions,
- fuse image and text representations,
- score all candidate test images,
- return top-K indices,
- exclude the source image,
- aggregate metrics over all benchmark cases,
- build a baseline,
- build your improved method,
- optionally train a lightweight fusion module,
- produce result tables, plots, and qualitative grids,
- write the scientific report inside the notebook.

## 10. End-To-End System Architecture

```text
CelebA test images
        |
        | one-time CLIP image encoding
        v
frozen candidate matrix E_test [19,962 x embedding_dim]

reference test image ----> CLIP image encoder ----> v_ref
text conditions ---------> CLIP text encoder -----> t_1, ..., t_n
                                                    |
                       your fusion method Phi <----+
                                                    |
                                                    v
                                         fused query vector q
                                                    |
                                  scores = E_test @ q
                                                    |
                                      remove source index
                                                    |
                                         top-K dataset indices
                                                    |
                           compare with JSON valid target indices
                                                    |
                              Recall@K and Precision@K
```

CLIP should normally remain frozen. The assignment is about the conditioning/fusion mechanism, not retraining a giant vision-language model.

## 11. Do You Need To Train?

Not necessarily.

The assignment explicitly permits:

- training-free methods,
- training-based methods.

### Minimum Viable Route

1. Freeze CLIP.
2. Extract embeddings once.
3. Build naive vector-arithmetic baseline.
4. Build an original training-free dynamic fusion/scoring method.
5. Compare both methods.

### Recommended Stronger Route

1. Freeze CLIP.
2. Train only a small fusion module on CelebA train.
3. Tune architecture/hyperparameters on CelebA validation.
4. Run the official JSON benchmark on test only after choices are fixed.

Never train on `celeba_evaluation.json` targets.

## 12. Recommended Development Stages

Do not begin by designing a complicated model. Build a trustworthy pipeline in this order.

### Stage 0: Group Setup

- Register the group if not already done.
- Put the notebook in a shared group-controlled location.
- Agree on one canonical notebook/repository to prevent incompatible copies.
- Record every experiment with method name, parameters, date, and metrics.
- Read the academic-integrity policy: group members may collaborate, but code cannot be shared across groups.

### Stage 1: Reproduce The Supplied Skeleton

Definition of done:

- dataset loads,
- test length is 19,962,
- JSON loads and has 14 query entries,
- `celeba[13]` displays the correct source,
- official target examples display,
- metric unit tests produce expected values.

Do this before loading CLIP.

### Stage 2: Explore Data And Benchmark

Produce:

- attribute prevalence table by split,
- source count per query,
- number of valid targets per source,
- visual examples for simple and composed queries,
- check that positive and negative constraints match target labels.

This catches data and indexing mistakes early.

### Stage 3: Extract Frozen CLIP Features

Required model:

```text
openai/clip-vit-base-patch32
```

Extract normalized image embeddings for the test split and cache them. For training a fusion module, separately cache train and validation embeddings.

Approximate float32 storage for 512-dimensional embeddings:

```text
test only: about 39 MB
all 202,599 images: about 396 MB
```

Cache metadata together with embeddings:

```python
{
    "embeddings": tensor,
    "filenames": dataset.filename,
    "split": "test",
    "model_id": "openai/clip-vit-base-patch32",
}
```

Definition of done:

- embedding count equals dataset count,
- no NaN values,
- rows are L2-normalized,
- cache reloads without recomputing,
- nearest-neighbor sanity examples look visually related.

### Stage 4: Implement The Vanilla Baseline

Use simple signed arithmetic:

```text
q = normalize(v_ref + alpha * positive_text - beta * negative_text)
```

For multiple conditions, sum all signed text embeddings.

Tune `alpha` and `beta` on validation or a separately constructed validation benchmark, not on official test results.

Definition of done:

- one source/query returns sensible top 10,
- source itself is excluded,
- full evaluator runs,
- baseline table contains all official queries and K values.

### Stage 5: Implement A Better Fusion Method

Your method must have a clear hypothesis. Examples:

- **Dynamic gates:** predict a weight for every condition from the reference and text embeddings.
- **Cross-attention:** let the reference attend to a variable number of signed condition tokens.
- **Nonlinear residual fusion:** predict a bounded edit vector conditioned on reference and conditions.
- **Hybrid scoring:** combine identity-preservation score and condition-satisfaction score with query-dependent weights.
- **CLAY-inspired modulation:** preserve fixed candidate image embeddings while dynamically changing the similarity space.

Avoid making the method large just to appear sophisticated. A small mechanism with clear ablations is usually stronger scientifically.

### Stage 6: Train Only If Your Method Requires It

Create training examples from CelebA train attributes:

1. Choose a train source image.
2. Sample one to three query attributes and signs.
3. Find positive targets that satisfy the conditions.
4. Prefer positives close on non-query attributes.
5. Sample hard negatives that violate at least one condition or lose source similarity.

Possible objective:

```text
contrastive loss(fused_query, positive_target, negative_targets)
+ identity-preservation regularizer
+ optional gate/weight regularization
```

A batch contrastive formulation is practical:

```python
logits = fused_queries @ positive_target_embeddings.T / temperature
labels = torch.arange(batch_size, device=logits.device)
loss = torch.nn.functional.cross_entropy(logits, labels)
```

Use:

```text
train split      -> parameter optimization
validation split -> hyperparameters, early stopping, method selection
test JSON        -> final locked evaluation only
```

### Stage 7: Evaluate And Diagnose

For every method:

- Recall@1, @5, @10,
- Precision@1, @5, @10,
- per-query rows,
- macro average,
- runtime and parameter count if relevant.

Analyze:

- simple versus composed queries,
- positive versus negative constraints,
- common versus rare attributes,
- successes,
- failures,
- whether identity preservation or condition satisfaction is failing.

### Stage 8: Ablations

Examples:

- equal weights versus learned/dynamic weights,
- reference residual on/off,
- positive conditions only versus signed conditions,
- one condition versus two versus three,
- different prompt templates,
- different fusion depth/width,
- training-free versus trained variant.

An ablation should answer why your method works, not merely add another table.

### Stage 9: Finish The Notebook Report

Required narrative:

- problem statement,
- related work,
- method and equations,
- experimental protocol,
- implementation details,
- baseline,
- main results,
- ablations,
- qualitative examples,
- limitations,
- conclusion.

## 13. Suggested Notebook And Code Skeleton

The final submission is one notebook, but the code should still be modular.

### Section 1: Setup And Configuration

```python
from dataclasses import dataclass
from pathlib import Path

import json
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import CelebA
from transformers import CLIPModel, CLIPProcessor


@dataclass
class Config:
    data_root: Path = Path("/content/datasets")
    annotation_path: Path = Path(
        "/content/drive/MyDrive/datasets/celeba_evaluation.json"
    )
    cache_root: Path = Path("/content/drive/MyDrive/dl_project_cache")
    model_id: str = "openai/clip-vit-base-patch32"
    batch_size: int = 128
    ks: tuple[int, ...] = (1, 5, 10)
    seed: int = 42


cfg = Config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

Purpose: keep paths and hyperparameters in one reproducible place.

### Section 2: Load Data And Benchmark

```python
celeba_train = CelebA(cfg.data_root, split="train", download=False)
celeba_valid = CelebA(cfg.data_root, split="valid", download=False)
celeba_test = CelebA(cfg.data_root, split="test", download=False)

with open(cfg.annotation_path) as f:
    annotations = json.load(f)

assert len(celeba_test) == 19_962
assert len(annotations) == 14
```

Purpose: create clearly separated datasets and load official test cases.

If you implement a training-free method, train and validation datasets may only be needed for analysis or hyperparameter construction.

### Section 3: Load Frozen CLIP

```python
processor = CLIPProcessor.from_pretrained(cfg.model_id)
clip_model = CLIPModel.from_pretrained(cfg.model_id).to(device).eval()

for parameter in clip_model.parameters():
    parameter.requires_grad_(False)
```

Purpose: use the required pretrained backbone without expensive full-model training.

### Section 4: Feature Extraction

```python
@torch.inference_mode()
def encode_images(images):
    inputs = processor(images=images, return_tensors="pt").to(device)
    features = clip_model.get_image_features(
        pixel_values=inputs["pixel_values"]
    )
    return F.normalize(features, dim=-1)


@torch.inference_mode()
def encode_texts(texts):
    inputs = processor(
        text=texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(device)
    features = clip_model.get_text_features(
        input_ids=inputs["input_ids"],
        attention_mask=inputs.get("attention_mask"),
    )
    return F.normalize(features, dim=-1)
```

Purpose: map images and text into the same CLIP embedding space.

Build a batched extraction function and cache split embeddings. Do not repeatedly run CLIP inside every evaluation case.

### Section 5: Query Parsing

```python
def parse_query(query: str) -> list[tuple[int, str]]:
    conditions = []
    for raw_part in query.split(","):
        part = raw_part.strip()
        if not part or part[0] not in "+-":
            raise ValueError(f"Invalid condition: {part!r}")
        sign = 1 if part[0] == "+" else -1
        attribute = part[1:].strip()
        conditions.append((sign, attribute))
    return conditions
```

Purpose: convert strings such as `+Black_Hair, -Wavy_Hair` into structured signed conditions.

Use human-readable prompt text by replacing underscores and optionally defining attribute-specific templates.

### Section 6: Baseline Fusion

```python
def baseline_fusion(ref_embedding, condition_embeddings, signs, alpha=1.0):
    signed_edit = sum(
        sign * embedding
        for sign, embedding in zip(signs, condition_embeddings)
    )
    query = ref_embedding + alpha * signed_edit
    return F.normalize(query, dim=-1)
```

Purpose: provide the required lower bound and validate the pipeline.

### Section 7: Retrieval

```python
def retrieve(query_embedding, candidate_embeddings, source_idx, max_k=10):
    scores = candidate_embeddings @ query_embedding
    scores = scores.clone()
    scores[source_idx] = -torch.inf
    values, indices = torch.topk(scores, k=max_k)
    return indices.tolist(), values.tolist()
```

Purpose: rank the fixed test image database and prevent trivial self-retrieval.

### Section 8: Single-Case Evaluation

```python
def evaluate_retrieval(retrieved, ground_truth, k):
    top_k = retrieved[:k]
    hits = set(top_k) & set(ground_truth)
    return {
        f"Recall@{k}": int(bool(hits)),
        f"Precision@{k}": len(hits) / k,
    }
```

Purpose: use the exact metric definition supplied by the instructors.

### Section 9: Full Benchmark Runner

```python
def evaluate_method(method, annotations, candidate_embeddings, ks=(1, 5, 10)):
    rows = []

    for query_item in annotations:
        query_text = query_item["query"]
        gt_by_source = query_item["ground_truth"]

        totals = {f"Recall@{k}": 0.0 for k in ks}
        totals.update({f"Precision@{k}": 0.0 for k in ks})

        for source_key, valid_targets in gt_by_source.items():
            source_idx = int(source_key)
            retrieved = method.retrieve(source_idx, query_text)

            for k in ks:
                result = evaluate_retrieval(retrieved, valid_targets, k)
                for name, value in result.items():
                    totals[name] += value

        count = len(gt_by_source)
        rows.append({
            "query": query_text,
            "sources": count,
            **{name: value / count for name, value in totals.items()},
        })

    return rows
```

Purpose: produce one result row per official query. In the actual implementation, batch sources for speed rather than scoring them one at a time.

### Section 10: Improved Fusion Module

A compact trained example could use signed condition tokens and dynamic gates:

```python
class SignedGatedFusion(torch.nn.Module):
    def __init__(self, dim=512, hidden_dim=256):
        super().__init__()
        self.gate = torch.nn.Sequential(
            torch.nn.Linear(dim * 2 + 1, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, 1),
            torch.nn.Sigmoid(),
        )
        self.output = torch.nn.Sequential(
            torch.nn.Linear(dim, dim),
            torch.nn.GELU(),
            torch.nn.Linear(dim, dim),
        )

    def forward(self, ref, condition_embeddings, signs, mask):
        batch, conditions, dim = condition_embeddings.shape
        ref_expanded = ref[:, None, :].expand(-1, conditions, -1)
        sign_feature = signs[..., None].float()

        gate_input = torch.cat(
            [ref_expanded, condition_embeddings, sign_feature], dim=-1
        )
        weights = self.gate(gate_input).squeeze(-1) * mask
        signed_conditions = condition_embeddings * signs[..., None]
        edit = (weights[..., None] * signed_conditions).sum(dim=1)

        fused = ref + self.output(edit)
        return F.normalize(fused, dim=-1), weights
```

Purpose:

- conditions no longer receive identical fixed weights,
- weights depend on both the reference and each condition,
- positive and negative signs are explicit,
- CLIP and the candidate database remain frozen,
- learned parameters stay small.

This is an example structure, not an instruction to use it unchanged. Your group should justify and test its own design.

## 14. Prompt Design

CLIP sees natural language, not raw attribute syntax. Convert attributes into readable prompts.

Examples:

```text
Smiling          -> "a photo of a smiling face"
Eyeglasses       -> "a photo of a face wearing eyeglasses"
Heavy_Makeup     -> "a photo of a face with heavy makeup"
Black_Hair       -> "a photo of a person with black hair"
Wearing_Hat      -> "a photo of a person wearing a hat"
```

For a negative condition, a clean baseline is to subtract the embedding of the positive concept:

```text
-Wavy_Hair -> subtract embedding("a person with wavy hair")
```

Also test explicit negative prompts:

```text
"a person without wavy hair"
```

Compare prompt strategies as an ablation instead of assuming one is best.

## 15. Efficiency Rules

- Encode candidate images once.
- Cache train/validation/test embeddings separately.
- Normalize embeddings once before cosine scoring.
- Encode each distinct textual condition once and cache it.
- Batch query scoring on GPU.
- Use top-K directly; do not sort all 19,962 scores when only top 10 are needed.
- Save result tables after every experiment.
- Store model/config identifiers with every cache to prevent mismatches.

### Where Working Outputs Should Live

Recommended Colab/Drive layout:

```text
Google Drive
└── MyDrive/
    ├── datasets/
    │   ├── celeba.zip
    │   └── celeba_evaluation.json
    └── dl_project/
        ├── project_notebook.ipynb
        ├── cache/
        │   ├── clip_train.pt
        │   ├── clip_valid.pt
        │   ├── clip_test.pt
        │   └── text_embeddings.pt
        ├── checkpoints/
        │   └── fusion_best.pt
        └── results/
            ├── baseline_metrics.csv
            ├── proposed_metrics.csv
            └── qualitative_examples/
```

Inside the temporary Colab runtime:

```text
/content/datasets/       # unzipped fast working copy; disappears on disconnect
```

The final required output is still the single notebook. Caches and intermediate result files support reproducibility and speed, but the notebook must explain how they are created and used.

## 16. Common Failure Modes

### Wrong Dataset Indexing

Symptom: retrieved examples seem unrelated despite correct JSON lookup.

Cause: treating dataset index `13` as filename `000013.jpg`.

Fix: always use `celeba_test[13]`.

### Self-Retrieval

Symptom: source image is always top 1.

Fix: set `scores[source_idx] = -inf`.

### Data Leakage

Symptom: excellent test results that do not generalize.

Cause: training or tuning directly on JSON targets.

Fix: train on train, tune on validation, lock choices before test.

### Recomputing CLIP Constantly

Symptom: evaluation takes hours.

Fix: cache frozen image and text embeddings.

### Optimizing Only Average Performance

Symptom: common single attributes look good while rare composed queries fail.

Fix: inspect every query separately.

### Confusing Attribute Similarity With Same Person Identity

Symptom: trying to force the same CelebA identity label.

Fix: follow the official JSON's non-query attribute Hamming definition.

### Reporting Only Qualitative Images

Symptom: results look convincing but are not measurable.

Fix: report all required Recall and Precision values.

## 17. Recommended Group Division

For three people:

| Person | Primary ownership | Shared review |
| --- | --- | --- |
| A | Data loading, JSON validation, metrics, visualization | Review retrieval correctness |
| B | CLIP extraction, caching, baseline, efficient retrieval | Review model/evaluation interfaces |
| C | Improved fusion, training data generation, experiments | Review report and methodology |

For two people:

| Person | Primary ownership |
| --- | --- |
| A | Data, evaluator, CLIP cache, baseline |
| B | Improved method, training, experiment/report pipeline |

Everyone must understand the full pipeline. Ownership means coordinating a section, not creating isolated code nobody else can debug.

Define shared interfaces early:

```text
method.retrieve(source_idx: int, query: str) -> list[int]
evaluate_method(method, annotations, candidate_embeddings) -> table
```

Any method implementing that interface can be evaluated identically.

## 18. First Group Session Plan For June 11, 2026

### First 30 Minutes

- Everyone reads Sections 1-9 of this guide.
- Open the assignment and starter notebook together.
- Agree that JSON indices always refer to `CelebA(split="test")`.
- Decide where the canonical notebook and caches will live.

### Next 60 Minutes

- Run the supplied skeleton without modification.
- Display source 13 and three valid `+Smiling` targets.
- Test Recall/Precision with artificial predictions.
- Inspect one composed query.

### Next 90 Minutes

- Install/load CLIP ViT-B/32.
- Implement batched image encoding.
- Extract a small sample first.
- Verify normalized embeddings and nearest neighbors.
- Start full test feature extraction and cache it.

### Final 60 Minutes

- Implement query parsing and prompt templates.
- Implement naive signed arithmetic baseline.
- Run one source/query end to end.
- Run a small benchmark subset.
- Record blockers and divide next tasks.

Goal for the end of the first session:

```text
one reference image + one official query
        -> CLIP/fusion
        -> top 10 test indices
        -> displayed result grid
        -> Recall@K and Precision@K
```

Do not try to train the improved model during the first session. First prove the complete baseline pipeline.

## 19. Experiment Tracking

For every run, record:

```text
experiment_id
date
git/notebook version
model_id
prompt template
fusion method
hyperparameters
training split and sampling
random seed
per-query metrics
macro metrics
runtime
notes/failures
```

Minimum comparison table:

| Method | Query | R@1 | R@5 | R@10 | P@1 | P@5 | P@10 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Image-only CLIP | ... | | | | | | |
| Signed arithmetic | ... | | | | | | |
| Proposed method | ... | | | | | | |

An image-only retrieval baseline is useful because it shows what happens when the text edit is ignored.

## 20. Final Notebook Structure

Recommended order:

1. Title, group members, abstract.
2. Problem definition.
3. Related work: CLIP, compositionality/GDE, CLAY.
4. Dataset and official benchmark.
5. Evaluation metrics.
6. Reproducibility configuration.
7. CLIP feature extraction and caching.
8. Image-only baseline.
9. Signed arithmetic baseline.
10. Proposed method.
11. Training procedure, if applicable.
12. Experimental setup.
13. Quantitative results.
14. Ablation studies.
15. Qualitative success/failure cases.
16. Limitations and ethical considerations.
17. Conclusion.
18. References.

The notebook must run from top to bottom in a fresh Colab runtime, apart from loading explicitly documented caches.

## 21. Final Deliverable Checklist

- [ ] One self-contained Google Colab notebook.
- [ ] Complete executable code.
- [ ] CLIP ViT-B/32 included.
- [ ] Frozen feature database explained.
- [ ] Vanilla baseline implemented.
- [ ] Original fusion method implemented and justified.
- [ ] Positive and negative conditions supported.
- [ ] Multiple simultaneous conditions supported.
- [ ] Source image excluded from retrieval.
- [ ] Official JSON used correctly.
- [ ] Recall@1/5/10 reported.
- [ ] Precision@1/5/10 reported.
- [ ] Per-query and aggregate tables included.
- [ ] Qualitative success and failure examples included.
- [ ] Training/validation/test separation respected.
- [ ] Architecture, forward pass, and loss described mathematically.
- [ ] Hyperparameters and sampling strategy documented.
- [ ] Ablations explain the contribution.
- [ ] Code is readable and modular.
- [ ] External snippets, if any, are cited.
- [ ] No third-party full project repository used.

## 22. Definition Of Project Success

At minimum, you should be able to demonstrate:

```text
source test index + signed textual conditions
        -> fused query representation
        -> ranked CelebA test indices
        -> official top-K metrics
```

A strong submission additionally demonstrates:

- a clear reason why the proposed fusion is better,
- measurable gains over image-only and arithmetic baselines,
- stronger behavior on composed queries,
- careful no-leakage experiments,
- interpretable ablations and failure analysis,
- a polished notebook that reads like a small research paper.

## 23. Immediate Next Action

On June 11, 2026, begin with the starter notebook and finish one complete baseline case before discussing a trained architecture. The first technical milestone is not "train a model". It is:

```text
load -> encode -> fuse -> rank -> display -> evaluate
```

Once that works reliably, every later method becomes a replaceable implementation of the fusion step.
