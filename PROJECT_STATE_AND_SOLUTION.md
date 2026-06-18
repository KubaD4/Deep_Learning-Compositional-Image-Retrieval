# Project State, Task, and Proposed Solution

Status snapshot: June 13, 2026.

This document is the current authoritative summary of the project. It explains the assignment, data, benchmark, tools, work completed so far, selected solution proposal, implementation plan, evaluation, risks, and remaining work. A reader should be able to understand the complete project without first reading the original PDFs.

For a longer implementation tutorial and code skeleton, see [PROJECT_GUIDE.md](PROJECT_GUIDE.md).
For the proposed repository tree and exact per-batch/per-epoch execution flow, see [PROJECT_FOLDER_AND_TRAINING_SCHEMA.md](PROJECT_FOLDER_AND_TRAINING_SCHEMA.md).

## 1. Executive Summary

The project is a **compositional image retrieval** system for CelebA faces.

For each task, the system receives:

1. a reference image from the CelebA test split;
2. one to three signed textual conditions such as `+Smiling`, `-Young`, or `+Black_Hair, -Wavy_Hair`.

It must return a ranked list of **existing CelebA test images** that:

1. satisfy every requested positive and negative condition;
2. preserve the source image's other annotated characteristics as closely as possible.

The project does not generate or edit pixels. It predicts a retrieval-query embedding and ranks a fixed image gallery.

The required backbone is Hugging Face CLIP ViT-B/32:

```text
openai/clip-vit-base-patch32
```

The group currently proposes to freeze CLIP and train a small **signed gated residual composition network**. The network combines the source image embedding with one or more signed text-condition embeddings and predicts a residual movement in CLIP space. The resulting vector is compared with cached CelebA image embeddings using cosine similarity.

This proposed model has been designed and documented, but it has **not yet been implemented or trained**.

The frozen-CLIP baseline pipeline is now implemented and has been run end to
end on the official JSON. Five arithmetic methods have quantitative results;
the strongest macro method is Contrastive Sequential with Macro Recall@10
`0.18708`.

## 2. Current Project Status

### Completed

- Original assignment, slides, research papers, notebook, dataset metadata, images, and benchmark JSON have been inspected.
- Microsoft MarkItDown is installed locally and all supplied PDFs plus the starter notebook have Markdown derivatives.
- A maintained LLM wiki and detailed project guide have been created.
- CelebA split sizes, identity counts, and identity-disjoint partitions have been verified.
- The official JSON structure, queries, dataset-index mapping, and metric definitions have been verified.
- A local ground-truth dashboard has been implemented.
- The dashboard correctly maps PyTorch test indices to physical filenames, displays source/target attributes and identities, checks query constraints, and computes non-query Hamming distance.
- All four dashboard unit tests pass.
- A training-based solution using frozen CLIP, derived training tuples, dynamic condition fusion, residual prediction, and contrastive loss has been formally proposed.
- A git repository and GitHub remote exist on branch `main`.
- CelebA annotation files are committed under `data/celeba/annotations/`.
- A self-contained Steps 1-3 notebook and cluster execution bundle exist.
- The complete 19,962-image test embedding cache and 40-attribute text cache
  have been generated with Hugging Face CLIP ViT-B/32.
- Resumable Slurm extraction, embedding, baseline, and merge jobs are available.
- Four initial arithmetic baselines and one adaptive tangent experiment have
  been evaluated over all 14 official queries and 33,052 source-query cases.
- Per-query metrics, summaries, retrieval records, comparison CSVs, and plots
  have been generated.
- A local CLIP image-text diagnostic supports single prompts, binary prompt
  comparisons, and repeated positive/negative prompt-pair ensembles.

### Not Yet Completed

- Train and validation CLIP image caches have not yet been confirmed complete.
- The image-only CLIP retrieval baseline still needs to be reported alongside
  the completed arithmetic baselines.
- No training tuple generator or sampler has been implemented.
- No composition network has been implemented or trained.
- No learned-model checkpoints, learning curves, or trained-model results exist.
- Formal prompt calibration, confidence intervals, and learned-model ablations
  have not yet been run.
- The final consolidated report notebook is not yet complete.

Therefore, the project has completed the frozen-feature and arithmetic-baseline
stage and is ready for training-data generation and learned-model development.

## 3. Exact Task

For one evaluation case:

```text
input:
    source test image I_s
    signed query Q = {+attribute, -attribute, ...}
    candidate gallery = all 19,962 CelebA test images

output:
    ranked test dataset indices [i_1, i_2, ..., i_N]
```

Example:

```text
source: CelebA test dataset index 13
query: +Smiling

desired result:
faces with attributes close to source 13, but with Smiling present

model output example:
[579, 325, 456, 981, ...]
```

The integers are indices into `torchvision.datasets.CelebA(split="test")`, not image filenames.

## 4. What "Similar" Means

There are two distinct concepts.

### Model Similarity

The model produces a numerical score for each gallery image. With normalized CLIP vectors, this is normally cosine similarity:

```text
score(I_i) = z_i dot q
```

where:

- `z_i` is the cached CLIP embedding of candidate image `I_i`;
- `q` is the source-and-condition composition produced by the proposed method.

### Official Correctness

The model is graded against `celeba_evaluation.json`, not by manually judging cosine scores.

A target is officially valid when:

1. every positive/negative query condition is satisfied exactly;
2. among all non-query CelebA attributes, at most two differ from the source.

This is an attribute-based relaxed Hamming-distance rule.

Important: official ground truth does **not** require the same real person identity. Same-identity information is useful training supervision, but the benchmark's operational definition of preservation uses the 40 CelebA attributes.

## 5. Verified Concrete Example

For the official `+Smiling` query:

```text
source test index: 13
source filename: 182651.jpg
source Smiling: -1
```

One valid target is:

```text
target test index: 579
target filename: 183217.jpg
target Smiling: +1
```

Apart from the requested `Smiling` change, this target differs from the source only in `High_Cheekbones`. It therefore satisfies the query and remains within the maximum non-query Hamming distance of two.

This also demonstrates the indexing rule:

```text
dataset index 13 != filename 000013.jpg
dataset index 13 == filename 182651.jpg in the test split
```

Always load benchmark images through the PyTorch dataset object.

## 6. Supplied Material

### Assignment And Starter Code

| File | Purpose |
| --- | --- |
| `Project assignment - V1.2.pdf` | Canonical assignment specification |
| `Project - Introduction-2.pdf` | Instructor overview of background, task, evaluation, and deliverables |
| `Project Skeleton.ipynb` | Starter Colab for dataset loading and evaluation examples |
| `celeba_evaluation.json` | Official queries, valid source indices, and valid target indices |
| `VM usage.pdf` | Instructions for the provided Azure GPU machine |

### Research Background

| File | Purpose |
| --- | --- |
| `CLIP.pdf` | Required vision-language embedding backbone |
| `CLAY.pdf` | Conditional similarity method whose rigid multi-condition fusion motivates the assignment |
| `CLIP Compositionality - Davide Berasi.pdf` | Compositional structure in visual-language embeddings |
| `Compositionality.pdf` | Course explanation of compositionality and CLIP geometry |
| `Lab 2 - CNNs.pdf` | Course lab context and starter-code introduction |

### Dataset

The local extracted dataset is under `celeba/`:

```text
celeba/
├── img_align_celeba/                 # 202,599 aligned JPG images
├── identity_CelebA.txt               # filename -> person identity ID
├── list_attr_celeba.txt              # filename -> 40 binary attributes
├── list_eval_partition.txt           # filename -> train/valid/test
├── list_bbox_celeba.txt              # face bounding boxes
├── list_landmarks_align_celeba.txt   # aligned landmarks
└── list_landmarks_celeba.txt         # original-image landmarks
```

The git repository contains a shareable metadata-only copy under:

```text
data/celeba/annotations/
```

The image directory is intentionally represented by `data/celeba/images/.gitkeep`; the 202,599 JPG files are not committed.

### The Three Identifiers You Must Distinguish

The data uses three different identifiers:

| Identifier | Example | Meaning | Where it is used |
| --- | --- | --- | --- |
| Image filename | `000001.jpg` | Globally identifies one physical CelebA photograph | Joins all CelebA metadata files and locates the JPG |
| Person identity ID | `2880` | Identifies the person shown in one or more photographs | Groups same-person images for proposed training tuples |
| PyTorch split index | `13` | Position inside a loaded dataset split such as `CelebA(split="test")` | Used by `celeba_evaluation.json` and retrieval outputs |

These values are not interchangeable. In particular, test dataset index `13` maps to physical filename `182651.jpg`, not `000013.jpg`.

### Detailed Data Dictionary

#### `img_align_celeba/`: Physical Image Data

Each file is one aligned RGB face photograph:

```text
000001.jpg
000002.jpg
...
202599.jpg
```

How we use it:

- load the actual source, target, and candidate images;
- preprocess images with the CLIP processor;
- extract and cache one frozen CLIP image embedding per image;
- display qualitative retrieval results.

The model does not use the filename number as a semantic value. The filename is only a stable join key and file locator.

#### `identity_CelebA.txt`: Image Filename + Person ID

Row format:

```text
000001.jpg 2880
000002.jpg 2937
000003.jpg 8692
```

Meaning:

```text
image filename -> real-person identity group
```

If two rows have the same second value, CelebA considers them photographs of the same person.

How we want to use it:

- group training images by person ID;
- select a source and target photograph of the same person;
- use their attribute differences to derive a textual edit;
- teach the composition network to change requested attributes while preserving person-specific visual information.

This file is proposed training supervision. It is not part of the official correctness rule in `celeba_evaluation.json`.

#### `list_attr_celeba.txt`: Image Filename + 40 Binary Attributes

The first line is the image count. The second line contains 40 column names. Every later row contains one filename and 40 values:

```text
202599
5_o_Clock_Shadow Arched_Eyebrows ... Smiling ... Young
000001.jpg -1 1 1 ... 1 ... 1
```

Meaning:

- `1`: attribute present;
- `-1`: attribute absent.

Conceptually:

```text
image filename -> 40-dimensional binary semantic description
```

How we use it:

- inspect the semantic content of each source and target;
- derive signed edits from a source/target pair;
- build positive and negative text conditions;
- filter noisy pairs by counting non-query attribute differences;
- generate benchmark-style training and validation targets;
- validate dashboard query conditions and Hamming distances.

For source vector `a_s` and target vector `a_t`:

```text
d = (a_t - a_s) / 2
```

Then each component of `d` is:

- `+1`: add the attribute;
- `-1`: remove the attribute;
- `0`: leave it unchanged.

Example:

```text
source Eyeglasses = -1
target Eyeglasses = +1
derived query = +Eyeglasses
```

#### `list_eval_partition.txt`: Image Filename + Split ID

Row format:

```text
000001.jpg 0
000002.jpg 0
...
```

Meaning:

- `0`: training split;
- `1`: validation split;
- `2`: test split.

How we use it:

- ensure training tuples contain only partition `0` images;
- ensure hyperparameter selection uses only partition `1`;
- reserve partition `2` for the official retrieval benchmark;
- build the ordered test-filename list that maps JSON/PyTorch indices to physical files.

This file prevents data leakage. The identity sets are disjoint across the three official partitions.

#### `list_bbox_celeba.txt`: Image Filename + Face Bounding Box

Header and row format:

```text
image_id x_1 y_1 width height
000001.jpg 95 71 226 313
```

Meaning: top-left face position and bounding-box size in the original image coordinates.

Expected use:

- optional visualization or alternative face cropping;
- optional quality checks.

It is not required for the main approach because `img_align_celeba/` already contains aligned faces and the CLIP processor performs its own resize/crop preprocessing.

#### `list_landmarks_align_celeba.txt`: Image Filename + Five Aligned Landmarks

Row contents:

```text
filename
left eye (x,y)
right eye (x,y)
nose (x,y)
left mouth corner (x,y)
right mouth corner (x,y)
```

The coordinates refer to aligned images.

Expected use:

- optional pose/alignment analysis;
- optional visualization or pair-quality filtering.

They are not currently part of the proposed baseline or composition network.

#### `list_landmarks_celeba.txt`: Image Filename + Five Original-Image Landmarks

It has the same ten coordinate values as the aligned-landmark file, but in original-image coordinates.

Expected use: optional original-image geometry analysis. It is not needed for the planned CLIP retrieval pipeline.

#### `celeba_evaluation.json`: Query + Test Dataset Index Mapping

Conceptual structure:

```text
query string
    -> source test dataset index
        -> list of valid target test dataset indices
```

Example:

```python
{
    "query": "+Smiling",
    "ground_truth": {
        "13": [325, 456, 579, 685]
    }
}
```

How we use it:

- obtain the official source/query cases;
- evaluate ranked model outputs;
- compute Recall@K and Precision@K;
- inspect valid targets with the dashboard.

How we do **not** use it:

- not for model training;
- not for selecting model hyperparameters;
- not as a list of filenames;
- not as a ranking, because JSON target order has no model-score meaning.

### How The Metadata Files Are Joined

The common key across CelebA text files is the image filename:

```text
identity_CelebA.txt          000001.jpg -> person 2880
list_attr_celeba.txt         000001.jpg -> 40 attributes
list_eval_partition.txt      000001.jpg -> train split
list_bbox_celeba.txt         000001.jpg -> bounding box
landmark files               000001.jpg -> facial keypoints
img_align_celeba/            000001.jpg -> actual pixels
```

After joining, one conceptual row becomes:

```text
filename:       000001.jpg
person_id:      2880
split:          train
attributes:     {Smiling: +1, Eyeglasses: -1, ...}
image_path:     celeba/img_align_celeba/000001.jpg
bounding_box:   [95, 71, 226, 313]
landmarks:      {...}
```

The core training table only needs:

```text
filename + person_id + split + 40 attributes + image/embedding reference
```

Bounding boxes and landmarks remain optional metadata.

### How We Want To Convert Joined Rows Into Training Examples

Suppose two train rows share `person_id = 2880`:

```text
source:
    filename = image_A.jpg
    Eyeglasses = -1
    Smiling = -1

target:
    filename = image_B.jpg
    Eyeglasses = +1
    Smiling = -1
```

We derive:

```text
source image: image_A.jpg
condition: +Eyeglasses
target image: image_B.jpg
```

CLIP converts the images and text into embeddings:

```text
image_A.jpg -> z_source
"a face wearing eyeglasses" -> c_text
image_B.jpg -> z_target
```

The composition network receives `z_source` and `c_text`, predicts query vector `q`, and training encourages `q` to rank `z_target` highly.

For multiple changes, the condition list might be:

```text
[+Eyeglasses, +Smiling, -Heavy_Makeup]
```

We keep examples only when unrequested attribute differences remain small. We also plan a cross-identity benchmark-style sampler because the official JSON does not require same-person targets.

### Which Files Enter Each Project Phase

| Phase | Files used | Purpose |
| --- | --- | --- |
| Data loading | images, attributes, partition file | Build train/valid/test datasets |
| CLIP feature cache | aligned images, partition file | Produce frozen image embeddings per split |
| Same-identity training tuples | identity + attributes + partitions | Preserve person while deriving signed edits |
| Benchmark-style tuples | attributes + partitions | Match official attribute/Hamming rule |
| Optional quality analysis | bounding boxes + landmarks | Inspect crop, pose, and alignment effects |
| Official test evaluation | test dataset + evaluation JSON | Rank test images and compute required metrics |
| Qualitative reporting | aligned images + attributes + identities | Display successes, failures, and same-person cases |

## 7. Verified Dataset Statistics

| Item | Count |
| --- | ---: |
| Images | 202,599 |
| Binary attributes | 40 |
| Identities | 10,177 |
| Train images | 162,770 |
| Validation images | 19,867 |
| Test images | 19,962 |
| Train identities | 8,192 |
| Validation identities | 985 |
| Test identities | 1,000 |

The identity overlap between train, validation, and test is zero.

The train split contains:

- 1,861,685 unordered same-identity image pairs;
- 3,723,370 directional same-identity pairs.

Across the complete CelebA dataset, before respecting split boundaries, the identity file yields:

- 2,320,695 unique unordered same-identity pairs;
- 4,641,390 directional source-to-target pairs.

These complete-dataset totals are descriptive only. Model training must use train-only pairs to avoid mixing validation/test identities into training. The reproducible read-only calculation is implemented in `scripts/count_identity_pairs.py`.

Because the full pair set is large and noisy, training should use filtering and balanced sampling rather than materializing every possible pair.

## 8. Official Benchmark

The JSON contains 14 query entries and 33,052 source-query cases.

| Index | Query | Sources |
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

`-Young` appears twice with identical ground truth. The actual JSON is authoritative unless the instructors provide a corrected version.

Simplified JSON structure:

```python
{
    "query": "+Smiling",
    "ground_truth": {
        "13": [325, 456, 579, ...]
    }
}
```

- `"13"` is the source test dataset index, stored as a string.
- `[325, 456, 579, ...]` contains acceptable target test dataset indices.
- JSON target order is not a model ranking.

## 9. Evaluation Metrics

For top-K retrieved indices `R_K` and valid targets `G`:

```text
Recall@K = 1 if R_K intersects G, otherwise 0
Precision@K = size(R_K intersect G) / K
```

Report both metrics at K = 1, 5, and 10, averaged across all valid sources for each query.

Recall@K is the primary metric. The assignment's Recall is a hit rate, not the fraction of all valid targets recovered.

The source image itself must be excluded from retrieval.

## 10. What The Starter Notebook Provides

`Project Skeleton.ipynb` currently:

- mounts Google Drive;
- unzips CelebA into the Colab runtime;
- loads `CelebA(split="test")`;
- verifies the test length is 19,962;
- defines single-case Recall@K and Precision@K;
- loads `celeba_evaluation.json`;
- explains the JSON structure;
- displays source index 13 and valid `+Smiling` targets;
- creates attribute-name/index dictionaries.

It does not contain CLIP, feature extraction, retrieval, training, or the proposed method.

## 11. Ground-Truth Dashboard Reached So Far

An implemented local dashboard exists in `ground_truth_dashboard/`.

Start it with:

```bash
./ground_truth_dashboard/run_dashboard.sh
```

Then open:

```text
http://127.0.0.1:8876
```

It provides:

- all 14 benchmark queries;
- source navigation and direct source-index search;
- correct dataset-index-to-filename mapping;
- fixed source image plus all valid targets;
- query-constraint checks;
- active attributes and all 40 attribute values;
- non-query Hamming distance;
- person identity IDs;
- same-identity target counts and navigation.

The dashboard is a data/benchmark inspection tool. It does not run a retrieval model.

Verification on June 11, 2026:

```text
4 dashboard unit tests passed
```

## 12. Selected Proposed Solution

### Design Goal

Replace naive equal-weight text arithmetic and CLAY's rigid pre-SVD stacking with a small module that dynamically determines how each signed condition should modify the source representation.

### Frozen CLIP Backbone

Use CLIP only as a fixed coordinate system:

```text
source image -> frozen CLIP image encoder -> z_s in R^512
condition text -> frozen CLIP text encoder -> c_j in R^512
gallery image -> frozen CLIP image encoder -> z_i in R^512
```

All final embeddings are L2-normalized.

### Signed Gated Residual Composition

For condition embeddings `c_1 ... c_m`:

```text
alpha_j = gate(z_s, c_j, sign_j)
c       = aggregate(alpha_j * signed(c_j))
Delta_z = MLP([z_s, c])
q       = normalize(z_s + Delta_z)
```

The residual form begins from the source embedding, helping preserve source information. Dynamic gates allow different conditions to receive different weights depending on the source and query.

### Retrieval

Precompute the test gallery matrix:

```text
G in R^(19962 x 512)
```

For each query:

```text
scores = G @ q
scores[source_index] = -infinity
top_indices = topk(scores)
```

Compare `top_indices` with the JSON ground truth.

## 13. Proposed Training Data

### Same-Identity Tuples

Join identity, attribute, and partition files by filename. Within each train identity, construct directional pairs:

```text
(source image, signed attribute edits, target image)
```

For source and target attribute vectors `a_s` and `a_t`:

```text
d = (a_t - a_s) / 2
```

Each value is:

- `+1`: add attribute;
- `-1`: remove attribute;
- `0`: unchanged.

Select one to three changed attributes as explicit query conditions. Keep examples with at most two additional non-query differences so training resembles the benchmark tolerance.

### Benchmark-Style Tuples

Same-identity pairs teach visual/identity preservation but do not exactly match official evaluation, which allows different people. A second sampler should create cross-identity pairs using the same attribute rule as the benchmark.

### Recommended Hybrid Sampler

Mix:

1. same-identity pairs for strong preservation supervision;
2. benchmark-style attribute pairs for distribution alignment.

Balance:

- positive and negative directions;
- query lengths one, two, and three;
- official benchmark attributes;
- identities, so people with many photos do not dominate.

## 14. Proposed Loss

For batch size `B`, let predicted query embeddings be `q_i` and target embeddings be `z_ti`:

```text
S_ij = cosine(q_i, z_tj) / temperature
L_contrastive = CrossEntropy(S, diagonal targets)
```

Optional stabilization:

```text
L_cos = 1 - cosine(q_i, z_ti)
L = L_contrastive + lambda * L_cos
```

The contrastive loss is primary because the final task is ranking.

A later improvement can use a multi-positive loss because another target in the same batch may also be valid, making ordinary InfoNCE treat a valid image as a false negative.

## 15. Required Baselines

Before training the proposed model, implement:

### Image-Only CLIP

```text
q = z_s
```

This measures retrieval when the textual edit is ignored.

### Zero-Shot Signed Arithmetic

```text
q = normalize(z_s
              + alpha * sum(positive text embeddings)
              - beta  * sum(negative text embeddings))
```

This is the required practical lower bound and verifies the full evaluation pipeline.

The proposed learned model must be compared against both baselines.

## 16. Train, Validation, And Test Policy

```text
train split:
    build tuples and optimize composition-network parameters

validation split:
    tune prompts, sampler, architecture, alpha/beta, temperature,
    learning rate, and stopping point

official test JSON:
    final evaluation only after decisions are fixed
```

Never train on `celeba_evaluation.json` target lists. Doing so leaks official answers.

CLIP should remain frozen for the first experiments. Only the small composition module is trained.

## 17. Implementation Order

1. Re-run and verify the supplied skeleton.
2. Load frozen CLIP ViT-B/32.
3. Extract and cache normalized test image embeddings.
4. Implement prompt templates and text-embedding cache.
5. Implement image-only retrieval.
6. Implement zero-shot signed arithmetic retrieval.
7. Build a batched full-benchmark evaluator.
8. Save baseline per-query and macro metrics.
9. Implement train/validation tuple generation.
10. Implement same-identity sampler, then benchmark-style/hybrid sampling.
11. Implement signed gated residual composition network.
12. Train on train and choose settings on validation.
13. Evaluate once on the official JSON.
14. Run ablations and produce qualitative success/failure grids.
15. Integrate all code and report text into the final Colab notebook.

## 18. Experiments To Run

Minimum experiment matrix:

| Experiment | Purpose |
| --- | --- |
| Image-only CLIP | Tests source preservation without text |
| Zero-shot signed arithmetic | Required lower-bound composition baseline |
| Learned same-identity residual model | Tests identity-focused supervision |
| Learned benchmark-style model | Tests alignment with official attribute rule |
| Hybrid learned model | Combines preservation and benchmark alignment |

Recommended ablations:

- fixed averaging versus dynamic gates;
- residual connection on/off;
- explicit negative prompts versus vector subtraction;
- one versus two versus three conditions;
- single-positive versus multi-positive contrastive loss;
- same-identity versus benchmark-style versus hybrid sampler;
- prompt-template variants.

### Completed Arithmetic Results

All 14 official query entries and 33,052 source-query cases were evaluated with
the cached test gallery.

| Method | Macro R@1 | Macro R@5 | Macro R@10 | Macro P@10 |
| --- | ---: | ---: | ---: | ---: |
| Direct Sum | 0.02397 | 0.07183 | 0.10842 | 0.01471 |
| Direct Sequential | 0.02807 | 0.07327 | 0.11206 | 0.01530 |
| Contrastive Sum | 0.03912 | 0.12039 | 0.17074 | 0.02474 |
| **Contrastive Sequential** | **0.04064** | **0.12381** | **0.18708** | **0.02698** |
| Adaptive Tangent Sequential | 0.04061 | 0.12353 | 0.18693 | 0.02687 |

Contrastive Sequential improves Macro Recall@10 by 72.54% relative to Direct
Sum. Adaptive Tangent Sequential is effectively tied but does not improve macro
performance. Full micro/per-query results and prompt experiments are in
`llm-wiki/cluster-baselines-and-prompt-experiments.md`.

## 19. Main Risks

### Training/Evaluation Mismatch

Same-identity supervision is not identical to the attribute-based benchmark. Address this with benchmark-style and hybrid sampling.

### Noisy Attribute Labels

Attributes such as `Male`, `Young`, and `Attractive` may vary because of annotation noise. Analyze rather than blindly treating all differences as real edits.

### Negation In CLIP

CLIP may not represent `without X` reliably. Compare explicit negative prompts, signed vector subtraction, and learned sign embeddings.

### False Negatives In Contrastive Batches

Multiple targets can be valid for the same query. Ordinary diagonal InfoNCE can penalize valid alternatives. Consider a multi-positive mask.

### Rare Composed Queries

Some official queries have only 27, 34, or 79 valid sources. Report per-query results and avoid hiding failure behind macro averages.

### Computational Cost

Do not repeatedly encode all images. Cache frozen embeddings. The complete 512-dimensional image cache is approximately 396 MiB in float32 or 198 MiB in float16.

## 20. Artifact Status

```text
completed:
    Steps 1-3 project notebook
    CLIP test image and attribute text caches
    baseline metrics tables and retrieval records
    per-query comparison plots and numeric comparison CSVs

still required:
    confirmed train/validation CLIP caches
    training/validation tuple indices
    trained model checkpoints
    proposed learned-method metrics tables
    learned-model ablation tables
    qualitative learned-model retrieval figures
    learning curves
    final report text inside one self-contained notebook
```

## 21. Repository And Knowledge State

Git:

```text
branch: main
remote: GitHub origin configured
commits currently present: 2
latest commit: Add CelebA annotation metadata
```

Committed project content currently includes:

- README and starter notebook;
- official evaluation JSON;
- CelebA annotation metadata under `data/celeba/annotations/`;
- image-directory placeholder.

At the time of this snapshot, the guides, wiki, PDFs, dashboard, and local MarkItDown setup are present locally but have not all been committed. Before group development diverges, decide which non-wiki documentation and dashboard files belong in the shared repository. Large raw images, virtual environments, caches, and cloned tool repositories should remain untracked.

`llm-wiki/` is deliberately ignored by git and must remain local-only for now. It is the private persistent knowledge layer for local coding-agent sessions, not part of the GitHub repository. Do not force-add or publish it unless the user explicitly changes this policy later.

## 22. Final Deliverable

The assignment requires one self-contained Google Colab notebook containing:

- complete executable code;
- methodological description and equations;
- training and evaluation strategy;
- justified hyperparameters and data sampling;
- baseline and proposed-method results;
- Recall@1/5/10 and Precision@1/5/10;
- comparative tables;
- learning curves if training is used;
- qualitative successes and failures;
- discussion, limitations, conclusion, and citations.

The notebook should read like a small scientific report interwoven with reproducible code.

## 23. Definition Of The Next Milestone

The baseline path now works end to end. The next milestone is the first trained
composition model:

```text
generate train/validation caches and tuple manifests
    -> validate prompt-margin quality on validation attributes
    -> implement signed condition gate and residual MLP
    -> train with frozen CLIP target embeddings
    -> choose settings on validation only
    -> compare against Contrastive Sequential
    -> report confidence intervals and qualitative failures
```

Contrastive Sequential, not Direct Sum, is now the practical baseline that the
learned proposal should aim to beat, while Direct Sum remains the required
assignment lower bound.
