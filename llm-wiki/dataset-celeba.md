# CelebA Dataset

## Local Paths

- Dataset root: `celeba/`
- Aligned images: `celeba/img_align_celeba/`
- Original zip: `celeba.zip.download/celeba.zip`
- Identity labels: `celeba/identity_CelebA.txt`
- Attribute metadata: `celeba/list_attr_celeba.txt`
- Split metadata: `celeba/list_eval_partition.txt`
- Bounding boxes: `celeba/list_bbox_celeba.txt`
- Aligned landmarks: `celeba/list_landmarks_align_celeba.txt`
- Original-image landmarks: `celeba/list_landmarks_celeba.txt`
- Evaluation benchmark: `celeba_evaluation.json`

See also:

- [Practical Input and Output](practical-input-output.md)
- [Evaluation Protocol](evaluation-protocol.md)
- [Method Roadmap](method-roadmap.md)
- [Source Inventory](source-inventory.md)

## Image Corpus

There are 202599 aligned JPG images:

- First filenames: `000001.jpg`, `000002.jpg`, `000003.jpg`, `000004.jpg`, `000005.jpg`
- Last filenames: `202595.jpg`, `202596.jpg`, `202597.jpg`, `202598.jpg`, `202599.jpg`

## Attribute File

`celeba/list_attr_celeba.txt` has:

- line 1: number of images, `202599`
- line 2: 40 attribute names
- remaining lines: filename followed by 40 binary values

Important attributes for the provided queries include:

- `Smiling`
- `Eyeglasses`
- `Heavy_Makeup`
- `Male`
- `Young`
- `Blond_Hair`
- `Mustache`
- `Black_Hair`
- `Wavy_Hair`
- `Chubby`
- `Wearing_Hat`
- `Wearing_Lipstick`

The attribute values are binary:

- `1`: the attribute is present
- `-1`: the attribute is absent

## Identity File

`celeba/identity_CelebA.txt` maps each image filename to a person ID:

```text
000001.jpg 2880
000002.jpg 2937
```

This is the key file for finding more images of the same person. Two images belong to the same identity if they share the same identity ID.

Useful local statistics from the current dataset copy:

- total images: `202599`
- total identities: `10177`
- identities with at least 2 images: `10133`
- identities with at least 3 images: `9809`
- identities with at least 5 images: `9343`

The official partitions are identity-disjoint:

- train identities: `8192`
- validation identities: `985`
- test identities: `1000`
- identity overlap between any two splits: `0`

This makes it possible to train on same-identity pairs without exposing test identities during training.

## Split File

`celeba/list_eval_partition.txt` maps each image to its split:

- `0`: train
- `1`: validation
- `2`: test

Current counts:

- train: `162770`
- validation: `19867`
- test: `19962`

## How The Files Work Together

Join the metadata files by image filename:

- `identity_CelebA.txt`: image -> identity
- `list_attr_celeba.txt`: image -> 40 attributes
- `list_eval_partition.txt`: image -> split

The filename is the primary join key. It is different from both the person identity ID and the PyTorch split-local dataset index.

### Complete File Formats And Planned Uses

| File | Row shape | Planned use |
| --- | --- | --- |
| `img_align_celeba/<filename>.jpg` | RGB pixels | CLIP inputs, retrieval gallery, qualitative figures |
| `identity_CelebA.txt` | `filename person_id` | Group same-person images for preservation-focused training tuples |
| `list_attr_celeba.txt` | `filename` + 40 values in `{-1,+1}` | Derive signed conditions, filter pairs, reproduce benchmark-style similarity |
| `list_eval_partition.txt` | `filename split_id` | Enforce train/validation/test separation and map test indices to filenames |
| `list_bbox_celeba.txt` | `filename x y width height` | Optional original-face crop and quality analysis |
| `list_landmarks_align_celeba.txt` | `filename` + 5 aligned `(x,y)` points | Optional alignment/pose analysis |
| `list_landmarks_celeba.txt` | `filename` + 5 original `(x,y)` points | Optional original-coordinate geometry analysis |
| `celeba_evaluation.json` | query -> source test index -> valid target test indices | Official test evaluation only |

Core training join:

```text
filename
  + person_id
  + split_id
  + 40 attributes
  + image path or cached CLIP embedding
```

Example conceptual record:

```python
{
    "filename": "000001.jpg",
    "person_id": 2880,
    "split": 0,
    "attributes": {
        "Smiling": 1,
        "Eyeglasses": -1,
        "Young": 1,
    },
    "image_path": "celeba/img_align_celeba/000001.jpg",
}
```

Use only the core join for the first model. Bounding boxes and landmarks should remain optional until an experiment demonstrates a need for them.

After that join, group rows by identity. Within each identity group, look for pairs where one chosen attribute flips value.

Example for `Eyeglasses`:

- source image: same identity, `Eyeglasses = -1`
- target image: same identity, `Eyeglasses = 1`
- edit text: `add eyeglasses`

This gives an automatic training tuple:

```text
(reference image, text edit, target image)
```

After frozen CLIP encoding, that tuple becomes:

```text
(source embedding, signed text embedding(s), target embedding)
```

Only source and text embeddings enter the composition network. The target embedding is supervision for the loss.

The same logic can be used for edits such as:

- `add smiling`
- `remove smiling`
- `add bangs`
- `remove eyeglasses`
- `change hair color`

## Supervision For Training

CelebA does not directly ship with natural-language triplets, but it does provide enough structure to derive them automatically:

- identity tells you what should stay the same
- attributes tell you what should change
- partition tells you whether the pair belongs to train, validation, or test

This means the project can train a lightweight composition module without manually labeling examples.

## Caveat About Pair Quality

CelebA pairs are useful but noisy. Two images of the same person may differ in more than one attribute at once, for example:

- pose
- lighting
- makeup
- expression
- hairstyle

To build cleaner pairs, it is helpful to keep only candidate pairs where:

- the chosen target attribute flips
- the number of other attribute differences is small

### Pair Counts In The Training Split

The training split contains:

- `1,861,685` unordered same-identity pairs
- `3,723,370` directional same-identity pairs

The median same-identity pair differs in 6 attributes. Counts for cleaner unordered pairs are:

- Hamming distance <= 1: `52,566`
- Hamming distance <= 3: `333,769`
- Hamming distance <= 5: `855,128`

Therefore, creating every possible pair is unnecessarily large and noisy. Prefer filtering plus balanced sampling. The full proposed procedure is documented in [Proposed Training Strategy](training-strategy.md).

### Pair Counts Across The Complete Identity File

The read-only script `scripts/count_identity_pairs.py` groups all `202599` image rows by person ID and computes `n choose 2` for every identity.

Verified totals on 2026-06-11:

- person IDs: `10177`
- person IDs with at least two images: `10133`
- unique unordered same-person pairs: `2320695`
- directional source-to-target pairs: `4641390`

Run the summary:

```bash
python3 scripts/count_identity_pairs.py
```

Print every person ID:

```bash
python3 scripts/count_identity_pairs.py --per-identity
```

The complete-dataset total is not the training total. Training must remain inside partition `0`, which contains `1861685` unique unordered same-identity pairs.

## Closed Attributes vs Open-Ended Edits

The safest supervised setup is to use the listed CelebA attributes as the allowed edits.

Examples:

- `Eyeglasses`
- `Smiling`
- `Bangs`
- `Black_Hair`
- `Young`

Edits outside the listed 40 attributes may still work qualitatively with CLIP-style text embeddings, but they are not directly supervised by CelebA metadata. Those should be treated as optional generalization experiments rather than the main benchmark.

## Critical Indexing Rule

The evaluation JSON uses PyTorch CelebA dataset indices, not image filenames.

Correct:

```python
image, label = celeba[int(source_key)]
```

Incorrect:

```python
Image.open("celeba/img_align_celeba/000013.jpg")
```

Why: the test split is a subset of the full dataset. Dataset index `13` can point to a different physical filename, for example `182651.jpg` in the assignment example.

## Practical Loading Pattern

```python
from pathlib import Path
from torchvision.datasets import CelebA

data_root = Path("/content/datasets")
celeba = CelebA(root=data_root, split="test", download=False)
```

The skeleton notebook expects `len(celeba) == 19962` for the test split.

## Relationship To Retrieval

For each source image in the JSON, the model must search within the CelebA test split and return top-ranked target indices. A valid target must:

- satisfy all positive query attributes,
- violate all negative query attributes,
- stay close to the reference on remaining attributes, using relaxed Hamming distance <= 2 as precomputed in the JSON.
