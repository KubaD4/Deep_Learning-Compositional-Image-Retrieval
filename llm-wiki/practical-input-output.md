# Practical Input and Output

## One Retrieval Example

The system receives:

```text
source_idx = 13
query = "+Smiling"
```

This means:

- Load source image `13` from the PyTorch CelebA test dataset with `celeba[13]`.
- Search the CelebA test images.
- Return a ranked list of dataset indices that look like the source image but are smiling.

Example output shape:

```python
[325, 456, 579, 981, 1363, 1646, 2142, 2747, 3318, 3536]
```

Those numbers are not filenames. They are CelebA test dataset indices. To display the first retrieved image:

```python
image, attrs = celeba[325]
display(image)
```

## Where Inputs Come From

There are three practical input sources:

| Input | Where | Meaning |
| --- | --- | --- |
| Source image index | `celeba_evaluation.json` keys | Which image starts the retrieval task. |
| Text query | `celeba_evaluation.json["query"]` | The requested attribute edit, such as `+Eyeglasses` or `-Young`. |
| Image database | `celeba` test split | The pool of candidate target images to rank. |

## Where Outputs Go

During experiments, outputs usually live inside the final notebook:

- retrieval lists,
- metrics tables,
- qualitative image grids,
- plots,
- Markdown explanation.

Optional caches can go to Google Drive or local files, for example:

- cached CLIP image embeddings,
- per-method CSV/JSON metric tables,
- saved qualitative result figures.

The final required output is a single self-contained Colab notebook.

## What The Model Actually Computes

The model does not directly create new images. It retrieves existing CelebA images.

Typical flow:

```text
source image + text condition
        ↓
CLIP image/text embeddings
        ↓
your fusion/scoring method
        ↓
similarity score for every candidate image
        ↓
sorted target indices
        ↓
Recall@K and Precision@K evaluation
```

## Checking If Results Are Similar

To check a retrieved result, compare its dataset index against the ground-truth list for the same source/query in `celeba_evaluation.json`.

```python
query_item = annotations[0]  # for example "+Smiling"
valid_targets = set(query_item["ground_truth"]["13"])

retrieved = [325, 999, 456, 1234, 579]

correct = [idx for idx in retrieved if idx in valid_targets]
print(correct)
```

If `correct` is non-empty in the top K, Recall@K is 1 for that source/query. Precision@K is `len(correct) / K`.

## Important Distinction

Filename `000013.jpg` is not the same thing as dataset index `13`.

Use:

```python
source_image, source_attrs = celeba[13]
```

Do not use:

```python
Image.open("celeba/img_align_celeba/000013.jpg")
```
