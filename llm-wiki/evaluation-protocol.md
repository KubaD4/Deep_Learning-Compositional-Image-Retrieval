# Evaluation Protocol

## Benchmark File

File: `celeba_evaluation.json`

Structure:

```python
[
    {
        "query": "+Smiling",
        "ground_truth": {
            "13": [325, 456, ...],
            "...": [...]
        }
    },
    ...
]
```

Each top-level item is one textual query. `ground_truth` maps source dataset indices, stored as strings, to lists of acceptable target dataset indices.

## Official Queries Present Locally

The local JSON contains 14 query entries:

| # | Query | Source count | Avg valid targets/source |
| --- | --- | ---: | ---: |
| 0 | `+Smiling` | 4786 | 25.96 |
| 1 | `+Eyeglasses` | 2196 | 20.30 |
| 2 | `-Heavy_Makeup` | 4087 | 25.16 |
| 3 | `+Male` | 1595 | 64.77 |
| 4 | `-Young` | 5355 | 23.25 |
| 5 | `+Blond_Hair` | 5469 | 25.07 |
| 6 | `+Mustache` | 301 | 7.82 |
| 7 | `-Young` | 5355 | 23.25 |
| 8 | `+Eyeglasses, +Smiling` | 612 | 11.57 |
| 9 | `+Black_Hair, -Wavy_Hair` | 2572 | 28.67 |
| 10 | `-Male, -Mustache` | 27 | 16.07 |
| 11 | `+Chubby, -Young` | 584 | 9.23 |
| 12 | `-Smiling, +Eyeglasses, +Wearing_Hat` | 79 | 7.68 |
| 13 | `+Wearing_Lipstick, -Heavy_Makeup, +Smiling` | 34 | 10.85 |

Note: `-Young` appears twice in the local JSON. Treat the JSON as authoritative unless the instructor provides an updated file.

## Ground Truth Definition

A target image is valid for a source/query pair if:

1. It strictly satisfies the positive and negative constraints.
2. Its remaining attributes have Hamming distance <= 2 from the source image.

The JSON already precomputes these acceptable targets. Do not recompute ground truth unless checking the data or building custom queries.

## What "Similar Face" Means In Practice

There are two different notions of similarity:

1. Model similarity: the score your method computes, usually cosine similarity between a fused query embedding and each candidate image embedding.
2. Official validation similarity: whether the retrieved candidate appears in `celeba_evaluation.json` for that source/query pair.

For grading, the second one matters. The provided JSON defines "similar enough" as: satisfy the requested attribute edits, while keeping all non-queried CelebA attributes close to the source with relaxed Hamming distance <= 2.

This does not guarantee that the target is the same person or even perceptually similar to the source for a human observer. Real identity IDs, facial recognition features, landmarks, and pixel-level similarity are not used when constructing the official ground truth. The phrase "core visual identity" in the assignment refers operationally to preservation of the non-queried CelebA attributes.

Example:

```text
source = celeba[13]
query = "+Smiling"
retrieved = [325, 999, 456, 1234, 579]
valid_targets = annotations[0]["ground_truth"]["13"]
```

Then the retrieved faces are officially similar/correct when their indices are in `valid_targets`.

You can still use CLIP cosine similarity, attribute classifiers, or visual inspection to develop and debug the model, but final validation should use Recall@K and Precision@K against the JSON.

## Metrics

Compute and report at K = 1, 5, 10.

Recall@K, primary metric:

```text
1 if any retrieved top-K index is in the ground-truth target set, else 0
```

Precision@K, secondary metric:

```text
number of valid retrieved top-K indices / K
```

Average both metrics over all source images for each query, then present per-query tables. It is also useful to report macro averages across queries.

Important interpretation: in this assignment, Recall@K is explicitly a hit-rate, not classical IR recall over all valid targets. If a source/query has 80 valid targets and the system retrieves exactly one valid target in the top 10, Recall@10 is 1 but Precision@10 is 0.1. Therefore:

```text
high Recall@10 + low Precision@10
= the query vector often reaches a valid region,
  but many of the remaining nearest neighbours are not official-valid targets.
```

This is the current behavior of the best learned system. The next likely improvement is not only to make a better single `q_final`, but to use a two-stage retrieval:

```text
1. retrieve a larger candidate pool with q_final, e.g. top 100/top 500 by cosine;
2. rerank or filter candidates using attribute-query satisfaction and non-query Hamming/source-preservation estimates;
3. return the final top 10.
```

For fair final evaluation, any reranker should avoid using the official test JSON target lists directly. A safe version would learn/predict attribute satisfaction from train data, while an oracle Hamming filter can be used only as an analysis upper bound.

## Evaluation Loop Sketch

```python
for query_item in annotations:
    query = query_item["query"]
    gt_by_source = query_item["ground_truth"]

    for source_key, valid_targets in gt_by_source.items():
        source_idx = int(source_key)
        source_image, source_attrs = celeba[source_idx]

        ranked_indices = retrieve(source_image, query)

        for k in [1, 5, 10]:
            recall = int(bool(set(ranked_indices[:k]) & set(valid_targets)))
            precision = len(set(ranked_indices[:k]) & set(valid_targets)) / k
```

Important: exclude the source image itself if the retrieval corpus includes it and it would trivially rank at the top.

## First Completed Official Baselines

The cluster evaluation completed all 14 JSON entries and 33,052 source-query
cases with frozen `openai/clip-vit-base-patch32` embeddings.

| Method | Macro R@1 | Macro R@5 | Macro R@10 | Macro P@10 |
| --- | ---: | ---: | ---: | ---: |
| Direct Sum | 0.02397 | 0.07183 | 0.10842 | 0.01471 |
| Direct Sequential | 0.02807 | 0.07327 | 0.11206 | 0.01530 |
| Contrastive Sum | 0.03912 | 0.12039 | 0.17074 | 0.02474 |
| **Contrastive Sequential** | **0.04064** | **0.12381** | **0.18708** | **0.02698** |
| Adaptive Tangent Sequential | 0.04061 | 0.12353 | 0.18693 | 0.02687 |

Contrastive Sequential improves Macro R@10 by 72.54% relative to Direct Sum.
Adaptive Tangent is essentially tied but slightly lower on macro metrics. See
[Cluster Baselines and Prompt Experiments](cluster-baselines-and-prompt-experiments.md)
for micro metrics, per-query results, runtime, prompt diagnostics, and caveats.
