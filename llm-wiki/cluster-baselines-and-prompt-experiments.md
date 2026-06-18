# Cluster Baselines and CLIP Prompt Experiments

Status: June 13, 2026.

This page records the first complete quantitative experiment cycle: cluster
packaging, frozen CLIP feature extraction, five arithmetic retrieval methods,
official benchmark results, and local image-text prompt diagnostics.

## 1. Cluster Bundle And Execution

The operational bundle is under `cluster/` and was transferred to:

```text
/mnt/meditech/group1/deep_learning/cluster
```

It contains only execution material: CelebA data and annotations, the official
evaluation JSON, Python scripts, Slurm jobs, requirements, the Steps 1-3
notebook, checkpoints, logs, and result directories. PDFs and the local wiki are
not part of the cluster bundle.

Relevant scripts:

| Job/script | Purpose | Main output |
| --- | --- | --- |
| `jobs/00_setup_data.sh` | Resumable CelebA ZIP extraction | `data/celeba/img_align_celeba/` |
| `jobs/01_embeddings_test.sh` | Frozen CLIP test embeddings | `test_image_embeddings.pt`, text cache |
| `jobs/02_embeddings_train.sh` | Frozen CLIP train embeddings | `train_image_embeddings.pt` |
| `jobs/03_embeddings_valid.sh` | Frozen CLIP validation embeddings | `valid_image_embeddings.pt` |
| `jobs/10...13_baseline_*.sh` | Four initial arithmetic methods | one result folder per method |
| `jobs/14_baseline_adaptive_tangent_sequential.sh` | Source-adaptive arithmetic experiment | fifth result folder |
| `jobs/20_merge_results.sh` | Merge summaries | `all_methods_summary.csv` |

The cluster requires CPU-only jobs to state `--gres=gpu:0`. GPU jobs request
`--gres=gpu:1`. Slurm copies submitted scripts to `/var/spool`, so job roots
must use `SLURM_SUBMIT_DIR`; deriving paths from `$0` incorrectly points to the
temporary Slurm script.

The cluster Python environment used PyTorch `2.8.0+cu128`. CUDA is expected to
be unavailable on the login node and available only inside GPU allocations.

## 2. Frozen CLIP Caches

Required model:

```text
openai/clip-vit-base-patch32
```

The test embedding job completed successfully in 2 minutes 9 seconds. It
processed all 19,962 CelebA test images and produced:

| Artifact | Approximate size | Contents |
| --- | ---: | --- |
| `test_image_embeddings.pt` | 20 MiB | normalized float16 test embeddings and filenames |
| `attribute_text_embeddings.pt` | 126 KiB | 40 positive prompts, negative prompts, and directions |
| test checkpoint chunks | 78 files | resumable image batches |

The final verifier reported both caches ready. Checkpoint files remain useful
for provenance and restart safety, although the final cache is sufficient for
retrieval.

## 3. Evaluation Setup

All methods use:

- the 14 queries in `celeba_evaluation.json`;
- 33,052 source-query cases;
- the complete 19,962-image test split as gallery;
- source-image exclusion from its own ranking;
- Recall@K and Precision@K for K = 1, 5, 10;
- the instructor's hit-rate Recall definition.

Macro metrics give equal weight to each of the 14 query entries. Micro metrics
weight every source-query case equally, so high-frequency queries contribute
more. This distinction matters because source counts range from 27 to 5,469.

## 4. Arithmetic Methods

### 4.1 Direct Sum

The assignment-required baseline:

```text
q = normalize(z_source + sum(sign_j * t_positive_j))
```

Negative conditions subtract the positive text embedding.

### 4.2 Direct Sequential

The same edits are applied one at a time, normalizing after every condition.
For one-condition queries it is mathematically equivalent to Direct Sum.

### 4.3 Contrastive Sum

Each attribute uses a contrastive direction:

```text
d_j = normalize(t_positive_j - t_negative_j)
q   = normalize(z_source + sum(sign_j * d_j))
```

### 4.4 Contrastive Sequential

Contrastive directions are applied one at a time with normalization after each
condition.

### 4.5 Adaptive Tangent Sequential

For the signed requested direction `d` and current normalized query `q`:

```text
a       = dot(q, d)
lambda  = clamp(1 - a, 0.25, 1.75)
tangent = d - a*q
q_new   = normalize(q + lambda*tangent)
```

The intent is to use a smaller step when the source is already aligned with the
requested condition, a larger step when it is opposed, and to remove the radial
component before moving on the normalized CLIP sphere. This method is still
training-free and uses no evaluation targets to set its per-image weights.

## 5. Aggregate Results

### 5.1 Macro Metrics

| Method | R@1 | R@5 | R@10 | P@10 |
| --- | ---: | ---: | ---: | ---: |
| Direct Sum | 0.02397 | 0.07183 | 0.10842 | 0.01471 |
| Direct Sequential | 0.02807 | 0.07327 | 0.11206 | 0.01530 |
| Contrastive Sum | 0.03912 | 0.12039 | 0.17074 | 0.02474 |
| **Contrastive Sequential** | **0.04064** | **0.12381** | **0.18708** | **0.02698** |
| Adaptive Tangent Sequential | 0.04061 | 0.12353 | 0.18693 | 0.02687 |

Relative to Direct Sum, Contrastive Sequential improves:

- Macro R@1 by 69.55%;
- Macro R@5 by 72.37%;
- Macro R@10 by 72.54%;
- Macro P@10 by 83.36%.

The absolute Macro R@10 gain is `0.07865`, or 7.865 percentage points.

### 5.2 Micro Metrics

| Method | Micro R@10 | Micro P@10 | R@10 change vs Direct Sum |
| --- | ---: | ---: | ---: |
| Direct Sum | 0.12477 | 0.01768 | reference |
| Direct Sequential | 0.12635 | 0.01786 | +1.26% |
| Contrastive Sum | 0.16650 | 0.02422 | +33.44% |
| Contrastive Sequential | 0.16653 | 0.02418 | +33.46% |
| **Adaptive Tangent Sequential** | **0.16677** | 0.02420 | **+33.66%** |

The adaptive method has the highest Micro R@10, but only by `0.00024` over
Contrastive Sequential. This difference is too small to claim a meaningful
improvement without uncertainty estimates or repeated/bootstrapped evaluation.

## 6. Per-Query Findings

The strongest result appears on the three-condition query:

```text
-Smiling, +Eyeglasses, +Wearing_Hat
```

| Method | R@10 | P@10 |
| --- | ---: | ---: |
| Direct Sum | 0.1013 | 0.0101 |
| Direct Sequential | 0.1266 | 0.0127 |
| Contrastive Sum | 0.2405 | 0.0354 |
| Contrastive Sequential | 0.4304 | 0.0658 |
| Adaptive Tangent Sequential | 0.4304 | 0.0646 |

Other observations:

- Contrastive methods strongly improve `+Eyeglasses` and `+Male`.
- For `+Eyeglasses, +Smiling`, Contrastive Sequential reaches R@10 `0.2712`
  versus Direct Sum `0.0784`.
- For `+Black_Hair, -Wavy_Hair`, Contrastive Sum (`0.2107`) is better than both
  sequential contrastive variants.
- For `+Wearing_Lipstick, -Heavy_Makeup, +Smiling`, no method clearly wins;
  several methods tie at R@10 `0.0882`.
- Queries with 27, 34, or 79 source images have high sampling uncertainty and
  can influence the macro average disproportionately.
- Adaptive Tangent wins R@10 outright on `+Smiling`, `+Male`, and the duplicated
  `-Young` entries, but loses enough elsewhere to remain essentially tied with
  Contrastive Sequential overall.

## 7. Main Baseline Conclusions

1. Contrastive attribute directions are the largest successful change. Using
   `t_positive - t_negative` is much better than adding/subtracting only the
   positive prompt embedding.
2. Sequential normalization matters mainly for multi-condition queries. It has
   no effect for a single edit.
3. The best original method is Contrastive Sequential by macro metrics.
4. The adaptive tangent rule does not produce a clear aggregate improvement.
   Its source score is too compressed for `lambda = 1 - a` to change the step
   strongly; its macro result is slightly below Contrastive Sequential.
5. Arithmetic remains query-dependent: no fixed composition rule dominates all
   14 queries.

Generated local analysis artifacts:

```text
cluster/artifacts/results/baselines/plots/method_comparison.csv
cluster/artifacts/results/baselines/plots/per_query_winners.csv
cluster/artifacts/results/baselines/plots/query_*.png
```

## 8. Absolute Cosine Similarity Diagnostics

The local script `scripts/clip_image_text_similarity.py` runs CLIP inference on
the Mac using CPU or Apple MPS. Hugging Face supplies and caches model files;
the actual image/text forward pass is local. With `--offline` and
`HF_HUB_OFFLINE=1`, no network access is required after the cache exists.

Absolute image-text cosine values were compressed. For image `000366.jpg`:

```text
"a black person smiling, wearing a hat"  -> 0.270775
"a dog"                                  -> 0.202673
```

The difference `0.068102` is much more informative than either absolute value.
CLIP embeddings are anisotropic and share broad semantic components, so `0.20`
does not imply strong human-level similarity and `0.27` is not necessarily low.

## 9. Binary Prompt Comparison

A binary comparison asks CLIP to choose between exactly two descriptions:

```text
s_positive = cosine(image, t_positive)
s_negative = cosine(image, t_negative)
margin     = s_positive - s_negative
```

The direction score is:

```text
cosine(image, normalize(t_positive - t_negative))
```

It has the same sign as the margin and is the margin divided by the norm of the
text difference. The printed pair preference is a two-prompt softmax using
CLIP's learned logit scale. It is not a calibrated probability that the
attribute exists.

For `000366.jpg`:

| Positive prompt | Negative prompt | Cos+ | Cos- | Margin | Pair preference+ |
| --- | --- | ---: | ---: | ---: | ---: |
| person wearing a hat | bareheaded person | 0.239657 | 0.202803 | +0.036853 | 97.55% |
| person smiling | very serious person | 0.238512 | 0.207516 | +0.030995 | 95.69% |
| Black person smiling, wearing a hat | serious person, without a hat | 0.270775 | 0.216572 | +0.054203 | 99.56% |

The multi-attribute prompt cannot isolate which property caused the margin and
should not be used to estimate individual attribute presence.

## 10. Multi-Pair Prompt Ensembles

The script now accepts repeated pairs:

```bash
python3 scripts/clip_image_text_similarity.py 000366.jpg \
  --pair "positive prompt 1" "negative prompt 1" \
  --pair "positive prompt 2" "negative prompt 2" \
  --pair "positive prompt 3" "negative prompt 3" \
  --device mps
```

It reports every pair, then builds normalized mean positive and negative text
prototypes. It also reports mean pair margin, standard deviation, number of
positive-voting pairs, ensemble margin, direction score, and two-prompt
ensemble preference.

The official CelebA TXT row for `000366.jpg` marks these attributes present:

```text
Big_Lips, Big_Nose, Chubby, Double_Chin, Goatee, Male,
Mustache, Narrow_Eyes, Smiling, Wearing_Hat
```

### Wearing Hat

Using concrete lexical opposites:

```text
wearing a hat                 vs bareheaded
hat covering the head        vs uncovered head
wearing a baseball cap       vs uncovered hair
```

Results:

```text
positive pairs:   3/3
ensemble margin:  +0.027074
direction score:  +0.052661
mean pair margin: +0.029934
margin std:        0.009884
ensemble pref+:   93.75%
```

However, an earlier ensemble using `not wearing`, `without`, and `no headwear`
gave only `1/3` positive pairs and an ensemble preference of `49.86%`. This is
direct evidence that CLIP ViT-B/32 is sensitive to prompt wording and weak at
linguistic negation.

### Smiling

Three pairs comparing smiling/happy against serious/neutral/stern produced:

```text
positive pairs:   1/3
ensemble margin:  -0.000389
direction score:  -0.000963
mean pair margin: +0.004447
margin std:        0.008272
ensemble pref+:   49.03%
```

Despite the positive CelebA label, the ensemble is effectively undecided.

### Mustache

Three mustache versus clean-shaven/no-facial-hair pairs produced:

```text
positive pairs:   0/3
ensemble margin:  -0.017345
direction score:  -0.044188
mean pair margin: -0.017694
margin std:        0.011292
ensemble pref+:   15.00%
```

CLIP predicts the wrong side despite the positive CelebA label and visible
facial hair.

## 11. Consequences For The Proposed Model

- Image-text cosine is useful as a relative signal, not an absolute presence
  probability.
- A single prompt pair is too brittle to control a learned or fixed gate.
- Prompt ensembles expose disagreement but do not guarantee correct attribute
  detection.
- Negation should prefer concrete semantic opposites over constructions such as
  `not X` or `without X`, while acknowledging that the best wording is
  attribute-specific.
- Before using prompt margins to control `lambda`, evaluate each attribute on
  train/validation CelebA labels and measure accuracy, ROC-AUC, threshold,
  margin distributions, and calibration. Do not calibrate on the official test
  JSON.
- The failure of fixed source-adaptive arithmetic strengthens the case for a
  learned gate/residual module trained on CelebA supervision.

## 12. Immediate Next Steps

1. Generate train and validation image caches on the cluster.
2. Build prompt ensembles for the official-query attributes.
3. Evaluate attribute-margin quality on validation labels before using it as a
   gate input.
4. Implement the signed gated residual model and compare it with Contrastive
   Sequential, the strongest macro baseline.
5. Add bootstrap confidence intervals or paired significance tests for small
   differences between arithmetic methods.
