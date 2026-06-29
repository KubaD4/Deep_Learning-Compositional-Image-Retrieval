# Method Roadmap

This page is a practical plan for turning the assignment into a working notebook.

For the complete group-ready explanation, detailed code skeleton, training decision, first-session plan, and final checklist, read [PROJECT_GUIDE.md](../PROJECT_GUIDE.md).

The current proposed learned method is formalized in [Proposed Training Strategy](training-strategy.md).

## Stage 1: Reproduce The Skeleton

Goals:

- Load CelebA test split correctly.
- Load `celeba_evaluation.json`.
- Verify the provided evaluation helper.
- Visualize source images and some valid target examples.

Deliverable inside notebook:

- Dataset setup cell.
- JSON loading cell.
- Small sanity-check section proving indices are dataset indices, not filenames.

## Stage 2: Offline CLIP Feature Extraction

Use HuggingFace `openai/clip-vit-base-patch32`.

Extract once:

- image embeddings for all CelebA test images,
- text embeddings for attribute prompts and query conditions.

Normalize embeddings if using cosine similarity.

Possible prompt templates:

- `"a face with eyeglasses"`
- `"a photo of a smiling face"`
- `"a face without heavy makeup"`
- `"a face with black hair"`

Cache features in memory or to Drive so experiments are fast.

## Stage 3: Vanilla Zero-Shot Baseline

Implement a lower-bound retrieval method:

```text
query_embedding = ref_image_embedding
for positive condition:
    query_embedding += alpha * text_embedding(condition)
for negative condition:
    query_embedding -= beta * text_embedding(condition)
```

Rank targets by cosine similarity between `query_embedding` and candidate image embeddings.

This baseline is important because:

- it validates the retrieval and evaluation pipeline,
- it gives a comparison point for any fusion mechanism,
- it is simple enough to debug visually.

Status: completed on the cluster. Four initial arithmetic variants and one
adaptive tangent variant were evaluated over all official cases. Contrastive
Sequential is currently the strongest macro method with Macro Recall@10
`0.18708`, a 72.54% relative improvement over Direct Sum. The adaptive variant
does not clearly improve it. Detailed results are recorded in
[Cluster Baselines and Prompt Experiments](cluster-baselines-and-prompt-experiments.md).

## Stage 4: Improve Fusion

The assignment asks for a dynamic or hybrid conditioning mechanism, not just naive arithmetic.

Possible training-free ideas:

- condition-specific weights based on similarity between reference image and condition text,
- separate identity-preservation score plus condition-satisfaction score,
- positive and negative margins instead of direct vector subtraction,
- CLAY-inspired conditional similarity modulation using fixed image embeddings,
- query-adaptive weighting that downweights conflicting or already-satisfied conditions.

Possible lightweight training-based ideas:

- small MLP/gating module over `[image_emb, positive_text_embs, negative_text_embs]`,
- cross-attention between image embedding tokens and condition embeddings,
- learned scalar weights for positive/negative conditions,
- contrastive adapter trained on attribute-generated pairs from CelebA.

### Selected First Learned Model

The current group proposal is a frozen-CLIP residual composition network:

```text
source embedding + signed condition embeddings
                    -> gated aggregation
                    -> residual MLP
                    -> predicted retrieval embedding
```

Training tuples are derived from CelebA metadata. Same-identity source/target pairs provide identity-preservation supervision, while the signed attribute differences provide positive and negative text conditions. The primary loss is contrastive retrieval loss against the frozen target image embedding.

See [Proposed Training Strategy](training-strategy.md) for tuple generation, equations, sampling rules, loss functions, and the mismatch between same-identity training and the attribute-based official benchmark.

## Stage 5: Evaluation and Analysis

For each method:

- report Recall@1, Recall@5, Recall@10,
- report Precision@1, Precision@5, Precision@10,
- compare per query,
- show qualitative examples for wins and failures.

Recommended result table columns:

```text
method | query | R@1 | R@5 | R@10 | P@1 | P@5 | P@10
```

## Stage 6: Notebook Report

The final notebook should read like a paper/report:

- Introduction and task definition.
- Related work: CLIP, GDE/compositionality, CLAY.
- Method: math and architecture.
- Experimental setup.
- Results tables and plots.
- Qualitative retrieval examples.
- Error analysis and conclusions.

### Report Framing Reminder: Hybrid Compositionality vs Full Retrieval System

When preparing the final notebook/report, separate the contribution into two
levels.

Level 1 is the assignment-facing hybrid compositionality method:

```text
q_model  = learned gate / sequential composer(source, signed query)
q_sum    = explicit CLIP arithmetic composition(source, signed query)
q_hybrid = normalize(q_model + beta * (q_sum - source))
```

This is the clean answer to the requested hybrid compositionality: the system
combines a learned source-conditioned composer with explicit semantic CLIP
directions, so multi-attribute queries are not handled by a single naive sum.
This is the result to report as the hybrid-compositionality model, with its
official Recall@K and Precision@K.

Level 2 is the full retrieval system built on top of that hybrid query vector:

```text
q_hybrid -> retrieve broad candidate pool -> optional probe/filter/reranker
```

Probe-based filters, CLIP prompt filters, oracle top-pool diagnostics, or future
rerankers should be described as optional system-level extensions built on top
of hybrid compositionality. Do not present those as replacing the core method
unless we decide to include them in the final submission after seeing their
results. If included, report both:

```text
1. hybrid compositionality alone
2. complete hybrid + reranker/filter system
```

This prevents the report from blurring the assignment requirement with later
engineering improvements aimed at filling the final top-k with cleaner valid
targets.
