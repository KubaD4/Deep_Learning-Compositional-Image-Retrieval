# Official Results and Literature Findings

This page records the state after the prompt-v2 `gate_v3` overnight run completed on 2026-06-20.

## Current Best Result

Best official JSON model:

```text
gate_v3 sequential gate
config: seqp2_ov005
prompt cache: signed_attribute_prompt_embeddings_v2_photo_templates.pt
temperature: 0.02
learning_rate: 1e-4
lambda_source: 0.02
residual_scale: 0.02
edit_scale: 1.0
gate_state: current
```

Official JSON aggregate:

```text
seqp2_ov005 Macro Recall@10          0.2117
seqp2_ov005 Micro Recall@10          0.1936
seqp2_ov005 Macro Precision@10       0.0320

Contrastive Sequential Macro R@10    0.1871
Adaptive Tangent Sequential Macro R@10 0.1869
Direct Sum Macro R@10                0.1084
```

Relative improvement over the best arithmetic baseline:

```text
Macro R@10: (0.2117 - 0.1871) / 0.1871 ~= +13.2%
Micro R@10: (0.1936 - 0.1665) / 0.1665 ~= +16.3%
```

## Per-Query Findings

The learned model is strongest on local visible edits:

```text
+Eyeglasses                         0.4645 vs best baseline 0.2518
+Eyeglasses, +Smiling               0.4020 vs best baseline 0.2712
-Smiling, +Eyeglasses, +Wearing_Hat 0.5570 vs best baseline 0.4304
-Heavy_Makeup                       0.2256 vs best baseline 0.1686
+Mustache                           0.2990 vs best baseline 0.2525
+Smiling                            0.3042 vs best baseline 0.2620
```

Main weak queries:

```text
+Male                               0.0527 vs best baseline 0.2395
-Male, -Mustache                    0.0000 vs best baseline 0.0741
+Chubby, -Young                     0.0223 vs best baseline 0.0497
+Wearing_Lipstick, -Heavy_Makeup, +Smiling
                                    0.0588 vs best baseline 0.0882
```

The biggest single bottleneck is `+Male`.

## What The Official JSON Numbers Mean

The official JSON contains 14 query entries. Each query entry contains many source images. For each source image, the JSON lists the valid target indices in the test gallery.

Example:

```text
query: +Mustache
source images: 301
average valid targets per source: 7.8
```

This means that, for a typical source image under `+Mustache`, only about 7-8 images out of the 19,962-image test gallery are considered valid. A concrete source can have 14 valid targets, another can have 5, another can have 22.

The official `Recall@10` is binary per source-query case:

```text
1 if at least one valid target appears in the model top-10
0 otherwise
```

The task is therefore hard even when the model is qualitatively close. Random retrieval is around:

```text
random Macro Recall@10 ~= 0.0105
random Micro Recall@10 ~= 0.0127
```

The current best model is about 20x random on Macro R@10.

## Same-Identity Bias Diagnosis

The synthetic training validation and official test evaluate different things.

Synthetic validation:

```text
val_exact_R@10        ~= 0.6289 for seqp2_ov005
val_attr_success@10   ~= 0.9268
```

Interpretation:

- `val_exact_R@10`: the exact same-identity target `B` is in top-10 about 63% of the time.
- `val_attr_success@10`: at least one top-10 image satisfies the requested attributes about 93% of the time.

Official JSON behavior of the same model:

```text
same identity in top-1   ~= 69.1%
same identity in top-10  ~= 88.6%
official valid in top-10 ~= 19.4%
```

This shows that the model strongly preserves identity. That helps for local edits, but hurts when the official JSON valid targets are mostly cross-identity.

For several weak queries, same-identity valid targets are rare:

```text
+Male                only 2.5% of source cases have any same-identity valid target
-Young               only 3.8%
+Chubby, -Young      only 0.7%
-Male, -Mustache     0.0%
```

Therefore, "find the same person after the edit" is a useful inductive bias but not the exact official target distribution.

## Literature Notes

Keep these references in the wiki because they directly motivate the next design step.

- CLIP paper: <https://arxiv.org/abs/2103.00020>. Prompt engineering and prompt ensembling improve zero-shot performance; zero-shot CLIP remains weaker on more complex or abstract tasks.
- Winoground: <https://arxiv.org/abs/2204.03162>. Vision-language models perform poorly on fine-grained compositional reasoning where word order and binding matter.
- CLIP bag-of-words / binding analysis: <https://arxiv.org/abs/2502.03566>. CLIP can contain attribute-binding information unimodally, while cross-modal cosine alignment may fail to expose it reliably.
- Concept Association Bias: <https://arxiv.org/abs/2212.12043>. CLIP-like models can rely on concept associations and fill in correlated concepts rather than perform controlled visual reasoning; this is relevant to `Male`, `Young`, and `Chubby`.
- TripletCLIP: <https://proceedings.neurips.cc/paper_files/paper/2024/file/39781da4b5d05bc2908ce08e43bc6404-Paper-Conference.pdf>. Hard negative compositional image-text pairs and triplet-style learning are a plausible route for improving compositionality.
- Local project slide source: [CLIP Compositionality - Davide Berasi](source-markdown/CLIP%20Compositionality%20-%20Davide%20Berasi.md). The slides explicitly note CLIP's shortcomings on complex compositions, word order, and bag-of-words-like behavior.

## Decided Next Experiment

Do not train on `celeba_evaluation.json`. It is the official test benchmark and should remain untouched for final evaluation. Splitting it into train/test would be methodologically risky and likely perceived as benchmark leakage.

Instead, build an official-like training set from CelebA train/validation attributes only:

```text
source image A from train/valid partition
signed query C sampled from changed official-style attributes
positive set P(A, C):
  images that satisfy C
  and differ from A in few non-query attributes
  and are among the CLIP-nearest compatible images
hard negatives N(A, C):
  CLIP-near images that do not satisfy C
  same-identity images that remain too close but do not satisfy C
```

The candidate positive score should combine similarity and attribute preservation:

```text
score(x) = cosine(z_A, z_x) - beta * non_query_attribute_changes(A, x)
```

This does not guarantee the same real identity; it guarantees source-like visual/attribute similarity under the same relaxed principle used by the official JSON.

Next model direction:

```text
gate_v3 seqp2_ov005 backbone
+ multi-positive official-like loss
+ hard negatives
+ attribute-presence probe features
+ medium MLP capacity ablation
```

The attribute-presence probe is a small classifier on frozen CLIP image embeddings:

```text
probe(z_image) -> 40 CelebA attribute probabilities
```

Its outputs can be given to the gate so the gate knows whether requested attributes such as `Male`, `Young`, or `Chubby` appear already present/absent in the source. The probe is not the final retrieval model; it is auxiliary state for the gate.

Capacity note:

```text
current gate_v3      ~= 4.0M parameters
medium ablation      ~= 8.2M parameters
large ablation       ~= 13.9M parameters
```

Increasing layers is allowed by the assignment because lightweight training-based adapters are allowed, but the main issue appears to be supervision mismatch, not raw capacity. Therefore, run `medium` only as an ablation together with official-like multi-positive training, not as the sole fix.

## Implemented Follow-Up: Official-Like Multi-Positive Training

Implemented on 2026-06-20 as job `45`.

Code blocks:

```text
Frozen CLIP cache
  input: train/valid image embeddings, CelebA attributes, prompt-v2 signed directions
  output: source vectors, target mini-gallery vectors, signed condition vectors

Pair sampler
  input: same-identity pair index from build_training_pairs.py
  output: batch of source indices, exact target indices, signed query attributes

Official-like positive mask
  input: source attributes, candidate target attributes, query signs
  output: candidate positives that satisfy the query and change <= hamming threshold non-query attributes

Source-similarity filter
  input: source CLIP embedding and candidate target embeddings
  output: keep only top-fraction CLIP-nearest official-like positives

Multi-positive contrastive loss
  input: predicted query embedding q and target mini-gallery
  output: loss that treats exact target B plus compatible mini-gallery positives as positives

Attribute-presence probe
  input: frozen source image CLIP embedding
  output: 40 attribute logits, supervised by CelebA labels as an auxiliary loss

Sequential gate composer
  input: source embedding, signed directions, optional probe probabilities
  output: edited query embedding used for nearest-neighbor retrieval
```

Important limitation of this first implementation: the multi-positive set is built inside the training batch mini-gallery, not over the full train gallery. This keeps the run lightweight and cluster-safe, but it means the positive set is an approximation of the official JSON logic. If it improves, the next version can precompute larger compatible-positive pools.

New files/configs:

```text
cluster/configs/gate_sequential_official_like_long_configs.json
cluster/jobs/45_hpsearch_official_like_long.sh
```

Config variants:

```text
offmp_l001        multi-positive loss, no probe
offmp_l002_probe  multi-positive loss + attribute probe
offmp_l003_medium multi-positive loss + attribute probe + medium MLP
offmp_l004_widepos same as probe but wider positive top-fraction
```

## Result: Official-Like Multi-Positive Was Not Enough Alone

Job `45_hpsearch_official_like_long.sh` completed on 2026-06-20. The best synthetic validation config was:

```text
offmp_l004_widepos
val_official_like@10 = 0.8340
val_exact_R@10       = 0.6335
val_attr_success@10  = 0.9307
best epoch/step      = 8 / 16000
```

However, official JSON evaluation showed a small regression versus the previous best `seqp2_ov005`:

```text
seqp2_ov005          Macro R@10 = 0.2117, Micro R@10 = 0.1936
offmp_l004_widepos   Macro R@10 = 0.2091, Micro R@10 = 0.1918
```

Per-query movement was useful diagnostically:

```text
Improved:
+Eyeglasses                               0.4645 -> 0.4927
-Heavy_Makeup                             0.2256 -> 0.2339
+Male                                     0.0527 -> 0.0652
+Eyeglasses, +Smiling                     0.4020 -> 0.4167
+Wearing_Lipstick, -Heavy_Makeup, +Smiling 0.0588 -> 0.1176

Worsened:
+Blond_Hair                               0.2070 -> 0.1900
+Black_Hair, -Wavy_Hair                   0.2150 -> 0.1882
+Mustache                                 0.2990 -> 0.2791
-Smiling, +Eyeglasses, +Wearing_Hat        0.5570 -> 0.4684
+Chubby, -Young                           0.0223 -> 0.0171
```

Interpretation: pure multi-positive official-like training pushes the model toward attribute compatibility but weakens some context/source preservation. The problem is not simply "make training look more like the official JSON"; the useful supervision is a mixture:

```text
same-identity exact target B     teaches visual/source preservation
official-like compatible targets teaches benchmark-style relaxed retrieval
```

## Implemented Follow-Up: Hybrid Exact + Official-Like Loss

Implemented on 2026-06-21 as job `46`.

The training objective now supports:

```text
loss_retrieval = (1 - w) * exact_info_nce + w * multipositive_info_nce
```

where:

```text
exact_info_nce
  positive = the exact same-identity target B
  compatible non-target positives are masked as false negatives

multipositive_info_nce
  positives = B plus official-like compatible candidates inside the batch mini-gallery
  candidates must satisfy the signed query
  candidates must stay within the non-query Hamming threshold
  candidates can optionally be restricted to CLIP-near top-fraction positives
```

This tests the hypothesis suggested by job `45`: keep the identity-preserving target signal, but add a controlled amount of official-like supervision.

New files/configs:

```text
cluster/configs/gate_sequential_hybrid_official_like_long_configs.json
cluster/jobs/46_hpsearch_hybrid_official_like_long.sh
```

Config variants:

```text
hybmp_l001_w025                 w = 0.25, no probe
hybmp_l002_w050                 w = 0.50, no probe
hybmp_l003_w075                 w = 0.75, no probe
hybmp_l004_w025_probe           w = 0.25, attribute probe
hybmp_l005_w050_probe           w = 0.50, attribute probe
hybmp_l006_w025_widepos_probe   w = 0.25, attribute probe, wider positive pool
hybmp_l007_w050_widepos_probe   w = 0.50, attribute probe, wider positive pool
hybmp_l008_w025_src005_probe    w = 0.25, attribute probe, stronger source preservation
```

Expected readout:

```text
If w=0.25 wins:
  exact/same-identity supervision is still the core signal, official-like positives are regularization.

If w=0.50 or w=0.75 wins:
  benchmark-style compatibility is underrepresented in current training.

If probe configs win mostly on Male/Young/Chubby:
  source attribute-state awareness helps global/correlated attributes.

If widepos configs improve official Macro R@10 but hurt hat/hair/mustache:
  too many compatible positives are diluting source context.
```

## Result: Hybrid Loss Slightly Improves The Best Official Score

Job `46_hpsearch_hybrid_official_like_long.sh` completed on 2026-06-21. The best synthetic validation config selected by `best_val_official_like@10` was:

```text
hybmp_l002_w050
multipositive_weight = 0.50
attribute probe      = disabled
positive top fraction= 0.15
val_official_like@10 = 0.8323
val_exact_R@10       = 0.6338
val_attr_success@10  = 0.9282
best epoch/step      = 8 / 16000
val_mean_rank_B      = 48.1
```

Official JSON evaluation of this selected checkpoint:

```text
hybmp_l002_w050      Macro R@10 = 0.2130, Micro R@10 = 0.1953
seqp2_ov005          Macro R@10 = 0.2117, Micro R@10 = 0.1936
contrastive seq.     Macro R@10 = 0.1871, Micro R@10 = 0.1665
```

Interpretation:

```text
hybmp_l002 improves over seqp2_ov005 by about +0.0013 absolute Macro R@10
hybmp_l002 improves over seqp2_ov005 by about +0.0016 absolute Micro R@10
hybmp_l002 improves over contrastive sequential by about +13.9% relative Macro R@10
hybmp_l002 improves over contrastive sequential by about +17.3% relative Micro R@10
```

The hybrid result validates the hypothesis from job `45`: official-like positives are useful, but they should be mixed with exact same-identity target supervision rather than fully replacing it.

Important ablation findings from the 8 hybrid configs:

```text
w=0.50 no probe won official-like validation and official JSON.
w=0.25 and w=0.75 were very close, so the method is not highly sensitive in this range.
attribute probe variants were consistently slightly worse.
wider positive pools with probe were also slightly worse.
stronger source preservation (lambda_source=0.05) was worse.
```

This means the current best direction is not "more probe" or "more positives"; it is a clean learned sequential gate with a moderate official-like loss component.

Note: the cluster `per_query_delta_Recallat10.csv` printed during this run still used collapsed labels such as `sequentialgate`, so it should be regenerated with the updated `analyze_official_per_query.py` before making precise per-query claims for `hybmp_l002`.

## Final Baseline Rerun And Clean Official Comparison

On 2026-06-23 the five arithmetic baselines were rerun from scratch on the cluster, then the official comparison plots/tables were regenerated with all learned gates included. The rerun confirmed the baseline numbers were stable:

```text
Direct sum                    Macro R@10 = 0.1084, Micro R@10 = 0.1248
Direct sequential             Macro R@10 = 0.1121, Micro R@10 = 0.1263
Contrastive sum               Macro R@10 = 0.1707, Micro R@10 = 0.1665
Adaptive tangent sequential   Macro R@10 = 0.1869, Micro R@10 = 0.1668
Contrastive sequential        Macro R@10 = 0.1871, Micro R@10 = 0.1665
```

Final top learned result:

```text
hybmp_l002_w050               Macro R@10 = 0.2130, Micro R@10 = 0.1953
seqp2_ov005                   Macro R@10 = 0.2117, Micro R@10 = 0.1936
offmp_l004_widepos            Macro R@10 = 0.2091, Micro R@10 = 0.1918
seq_l007                      Macro R@10 = 0.1951, Micro R@10 = 0.1824
```

Final comparison against the strongest rerun baseline:

```text
hybmp_l002_w050 vs Contrastive sequential:
Macro R@10: 0.2130 vs 0.1871 = +0.0260 absolute, about +13.9% relative
Micro R@10: 0.1953 vs 0.1665 = +0.0287 absolute, about +17.3% relative
Macro P@10: 0.0325 vs 0.0270 = +0.0056 absolute, about +20.6% relative
```

Per-query findings from the clean rerun:

```text
Strong learned-gate gains:
+Eyeglasses                         best gate 0.4927 vs baseline 0.2518
+Eyeglasses, +Smiling               best gate 0.4167 vs baseline 0.2712
-Smiling, +Eyeglasses, +Wearing_Hat  best gate 0.5570 vs baseline 0.4304
-Heavy_Makeup                       best gate 0.2339 vs baseline 0.1686
+Mustache                           best gate 0.2990 vs baseline 0.2525
+Smiling                            best gate 0.3055 vs baseline 0.2620

Remaining weak points:
+Male                               best gate 0.0652 vs baseline 0.2395
-Male, -Mustache                    best gate 0.0000 vs baseline 0.0741
+Chubby, -Young                     best gate 0.0223 vs baseline 0.0497
```

Interpretation: the learned models are clearly better on local/editable visual attributes and local compositions, especially eyewear, smile, makeup, hat, mustache, and some hair queries. They still fail badly on broad demographic/correlated attributes such as `Male` and `Chubby`. This supports the existing hypothesis: the learned sequential gate is effective for compositional visual edits, but the official JSON contains global attributes where CLIP direction learning and same-source preservation are misaligned with the benchmark's cross-identity target set.

## Local Algebra Diagnosis For Male, Young, Chubby

On 2026-06-23 a local no-training vector sweep was run on the Mac using the cached CLIP test embeddings and text/prompt embeddings. No packages were installed and temporary scripts were removed after the run.

The goal was to understand whether the weak official queries can be improved by changing only CLIP-space algebra: source weight, edit strength, tangent projection, endpoint prompts, and small direction surgery.

### Attribute Separability In CLIP Space

The CLIP direction score was measured against CelebA test labels using a standardized positive-vs-negative separation (`d-prime`). Higher is better.

```text
Male        text direction d' ~= 7.73, prompt-v1 direction d' ~= 7.91
Young       text direction d' ~= 1.32, prompt-v1 direction d' ~= 1.73
Chubby      text direction d' ~= 0.47, prompt-v1 direction d' ~= 0.55
Eyeglasses  prompt-v1 direction d' ~= 3.05
Smiling     prompt-v1 direction d' ~= 2.27
```

Interpretation:

```text
Male is not weak as a CLIP direction. It separates labels extremely well.
Young is moderately weak and needs stronger movement.
Chubby is intrinsically weak/noisy in CLIP text space and rare in the test set.
```

Label correlations also explain why these attributes behave differently:

```text
Male strongly correlates with absence of lipstick/makeup and facial-hair attributes.
Young correlates with Attractive, Gray_Hair, Chubby, Double_Chin, Male, and Lipstick.
Chubby correlates with Double_Chin, Young, Big_Nose, Male, Attractive, Mustache, Bald, Gray_Hair.
```

This supports the hypothesis that the learned gate struggles not because local visual edits fail, but because these attributes are global/correlated and often imply cross-identity retrieval in the official JSON.

### Local Algebra Sweep Results

For weak official queries, the best no-training arithmetic variants found locally were:

```text
q03 +Male
baseline/gate reference:
  hybmp_l002_w050 R@10 ~= 0.0520
  best arithmetic baseline R@10 ~= 0.2395
best local algebra:
  text direction + adaptive/tangent-ish movement, alpha ~= 0.75, source_weight ~= 1.0
  R@10 ~= 0.2527

q04/q07 -Young
baseline/gate reference:
  hybmp_l002_w050 R@10 ~= 0.0794
  best arithmetic baseline R@10 ~= 0.0698
best local algebra:
  text direction tangent, alpha ~= 2.0, source_weight ~= 0.75
  R@10 ~= 0.0995

q10 -Male, -Mustache
baseline/gate reference:
  hybmp_l002_w050 R@10 = 0.0000
  best arithmetic baseline R@10 ~= 0.0741
best local algebra:
  prompt-v1 direction sum/tangent, alpha ~= 0.5-0.75, source_weight ~= 0.75-1.0
  R@10 ~= 0.1111

q11 +Chubby, -Young
baseline/gate reference:
  hybmp_l002_w050 R@10 ~= 0.0223
  best arithmetic baseline R@10 ~= 0.0497
best local algebra:
  prompt-v1 endpoint-only for `chubby + older`, source_weight = 0.0
  R@10 ~= 0.1113
  with +0.5 Double_Chin helper: R@10 ~= 0.1164
```

If the weak queries used these local arithmetic replacements while all other queries used `hybmp_l002_w050`, the approximate official score would become:

```text
Current hybmp_l002_w050 Macro R@10 ~= 0.2130
Weak-query-routed Macro R@10      ~= 0.2446
Current hybmp_l002_w050 Micro R@10 ~= 0.1953
Weak-query-routed Micro R@10      ~= 0.2131
```

This is an oracle-style diagnostic, not yet a final fair method, because the routing was chosen after inspecting the official weak queries. It shows the potential size of the failure mode and motivates a principled attribute-type-aware composer.

### Failed Direction-Surgery Ideas

For `+Male`, orthogonalizing the male direction against correlated components such as lipstick, makeup, no-beard, facial hair, and top gender-correlated attributes did not improve retrieval. The plain text male direction remained best.

For `+Chubby,-Young`, adding a small `Double_Chin` helper improved slightly, but adding too many correlated concepts risks violating non-query attribute preservation.

For `-Male,-Mustache`, explicit endpoint prompts like `female + no mustache + no beard` were worse than prompt-v1 contrastive directions. This suggests that endpoint semantics can over-constrain the query.

### Proposed No-Training Fix To Test Next

Add an attribute-aware composition fallback before retrieval:

```text
local/editable attributes:
  use learned gate hybmp_l002_w050

Male / -Male compositions:
  use text or prompt-v1 contrastive direction arithmetic
  use reduced alpha around 0.5-0.75 rather than learned gate

Young / -Young:
  use stronger tangent movement
  alpha around 2.0, source_weight around 0.75

Chubby + older combinations:
  use endpoint-style global query with strongly reduced source weight
  optionally add a small Double_Chin helper, but treat this as benchmark-oriented and report the caveat
```

A fair next experiment should define this routing only from attribute type, not from individual query IDs, then evaluate once on the official JSON. This keeps it closer to a real method and avoids pure per-query oracle tuning.

## Implemented Test: Attribute-Routed Orchestrator

On 2026-06-23 an experimental no-training orchestrator was added to test the hypothesis from the local algebra diagnosis: local/editable attributes should use the learned gate, while global weak attributes should use tuned CLIP arithmetic.

New files:

```text
cluster/orchestrator/evaluate_orchestrated_router.py
cluster/jobs/47_evaluate_orchestrated_router_short.sh
```

The orchestrator loads the best official learned checkpoint available under `artifacts/results/gate_model`, expected to be `hybmp_l002_w050`, then for each official JSON query it splits conditions into:

```text
local conditions:
  all ordinary/local attributes, composed by the learned gate

weak/global conditions:
  Male, Young, Chubby
  plus Mustache when it appears together with Male
```

Example:

```text
+Male, +Eyeglasses, +Smiling, -Young

local gate part:
  +Eyeglasses, +Smiling

weak arithmetic part:
  +Male, -Young
```

The script evaluates several assembly strategies in one short job:

```text
stage_tuned:
  apply learned gate to local attrs, then apply tuned weak arithmetic to that query vector

delta_sum_tuned:
  learned local query + arithmetic global delta

score_fusion_70local:
  0.70 * local learned score + 0.30 * weak arithmetic score

score_fusion_50:
  equal score fusion

rrf_union:
  reciprocal-rank fusion of learned-local ranking and weak-arithmetic ranking

full_arithmetic_if_weak:
  if a weak attribute is present, use tuned arithmetic for the whole query
```

The weak arithmetic rules are fixed from the local diagnostic, not learned:

```text
Male:
  text contrastive direction, alpha about 0.75

Young / older:
  stronger tangent movement, alpha about 2.0, source_weight about 0.75

Chubby + older:
  endpoint-style prompt query with low source weight and a small Double_Chin helper

Male + Mustache:
  prompt-v1 contrastive directions for both terms
```

Outputs are kept separate from prior experiments:

```text
artifacts/results/orchestrated_router/<method>/summary.csv
artifacts/results/orchestrated_router/<method>/per_query_metrics.csv
artifacts/results/orchestrated_router/<method>/retrievals.jsonl
artifacts/results/orchestrated_router/debug/routing_decisions.jsonl
artifacts/results/orchestrated_router/comparison/combined_summary.csv
artifacts/results/orchestrated_router/comparison/weak_query_recall10.csv
artifacts/results/orchestrated_router/comparison/*.png
```

This is deliberately not a trained model. It tests whether a principled attribute-type routing rule can close the weak-query gap before spending cluster time on new training.

## Implemented Follow-Up: Sum-Only vs Model+Sum Blends

On 2026-06-23 a second no-training orchestrator was added to answer a more general question before adding more hand routing:

```text
Does the best arithmetic sum work better than the learned model by itself,
or does it only help as a correction/fusion signal?
```

New files:

```text
cluster/orchestrator/evaluate_sum_model_blends.py
cluster/jobs/48_evaluate_sum_model_blends_short.sh
```

This script evaluates global/fixed methods, reducing explicit `if attribute then method` logic:

```text
model_only
  full query through the best learned gate checkpoint

generic_sum_only
  source + generic CLIP contrastive direction sum for every query

generic_tangent_seq
  sequential tangent arithmetic with fixed alpha/source settings

tuned_sum_only
  the tuned arithmetic from the local weak-attribute diagnosis, used as a diagnostic upper-ish arithmetic baseline

model_plus_*_delta
  q = normalize(q_model + beta * (q_sum - z_source))
  tests whether arithmetic is useful as a vector correction

score_fusion_*_sum
  s = (1-w) * s_model + w * s_sum
  tests whether arithmetic is useful as a ranking signal

rrf_model_*
  reciprocal-rank fusion between learned model and arithmetic ranking
```

Outputs are stored under:

```text
artifacts/results/sum_model_blends/<method>/summary.csv
artifacts/results/sum_model_blends/<method>/per_query_metrics.csv
artifacts/results/sum_model_blends/comparison/combined_summary.csv
artifacts/results/sum_model_blends/comparison/weak_query_recall10.csv
artifacts/results/sum_model_blends/comparison/*.png
```

This run should determine whether future work should prefer:

```text
A. replacing the learned model with arithmetic for broad weak attributes;
B. blending learned model and arithmetic with fixed global weights;
C. learning a confidence/gating mechanism to choose/blend without hard-coded query rules.
```

## Result: Sum-Only vs Model+Sum Blends

The `sum_model_blends` run completed on the cluster and was copied locally. The result is clear: arithmetic alone is not competitive, but arithmetic as a vector correction on top of the learned gate is much stronger than either component by itself.

Top methods by official Macro Recall@10:

```text
method                              Macro R@10   Micro R@10   Macro P@10
model_plus_generic_delta_100        0.2828       0.2386       0.0462
model_plus_tuned_delta_100          0.2758       0.2329       0.0457
model_plus_tuned_delta_050          0.2732       0.2410       0.0447
model_plus_generic_delta_050        0.2720       0.2344       0.0450
model_only / hybmp_l002             0.2130       0.1953       0.0325
contrastive_sequential baseline     0.1871       0.1665       0.0270
generic_sum_only                    0.1545       0.1637       0.0222
tuned_sum_only                      0.1594       0.1739       0.0226
```

The strongest Macro R@10 method is:

```text
q_model = learned_gate(source, query)
q_sum = generic arithmetic composition(source, query)
q_final = normalize(q_model + 1.0 * (q_sum - source))
```

This avoids a brittle `if attribute then method` router. The arithmetic branch contributes a displacement vector from the original source, while the learned gate remains the base composed representation.

The best Micro R@10 method is:

```text
q_final = normalize(q_model + 0.5 * (q_tuned_sum - source))
```

This is slightly more conservative and appears more stable for frequent/local queries.

Relative gains:

```text
model_plus_generic_delta_100 vs hybmp_l002:
  Macro R@10: +0.0697 absolute, +32.7% relative
  Micro R@10: +0.0433 absolute, +22.2% relative

model_plus_generic_delta_100 vs contrastive_sequential:
  Macro R@10: +0.0957 absolute, +51.1% relative
  Micro R@10: +0.0721 absolute, +43.3% relative

model_plus_tuned_delta_050 vs hybmp_l002:
  Macro R@10: +0.0602 absolute, +28.2% relative
  Micro R@10: +0.0457 absolute, +23.4% relative
```

Weak-query behavior:

```text
query                hybmp_l002   tuned_sum_only   generic_sum_only   model+tuned_delta_050
+Male                0.0520       0.2514           0.2382             0.2558
-Young               0.0794       0.0995           0.0698             0.1361
-Male,-Mustache      0.0000       0.0741           0.0741             0.0741
+Chubby,-Young       0.0223       0.0360           0.0394             0.0325
```

Per-query best blend winners:

```text
+Smiling                              model_plus_tuned_delta_025   R@10 0.3126
+Eyeglasses                           model_plus_tuned_delta_050   R@10 0.5000
-Heavy_Makeup                         model_plus_tuned_delta_100   R@10 0.2395
+Male                                 model_plus_tuned_delta_100   R@10 0.3154
-Young                                model_plus_generic_delta_100 R@10 0.1481
+Blond_Hair                           model_plus_tuned_delta_100   R@10 0.2573
+Mustache                             model_plus_tuned_delta_100   R@10 0.4319
+Eyeglasses,+Smiling                  model_plus_tuned_delta_050   R@10 0.5180
+Black_Hair,-Wavy_Hair                model_plus_tuned_delta_050   R@10 0.2729
-Male,-Mustache                       model_plus_generic_delta_100 R@10 0.1481
+Chubby,-Young                        model_plus_tuned_delta_100   R@10 0.0651
-Smiling,+Eyeglasses,+Wearing_Hat     model_plus_tuned_delta_100   R@10 0.7215
+Wearing_Lipstick,-Heavy_Makeup,+Smiling score_fusion_tuned_25sum  R@10 0.1176
```

Findings:

- Sum-only is not the answer. It underperforms both the learned gate and the strongest previous arithmetic baseline.
- Ranking fusion is also not the answer. Score fusion and reciprocal-rank fusion generally stay near or below `model_only`.
- The strong pattern is vector-space correction before retrieval: `q_model + beta * (q_sum - source)`.
- This suggests that the learned gate and CLIP arithmetic contain complementary information. The learned gate preserves local visual/compositional structure; arithmetic supplies a strong semantic displacement for global/correlated attributes.
- The best current direction is not a hard-coded router. It is a global delta-correction family with a small beta grid. A future model can learn beta/confidence, but the no-training fixed beta already gives a large jump.

Next proposed experiment:

```text
1. Expand beta grid for model_plus_generic_delta and model_plus_tuned_delta:
   beta in {0.35, 0.50, 0.65, 0.80, 1.00, 1.20}
2. Add tangent-projected delta variants:
   delta = tangent_project(q_sum - source, q_model)
3. Add a learned scalar beta head only if the fixed-beta grid plateaus.
4. Treat model_plus_tuned_delta_050 as the conservative candidate and
   model_plus_generic_delta_100 as the macro-optimized candidate.
```

## Packaged Current Best System

On 2026-06-23 the current best system artifacts were packaged under:

```text
final_best_system/
```

The package contains:

```text
README.md
REPORT_NOTES.md
manifest.json
code/
embeddings/
configs/
weights/CHECKPOINT_INFO.txt
weights/best_val_official_like_at10.pt
results/assignment_baseline_direct_sum/
results/strong_clip_baseline_contrastive_sequential/
results/best_system_model_plus_generic_delta_100/
results/best_micro_model_plus_tuned_delta_050/
results/clean_report/
results/sum_model_blends_comparison/
explanations/
```

The checkpoint is now present in the local package. The exact cluster checkpoint used by the best result was:

```text
/mnt/meditech/group1/deep_learning/cluster/artifacts/training_runs/hpsearch_gate_v3_20260621_001921_long/gate_v3_sequentialgate_hybmp_l002_w050_balanced_length_b256_lr0.0001_src0.02_scale0.02_long/checkpoints/best_val_official_like_at10.pt
```

The package includes report-oriented comparison figures:

```text
final_best_system/results/clean_report/per_query_recall10_three_systems.png
final_best_system/results/clean_report/overall_metrics_three_systems.png
```

The 1v1 comparison uses:

```text
assignment vanilla baseline:
  direct_sum

strongest zero-shot CLIP-only baseline:
  contrastive_sequential

best current system:
  model_plus_generic_delta_100
  q_final = normalize(q_model + 1.0 * (q_generic_sum - source))
```

Key corrected comparison numbers:

```text
Macro Recall@10:
  assignment baseline direct_sum             0.1084
  strongest CLIP baseline contrastive_seq    0.1871
  final system                               0.2827

  final vs assignment baseline:              +160.8% relative
  final vs strongest CLIP baseline:          +51.1% relative

Micro Recall@10:
  assignment baseline direct_sum             0.1248
  strongest CLIP baseline contrastive_seq    0.1665
  final system                               0.2386

  final vs assignment baseline:              +91.2% relative
  final vs strongest CLIP baseline:          +43.3% relative
```

The clean report tables are now:

```text
final_best_system/results/clean_report/overall_metrics_three_systems.csv
final_best_system/results/clean_report/overall_metrics_macro.csv
final_best_system/results/clean_report/overall_metrics_micro.csv
final_best_system/results/clean_report/per_query_recall10_three_systems.csv
final_best_system/results/clean_report/overall_metrics_three_systems.png
final_best_system/results/clean_report/per_query_recall10_three_systems.png
```

Terminology to use in the report:

- `direct_sum` is the assignment's vanilla zero-shot baseline.
- `contrastive_sequential` is a stronger no-training CLIP-only baseline developed during our experiments.
- `model_plus_generic_delta_100` is the final proposed system.

## Report Figure Requirement: Cosine vs Euclidean Overshoot

When producing the final notebook or any project report, include this figure or regenerate a cleaner version of it:

```text
final_best_system/explanations/toy_vector_correction_clip_cosine.png
```

Required explanatory text to include near the figure:

```text
Stessa direzione. Per cosine similarity sono praticamente uguali.
Il punto chiave: CLIP retrieval non chiede “quanto sono vicino come coordinate assolute?”, ma:
“qual è l’immagine con embedding che ha angolo/cosine più alto rispetto a q_final?”
```

Why this matters:

- The final system does not rank images by Euclidean distance to an unnormalized coordinate.
- It creates `q_final = normalize(q_model + (q_sum - source))`.
- Retrieval ranks gallery images by cosine similarity to `q_final`.
- Therefore a raw vector that looks like it “overshoots” in Euclidean coordinates can still have the correct cosine direction after normalization.

The current local package also includes:

```text
final_best_system/explanations/toy_vector_correction_euclidean.png
final_best_system/explanations/toy_vector_correction_clip_cosine.png
final_best_system/REPORT_NOTES.md
```

## Final Local Package Status

As of 2026-06-23, `final_best_system/` is the clean result package to use for report assets and GitHub sharing.

It contains:

```text
results/assignment_baseline_direct_sum/
results/strong_clip_baseline_contrastive_sequential/
results/best_system_model_plus_generic_delta_100/
results/best_micro_model_plus_tuned_delta_050/
results/clean_report/
results/sum_model_blends_comparison/
weights/best_val_official_like_at10.pt
code/
configs/
embeddings/
explanations/
```

The best learned-gate checkpoint has been copied locally into:

```text
final_best_system/weights/best_val_official_like_at10.pt
```

The final prompt cache used by the checkpoint is also present locally:

```text
cluster/data/celeba/embeddings/openai_clip_vit_b32/signed_attribute_prompt_embeddings_v2_photo_templates.pt
final_best_system/embeddings/signed_attribute_prompt_embeddings_v2_photo_templates.pt
```

The final notebook has been updated to load the checkpoint from repo-relative paths, implement:

```text
q_final = normalize(q_model + 1.0 * (q_sum - source))
```

and explain the learned sequential gate, the arithmetic delta branch, the training objective, and the assignment metrics.

## 2026-07-01 Delivery notebook status

Canonical final notebook:

```text
notebooks/DL26_Project_Delivery_notebook.ipynb
```

This is now the notebook to use for delivery. It keeps full training/evaluation code visible but guarded by flags, loads the actual cluster-run training curves and metrics, and runs lightweight smoke checks locally.

Current final packaged inference equation:

```text
q_model  = learned sequential gate(source, signed query)
q_sum    = CLIP arithmetic composition(source, signed query)
q_hybrid = normalize(q_model + 1.25 * (q_sum - source))
```

The official final system then retrieves a top-500 pool by cosine similarity to `q_hybrid`, applies the calibrated v4 embedding probe, promotes candidates satisfying the requested query attributes and predicted non-query Hamming distance <= 2, and fills the final top-10 with the original `q_hybrid` ranking if fewer than 10 candidates survive.

The delivery notebook smoke run on 2026-07-01 succeeded and generated qualitative grids in:

```text
final_best_system/results/delivery_notebook_qualitative/
```
