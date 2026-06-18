# Proposed Training Strategy

## Status

This page records the training-based approach discussed by the group on 2026-06-11. It is the recommended first learned method to compare against the zero-shot CLIP arithmetic baseline.

For the concrete target directory tree, file schemas, code ownership, and call-by-call batch/epoch flow, see [PROJECT_FOLDER_AND_TRAINING_SCHEMA.md](../PROJECT_FOLDER_AND_TRAINING_SCHEMA.md).

For a full worked example using the real images `000023.jpg` and `145590.jpg`, see [PRACTICAL_TRAINING_CYCLE.md](../PRACTICAL_TRAINING_CYCLE.md).

Related pages:

- [Project Overview](project-overview.md)
- [CelebA Dataset](dataset-celeba.md)
- [Evaluation Protocol](evaluation-protocol.md)
- [Method Roadmap](method-roadmap.md)
- [CLIP Data Flow](clip-data-flow.md)

## Core Idea

Use frozen CLIP encoders to represent images and textual modifications in the same latent space. Train only a small composition network that learns how a signed set of conditions should move the source image embedding toward a suitable target embedding.

The model input is always a source image plus the desired modifications, not a full textual description of the desired target. Therefore, training queries must contain only attributes that actually change from source to target. This matches inference time, where the user provides an image and asks for edits such as `+Eyeglasses` or `-Smiling`, rather than describing the entire resulting face.

The network is not asked to generate an image. It predicts a retrieval query vector.

```text
source image + signed text conditions
                |
                v
        composition network
                |
                v
       predicted target embedding
                |
                v
 cosine similarity against cached gallery embeddings
```

Decision for implementation: precompute all frozen CLIP embeddings before training. The training loop should not run the CLIP image encoder or text encoder. It should load cached image embeddings, cached signed-attribute prompt embeddings, and a precomputed tuple index.

Required cached inputs:

```text
image_embeddings_train.pt
image_embeddings_valid.pt
signed_attribute_prompt_embeddings.pt
training_pairs_index.jsonl/parquet
validation_pairs_index.jsonl/parquet
```

This keeps the cluster training job focused on the learned composition network and makes the short-queue smoke test faster and easier to debug.

## Notation

For a source image `I_s` and target image `I_t`:

```text
z_s = normalize(CLIP_image(I_s))
z_t = normalize(CLIP_image(I_t))
```

Let `a_s` and `a_t` be their 40-dimensional CelebA attribute vectors, with values in `{-1, +1}`.

The complete signed attribute difference is:

```text
d = (a_t - a_s) / 2
```

Each component of `d` belongs to `{-1, 0, +1}`:

- `+1`: add the attribute
- `-1`: remove the attribute
- `0`: no requested change

Example:

```text
source: Eyeglasses = -1
target: Eyeglasses = +1
d[Eyeglasses] = +1
query condition: +Eyeglasses
```

The signed conditions should be converted into natural-language prompts and encoded with the frozen CLIP text encoder, for example:

- `+Eyeglasses` -> `"a face with eyeglasses"`
- `-Eyeglasses` -> `"a face without eyeglasses"`
- `+Smiling` -> `"a smiling face"`

Using CLIP text embeddings keeps the method genuinely multimodal. Feeding only the 40-dimensional attribute vector would create a closed-set attribute model and would not test textual conditioning properly.

## Prompt Construction Policy

Prompt wording matters for CLIP. Radford et al. report that replacing contextless labels with a natural prompt such as `"A photo of a {label}."` improves ImageNet zero-shot accuracy, and that embedding-space prompt ensembling improves results further. OpenAI's CLIP reference example also uses the template `"a photo of a {class}"` for zero-shot classification.

Sources:

- CLIP paper, Section 3.1.4, prompt engineering and ensembling: <https://arxiv.org/pdf/2103.00020>
- OpenAI CLIP reference code: <https://github.com/openai/CLIP>
- CoOp prompt-learning paper: <https://arxiv.org/abs/2109.01134>
- CoCoOp conditional prompt-learning paper: <https://arxiv.org/abs/2203.05557>

Decision for the first learned run: use a fixed, versioned manual prompt dictionary rather than generating prompts dynamically. Each CelebA attribute must have a prompt ensemble with two or three positive variants and two or three negative variants, for example:

```yaml
Eyeglasses:
  positive:
    - "a face with eyeglasses"
    - "a portrait of a person wearing glasses"
    - "a close-up face with glasses"
  negative:
    - "a face without eyeglasses"
    - "a portrait of a person without glasses"
    - "a close-up face with no glasses"
```

For an attribute sign, encode all variants with the frozen CLIP text encoder, average their normalized embeddings, and normalize the result again. This produces one stable text vector per signed attribute. A single-prompt representation is not the default method; it can be kept only as an ablation.

```text
c(+A) = normalize(mean(normalize(CLIP_text(prompt_k(+A)))))
```

For a multi-attribute query such as `+Eyeglasses -Beard`, keep attributes separate:

```text
[c(+Eyeglasses), c(-Beard)]
```

The gate/attention module can then learn how much each signed condition should influence the source embedding. A single combined sentence such as `"a face with eyeglasses and without beard"` should be kept only as an ablation, because previous local diagnostics showed that CLIP can be fragile with long compositions and negations.

Learned soft prompts, such as CoOp/CoCoOp-style context vectors, are a promising later experiment. They are not recommended for the first run because they add a second learning mechanism and make it harder to isolate whether improvements come from the composition model or from prompt adaptation.

## Building Training Tuples

Join the following files on image filename:

- `identity_CelebA.txt`
- `list_attr_celeba.txt`
- `list_eval_partition.txt`

Use only partition `0` to build training tuples and partition `1` for validation. CelebA identities are disjoint across the official train, validation, and test splits, so this does not leak test identities into training.

For every identity in the training split:

1. collect all images of that identity;
2. create directional candidate pairs `(I_s, I_t)`;
3. find the set `D` of attributes that differ between source and target;
4. use the whole difference set as the edit query, `Q = D`;
5. derive the signs from the direction source -> target;
6. keep the tuple only if the query length is inside the chosen training range.

This produces tuples of the form:

```text
(source image, signed textual conditions, target image)
```

The signed textual conditions are edit instructions only. They should not include attributes that are already shared by source and target, because those would turn the task into target description rather than compositional image retrieval from a reference image.

For the first learned run, use multi-attribute edit queries rather than forcing every tuple to one condition. If source and target differ on multiple attributes, the query should contain all those changes. This better matches inference time: the model receives a source image plus the desired modifications, not a full target description and not a partial edit that omits changes actually present in the target.

The official evaluation JSON contains 14 query templates:

- 8 single-attribute queries;
- 4 two-attribute queries;
- 2 three-attribute queries.

Therefore, the first training run should focus on same-identity pairs where `|D|` is 1, 2, or 3. This still gives a multi-attribute model while staying aligned with the benchmark. Longer same-identity differences exist and are common, but they are more likely to combine many incidental annotation changes and should be introduced later as a curriculum or robustness experiment.

Observed same-identity directional pair counts in the local CelebA train split:

```text
|D| = 1:  85,766
|D| = 2: 206,666
|D| = 3: 355,740
```

This gives about 648k train tuples before additional balancing/subsampling.

Do not materialize every possible tuple for every epoch. The training split contains about 3.72 million directional same-identity pairs before filtering. Use an offline filtered index or an on-the-fly balanced sampler.

## Sampling Rules

The sampler should:

- balance positive and negative directions for each attribute;
- balance query lengths of one, two, and three conditions;
- avoid identities with many images dominating the dataset;
- avoid pairs with no attribute changes;
- prioritize attributes used by the official benchmark;
- sample only a limited number of tuples per identity and epoch;
- keep train and validation tuples strictly inside their official partitions.

Do not randomly split generated pairs across train/validation/test. Keep the official CelebA partitions: partition `0` for training tuples, partition `1` for validation tuples, and partition `2` only for official-style retrieval evaluation. Balance query lengths and attributes inside each partition through sampling.

Decision for the first learned run: use a balanced query-length sampler by default:

```text
1/3 tuples with |D| = 1
1/3 tuples with |D| = 2
1/3 tuples with |D| = 3
```

Also support a `natural` sampling mode in config, where tuple lengths follow the observed dataset distribution. This enables a direct ablation:

```text
gate_v1 balanced_length
gate_v1 natural_length
```

If both are run, compare validation and official JSON metrics to decide whether balancing improves generalization or distorts the real distribution.

Useful first-pass editable attributes are:

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

Some labels such as `Male`, `Young`, and `Attractive` may change across photographs because of annotation noise rather than a real physical edit. These should be analyzed carefully rather than assumed to be clean transformations.

## Composition Network

Decision for the first learned run: use the gated multi-attribute architecture immediately, because the first training tuples already contain one to three edit attributes.

Recommended first architecture:

1. freeze the CLIP image and text encoders;
2. encode the source image as `z_s`;
3. encode each signed text condition as `c_j`;
4. use a small gating or attention module to compute condition weights `alpha_j` based on both `z_s` and `c_j`;
5. aggregate the conditions into `c`;
6. predict a residual movement `Delta z` with a small MLP;
7. add the residual to the source embedding and normalize.

```text
alpha_j = gate(z_s, c_j)
c       = aggregate(alpha_j * c_j)
Delta z = MLP([z_s, c])
q       = normalize(z_s + Delta z)
```

`q` is the predicted embedding used for retrieval. The residual form is preferable to predicting a completely new vector because it explicitly starts from the source and encourages preservation of its visual information.

For a query with one attribute, the gate reduces to a learned edit strength for that attribute. For a query with two or three attributes, it learns their relative importance conditioned on the source image. This is the first learned version of the earlier intuition that not all edits should have the same magnitude for every source image.

The first gate should produce independent sigmoid weights, not a softmax over attributes:

```text
alpha_j = sigmoid(MLP_gate([z_s, c_j, z_s * c_j, |z_s - c_j|]))
```

## 2026-06-18 Update: Residual-Only Gate vs Additive Gate

The first implemented learned model, now referred to as `gate_v1`, used the following residual-only composition:

```text
alpha_j    = sigmoid(gate(z_s, c_j))
c          = sum(alpha_j * c_j)
Delta z    = residual_mlp([z_s, c, z_s * c, |z_s - c|])
q          = normalize(z_s + residual_scale * Delta z)
```

This trained successfully and achieved strong synthetic validation scores (`val_official_like@10 = 0.7573` for the best long run), but it did not beat the best contrastive arithmetic baseline on the official JSON benchmark:

```text
Official JSON Macro R@10:
Contrastive Sequential        0.1871
Adaptive Tangent Sequential   0.1869
Contrastive Sum               0.1707
gate_v1 learned residual      0.1601
```

Per-query analysis showed that `gate_v1` works well for local visual edits:

```text
+Eyeglasses    gate_v1 0.3780 vs best baseline 0.2518
-Heavy_Makeup  gate_v1 0.1977 vs best baseline 0.1686
+Mustache      gate_v1 0.2791 vs best baseline 0.2525
```

but loses on global or correlated attributes:

```text
+Male                         gate_v1 0.0313 vs best baseline 0.2395
+Black_Hair, -Wavy_Hair       gate_v1 0.1326 vs best baseline 0.2107
-Smiling, +Eyeglasses, +Hat   gate_v1 0.3038 vs best baseline 0.4304
```

Interpretation: the residual-only model asks the MLP to rediscover useful CLIP edit directions from the prompt embeddings. The contrastive arithmetic baseline already has a strong inductive bias because it directly adds CLIP-space directions:

```text
d_A = normalize(t_positive(A) - t_negative(A))
q   = normalize(z_s + d_A)
```

Therefore the next architecture, `gate_v2`, should not replace this strong baseline. It should learn to weight and correct it:

```text
d_j        = signed contrastive CLIP direction
alpha_j    = sigmoid(gate(z_s, d_j)) * gate_max
c          = sum(alpha_j * d_j)
Delta z    = residual_mlp([z_s, c, z_s * c, |z_s - c|])
q          = normalize(z_s + edit_scale * c + residual_scale * Delta z)
```

This makes the learned model a source-conditioned extension of Contrastive Sequential rather than a separate residual-only regressor. The first short additive smoke test supports this change:

```text
add_s001: edit_scale=1.0, gate_max=1.5, residual_scale=0.02
          val_official_like@10 = 0.7207

add_s002: same but residual_scale=0.0
          val_official_like@10 = 0.6797
```

The residual-free ablation dropping badly suggests that a small learned correction is useful, but the direct additive contrastive direction should remain the main movement.

Recommended next experiment: run `gate_v2` long hpsearch and evaluate the best `add_l*` checkpoint on the official JSON. The target to beat is:

```text
Macro R@10 >= 0.1871
Micro R@10 >= 0.1668
```

Softmax is differentiable, so the reason to avoid it is not differentiability. The problem is that softmax forces attributes to compete: if one edit gets a larger weight, the others must receive smaller weights. Independent sigmoid gates are more appropriate because multiple requested edits can all be important at the same time.

The gate is not given ground-truth scalar weights. It learns them indirectly from retrieval loss. If increasing the weight of `+Eyeglasses` helps move `q` toward targets with eyeglasses and away from safe negatives, gradients will increase the gate behavior for similar source/query cases. If an edit is already visually present or weakly relevant for a source image, the best retrieval direction may require a smaller residual, and the gate can learn to reduce that edit strength.

The gate can learn approximate source-conditioned notions such as:

- whether the requested attribute appears already present in the source embedding;
- whether the attribute direction is reliable for this source;
- whether several requested edits conflict or reinforce each other;
- whether an edit tends to require a large or small movement in CLIP space.

However, the gate weight is not a calibrated probability that an attribute is present. It is an internal learned edit strength optimized for retrieval. If calibrated attribute presence is needed, train or evaluate a separate attribute probe on frozen CLIP embeddings using CelebA labels.

Future extension: add an attribute-presence probe on top of frozen CLIP image embeddings:

```text
p_attr = sigmoid(MLP_attr_probe(z_s))  # 40 CelebA attribute probabilities
```

This probe can be trained with the CelebA attribute labels and then used in two ways:

- diagnostic only: estimate whether CLIP embeddings contain enough information to detect each attribute;
- gate input: provide `p_attr[j]` or signed presence features to the gate, so it can better decide whether a requested edit is already present, absent, or uncertain in the source image.

This should not be part of the first run, because it adds another supervised component. It is a good second-stage improvement if the learned gate needs a clearer source-conditioned estimate of attribute presence.

For CLIP ViT-B/32, the projected embedding dimension is normally 512, but the implementation should read the model projection dimension rather than hard-code it.

For a detailed distinction between JPEG images, preprocessed tensors, internal encoder outputs, projected embeddings, cached features, and gallery scores, see [CLIP Data Flow](clip-data-flow.md).

### First-Run Model Size

Use a small MLP architecture. The task is to learn a controlled residual in an already meaningful CLIP space, not to learn a large vision-language model from scratch.

Recommended first configuration:

```text
clip_dim = 512

gate input per attribute:
  [z_s, c_j, z_s * c_j, abs(z_s - c_j)]  -> 2048 dims

gate MLP:
  LayerNorm(2048)
  Linear(2048, 512)
  GELU
  Dropout(0.1)
  Linear(512, 128)
  GELU
  Linear(128, 1)
  Sigmoid

residual input:
  [z_s, c_agg, z_s * c_agg, abs(z_s - c_agg)] -> 2048 dims

residual MLP:
  LayerNorm(2048)
  Linear(2048, 1024)
  GELU
  Dropout(0.1)
  Linear(1024, 512)
  GELU
  Dropout(0.1)
  Linear(512, 512)

q = normalize(z_s + residual_scale * Delta_z)
```

Start with `residual_scale = 0.2` or make it a small learned scalar initialized near `0.1`. This avoids early training steps moving query embeddings too aggressively away from the source image.

This gives:

- 2 hidden layers in the gate;
- 2 hidden layers in the residual MLP;
- enough capacity to model source-conditioned edit strength;
- low enough capacity to debug quickly on the short queue.

If the model underfits, increase hidden sizes or add one more residual layer. If it overfits or destabilizes retrieval, reduce residual scale, add dropout, or add stronger validation-based checkpointing.

## Training Objective

The primary loss should be contrastive retrieval loss. For a batch of `B` tuples, compare every predicted query `q_i` with every frozen target embedding `z_tj`:

```text
S_ij = cosine(q_i, z_tj) / temperature
L_contrastive = CrossEntropy(S, diagonal_targets)
```

This teaches the model to place the correct target above the other targets in the batch.

This should be interpreted as a ranking objective, not as forcing `q_i` to collapse exactly onto one isolated target point. The desired practical behavior is:

- the known target `z_ti` should rank high;
- visually/attribute-wise similar images should remain nearby in the global CLIP gallery;
- unrelated batch targets should be pushed lower.

For the first learned run, use a single explicit positive target `B` per tuple. This keeps the training signal simple and debuggable. The final retrieval system, however, must still be top-k/KNN-style: after predicting `q`, rank the full gallery by cosine similarity and return the nearest images.

In this context, "near the target" means that the known target should receive a higher cosine score than the safe negatives in the batch. It does not mean enforcing an absolute cosine radius around the target, and it does not mean that all other images must be far away in the final gallery.

The batch is a set of training tuples:

```text
(source_image_i, signed_edit_conditions_i, target_image_i, metadata_i)
```

For each tuple:

```text
q_i = model(source_image_i, signed_edit_conditions_i)
z_ti = CLIP_image(target_image_i)
```

The score matrix compares every predicted query with every target in the batch:

```text
S_ij = cosine(q_i, z_tj) / temperature
```

The diagonal `S_ii` is the explicit positive. Off-diagonal targets `S_ij` are useful negatives only if they are not also valid answers for query `i`.

To avoid pushing away images that the assignment would consider valid, the first training code should include a false-negative guard. For each off-diagonal candidate target `j`, use the CelebA attributes to check whether it already satisfies the requested edit for source `i` and whether its non-query attributes are within the relaxed Hamming threshold used by the assignment. If so, do not treat it as a negative:

- simplest implementation: avoid placing that pair in the same batch;
- better implementation: mask `S_ij` out of the cross-entropy denominator;
- later implementation: treat all such images as multi-positive targets.

For the first run, the recommended compromise is masked InfoNCE with a simple attribute-based false-negative mask when feasible; otherwise use a sampler that avoids obvious ambiguous negatives.

An optional auxiliary cosine loss can stabilize early training:

```text
L_cos = 1 - cosine(q_i, z_ti)
L = L_contrastive + lambda * L_cos
```

The contrastive loss should remain primary because the final task is ranking, not exact embedding regression.

Decision for the first learned run: use a configurable small source-preservation regularizer:

```text
L = L_masked_InfoNCE
    + lambda_target * (1 - cosine(q_i, z_ti))
    + lambda_source * (1 - cosine(q_i, z_si))
```

Recommended defaults:

```text
lambda_target = 0.1
lambda_source = 0.02
```

`lambda_source` must be configurable from `config.json`, including `0.0`, because too much source preservation can block real edits. Monitor whether this regularizer is hurting learning by logging:

- average `cosine(q, z_source)`;
- average `cosine(q, z_target)`;
- validation `attr_success@K`;
- validation `official_like@K`;
- per-loss components.

Warning sign: `cosine(q, z_source)` stays very high, while `attr_success@K` remains low. This suggests the model is preserving the source too strongly and not applying the edit. In that case reduce `lambda_source`, reduce residual-scale constraints, or temporarily disable the preservation term.

One caveat is that another target in the batch may also be valid for the same source/query. Such an image becomes a false negative under ordinary diagonal InfoNCE. The first run should at least avoid or mask these false negatives when they can be detected from attributes. A later improvement can use a full multi-positive mask based on the attribute ground-truth rule.

Do not use `celeba_evaluation.json` during training or hyperparameter tuning. The official JSON should be used only after the model pipeline is ready: feed each JSON source image and query into the trained model, retrieve top-k test images from the gallery, and compare those rankings with the official target sets.

Do not add an explicit triplet loss in the first run. InfoNCE already provides in-batch negatives and is easier to debug. Hard-negative mining, semi-hard negatives, triplet loss, or multi-positive losses should be second-stage experiments after the single-positive retrieval pipeline is working.

## Retrieval At Test Time

Precompute and freeze CLIP image embeddings for all 19,962 test images.

For each source/query entry in `celeba_evaluation.json`:

1. obtain the source image with `celeba[source_index]`;
2. encode the source and query conditions;
3. compute `q` with the composition network;
4. rank every test image by cosine similarity with `q`;
5. exclude the source image itself;
6. evaluate the ranked dataset indices against the JSON ground truth using Recall@K and Precision@K.

This KNN-style retrieval step is part of the final system even if the first training objective uses one target image per tuple. The model produces a query vector; nearest-neighbor search over cached CLIP image embeddings produces the ranked result list.

## Validation Gallery

A gallery is the set of candidate images searched by nearest-neighbor retrieval. The model predicts one query embedding `q`; the gallery contains precomputed CLIP image embeddings; retrieval ranks gallery images by cosine similarity to `q`.

During validation, the validation gallery should be all images from the official CelebA validation partition (`list_eval_partition.txt == 1`) for which embeddings are available. Validation tuples are generated only from that same validation partition. This gives a retrieval setting analogous to test-time evaluation without touching the official test JSON.

For each validation tuple:

```text
source image A + edit query Q -> predicted q
rank all validation gallery images by cosine(q, z_gallery)
exclude source A from the ranked list
measure whether target B appears in top-k
measure whether top-k images satisfy the requested edit attributes
```

Recommended validation metrics:

- `val_exact_R@K`: whether the known target `B` appears in the top-k;
- `val_attr_success@K`: whether retrieved images satisfy the requested edits according to CelebA labels;
- `val_official_like@K`: whether retrieved images satisfy the requested edits and stay within the relaxed non-query Hamming threshold;
- `val_mean_rank_B`: average rank of the known target `B`.

`val_attr_success@K` is a soft diagnostic metric: it asks whether the model understood the requested edit attributes. `val_official_like@K` is stricter and closer to the assignment: it asks whether the model performs the edit while preserving the rest of the attribute profile. If `val_attr_success` is high but `val_official_like` is low, the model probably understands the edit but fails to preserve source-like attributes.

Save:

- `val_exact_R@1/5/10`;
- `val_attr_success@1/5/10`;
- `val_official_like@1/5/10`;
- `val_mean_rank_B`.

Use `best_val_official_like@10` as the main checkpoint selection criterion. Also save a debug checkpoint for `best_val_exact_R@10`.

Save validation metrics after each epoch in `metrics.csv` and append human-readable status updates to `progress.txt` with timestamps, because cluster jobs may not provide an interactive console.

Important progress events should be easy to find in a long log. Prefix improvement lines with a stable marker:

```text
[2026-06-17 15:48:03] BEST_OFFICIAL_LIKE epoch=4 val_official_like@10=0.1842 saved=checkpoints/best_val_official_like_at10.pt
[2026-06-17 16:02:11] BEST_EXACT epoch=5 val_exact_R@10=0.2310 saved=checkpoints/best_val_exact_at10.pt
```

This allows quick inspection with:

```bash
grep "BEST_" progress.txt
```

Each training run should create a timestamped output folder whose name ends with the queue/profile type:

```text
artifacts/training_runs/gate_v1_YYYYMMDD_HHMMSS_short/
artifacts/training_runs/gate_v1_YYYYMMDD_HHMMSS_long/
```

Recommended run directory:

```text
config.json
progress.txt
metrics.csv
checkpoints/latest.pt
checkpoints/best_val_official_like_at10.pt
checkpoints/best_val_exact_at10.pt
samples/val_examples_epoch_XXX.jsonl
plots/training_curves.png
plots/validation_metrics.png
plots/query_length_breakdown.png
plots/attribute_breakdown.png
plots/gate_weights_heatmap.png
plots/retrieval_examples_epoch_XXX.png
logs/stdout.log
logs/stderr.log
```

The PNG files are report-oriented artifacts. Generate cheap metric plots after each validation. Generate qualitative retrieval grids less frequently, for example every validation epoch or only for the best checkpoint:

- `training_curves.png`: train loss and learning rate over time;
- `validation_metrics.png`: `exact`, `attr_success`, and `official_like` at K over epochs;
- `query_length_breakdown.png`: performance for 1-, 2-, and 3-attribute queries;
- `attribute_breakdown.png`: performance grouped by edited attribute;
- `gate_weights_heatmap.png`: average learned gate weights by attribute/sign;
- `retrieval_examples_epoch_XXX.png`: source image, query text, target image, and top-k retrieved images.

The short queue profile can run more than one epoch if needed. It should be limited by `max_steps` and wall time rather than by exactly one epoch. A good default is 20-40 tiny epochs or a small `max_steps` cap, whichever finishes first. The long profile should resume automatically from `checkpoints/latest.pt`.

## Hyperparameter Search

The cluster code should support sequential hyperparameter search: one Slurm job can run multiple training configurations one after another, each in its own output directory. This matches the previous workflow used on the cluster and makes it easy to explore a direction without manually submitting every configuration.

Each config should create a directory name containing both the run profile and a compact config id:

```text
artifacts/training_runs/gate_v1_cfg001_balanced_b256_lr1e-4_src0.02_short/
artifacts/training_runs/gate_v1_cfg002_balanced_b256_lr3e-4_src0.00_short/
artifacts/training_runs/gate_v1_cfg003_natural_b256_lr1e-4_src0.02_long/
```

The search runner should write a global summary file:

```text
artifacts/training_runs/hpsearch_YYYYMMDD_HHMMSS/summary.csv
```

with one row per configuration:

```text
config_id, run_dir, status, best_val_official_like@10, best_val_exact_R@10, best_epoch, notes
```

Initial hyperparameters worth sweeping:

- learning rate: `1e-4`, `3e-4`;
- `lambda_source`: `0.0`, `0.02`;
- sampler mode: `balanced_length`, `natural_length`;
- residual scale initialization: `0.1`, `0.2`;
- dropout: `0.0`, `0.1`.

Keep the first search small. Use the short queue to validate that each config starts, logs, checkpoints, and produces plots. Use the long queue for a smaller set of promising configurations.

Decision for the first hyperparameter search: use a small manual list of 6-8 configurations rather than a full grid. This avoids wasting cluster time before the pipeline is proven stable.

Suggested first configs:

```text
cfg001 balanced lr1e-4 src0.02 scale0.1 drop0.1
cfg002 balanced lr3e-4 src0.02 scale0.1 drop0.1
cfg003 balanced lr1e-4 src0.00 scale0.1 drop0.1
cfg004 natural  lr1e-4 src0.02 scale0.1 drop0.1
cfg005 balanced lr1e-4 src0.02 scale0.2 drop0.1
cfg006 balanced lr1e-4 src0.02 scale0.1 drop0.0
```

After the short-queue smoke search, select a smaller set for long training based primarily on `best_val_official_like@10`, while also checking `attr_success@10`, `exact_R@10`, and retrieval example PNGs.

## Important Mismatch With The Benchmark

Same-identity pairs provide strong supervision for identity preservation, but the official JSON does not require the target to have the same real identity. Its notion of similarity is based only on the 40 CelebA attributes and the relaxed Hamming-distance rule.

Therefore, same-identity training is a useful inductive bias, not a perfect reproduction of the evaluation target distribution.

The recommended experiments are:

- zero-shot CLIP arithmetic baseline;
- learned residual model trained on same-identity pairs;
- learned model trained on benchmark-style attribute pairs, which may use different identities;
- optional hybrid sampler mixing both pair types.

Decision for the first learned run: use same-identity pairs only. This is the cleanest way to define a concrete source-target edit because CelebA provides image identity, image attributes, and partition metadata. In this setting, the target image is another photo of the same person where the queried attribute actually changes.

Benchmark-style or hybrid pairs are still possible later, but their target image has a different meaning. For a source image and signed edit query, a benchmark-style target would be sampled from any training image whose attributes satisfy the requested edit and whose non-query attributes remain within the relaxed Hamming threshold. This matches the official evaluation logic better, but it no longer represents the same person after an edit. It is therefore noisier for learning identity-preserving composition and should be treated as a second-stage experiment.

The hybrid sampler remains a later candidate: same-identity pairs teach visual preservation, while benchmark-style pairs align training with the official evaluation definition. It should not be used in the first debug run.

## Minimal First Experiment

Start small:

- frozen CLIP ViT-B/32;
- multi-attribute edit tuples where the query contains all attributes that truly change from source to target;
- same-identity pairs only, with total attribute Hamming distance from 1 to 3;
- gated residual MLP composition network;
- InfoNCE plus a small cosine auxiliary loss, trained with one explicit positive target per tuple;
- validation on the official validation split with synthetically generated queries;
- final comparison against the zero-shot baseline on the untouched JSON test benchmark.

After this pipeline works, test longer edit sets, benchmark-style targets, hybrid sampling, and more advanced multi-positive losses.

The first run deliberately does not plug KNN/multi-positive targets into the loss. KNN is used after prediction for retrieval. Multi-positive training can be added as a second experiment once the single-positive pipeline is verified.
