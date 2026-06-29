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

The long `gate_v2` hpsearch completed on 2026-06-19. The best validation run was `add_l009`:

```text
add_l009:
val_official_like@10 = 0.7969
val_exact_R@10       = 0.6016
val_attr_success@10  = 0.8987
best epoch/step      = 2 / 4000
mean_rank_B          = 50.6
```

On the official JSON benchmark, `add_l009` reached:

```text
gate_v2/add_l009 official JSON:
Macro R@10 = 0.1790
Micro R@10 = 0.1645
Macro P@10 = 0.0269
Micro P@10 = 0.0249
```

This is a strong improvement over `gate_v1` (`Macro R@10 = 0.1601`) and beats `Contrastive Sum` (`Macro R@10 = 0.1707`), but it is still slightly below the best arithmetic baselines:

```text
Contrastive Sequential        Macro R@10 = 0.1871
Adaptive Tangent Sequential   Macro R@10 = 0.1869
gate_v2/add_l009              Macro R@10 = 0.1790
```

Per-query official JSON inspection showed the shape of the improvement:

```text
+Eyeglasses                       add_l009 0.4335 vs best baseline 0.2518
+Eyeglasses, +Smiling             add_l009 0.3382 vs best baseline 0.2712
-Heavy_Makeup                     add_l009 0.2102 vs best baseline 0.1686
+Mustache                         add_l009 0.2824 vs best baseline 0.2525
+Wearing_Lipstick, -Heavy_Makeup,
  +Smiling                         add_l009 0.1176 vs best baseline 0.0882
```

but the model still loses on several global, correlated, or hair-related edits:

```text
+Male                             add_l009 0.0414 vs best baseline 0.2395
+Blond_Hair                       add_l009 0.1454 vs best baseline 0.1938
+Black_Hair, -Wavy_Hair           add_l009 0.1621 vs best baseline 0.2107
-Male, -Mustache                  add_l009 0.0000 vs best baseline 0.0741
-Smiling, +Eyeglasses, +Hat       add_l009 0.3544 vs best baseline 0.4304
```

Interpretation: `gate_v2` fixed the main weakness of `gate_v1` by using real contrastive CLIP directions, but it still applies all directions as one weighted sum and normalizes only once at the end.

## 2026-06-19 Update: Learned Sequential Gate

The best non-learned method is `Contrastive Sequential`, not `Contrastive Sum`. The key difference is the normalization schedule:

```text
Contrastive Sum:
q = normalize(z_s + d_1 + d_2 + ... + d_n)

Contrastive Sequential:
q_0 = z_s
q_1 = normalize(q_0 + d_1)
q_2 = normalize(q_1 + d_2)
...
q_n = normalize(q_{n-1} + d_n)
```

`gate_v2` is closer to `Contrastive Sum`:

```text
q = normalize(z_s + edit_scale * sum(alpha_j * d_j) + residual_scale * Delta z)
```

The next architecture, `gate_v3`, is a learned version of `Contrastive Sequential`:

```text
q_0       = z_s
alpha_j   = sigmoid(gate(q_{j-1}, d_j)) * gate_max
q_j       = normalize(q_{j-1} + edit_scale * alpha_j * d_j)
Delta z   = residual_mlp([z_s, sum(alpha_j * d_j), z_s * c, |z_s - c|])
q         = normalize(q_n + residual_scale * Delta z)
```

Two variants should be tested:

```text
gate_state = current  # alpha_j sees q_{j-1}; most faithful to sequential arithmetic
gate_state = source   # alpha_j always sees z_s; simpler source-conditioned weights
```

This experiment answers a narrow question: can the learned system keep the strong inductive bias of `Contrastive Sequential` while improving it through source-conditioned step sizes? If yes, it should close the remaining gap between `gate_v2/add_l009` and the best arithmetic baseline.

If `gate_v3` improves validation but not official JSON, the likely issue is still training/evaluation mismatch: same-identity tuple supervision learns exact target movement, while official JSON accepts many cross-identity valid answers. In that case the next strategies are:

- add official-style multi-positive validation/training masks using attribute compatibility;
- add a source attribute-presence probe as explicit gate input;
- train attribute-family-specific gates for global attributes (`Male`, `Young`, `Chubby`) and local attributes (`Eyeglasses`, `Smiling`, `Mustache`);
- use query-order ensembling for sequential methods, because multi-attribute sequential composition can be order-sensitive.

## 2026-06-19 Update: Prompt v2 Ablation

The first `gate_v3` official JSON result beat the best arithmetic baseline:

```text
gate_v3/seq_l007              Macro R@10 = 0.1951
Contrastive Sequential        Macro R@10 = 0.1871
Adaptive Tangent Sequential   Macro R@10 = 0.1869
```

However, the per-query profile remains uneven. The learned sequential gate is strong on local, visually concrete attributes such as `+Eyeglasses`, `+Smiling`, `-Heavy_Makeup`, and `+Mustache`, but weak on global/correlated or prompt-sensitive attributes:

```text
+Male                         gate_v3 0.0464 vs best baseline 0.2395
-Male, -Mustache              gate_v3 0.0000 vs best baseline 0.0741
+Chubby, -Young               gate_v3 0.0205 vs best baseline 0.0497
-Smiling, +Eyeglasses, +Hat   gate_v3 0.3797 vs best baseline 0.4304
+Black_Hair, -Wavy_Hair       gate_v3 0.1963 vs best baseline 0.2107
```

The next controlled experiment is `prompt_v2`: keep the same model, training data, and best hyperparameters from `seq_l007`, but rebuild the signed text direction cache from a revised prompt ensemble.

Motivation:

- CLIP prompt engineering literature reports that natural templates such as `"a photo of a {label}"` and prompt ensembling improve zero-shot behavior compared with bare labels.
- Our original prompts were already contextual, but not always uniformly photographic.
- Some problematic prompts, especially `Male`, `Young`, and `Chubby`, used semantically broad labels that may trigger CLIP correlations rather than localized visual evidence.

Prompt v2 policy:

- use more uniform templates: `"a close-up portrait photo of ..."`, `"an image of ..."`, `"a face photo ..."`;
- avoid over-forcing semantic labels where possible;
- describe visible facial evidence for `Male`, `Young`, and `Chubby` more carefully;
- keep the experiment controlled by training only one long config with the same parameters as `seq_l007`.

The intended comparison is:

```text
seq_l007      = gate_v3 with original prompt cache
seqp2_l007    = gate_v3 with prompt_v2 cache, same architecture and hyperparameters
```

When prompt v2 results are available, compare:

- synthetic validation: `best_val_official_like@10`, `best_val_exact_R@10`, `best_val_attr_success@10`;
- official JSON macro/micro Recall@10;
- per-query deltas, especially `Male`, `Young`, `Chubby`, hair queries, and hat queries.

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

The 2026-06-20 `seqp2_ov005` analysis quantified this mismatch:

```text
same identity in top-1   ~= 69.1%
same identity in top-10  ~= 88.6%
official valid in top-10 ~= 19.4%
```

This means the best model is very good at staying near the source identity, but many official JSON queries reward cross-identity targets. Same-identity valid targets are especially rare for the weak queries:

```text
+Male                only 2.5% of source cases have any same-identity valid target
-Young               only 3.8%
+Chubby, -Young      only 0.7%
-Male, -Mustache     0.0%
```

See [Official Results and Literature Findings](official-results-and-literature-findings.md) for the full diagnosis and paper links.

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

Decision after `seqp2_ov005`: the next experiment should be an official-like multi-positive training objective built from train/validation attributes, not from `celeba_evaluation.json`. Candidate positives should satisfy the query, remain close to the source in CLIP image space, and differ in few non-query attributes. This better matches the official benchmark without leaking the official JSON into training.

Decision after job `45`: pure official-like multi-positive training slightly underperformed the previous best `seqp2_ov005` on the official JSON despite improving the synthetic validation proxy. This suggests the official-like objective is useful but too strong when it replaces exact same-identity supervision.

The next training objective is therefore hybrid:

```text
loss_retrieval = (1 - w) * exact_info_nce + w * multipositive_info_nce
```

This keeps exact target `B` as a positive for identity/source preservation, while adding a controlled fraction of benchmark-style compatible positives. The first long grid should test `w = 0.25`, `0.50`, and `0.75`, with and without the attribute-presence probe, then compare official JSON Macro/Micro R@10 against `seqp2_ov005`.

## Post-Final Experiment: Train The Deployed Blend

The clean final system selected on 2026-06-23 is not the learned gate alone. It is:

```text
q_final = normalize(q_model + 1.0 * (q_generic_sum - source))
```

where:

- `q_model` is produced by the best learned sequential gate (`hybmp_l002_w050`);
- `q_generic_sum` is the generic CLIP arithmetic query;
- `(q_generic_sum - source)` is a zero-shot correction vector;
- retrieval ranks gallery images by cosine similarity to `q_final`.

This creates an important training mismatch: all previous learned-gate training optimized `q_model`, while the deployed system uses `q_final`.

The next experimental direction is therefore to fine-tune the current best checkpoint while optimizing the blended vector directly:

```text
q_train = normalize(q_model + beta * (q_sum - source))
```

This is not a replacement for the final packaged system. It is a separate v4 experiment. The final package remains the current reproducible reference until a new run beats it on the official JSON metrics.

### Loss

The proposed v4 loss keeps the successful hybrid multi-positive objective, but applies it to `q_train`:

```text
L_retrieval = (1 - w) * exact_info_nce(q_train)
            + w * multipositive_info_nce(q_train)
```

Add optional hard-negative / triplet pressure:

```text
L_triplet = max(0, margin + cos(q_train, hard_negative) - cos(q_train, target))
```

Negatives must mask:

- the diagonal target;
- false negatives satisfying the same requested attributes;
- official-like positives under the relaxed non-query Hamming threshold.

Total experimental objective:

```text
L = L_retrieval
  + lambda_triplet * L_triplet
  + lambda_target  * (1 - cos(q_train, target))
  + lambda_source  * (1 - cos(q_train, source))
  + lambda_distill * (1 - cos(q_train, q_current_best_final))
```

The distillation term is optional and should be used only in some configs. It prevents the model from drifting too far from the already-good final system while triplet loss pushes hard negatives away.

### Long Queue Grid

The v4 grid should be centered around the current best instead of doing a broad architecture search:

- initialize from `final_best_system/weights/best_val_official_like_at10.pt` or the cluster `hybmp_l002_w050` checkpoint;
- keep the same sequential-gate architecture and prompt-v2 directions;
- sweep `beta` over `0.75`, `1.0`, `1.25`;
- sweep learning rate over `2e-5`, `5e-5`;
- sweep triplet weight over `0`, `0.05`, `0.10`, `0.20`;
- test distillation weights `0.02` and `0.05`;
- keep `multipositive_weight = 0.50`, the best previous hybrid value.

The implementation lives in:

```text
cluster/experimental/train_blend_finetune_v4.py
cluster/experimental/run_blend_finetune_v4_hpsearch.py
cluster/configs/gate_v4_blend_finetune_long_configs.json
cluster/jobs/49_hpsearch_blend_finetune_v4_long.sh
```

The output must go to new experiment folders only:

```text
artifacts/training_runs/hpsearch_blend_finetune_v4_.../
artifacts/results/blend_finetune_v4/.../
```

Do not overwrite `final_best_system/` unless the new experiment is explicitly promoted after official JSON comparison.

## Future Data Augmentation: Weak-Attribute Official-Like Pairs

The current same-identity synthetic training has an important weakness: some
global/correlated attributes rarely change within the same CelebA identity.
This affects especially:

```text
Male
Young
Chubby
Male + Mustache
Chubby + Young
```

For these attributes, the model sees too few useful source-target edits if we
only train on pairs of the same person. The official JSON benchmark, however,
does not require the same identity. It asks for images that satisfy the query
and are close in the 40-attribute CelebA space.

Decision for the v5 mixed experiment: keep the current same-identity supervision as
the dominant signal, but add a weak-attribute official-like augmentation:

```text
70% same-identity pairs
30% weak-attribute official-like augmented pairs
```

Also test a more conservative ratio:

```text
85% same-identity pairs
15% weak-attribute official-like augmented pairs
```

The augmented pool must be generated only from train split images and not from
`celeba_evaluation.json`.

### How To Build The 30%

For each source image `A`, sample queries involving weak/global attributes:

```text
+Male
-Male
+Young
-Young
+Chubby
-Chubby
+Male, +Chubby
+Male, -Young
+Chubby, -Young
-Male, -Mustache
```

Use only queries that actually change at least one source attribute. For
example, if source `A` already has `+Male`, do not create query `+Male` from
that source unless another requested attribute changes.

For each source/query pair, select target candidates `B` from the train split
that satisfy:

```text
1. B satisfies all requested query signs.
2. B is not the same image as A.
3. B has low non-query Hamming distance from A.
4. B is reasonably close to A in CLIP image-embedding cosine.
5. B is not an extreme near-duplicate/trivial candidate unless it is a true
   same-identity pair already covered by the 70% pool.
```

The practical scoring rule can be:

```text
candidate_score =
    + clip_cosine(A, B)
    - lambda_hamming * nonquery_hamming(A, B)
```

Then keep the top `M` candidates per source/query, or sample among candidates
whose score is in the top percentile. This creates multi-positive official-like
targets rather than a single artificial "correct" person.

### Practical Example

Source `A`:

```text
Female, Young, Smiling, Brown_Hair, No_Beard, no Eyeglasses
```

Weak query:

```text
+Male
```

Valid augmented targets `B` should look like:

```text
Male, Young, Smiling, Brown_Hair, preferably No_Beard, no Eyeglasses
```

They do not need to be the same identity. What matters is that they satisfy
`+Male` while preserving as many non-query attributes as possible.

For a multi-attribute weak query:

```text
+Chubby, -Young
```

Source `A`:

```text
not Chubby, Young, Smiling, Black_Hair, no glasses
```

Target candidates `B` should satisfy:

```text
Chubby, older/not Young
```

and preserve non-query attributes where possible:

```text
Smiling, Black_Hair, no glasses
```

### Main Risk

This augmentation can teach the model to retrieve "attribute-similar people"
instead of "the same person after an edit." That is closer to the official JSON
but weaker as an identity-preserving CIR model.

Therefore it should be mixed, not substituted:

```text
70% same-identity pairs preserve the edit/identity bias.
30% weak official-like pairs teach global attributes that same-identity pairs
cannot cover well.
```

Evaluate both:

```text
synthetic validation: exact target B and official_like@10
official JSON: Recall@1/5/10 and Precision@1/5/10
```

If official JSON improves but same-identity exact retrieval collapses, reduce
the augmented percentage or use it only as a second-stage fine-tune.

### Implemented v5 Mixed-Weak Experiment

The implementation is isolated from the current final packaged system:

```text
cluster/experimental/build_weak_official_like_pairs.py
cluster/experimental/train_blend_finetune_v5_mixed.py
cluster/experimental/run_mixed_weak_hpsearch_v5.py
cluster/configs/gate_v5_mixed_weak_24h_configs.json
cluster/jobs/52_mixed_weak_v5_24h.sh
```

The job first builds or reuses:

```text
artifacts/training_pairs/weak_official_like_train_len1_3.pt
```

Then it trains 24 configurations:

```text
12 configs with weak_pair_fraction = 0.30
12 configs with weak_pair_fraction = 0.15
```

Each run optimizes the deployed blend family:

```text
q_final = normalize(q_model + beta * (q_sum - source))
```

and after each completed config it evaluates the checkpoint on the official
JSON using the beta/corrector sweep. This means the next-morning outputs should
already contain:

```text
artifacts/training_runs/hpsearch_mixed_weak_v5_<timestamp>_long/
artifacts/results/mixed_weak_v5/hpsearch_mixed_weak_v5_<timestamp>_long/
```

The key aggregate files are:

```text
summary.csv
progress.txt
artifacts/results/mixed_weak_v5/.../_aggregate/best_beta_sweeps.csv
artifacts/results/mixed_weak_v5/.../_aggregate/BEST_MIXED_WEAK_METHOD.txt
```

### Deferred Idea: Adaptive Beta

The beta sweep showed that different official queries prefer different delta
strengths. A future model could learn a scalar beta from source/query features:

```text
beta = f(source_embedding, query_directions, model_query, sum_query)
```

However, do not hardcode rules such as "if Male then beta=1.5". That would
overfit known JSON query names and reduce generality. For now, the v5
experiment keeps beta global/config-level and uses the same beta grid at
evaluation time for every query family. A learned beta head remains a later
research direction.

## Next Constraint Direction: Full Official-Like Positive Sets

The assignment language mentions preserving the "core identity" of the source,
but the official JSON operationalizes identity preservation through attributes:

```text
query attributes must match the requested signs
non-query attribute Hamming distance from source <= 2
```

Therefore, a future training objective can be closer to the official benchmark
by precomputing, for each train source and generated query, a set of valid
official-like positives:

```text
P(source, query) = {
  candidate images satisfying the query signs
  and non-query Hamming distance from source <= 2
}
```

Optionally score positives inside this set by CLIP image similarity to the
source:

```text
weight(candidate) ∝ exp(cos(source, candidate) / tau)
```

This would convert the current single-target/small-batch objective into a
multi-positive retrieval objective closer to the official JSON:

```text
L = -log sum_{p in P} exp(sim(q, p) / T)
        / sum_{g in gallery_or_batch} exp(sim(q, g) / T)
```

Keep same-identity supervision as a softer auxiliary term, not the only target:

```text
L_total =
  lambda_official * L_official_like_set
  + lambda_same_id * L_same_identity
  + lambda_source * (1 - cos(q, source))
  + lambda_triplet * L_hard_negative
```

This is more principled than checking only whether the predicted top-1 image is
inside a precomputed list, because retrieval training is differentiable through
similarity scores. The list defines positives; the gradient still flows through
the query vector and dot products.

Hard negatives should be candidates that are close to the source/model query but
violate at least one official constraint. Avoid using official-like positives as
negatives. Semi-hard negatives are preferred over random negatives because they
teach the model what it is confusing with the correct set.

Open design choice for the next experiment:

```text
Option A: keep v5 mixed tuples and add a stronger weighted multi-positive loss.
Option B: precompute full official-like positive lists per source/query and train
          with a listwise / sampled-softmax objective.
```

Option B is closer to the official metric, but heavier. It should be implemented
as a separate v6 experiment, not by overwriting the current best v5 system.

### No-Leakage Policy For Official-Like Positives

Precomputing official-like positive sets is acceptable only if the split is
strict:

```text
train positives      -> CelebA partition 0 only
validation positives -> CelebA partition 1 only
official JSON test   -> CelebA partition 2 only, never used for training
```

Do not build training positives from `celeba_evaluation.json`, from test-split
image indices, or from any target list derived from the official JSON. That
would leak the benchmark solution into training and invalidate the final JSON
metrics.

It is fine to reuse the official rule on the train/validation splits:

```text
query attributes match requested signs
non-query Hamming distance <= 2
```

because the model sees only train/validation images and attributes. The final
JSON benchmark should remain a held-out test. If we want the cleanest possible
report protocol, choose hyperparameters on validation official-like positives
and run the official JSON only for the final comparison. Iterating on JSON
results is not a direct data leak into the model, but it can still overfit
research decisions to the benchmark.

### Current Data Flow: Embeddings, Not Pixels

The current learned systems do not train CLIP and do not load raw images during
the gate training/evaluation loops. Raw images are used only once to create
frozen CLIP caches.

Training inputs are cached tensors:

```text
train_image_embeddings.pt
valid_image_embeddings.pt
signed_attribute_prompt_embeddings_v2_photo_templates.pt
pair / weak-pair index files
CelebA attribute tables
```

Official evaluation inputs are also cached tensors:

```text
test_image_embeddings.pt
celeba_evaluation.json
best learned checkpoint
prompt/direction embedding cache
```

At inference, the source "image" is represented by its normalized CLIP image
embedding. The model builds a query vector from that source embedding plus the
signed condition embeddings, applies the arithmetic corrector when configured,
and ranks all cached test-gallery embeddings by cosine similarity.

## Implemented v6 Experiment: Official-Like Multi-Positive Mix

The v6 experiment has been implemented as a separate line and does not modify
the current `final_best_system/` package.

Files:

```text
cluster/experimental/build_official_like_positive_sets_v6.py
cluster/experimental/train_official_mix_v6.py
cluster/experimental/run_official_mix_hpsearch_v6.py
cluster/configs/gate_v6_official_mix_3h_configs.json
cluster/jobs/53_official_mix_v6_3h.sh
```

The key difference from v5 is the official-like component. v5 sampled one
weak/global target per source-query. v6 builds multiple train-split positives
per source-query group:

```text
source_train + query
  -> positive set of train images satisfying query signs
  -> non-query Hamming distance <= 2
  -> ranked by CLIP similarity minus a small Hamming penalty
```

The training batch is mixed as:

```text
60% official-like multi-positive + 30% same-identity + 10% weak/global
70% official-like multi-positive + 20% same-identity + 10% weak/global
```

The config file uses paired hyperparameters: each HP setting appears twice, once
with `60/30/10` and once with `70/20/10`. This isolates the effect of the
mixture ratio.

No ultra-severe embargo on attribute signatures is used. The no-leakage rule is
instead:

```text
train positive sets are built only from CelebA train split
validation remains non-JSON
celeba_evaluation.json is used only after training for Recall/Precision
```

The model architecture and deployed inference formula remain the current best
family:

```text
q_model = learned_gate(source, query)
q_sum = generic CLIP arithmetic(source, query)
q_final = normalize(q_model + beta * (q_sum - source))
```

The v6 runner evaluates each completed checkpoint on the official JSON via the
existing beta sweep. Results are written under:

```text
artifacts/training_runs/hpsearch_official_mix_v6_<timestamp>_long/
artifacts/results/official_mix_v6/hpsearch_official_mix_v6_<timestamp>_long/
```

The aggregate winner file is:

```text
artifacts/results/official_mix_v6/.../_aggregate/BEST_OFFICIAL_MIX_METHOD.txt
```

## Probe/Reranker v1: Second Stage After `q_final`

This is not a replacement for the final gate model. It is a fair version of the
oracle top-pool filtering idea:

```text
frozen final system -> top-500 candidates -> learned CelebA probe -> rerank/filter
```

Training data:

```text
input  = frozen CLIP image embedding
label  = 40 CelebA binary attributes from list_attr_celeba.txt
split  = train for fitting, valid for model selection
```

Optional augmentation:

```text
horizontal flip image -> CLIP embedding -> same attribute label
```

Unsafe augmentations such as colour jitter are intentionally not used because
they can change labels like `Blond_Hair`, `Brown_Hair`, `Pale_Skin`, or visual
makeup cues.

Evaluation variants:

```text
A hard filter:
  predicted query satisfied AND predicted non-query Hamming <= 2

B soft reranker:
  cosine(q_final, candidate)
  + query satisfaction reward
  - predicted Hamming penalty
  + source similarity reward

C hybrid:
  hard query satisfaction filter + soft Hamming/source rerank
```

The official JSON remains held out from training and probe selection.
