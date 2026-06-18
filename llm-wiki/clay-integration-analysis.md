# CLAY Relationship And Integration Decision

## Decision

The project does not need to replace the proposed signed gated residual network with full CLAY.

The assignment requires CLIP ViT-B/32, a frozen offline visual database, a vanilla CLIP arithmetic baseline, and an original fusion mechanism that improves on CLAY's rigid multi-condition pre-SVD stacking. Full CLAY is therefore architectural context and a useful comparison, not a mandatory final model.

The proposed residual network already addresses the central assignment requirement:

```text
alpha_j = gate(z_source, c_j, sign_j)
c_fused = aggregate(alpha_j * c_j)
delta_z = residual_mlp([z_source, c_fused])
q = normalize(z_source + delta_z)
```

Unlike CLAY's unweighted concatenation of condition prompt banks before SVD, the gate can assign source-dependent weights to positive, negative, redundant, or conflicting conditions.

## Important Task Difference

Original CLAY performs conditional similarity retrieval. A condition such as `action`, `species`, or `color` specifies the aspect along which the query image and gallery images should be considered similar. CLAY does not normally edit the query from one attribute value to another.

The assignment instead uses signed target constraints such as `+Eyeglasses` or `-Young`. The method must move the source representation toward a requested target state while preserving non-requested properties. Therefore, applying original CLAY directly to the source and gallery would emphasize whether candidates share the source's value along an attribute axis; for an edit query, this can be the opposite of the desired behavior.

## Recommended Experimental Role

Use CLAY in three roles:

1. Related-work motivation: explain that CLAY keeps VLM image embeddings fixed and changes the similarity space using a condition-derived textual subspace.
2. Optional training-free baseline: implement the official or a clearly labelled simplified CLAY-style conditional projection, while noting that it is not naturally designed for signed attribute edits.
3. CLAY-inspired hybrid ablation: retain the residual composer and add a dynamically fused semantic-subspace score or loss.

The minimum required comparison remains:

```text
image-only CLIP
signed CLIP arithmetic
signed gated residual model (ours)
```

A CLAY baseline and hybrid variant strengthen the scientific comparison but should not block the first end-to-end implementation.

## Recommended Hybrid Placement

Insert a CLAY-inspired subspace module after the residual network produces `q` and before cosine ranking.

For each attribute, build a small prompt bank containing both semantic poles, for example variants of `with eyeglasses` and `without eyeglasses`. Encode these prompts with frozen CLIP and derive a low-rank attribute basis or projector `P_j`.

The existing condition gate produces `alpha_j`. Use those weights to construct a query-dependent positive-semidefinite metric rather than concatenating all prompts and treating them equally:

```text
M_Q = epsilon * I + sum_j alpha_j * P_j
```

The sign controls the desired movement in the residual composer; the projector identifies the semantic attribute subspace. This distinction matters because `+Eyeglasses` and `-Eyeglasses` concern opposite directions within the same attribute family.

Two practical integration variants are useful.

### Variant A: Dynamic Metric At Ranking

```text
q = residual_composer(z_source, conditions)
score_edit(i) = metric_cosine(q, z_i; M_Q)
score_global(i) = cosine(q, z_i)
score(i) = lambda * score_global(i) + (1 - lambda) * score_edit(i)
```

This is the safest first hybrid because it leaves the current training architecture intact and can be evaluated as a scoring ablation.

### Variant B: Edit/Preservation Decomposition

In a Euclidean first implementation:

```text
score_edit(i) = cosine(P_Q q, P_Q z_i)
score_preserve(i) = cosine((I - P_Q) z_source, (I - P_Q) z_i)
score(i) = lambda_edit * score_edit(i) + lambda_preserve * score_preserve(i)
```

This directly models the benchmark rule: satisfy requested attributes in the selected subspace and preserve the source outside it. A manifold-aware tangent-space version can be tested later, following CLAY's rotation and logarithmic map, but it should be an ablation rather than the first implementation.

## Training Integration

The CLAY-inspired module can also supply auxiliary losses:

```text
L = L_contrastive
    + lambda_edit * L_edit(P_Q q, P_Q z_target)
    + lambda_preserve * L_preserve((I - P_Q) q, (I - P_Q) z_source)
```

Keep contrastive retrieval loss primary. The auxiliary decomposition is useful only if validation shows that it improves signed multi-condition queries.

## Risks

- Projecting only onto requested attributes can discard the non-query information that the benchmark expects to preserve; use a global or complement-space term.
- A single text prompt is insufficient for CLAY-style SVD. Prompt banks must contain varied descriptions and both attribute poles.
- The official CLAY code concatenates prompt lists for multiple conditions and computes one SVD. Reproducing that behavior alone does not satisfy the assignment's requested improvement.
- Full hyperspherical geometry adds implementation and numerical risk. Start with the Euclidean dynamic-metric variant, then ablate the manifold-aware version.
- Do not train or tune with `celeba_evaluation.json`; construct train/validation tuples from the official CelebA partitions.

## Recommended Order

1. Complete frozen CLIP extraction and the official evaluator.
2. Run image-only and signed-arithmetic baselines.
3. Implement and train the documented signed gated residual network.
4. Add a CLAY or simplified CLAY-style baseline.
5. Add Variant A as the main CLAY-inspired hybrid ablation.
6. Try Variant B or manifold-aware geometry only if time and validation results justify it.

## Primary Sources

- Assignment: `Project assignment - V1.2.pdf`
- Local readable assignment: `source-markdown/Project assignment - V1.2.md`
- CLAY paper: <https://arxiv.org/abs/2604.11539>
- CLAY project page: <https://sohwi-lim.github.io/CLAY/>
- Official CLAY implementation: <https://github.com/kaist-ami/CLAY>

