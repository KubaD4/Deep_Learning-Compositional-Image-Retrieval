# V6 Correction Ablation

Official JSON comparison with exactly four systems:

1. Assignment baseline (`direct_sum`).
2. Learned gate only (`model_only`).
3. Generic CLIP arithmetic only (`generic_sum_only`).
4. Learned gate + generic corrective sum (`model_plus_generic_delta_beta_1p50`).

## Macro Recall@10

- Baseline: 0.108424
- Learned gate only: 0.149197
- Generic sum only: 0.154468
- Learned gate + correction: 0.391472

## Formula

```text
q_model = learned_gate(source, query)
q_sum = normalize(source + sum signed generic CLIP directions)
q_final = normalize(q_model + 1.5 * (q_sum - source))
```
