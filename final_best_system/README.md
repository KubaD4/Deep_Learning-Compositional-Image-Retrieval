# Final Best System Package

This is the single canonical final-best package for the current project.

## Winner

`model_plus_generic_delta_beta_1p50`

Formula:

```text
q_model = learned_gate(source, query)
q_sum = generic CLIP arithmetic(source, query)
q_final = normalize(q_model + 1.5 * (q_sum - source))
```

This is the v6 official-mix learned gate checkpoint plus the generic CLIP arithmetic displacement corrector.

## Current Values

```text
Assignment baseline Macro Recall@10:   0.108424
Strong CLIP baseline Macro Recall@10:  0.187076
Final system Macro Recall@10:          0.391472

Final vs assignment baseline:          261.06%
Final vs strong CLIP baseline:         109.26%

Assignment baseline Micro Recall@10:   0.124773
Strong CLIP baseline Micro Recall@10:  0.166525
Final system Micro Recall@10:          0.313657

Final vs assignment baseline:          151.38%
Final vs strong CLIP baseline:         88.35%
```

## Important Files

- `weights/best_val_official_like_at10.pt`: best learned gate checkpoint.
- `embeddings/signed_attribute_prompt_embeddings_v2_photo_templates.pt`: prompt cache used by the checkpoint.
- `results/final_best_model_plus_generic_delta_beta_1p50/summary.csv`: official JSON aggregate metrics.
- `results/final_best_model_plus_generic_delta_beta_1p50/per_query_metrics.csv`: official JSON per-query metrics.
- `results/final_best_model_only/summary.csv`: official JSON aggregate metrics for the learned gate without the corrective sum.
- `results/correction_ablation/`: report plots comparing assignment baseline, learned gate only, and learned gate with corrective sum.
- `results/clean_report/overall_metrics_three_systems.png`: clean report plot for all official metrics.
- `results/clean_report/per_query_recall10_three_systems.png`: per-query Recall@10 plot.
- `results/clean_report/per_query_recall10_improvement.png`: per-query improvement plot.
- `code/run_final_evaluation.sh`: reruns the final evaluation from the repository caches.

## Rerun

From the repository root:

```bash
bash final_best_system/code/run_final_evaluation.sh
```

The runner expects the repository `cluster/data` caches to exist, especially `test_image_embeddings.pt` and the CelebA annotations/evaluation JSON.

## Macro vs Micro

Macro average computes the metric per query first, then averages the 14 query scores equally. Micro average pools all source-query cases together, so queries with more source images count more.
