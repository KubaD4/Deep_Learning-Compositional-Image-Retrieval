# Final Best System Package

This folder collects the assignment baseline, strongest CLIP-only baseline, final system, and clean report plots.

## Winner For Primary Report Number

`model_plus_generic_delta_100`

Formula:

```text
q_model = learned_gate(source, query)
q_sum = generic CLIP arithmetic(source, query)
q_final = normalize(q_model + 1.0 * (q_sum - source))
```

Why this wins: it has the best current Macro Recall@10, which is the cleanest query-balanced summary of the assignment's primary Recall@K metric.

## Baselines

Assignment vanilla baseline:

```text
direct_sum
q = normalize(v_ref + t_A - t_B + ...)
```

Strongest zero-shot CLIP-only baseline:

```text
contrastive_sequential
q_i = normalize(q_{i-1} + d_i)
```

## Current Values

```text
Assignment baseline Macro Recall@10:   0.108424
Strong CLIP baseline Macro Recall@10:  0.187076
Final system Macro Recall@10:          0.282750

Final vs assignment baseline:          160.78%
Final vs strong CLIP baseline:         51.14%

Assignment baseline Micro Recall@10:   0.124773
Strong CLIP baseline Micro Recall@10:  0.166525
Final system Micro Recall@10:          0.238594

Final vs assignment baseline:          91.22%
Final vs strong CLIP baseline:         43.28%
```

## Important Files

- `results/clean_report/overall_metrics_three_systems.csv`
- `results/clean_report/overall_metrics_macro.csv`
- `results/clean_report/overall_metrics_micro.csv`
- `results/clean_report/per_query_recall10_three_systems.csv`
- `results/clean_report/overall_metrics_three_systems.png`
- `results/clean_report/per_query_recall10_three_systems.png`
- `results/assignment_baseline_direct_sum/summary.csv`
- `results/strong_clip_baseline_contrastive_sequential/summary.csv`
- `results/best_system_model_plus_generic_delta_100/summary.csv`
- `code/evaluate_sum_model_blends.py`

## Macro vs Micro

Macro average:

```text
1. compute the metric inside each query;
2. average the 14 query-level scores equally.
```

Micro average:

```text
pool all source-query cases together, so queries with more source images count more.
```

For the report, Macro Recall@K is the cleanest query-balanced number. Micro Recall@K is still useful because it shows performance over all evaluated source cases.

## Checkpoint Note

If present, the best learned-gate checkpoint is stored at:

```text
weights/best_val_official_like_at10.pt
```

If that file is missing after cloning, place a compatible checkpoint at that path.
The committed package is expected to include it, so this should normally not be needed.
