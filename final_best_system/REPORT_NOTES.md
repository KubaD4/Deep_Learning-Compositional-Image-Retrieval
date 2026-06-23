# Report Notes

Use this folder as the clean source of final report assets.

## Required Cosine Figure

Include this image when explaining why the vector-delta correction is valid for
CLIP retrieval:

```text
final_best_system/explanations/toy_vector_correction_clip_cosine.png
```

Required nearby explanation:

```text
Stessa direzione. Per cosine similarity sono praticamente uguali.
Il punto chiave: CLIP retrieval non chiede "quanto sono vicino come coordinate assolute?", ma:
"qual è l'immagine con embedding che ha angolo/cosine più alto rispetto a q_final?"
```

## Final System Formula

```text
q_model = learned_gate(source, query)
q_sum = generic CLIP arithmetic(source, query)
q_final = normalize(q_model + 1.0 * (q_sum - source))
```

## Report Metrics

Report all assignment metrics:

```text
Recall@1, Recall@5, Recall@10
Precision@1, Precision@5, Precision@10
```

Use `results/clean_report/overall_metrics_macro.csv` and
`results/clean_report/overall_metrics_micro.csv` for the final tables.
