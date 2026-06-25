# Report Notes

Final system: learned sequential gate plus generic CLIP arithmetic delta corrector.

Mathematical form:

```text
q_model = Gate(source, signed query conditions)
q_sum = GenericCLIPSum(source, signed query conditions)
q_final = normalize(q_model + 1.5 * (q_sum - source))
```

The correction is directional. Retrieval uses cosine similarity after normalization, so the key question is not absolute Euclidean position but whether `q_final` has a better angle toward valid target embeddings.

Use the clean report plots in `results/clean_report/` and the toy cosine plot in `explanations/` for the notebook/report explanation.
