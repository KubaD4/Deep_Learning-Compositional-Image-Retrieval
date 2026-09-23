# Report Notes

Final system: learned sequential gate plus generic CLIP arithmetic delta corrector, followed by calibrated CelebA attribute-probe reranking.

Mathematical form:

```text
q_model = Gate(source, signed query conditions)
q_sum = GenericCLIPSum(source, signed query conditions)
q_final = normalize(q_model + 1.25 * (q_sum - source))
```

The correction is directional. Retrieval uses cosine similarity after normalization, so the key question is not absolute Euclidean position but whether `q_final` has a better angle toward valid target embeddings.

Use the current final summary in `results/probe_embedding_v4_m04/A_cal_query_hardh2_accuracy/`, the top-pool diagnostic in `explanations/`, and the toy cosine plot in `explanations/` for the notebook/report explanation.
