# Final Best System Package

This folder is the canonical final-best package for the project. It is designed to run from a cloned repository without relying on `/Users/kuba/...` or the Baldo cluster path.

## Winner

```text
v4_m04_deep_asl_lr2e4_d00_query_hamming_fill_accuracy
```

The system has two stages.

Stage 1: hybrid compositional retrieval vector:

```text
q_model  = learned sequential gate(source, signed query)
q_sum    = CLIP arithmetic composition(source, signed query)
q_hybrid = normalize(q_model + 1.25 * (q_sum - source))
```

Stage 2: calibrated CelebA attribute probe reranking:

```text
1. retrieve top-500 candidates with q_hybrid
2. predict candidate/source attributes with the trained probe
3. promote candidates that satisfy:
   - requested query attributes
   - predicted non-query Hamming distance <= 2
4. if fewer than top-10 pass, fill the remaining slots with q_hybrid ranking
```

## Current Official JSON Values

```text
Macro Recall@10     0.4786552757255102
Micro Recall@10     0.4054520150066562
Macro Precision@10  0.08641455658635512
Micro Precision@10  0.06939065714631452
avg kept from 500   33.64725281374803
```

The saved aggregate is:

```text
results/probe_embedding_v4_m04/A_cal_query_hardh2_accuracy/summary.csv
```

If `results/probe_embedding_v4_m04/` is not present, the runnable code falls back to the previous packaged probe in `results/probe_reranker_v2_calibrated/`, whose Macro Recall@10 is `0.47297`.

## Important Files

- `weights/best_val_official_like_at10.pt`: best v7 learned gate checkpoint.
- `results/probe_embedding_v4_m04/probe/best_probe.pt`: best embedding MLP CelebA attribute probe checkpoint.
- `results/probe_embedding_v4_m04/probe/calibrated_thresholds.pt`: per-attribute calibrated thresholds.
- `results/probe_embedding_v4_m04/probe/test_probe_probs.pt`: cached probe probabilities for CelebA test images.
- `results/probe_reranker_v2_calibrated/`: older fallback probe package kept for reproducibility.
- `data/celeba/embeddings/openai_clip_vit_b32/test_image_embeddings.pt`: test gallery CLIP image embeddings.
- `data/celeba/embeddings/openai_clip_vit_b32/signed_attribute_prompt_embeddings_v2_photo_templates.pt`: prompt cache used by the learned gate.
- `data/celeba/annotations/list_attr_celeba.txt`: CelebA attributes used by probe/Hamming checks.
- `data/celeba_evaluation.json`: official evaluation JSON.

## Setup

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r final_best_system/code/requirements.txt
```

The final metrics can be recomputed with the packaged tensors and annotations. Rendering image grids additionally requires the CelebA aligned image folder:

```text
celeba/img_align_celeba/
```

If your images are elsewhere, pass `--images-dir /path/to/img_align_celeba`.

## Smoke Test

```bash
bash final_best_system/code/run_current_best_smoke.sh
```

This runs one source-query case on CPU and prints progress plus a summary. The metric values in the smoke test are not meaningful; it only verifies that the full package loads correctly.

## Full Local Evaluation

```bash
bash final_best_system/code/run_current_best_full_eval.sh
```

or:

```bash
bash final_best_system/code/run_final_evaluation.sh
```

On CPU this is slower than the cluster run because it evaluates all official source-query cases.

## Official JSON Visualization

```bash
.venv/bin/python final_best_system/code/show_json_retrieval_example.py \
  --query-id 5 \
  --source-index 3 \
  --top-k 10
```

The script draws:

- blue: source image;
- light green: official JSON-valid target;
- yellow: satisfies requested query attributes but is not official JSON-valid;
- red: fails at least one requested query attribute.

## Free Known-Attribute Query

```bash
.venv/bin/python final_best_system/code/show_json_retrieval_example.py \
  --free-source-index 47 \
  --free-query "+Eyeglasses" \
  --top-k 10 \
  --top-pool 50
```

Known attributes are checked with CelebA labels/probe logic.

## Free Open-Vocabulary Query

```bash
.venv/bin/python final_best_system/code/show_json_retrieval_example.py \
  --free-source-index 47 \
  --free-query "+Sunglasses" \
  --top-k 10 \
  --top-pool 50
```

Unknown attributes are converted into CLIP directions using prompt pairs such as:

```text
a portrait photo of a face with sunglasses
a portrait photo of a face without sunglasses
```

Open-vocabulary conditions are qualitative only because CelebA has no official labels for them.

## Mixed Multi-Attribute Query

```bash
.venv/bin/python final_best_system/code/show_json_retrieval_example.py \
  --free-source-index 66 \
  --free-query "-visible teeth +Eyeglasses" \
  --top-k 10 \
  --top-pool 50
```

Known attributes can be mixed with open-vocabulary attributes. The learned gate handles the known prompt-cache directions; unknown concepts use CLIP text-difference directions.

## Generated Free-Query Outputs

Local free-query visualizations are intentionally ignored by git:

```text
results/free_query_examples/
```

Regenerate them from `show_json_retrieval_example.py` whenever needed.

## Macro vs Micro

Macro averages compute each metric per query first, then average the 14 query scores equally. Micro averages pool all source-query cases together, so queries with more source images count more.
