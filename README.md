# Deep Learning - Compositional Image Retrieval

Course project for Deep Learning, UNITN 2025/2026.

We address the assignment on dynamic and hybrid conditioning for compositional image retrieval on CelebA. The repository contains the final notebook, the final packaged system, reusable evaluation scripts, and cluster utilities used during experimentation.

## Authors

- Di Quattro Kuba
- Giacomo Vettore
- Danilo Frailis

## Final Submission Notebook

The final delivery notebook to read/run is:

```text
notebooks/DL26_Project_Delivery_notebook.ipynb
```

It is written as a self-contained report: method, training strategy, results, ablations, qualitative examples, runnable code cells, and an appendix with the cluster source used for the final runs. Heavy training/evaluation cells are guarded by boolean flags so the notebook can be opened and partially rerun without launching hours of computation.

The older working notebook is kept for development history:

```text
notebooks/02_learned_gate_final_pipeline.ipynb
```

## Final Best System

The packaged final system lives in:

```text
final_best_system/
```

Current final method:

```text
v4_m04_deep_asl_lr2e4_d00_query_hamming_fill_accuracy
```

High-level pipeline:

```text
q_model  = learned sequential gate(source, signed query)
q_sum    = CLIP arithmetic composition(source, signed query)
q_hybrid = normalize(q_model + 1.25 * (q_sum - source))

top-500 candidates by cosine(q_hybrid, image)
-> calibrated CelebA attribute probe
-> promote candidates satisfying query + predicted non-query Hamming <= 2
-> fill remaining top-10 with original q_hybrid order
```

Best official JSON result found so far:

```text
Macro Recall@10     0.4786552757255102
Micro Recall@10     0.4054520150066562
Macro Precision@10  0.08641455658635512
```

The previous packaged probe (`current_A_query_hardh2_accuracy`) scored Macro Recall@10 `0.47297`. The newer v4 embedding MLP probe should be placed in `final_best_system/results/probe_embedding_v4_m04/`; the code automatically falls back to the older packaged probe if those files are not present.

The detailed package documentation is in:

```text
final_best_system/README.md
```

## Setup

Create a virtual environment from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r final_best_system/code/requirements.txt
```

The final evaluation package includes the lightweight artifacts needed for metric computation:

```text
final_best_system/data/celeba_evaluation.json
final_best_system/data/celeba/annotations/list_attr_celeba.txt
final_best_system/data/celeba/embeddings/openai_clip_vit_b32/test_image_embeddings.pt
final_best_system/weights/best_val_official_like_at10.pt
final_best_system/results/probe_embedding_v4_m04/probe/best_probe.pt
```

The full CelebA image folder is not stored in git. To render qualitative image grids, place aligned CelebA images at:

```text
celeba/img_align_celeba/
```

or pass a custom folder with `--images-dir`.

## Run The Final System

Smoke test, CPU-safe:

```bash
bash final_best_system/code/run_current_best_smoke.sh
```

Full official JSON evaluation:

```bash
bash final_best_system/code/run_current_best_full_eval.sh
```

Equivalent runner:

```bash
bash final_best_system/code/run_final_evaluation.sh
```

## Visualize An Official CelebA JSON Query

This uses an official query id and a valid source index from `celeba_evaluation.json`:

```bash
.venv/bin/python final_best_system/code/show_json_retrieval_example.py \
  --query-id 5 \
  --source-index 3 \
  --top-k 10
```

If a source is not valid for that query, the script tells you. A list of valid source examples is saved in:

```text
final_best_system/results/qualitative_examples/valid_source_indices_by_query.csv
```

## Visualize A Free Query

Known CelebA attributes can be used outside the official JSON:

```bash
.venv/bin/python final_best_system/code/show_json_retrieval_example.py \
  --free-source-index 47 \
  --free-query "+Eyeglasses" \
  --top-k 10 \
  --top-pool 50
```

Unknown/open-vocabulary CLIP concepts are also accepted. They are treated as CLIP text directions and are qualitative only because CelebA has no official labels for them:

```bash
.venv/bin/python final_best_system/code/show_json_retrieval_example.py \
  --free-source-index 47 \
  --free-query "+Sunglasses" \
  --top-k 10 \
  --top-pool 50
```

Mixed known/open multi-attribute query:

```bash
.venv/bin/python final_best_system/code/show_json_retrieval_example.py \
  --free-source-index 66 \
  --free-query "-visible teeth +Eyeglasses" \
  --top-k 10 \
  --top-pool 50
```

Generated free-query images are intentionally ignored by git under:

```text
final_best_system/results/free_query_examples/
```

so local qualitative experiments do not pollute the submitted repository.

## Baselines And Diagnostics

The final notebook and `final_best_system/results/` include the comparison against:

- assignment-style direct CLIP sum baseline;
- strongest zero-shot CLIP arithmetic baseline;
- learned hybrid system without probe filtering;
- final learned hybrid system with calibrated probe filtering.

The diagnostic scripts are kept because they explain why the final design changed over time:

```text
final_best_system/code/test_oracle_top500_rerank.py
final_best_system/code/test_oracle_filter_components.py
final_best_system/code/probe_filtering/
```

They are analysis tools, not extra hidden inference assumptions.
