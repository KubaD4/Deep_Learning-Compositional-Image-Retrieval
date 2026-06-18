# Deep Learning Cluster Bundle

This folder is self-contained for assignment Steps 1-3:

1. CelebA inspection and official JSON validation.
2. Frozen Hugging Face CLIP ViT-B/32 feature extraction.
3. Four zero-shot arithmetic baselines evaluated with official Recall@K and
   Precision@K.

## Folder Layout

```text
cluster/
├── data/
│   ├── celeba.zip
│   ├── celeba_evaluation.json
│   └── celeba/
│       ├── annotations/
│       └── embeddings/openai_clip_vit_b32/
├── scripts/
├── jobs/
├── notebooks/
├── artifacts/
│   ├── checkpoints/
│   └── results/baselines/
├── logs/
├── requirements.txt
└── setup_environment.sh
```

## Academic Integrity

The implementation does not use an existing third-party project repository.
It uses standard libraries and the required Hugging Face model. The metric
definitions follow the instructor starter notebook and are labelled in source.

## First Setup On Baldo

```bash
cd /mnt/meditech/group1/deep_learning/cluster
bash setup_environment.sh
sbatch jobs/00_setup_data.sh
```

After the setup job finishes:

```bash
source .venv/bin/activate
python3 scripts/verify_bundle.py
```

If extraction reaches the queue limit, submit jobs/00_setup_data.sh again or
use AUTO_RESUBMIT=1; existing files are validated and skipped.

If your cluster uses environment modules for CUDA/PyTorch, load those modules
before running setup_environment.sh. The virtual environment uses
--system-site-packages so it can reuse the cluster CUDA build.

## Embeddings

The embedding caches are generated on the cluster. Start with the test split,
which is required by all four baselines:

```bash
sbatch jobs/01_embeddings_test.sh
```

Create train and validation caches for later learned models:

```bash
sbatch jobs/02_embeddings_train.sh
sbatch jobs/03_embeddings_valid.sh
```

Every image batch is stored under:

```text
artifacts/checkpoints/embeddings/<split>/
```

If a 10-minute job ends before completion, submit the same job again. It resumes
from the last complete batch:

```bash
sbatch jobs/02_embeddings_train.sh
```

Optional automatic resubmission:

```bash
AUTO_RESUBMIT=1 sbatch jobs/02_embeddings_train.sh
```

Final caches are written to:

```text
data/celeba/embeddings/openai_clip_vit_b32/
```

Each file stores the model ID, split name, normalized embeddings, and filenames.
Embedding row i corresponds to dataset index i in that split.

## Four Baselines

Run them independently:

```bash
sbatch jobs/10_baseline_direct_sum.sh
sbatch jobs/11_baseline_direct_sequential.sh
sbatch jobs/12_baseline_contrastive_sum.sh
sbatch jobs/13_baseline_contrastive_sequential.sh
```

Or submit all four:

```bash
bash jobs/submit_all_baselines.sh
```

The methods are:

1. Required direct sum: source plus or minus positive text embeddings.
2. Direct sequential: same edits, normalizing after each condition.
3. Contrastive sum: directions built as positive prompt minus negative prompt.
4. Contrastive sequential: contrastive directions applied one at a time.

Checkpoint chunks are stored in:

```text
artifacts/checkpoints/baselines/<method>/
```

Final outputs for each method are stored separately under:

```text
artifacts/results/baselines/
├── 01_direct_sum/
├── 02_direct_sequential/
├── 03_contrastive_sum/
└── 04_contrastive_sequential/
```

Each completed method produces:

- per_query_metrics.csv
- summary.csv
- retrievals.jsonl
- config.json
- COMPLETE

After all methods complete:

```bash
sbatch jobs/20_merge_results.sh
```

This writes artifacts/results/baselines/all_methods_summary.csv.

## Adaptive Arithmetic Experiment

The fifth method keeps CLIP frozen and uses only embedding arithmetic. For each
signed contrastive direction, it scales the edit according to its alignment
with the current source and projects the edit onto the tangent plane of the
normalized CLIP space.

```bash
sbatch jobs/14_baseline_adaptive_tangent_sequential.sh
```

Results are written to:

```text
artifacts/results/baselines/05_adaptive_tangent_sequential/
```

## Learned Gate Training

The learned model keeps CLIP frozen and trains only a small gated residual MLP
on cached CLIP embeddings. The model input is:

```text
source image embedding + signed edit attributes -> retrieval query embedding
```

The first run uses same-identity training pairs where all actually changing
attributes become the query, with query length 1-3.

### 1. Required Caches

Make sure train, validation, and test image embeddings exist:

```bash
sbatch jobs/01_embeddings_test.sh
sbatch jobs/02_embeddings_train.sh
sbatch jobs/03_embeddings_valid.sh
```

Create the new prompt-ensemble text cache:

```bash
sbatch jobs/30_create_prompt_embeddings.sh
```

This writes:

```text
data/celeba/embeddings/openai_clip_vit_b32/signed_attribute_prompt_embeddings.pt
```

It is separate from `attribute_text_embeddings.pt`, which is kept for the
zero-shot baselines.

### 2. Build Training Pair Indices

```bash
sbatch jobs/31_build_training_pairs.sh
```

Outputs:

```text
artifacts/training_pairs/train_pairs_len1_3.pt
artifacts/training_pairs/valid_pairs_len1_3.pt
artifacts/training_pairs/summary.json
```

Each pair stores source index, target index, signed changed attributes, query
length, identity, filenames, and CelebA attributes.

### 3. Smoke Test On Short Queue

```bash
sbatch jobs/32_train_gate_short.sh
```

Optional explicit run directory, useful for resume:

```bash
RUN_DIR=artifacts/training_runs/gate_v1_manual_short \
sbatch --export=ALL,RUN_DIR jobs/32_train_gate_short.sh
```

If a job stops early, re-submit with the same `RUN_DIR`.

### 4. Sequential Hyperparameter Search

Short smoke search:

```bash
sbatch jobs/34_hpsearch_gate_short.sh
```

Run fewer configs:

```bash
LIMIT=2 sbatch --export=ALL,LIMIT jobs/34_hpsearch_gate_short.sh
```

Long search:

```bash
sbatch jobs/35_hpsearch_gate_long.sh
```

Each configuration creates its own run folder and the search writes:

```text
artifacts/training_runs/hpsearch_gate_v1_<timestamp>_<profile>/summary.csv
```

### 5. Long Training

```bash
sbatch jobs/33_train_gate_long.sh
```

The long queue name is set to `meditech-long` in the job file. If Baldo uses a
different partition name, check it with:

```bash
sinfo -s
```

Then either edit the `#SBATCH -p` line in `jobs/33_train_gate_long.sh` and
`jobs/35_hpsearch_gate_long.sh`, or override it at submission time with:

```bash
sbatch -p <real-long-partition> jobs/33_train_gate_long.sh
```

### 6. Official JSON Evaluation

Evaluate the latest `best_val_official_like_at10.pt` checkpoint:

```bash
sbatch jobs/36_evaluate_gate_json.sh
```

Or pass a checkpoint explicitly:

```bash
CHECKPOINT=artifacts/training_runs/<run>/checkpoints/best_val_official_like_at10.pt \
sbatch --export=ALL,CHECKPOINT jobs/36_evaluate_gate_json.sh
```

Outputs:

```text
artifacts/results/gate_model/<run>/
├── per_query_metrics.csv
├── retrievals.jsonl
├── summary.csv
├── config.json
└── COMPLETE
```

### Monitoring A Training Run

Every run writes:

```text
config.json
progress.txt
metrics.csv
checkpoints/latest.pt
checkpoints/best_val_official_like_at10.pt
checkpoints/best_val_exact_at10.pt
plots/training_curves.png
plots/validation_metrics.png
plots/query_length_breakdown.png
plots/attribute_breakdown.png
plots/gate_weights_heatmap.png
plots/retrieval_examples_epoch_XXX.png
```

Important improvements are marked in `progress.txt`:

```bash
grep "BEST_" artifacts/training_runs/*/progress.txt
```

## Monitoring

```bash
squeue -u "$USER"
tail -f logs/<job-name>_<job-id>.out
```

## Notebook

notebooks/01_clip_steps_1_2_3.ipynb contains the same logic in alternating
Markdown and Python cells for professor-facing execution. Full benchmarks remain
disabled by default in the notebook; the Slurm scripts produce the result files
that can later be loaded into the final report.
