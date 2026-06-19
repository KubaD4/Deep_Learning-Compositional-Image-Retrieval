# Learned Gate Cluster Runbook

This page records the operational plan for the first learned gated residual training run on Baldo.

The conceptual strategy is documented in [Proposed Training Strategy](training-strategy.md). The implemented cluster scripts live under `cluster/scripts/`, `cluster/jobs/`, and `cluster/configs/`.

## Implemented Components

| Component | Path | Purpose |
|---|---|---|
| Prompt ensemble config | `cluster/configs/attribute_prompts.json` | 2-3 natural CLIP prompts for each positive/negative CelebA attribute. |
| Manual hpsearch config list | `cluster/configs/gate_hpsearch_configs.json` | Six initial short/long search configurations. |
| Prompt embedding cache script | `cluster/scripts/create_prompt_embeddings.py` | Builds `signed_attribute_prompt_embeddings.pt` from prompt ensembles. |
| Pair index script | `cluster/scripts/build_training_pairs.py` | Builds same-identity train/valid edit tuples with query length 1-3. |
| Learned model core | `cluster/scripts/learned_gate_core.py` | Gate + residual MLP architecture and checkpoint helpers. |
| Training script | `cluster/scripts/train_gate_model.py` | Trains the model, validates, logs progress, saves checkpoints and PNGs. |
| Hpsearch runner | `cluster/scripts/run_hpsearch.py` | Runs several configs sequentially and writes a search summary. |
| Official JSON evaluator | `cluster/scripts/evaluate_gate_on_json.py` | Evaluates a trained checkpoint against `celeba_evaluation.json`. |
| Additive-gate short configs | `cluster/configs/gate_additive_short_configs.json` | Two-config smoke test for `gate_v2`, which directly adds gated CLIP contrastive directions. |
| Additive-gate long configs | `cluster/configs/gate_additive_long_configs.json` | Ten long configs varying edit scale, gate max, residual scale, learning rate, and temperature. |
| Additive-gate short job | `cluster/jobs/39_hpsearch_additive_gate_short.sh` | Short queue smoke test for `gate_v2`. |
| Additive-gate long job | `cluster/jobs/40_hpsearch_additive_gate_long.sh` | Long queue hpsearch for `gate_v2`. |
| Sequential-gate short configs | `cluster/configs/gate_sequential_short_configs.json` | Two-config smoke test for `gate_v3`, the learned sequential contrastive composer. |
| Sequential-gate long configs | `cluster/configs/gate_sequential_long_configs.json` | Ten long configs varying sequential gate state, edit scale, gate max, residual scale, learning rate, and temperature. |
| Sequential-gate short job | `cluster/jobs/41_hpsearch_sequential_gate_short.sh` | Short queue smoke test for `gate_v3`. |
| Sequential-gate long job | `cluster/jobs/42_hpsearch_sequential_gate_long.sh` | Long queue hpsearch for `gate_v3`. |

## Slurm Corrections To Preserve

Baldo-specific issues previously observed:

- CPU-only jobs must explicitly request `--gres=gpu:0`.
- GPU jobs request `--gres=gpu:1`.
- Job scripts must use `SLURM_SUBMIT_DIR` as root because Slurm copies scripts into `/var/spool`.
- Python commands in Slurm jobs should use `python3 -u` so logs are not buffered.
- The `logs/` directory should exist before submission because Slurm opens output files before the script body runs.

The current `cluster/jobs/*.sh` files have been checked for these conditions.

## Execution Table

Run all commands from the cluster bundle root:

```bash
cd /mnt/meditech/group1/deep_learning/cluster
```

| Step | Command | What It Does | Expected Output / Where To Check |
|---|---|---|---|
| 0 | `python3 scripts/verify_bundle.py` | Checks Python, CUDA visibility, evaluation JSON, caches, prompt cache, and pair indices. | Terminal prints missing/ready items. Missing train/valid cache means submit embedding jobs. |
| 1 | `sbatch jobs/00_setup_data.sh` | Extracts CelebA ZIP if needed and verifies data. | `logs/00_setup_data.sh_<jobid>.out`; extracted images in `data/celeba/img_align_celeba/`. |
| 2 | `sbatch jobs/01_embeddings_test.sh` | Builds/resumes frozen CLIP test image embeddings. Required for baselines and final JSON evaluation. | `data/celeba/embeddings/openai_clip_vit_b32/test_image_embeddings.pt`; chunks in `artifacts/checkpoints/embeddings/test/`. |
| 3 | `sbatch jobs/02_embeddings_train.sh` | Builds/resumes frozen CLIP train image embeddings. Required for learned training. | `data/celeba/embeddings/openai_clip_vit_b32/train_image_embeddings.pt`; chunks in `artifacts/checkpoints/embeddings/train/`. |
| 4 | `sbatch jobs/03_embeddings_valid.sh` | Builds/resumes frozen CLIP validation image embeddings. Required for learned validation. | `data/celeba/embeddings/openai_clip_vit_b32/valid_image_embeddings.pt`; chunks in `artifacts/checkpoints/embeddings/valid/`. |
| 5 | `sbatch jobs/30_create_prompt_embeddings.sh` | Creates prompt-ensemble text embeddings for signed attributes. | `data/celeba/embeddings/openai_clip_vit_b32/signed_attribute_prompt_embeddings.pt`. |
| 6 | `sbatch jobs/31_build_training_pairs.sh` | Builds same-identity train/valid tuple indices where all changed attributes become the query and `|D| in {1,2,3}`. | `artifacts/training_pairs/train_pairs_len1_3.pt`, `valid_pairs_len1_3.pt`, `summary.json`. |
| 7 | `python3 scripts/verify_bundle.py` | Confirms all learned-training prerequisites are ready. | Should report train cache, validation cache, prompt ensemble cache, and training pair indices ready. |
| 8 | `sbatch jobs/32_train_gate_short.sh` | Short smoke training of gate + residual MLP. | New folder under `artifacts/training_runs/*_short/`; inspect `progress.txt`, `metrics.csv`, `plots/*.png`, `checkpoints/*.pt`. |
| 9 | `grep "BEST_" artifacts/training_runs/*/progress.txt` | Quickly finds checkpoint improvements. | Lines like `BEST_OFFICIAL_LIKE ...` and `BEST_EXACT ...`. |
| 10 | `LIMIT=2 sbatch --export=ALL,LIMIT jobs/34_hpsearch_gate_short.sh` | Runs first 2 hpsearch configs as a safer short test. | `artifacts/training_runs/hpsearch_gate_v1_*_short/summary.csv` plus one run folder per config. |
| 11 | `sbatch jobs/34_hpsearch_gate_short.sh` | Runs the full six-config short hpsearch. | Same hpsearch folder; compare `summary.csv` and report PNGs. |
| 12 | `sinfo -s` | Checks the real long partition name before long jobs. | If not `meditech-long`, submit with `sbatch -p <real-partition> ...` or edit jobs `33` and `35`. |
| 13 | `sbatch jobs/33_train_gate_long.sh` | Runs one long training with default config. | `artifacts/training_runs/*_long/`; monitor `progress.txt`, `metrics.csv`, checkpoint files, and plots. |
| 14 | `sbatch jobs/35_hpsearch_gate_long.sh` | Runs long hpsearch over the manual config list. | `artifacts/training_runs/hpsearch_gate_v1_*_long/summary.csv`; select best config by `best_val_official_like@10`. |
| 15 | `sbatch jobs/36_evaluate_gate_json.sh` | Evaluates latest `best_val_official_like_at10.pt` on official JSON. | `artifacts/results/gate_model/<run>/summary.csv`, `per_query_metrics.csv`, `retrievals.jsonl`, `COMPLETE`. |
| 16 | `CHECKPOINT=artifacts/training_runs/<run>/checkpoints/best_val_official_like_at10.pt sbatch --export=ALL,CHECKPOINT jobs/36_evaluate_gate_json.sh` | Evaluates a specific checkpoint. | Same as above, but tied to the selected run. |
| 17 | `sbatch jobs/39_hpsearch_additive_gate_short.sh` | Runs the `gate_v2` additive-gate smoke test. | `artifacts/training_runs/hpsearch_gate_v2_*_short/summary.csv`; `add_s001` completed with `val_official_like@10 = 0.7207`. |
| 18 | `sbatch jobs/40_hpsearch_additive_gate_long.sh` | Runs the long `gate_v2` hpsearch. | `artifacts/training_runs/hpsearch_gate_v2_*_long/summary.csv`; select best `add_l*` by `best_val_official_like@10`, then evaluate on JSON. |
| 19 | `python3 scripts/summarize_gate_runs.py --top 50` | Summarizes all gate runs across short/long and v1/v2/v3. | `artifacts/results/gate_model/all_gate_runs_summary.csv`; look for `add_l*` or `seq_l*` rows after long jobs. |
| 20 | `sbatch jobs/38_plot_official_comparison.sh` | Rebuilds official comparison plots after evaluating a new gate checkpoint. | `artifacts/results/official_comparison/combined_summary.csv`, `method_comparison.csv`, and PNG plots. |
| 21 | `sbatch jobs/41_hpsearch_sequential_gate_short.sh` | Runs the `gate_v3` learned-sequential smoke test. | `artifacts/training_runs/hpsearch_gate_v3_*_short/summary.csv`; check that both `seq_s*` configs complete. |
| 22 | `sbatch jobs/42_hpsearch_sequential_gate_long.sh` | Runs the long `gate_v3` hpsearch. | `artifacts/training_runs/hpsearch_gate_v3_*_long/summary.csv`; select best `seq_l*` by `best_val_official_like@10`, then evaluate on JSON. |

## Current Learned Results

As of 2026-06-18, the official JSON benchmark is still led by the contrastive arithmetic baselines:

```text
Official JSON Macro R@10:
Contrastive Sequential        0.1871
Adaptive Tangent Sequential   0.1869
Contrastive Sum               0.1707
gate_v1 learned residual      0.1601
Direct Sequential             0.1121
Direct Sum                    0.1084
```

The best `gate_v1` training validation run was:

```text
ov015:
val_official_like@10 = 0.7573
val_exact_R@10       = 0.5574
val_attr_success@10  = 0.8677
best epoch/step      = 2 / 4000
```

But its official JSON evaluation did not beat the best arithmetic methods:

```text
gate_v1 official JSON:
Macro R@10 = 0.1601
Micro R@10 = 0.1552
Macro P@10 = 0.0237
Micro P@10 = 0.0233
```

Per-query inspection showed that `gate_v1` is strong on local visual edits such as `+Eyeglasses`, `-Heavy_Makeup`, and `+Mustache`, but weak on global/correlated edits such as `+Male`, `+Black_Hair,-Wavy_Hair`, and `+Chubby,-Young`.

The next implemented architecture is `gate_v2` additive-gate:

```text
q = normalize(z_source + edit_scale * sum(alpha_j * d_j) + residual_scale * delta)
```

where `d_j` is a signed contrastive CLIP direction. The short smoke test completed:

```text
add_s001: residual_scale=0.02, edit_scale=1.0, gate_max=1.5
          val_official_like@10 = 0.7207

add_s002: residual_scale=0.0, edit_scale=1.0, gate_max=1.5
          val_official_like@10 = 0.6797
```

This suggests the direct contrastive movement plus a small residual is more promising than removing the residual entirely. The long additive hpsearch is the next decisive experiment.

As of 2026-06-19, the long `gate_v2` additive hpsearch completed. Best validation config:

```text
add_l009:
val_official_like@10 = 0.7969
val_exact_R@10       = 0.6016
val_attr_success@10  = 0.8987
best epoch/step      = 2 / 4000
mean_rank_B          = 50.6
```

Official JSON ranking after evaluating `add_l009`:

```text
Official JSON Macro R@10:
Contrastive Sequential        0.1871
Adaptive Tangent Sequential   0.1869
gate_v2/add_l009              0.1790
Contrastive Sum               0.1707
gate_v1/ov015                 0.1601
Direct Sequential             0.1121
Direct Sum                    0.1084
```

`gate_v2/add_l009` is a real improvement over `gate_v1` and beats `Contrastive Sum`, but it still trails `Contrastive Sequential`. The next test is `gate_v3`, which applies the learned weighted directions sequentially and normalizes after every edit:

```text
q_0 = z_source
q_j = normalize(q_{j-1} + edit_scale * alpha_j * d_j)
q   = normalize(q_n + residual_scale * delta)
```

This directly tests whether the remaining gap is caused by the normalization schedule rather than by the learned gate itself.

## Interpreting Training Outputs

Main files in a learned training run:

| File | Meaning |
|---|---|
| `progress.txt` | Human-readable progress log with timestamped `BEST_` markers. |
| `metrics.csv` | Per-validation metrics and loss components. |
| `checkpoints/latest.pt` | Resume checkpoint. |
| `checkpoints/best_val_official_like_at10.pt` | Main model selection checkpoint. |
| `checkpoints/best_val_exact_at10.pt` | Debug checkpoint focused on retrieving the exact target `B`. |
| `plots/training_curves.png` | Train loss curves. |
| `plots/validation_metrics.png` | Validation exact/attribute/official-like metrics over time. |
| `plots/query_length_breakdown.png` | Performance for query length 1, 2, and 3. |
| `plots/attribute_breakdown.png` | Hardest edited attributes. |
| `plots/gate_weights_heatmap.png` | Average learned gate weight by signed attribute. |
| `plots/retrieval_examples_epoch_XXX.png` | Qualitative source/query/target/top-k examples for the report. |

## Health Checks

Use these checks while jobs run:

```bash
squeue -u "$USER"
tail -f logs/<job-name>_<job-id>.out
grep "BEST_" artifacts/training_runs/*/progress.txt
find artifacts/training_runs -name COMPLETE
```

Warning signs:

- `cos_q_source` stays very high while `val_attr_success@10` stays low: source preservation may be too strong.
- `val_attr_success@10` improves but `val_official_like@10` stays low: the model learns edits but fails to preserve non-query attributes.
- `val_exact_R@10` is low while `val_official_like@10` is acceptable: the model may retrieve valid alternatives rather than the exact same-identity target.

Official JSON targets to beat:

```text
Macro R@10 >= 0.1871
Micro R@10 >= 0.1668
```
