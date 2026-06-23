#!/bin/bash
#SBATCH -p meditech-long
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -N 1
#SBATCH -t 0-12:00:00
#SBATCH --signal=B:USR1@120
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -u
ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$ROOT"; mkdir -p logs
[ -f .venv/bin/activate ] && source .venv/bin/activate

python3 -u scripts/create_prompt_embeddings.py \
  --device cuda \
  --prompt-config configs/attribute_prompts_v2_photo_templates.json \
  --output data/celeba/embeddings/openai_clip_vit_b32/signed_attribute_prompt_embeddings_v2_photo_templates.pt

python3 -u scripts/run_hpsearch.py \
  --profile long \
  --device cuda \
  --configs configs/gate_sequential_official_like_long_configs.json \
  --limit "${LIMIT:-0}" \
  --evaluate-best-json \
  --plot-after-json
