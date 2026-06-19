#!/bin/bash
#SBATCH -p meditech-short
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -N 1
#SBATCH -t 0-00:10:00
#SBATCH --signal=B:USR1@60
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -u
ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$ROOT"; mkdir -p logs
[ -f .venv/bin/activate ] && source .venv/bin/activate

python3 -u scripts/run_hpsearch.py \
  --profile short \
  --device cuda \
  --configs configs/gate_sequential_short_configs.json \
  --limit "${LIMIT:-0}"
