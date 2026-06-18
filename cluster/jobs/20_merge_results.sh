#!/bin/bash
#SBATCH -p meditech-short
#SBATCH --ntasks=1
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH -N 1
#SBATCH -t 0-00:10:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -u
ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$ROOT"; mkdir -p logs
[ -f .venv/bin/activate ] && source .venv/bin/activate
python3 -u scripts/merge_baseline_summaries.py
