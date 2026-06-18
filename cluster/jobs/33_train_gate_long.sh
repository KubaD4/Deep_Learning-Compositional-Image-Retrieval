#!/bin/bash
#SBATCH -p meditech-long
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -N 1
#SBATCH -t 0-06:00:00
#SBATCH --signal=B:USR1@120
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -u
ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$ROOT"; mkdir -p logs
[ -f .venv/bin/activate ] && source .venv/bin/activate

EXTRA_ARGS=()
[ -n "${RUN_DIR:-}" ] && EXTRA_ARGS+=(--run-dir "$RUN_DIR")
[ -n "${CONFIG:-}" ] && EXTRA_ARGS+=(--config "$CONFIG")

python3 -u scripts/train_gate_model.py \
  --profile long --device cuda \
  "${EXTRA_ARGS[@]}"
STATUS=$?
if [ "$STATUS" -eq 3 ]; then
  echo "Checkpoint saved. Re-submit with the same RUN_DIR to continue."
  exit 0
fi
exit "$STATUS"
