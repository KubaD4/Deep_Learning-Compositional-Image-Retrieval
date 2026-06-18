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

CHECKPOINT_ARGS=()
if [ -n "${CHECKPOINT:-}" ]; then
  CHECKPOINT_ARGS=(--checkpoint "$CHECKPOINT")
fi

python3 -u scripts/evaluate_gate_on_json.py \
  "${CHECKPOINT_ARGS[@]}" --device cuda \
  --source-batch-size 256 --time-budget-seconds 480
STATUS=$?
if [ "$STATUS" -eq 3 ]; then
  echo "Evaluation checkpoint saved. Submit this same job again to continue."
  exit 0
fi
exit "$STATUS"
