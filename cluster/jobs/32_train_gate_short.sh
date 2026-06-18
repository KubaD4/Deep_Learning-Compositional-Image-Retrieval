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

EXTRA_ARGS=()
[ -n "${RUN_DIR:-}" ] && EXTRA_ARGS+=(--run-dir "$RUN_DIR")

python3 -u scripts/train_gate_model.py \
  --profile short --device cuda --time-budget-seconds 480 \
  "${EXTRA_ARGS[@]}"
STATUS=$?
if [ "$STATUS" -eq 3 ]; then
  echo "Checkpoint saved. Re-submit with the same RUN_DIR to continue."
  if [ "${AUTO_RESUBMIT:-0}" = "1" ] && [ -n "${RUN_DIR:-}" ]; then
    sbatch --export=ALL,RUN_DIR="$RUN_DIR",AUTO_RESUBMIT=1 "$ROOT/jobs/32_train_gate_short.sh"
  fi
  exit 0
fi
exit "$STATUS"
