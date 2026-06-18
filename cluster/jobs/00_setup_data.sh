#!/bin/bash
#SBATCH -p meditech-short
#SBATCH --ntasks=1
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH -N 1
#SBATCH -t 0-00:10:00
#SBATCH --signal=B:USR1@60
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -u
ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$ROOT"
mkdir -p logs
[ -f .venv/bin/activate ] && source .venv/bin/activate

echo "=== Extracting and validating CelebA ==="
python3 -u scripts/setup_data.py --time-budget-seconds 480
STATUS=$?
if [ "$STATUS" -eq 3 ]; then
  echo "Extraction checkpoint saved. Submit this same job again."
  [ "${AUTO_RESUBMIT:-0}" = "1" ] && sbatch --gres=gpu:0 "$ROOT/jobs/00_setup_data.sh"
  exit 0
fi
[ "$STATUS" -ne 0 ] && exit "$STATUS"
python3 -u scripts/verify_bundle.py
