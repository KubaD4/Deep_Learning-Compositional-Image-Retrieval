#!/bin/bash
#SBATCH -p meditech-long
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -N 1
#SBATCH -t 0-05:00:00
#SBATCH --signal=B:USR1@300
#SBATCH -o logs/65_probe_arch_sweep_v4_finetune_long.sh_%j.out
#SBATCH -e logs/65_probe_arch_sweep_v4_finetune_long.sh_%j.err

set -euo pipefail

cd /mnt/meditech/group1/deep_learning/cluster
mkdir -p logs artifacts/results/probe_arch_sweep_v4_finetune

STAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_ROOT="artifacts/results/probe_arch_sweep_v4_finetune/probe_arch_sweep_v4_finetune_${STAMP}_long"

echo "=== Probe architecture sweep v4 fine-tune long ==="
echo "Output root: ${OUTPUT_ROOT}"
echo "Train 50 focused probe configs; after each config calibrate thresholds and evaluate frozen final system on official JSON."
echo "Also saves train_metrics.csv per probe and training-curve PNGs under ${OUTPUT_ROOT}/training_curves."

python3 -u experimental/probe_arch_sweep_v4_finetune.py \
  --profile long \
  --device cuda \
  --output-root "${OUTPUT_ROOT}" \
  --probe-batch-size 2048 \
  --eval-source-batch-size 256 \
  --top-pool 500 \
  --top-k 10 \
  --threshold-objectives accuracy f1 balanced_accuracy \
  --time-budget-seconds 17400 \
  --finalize-window-seconds 300 \
  --min-seconds-for-new-run 420

echo "=== Probe architecture sweep v4 fine-tune long completed ==="
