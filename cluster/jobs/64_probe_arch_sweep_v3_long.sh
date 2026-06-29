#!/bin/bash
#SBATCH -p meditech-long
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -N 1
#SBATCH -t 0-05:00:00
#SBATCH --signal=B:USR1@300
#SBATCH -o logs/64_probe_arch_sweep_v3_long.sh_%j.out
#SBATCH -e logs/64_probe_arch_sweep_v3_long.sh_%j.err

set -euo pipefail

cd /mnt/meditech/group1/deep_learning/cluster
mkdir -p logs artifacts/results/probe_arch_sweep_v3

echo "=== Probe architecture sweep v3 long ==="
echo "Train 18 probe configs; after each config calibrate thresholds and evaluate frozen final system on official JSON."
python3 -u experimental/probe_arch_sweep_v3.py \
  --profile long \
  --device cuda \
  --probe-batch-size 2048 \
  --eval-source-batch-size 256 \
  --top-pool 500 \
  --top-k 10 \
  --threshold-objectives accuracy f1 balanced_accuracy \
  --time-budget-seconds 17400 \
  --finalize-window-seconds 300 \
  --min-seconds-for-new-run 900

echo "=== Probe architecture sweep v3 long completed ==="
