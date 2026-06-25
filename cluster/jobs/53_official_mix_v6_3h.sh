#!/bin/bash
#SBATCH -p meditech-long
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -N 1
#SBATCH -t 0-03:00:00
#SBATCH --signal=B:USR1@300
#SBATCH -o logs/53_official_mix_v6_3h.sh_%j.out
#SBATCH -e logs/53_official_mix_v6_3h.sh_%j.err

set -euo pipefail

cd /mnt/meditech/group1/deep_learning/cluster
mkdir -p logs artifacts/training_pairs

echo "=== Build/reuse v6 official-like train positive sets ==="
python3 -u experimental/build_official_like_positive_sets_v6.py \
  --split train \
  --preset official \
  --device cuda \
  --output artifacts/training_pairs/official_like_v6_official_train_h2_top4.pt \
  --max-hamming-other 2 \
  --top-targets-per-source-query 4 \
  --max-sources-per-query 20000 \
  --source-chunk-size 128 \
  --candidate-chunk-size 4096

echo "=== Build/reuse v6 weak/global train positive sets ==="
python3 -u experimental/build_official_like_positive_sets_v6.py \
  --split train \
  --preset weak \
  --device cuda \
  --output artifacts/training_pairs/official_like_v6_weak_train_h2_top4.pt \
  --max-hamming-other 2 \
  --top-targets-per-source-query 4 \
  --max-sources-per-query 20000 \
  --source-chunk-size 128 \
  --candidate-chunk-size 4096

echo "=== Start official mix v6 hpsearch ==="
LIMIT_ARGS=()
if [[ "${LIMIT:-0}" != "0" ]]; then
  LIMIT_ARGS=(--limit "${LIMIT}")
fi

python3 -u experimental/run_official_mix_hpsearch_v6.py \
  --profile long \
  --configs configs/gate_v6_official_mix_3h_configs.json \
  --device cuda \
  --job-time-budget-seconds 10200 \
  --finalize-window-seconds 300 \
  --min-seconds-for-new-run 900 \
  --force \
  "${LIMIT_ARGS[@]}"

echo "=== Official mix v6 job completed ==="
