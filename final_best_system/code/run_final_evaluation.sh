#!/bin/bash
set -euo pipefail

PACKAGE_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPO_ROOT="$(cd "$PACKAGE_ROOT/.." && pwd)"
CLUSTER_ROOT="$REPO_ROOT/cluster"
OUTPUT_ROOT="$PACKAGE_ROOT/results/recomputed_final_eval"

export DL_PROJECT_ROOT="$CLUSTER_ROOT"

python3 "$PACKAGE_ROOT/code/orchestrator/evaluate_beta_sweep_blends.py"   --checkpoint "$PACKAGE_ROOT/weights/best_val_official_like_at10.pt"   --output-root "$OUTPUT_ROOT"   --betas 1.5   --device auto   --force

echo "Recomputed results saved in: $OUTPUT_ROOT"
