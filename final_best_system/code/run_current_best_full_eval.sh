#!/bin/bash
set -euo pipefail

PACKAGE_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPO_ROOT="$(cd "$PACKAGE_ROOT/.." && pwd)"
PYTHON="${PYTHON:-$REPO_ROOT/.venv/bin/python}"
DEVICE="${DEVICE:-auto}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PACKAGE_ROOT/results/recomputed_current_A_query_hardh2_accuracy}"

if [ ! -x "$PYTHON" ]; then
  echo "Python not found: $PYTHON" >&2
  echo "Set PYTHON=/path/to/python or create the project virtualenv." >&2
  exit 1
fi

export DL_PROJECT_ROOT="$PACKAGE_ROOT"

"$PYTHON" "$PACKAGE_ROOT/code/probe_filtering/evaluate_weighted_probe_reranker.py" \
  --device "$DEVICE" \
  --probe-results "$PACKAGE_ROOT/results/probe_reranker_v2_calibrated" \
  --checkpoint "$PACKAGE_ROOT/weights/best_val_official_like_at10.pt" \
  --probe-checkpoint "$PACKAGE_ROOT/results/probe_reranker_v2_calibrated/probe/best_probe.pt" \
  --source-batch-size 256 \
  --top-pool 500 \
  --top-k 10 \
  --threshold-objectives accuracy \
  --output-root "$OUTPUT_ROOT"

echo
echo "Full evaluation output: $OUTPUT_ROOT"
echo
echo "=== best method ==="
cat "$OUTPUT_ROOT/BEST_WEIGHTED_PROBE_RERANKER_METHOD.txt"
echo
echo "=== top rows ==="
head -20 "$OUTPUT_ROOT/combined_summary.csv"
