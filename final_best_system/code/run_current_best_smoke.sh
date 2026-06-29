#!/bin/bash
set -euo pipefail

PACKAGE_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPO_ROOT="$(cd "$PACKAGE_ROOT/.." && pwd)"
PYTHON="${PYTHON:-$REPO_ROOT/.venv/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/tmp/current_A_query_hardh2_smoke}"

if [ ! -x "$PYTHON" ]; then
  echo "Python not found: $PYTHON" >&2
  echo "Set PYTHON=/path/to/python or create the project virtualenv." >&2
  exit 1
fi

export DL_PROJECT_ROOT="$PACKAGE_ROOT"

rm -rf "$OUTPUT_ROOT"

"$PYTHON" "$PACKAGE_ROOT/code/probe_filtering/evaluate_weighted_probe_reranker.py" \
  --device cpu \
  --probe-results "$PACKAGE_ROOT/results/probe_reranker_v2_calibrated" \
  --checkpoint "$PACKAGE_ROOT/weights/best_val_official_like_at10.pt" \
  --probe-checkpoint "$PACKAGE_ROOT/results/probe_reranker_v2_calibrated/probe/best_probe.pt" \
  --query-ids 0 \
  --max-sources-per-query 1 \
  --source-batch-size 1 \
  --top-pool 20 \
  --top-k 10 \
  --threshold-objectives accuracy \
  --output-root "$OUTPUT_ROOT"

echo
echo "Smoke output: $OUTPUT_ROOT"
echo
echo "=== progress ==="
cat "$OUTPUT_ROOT/progress.txt"
echo
echo "=== best method in tiny smoke, not a real metric ==="
cat "$OUTPUT_ROOT/BEST_WEIGHTED_PROBE_RERANKER_METHOD.txt"
echo
echo "=== current_A_query_hardh2_accuracy smoke summary ==="
cat "$OUTPUT_ROOT/current_A_query_hardh2_accuracy/summary.csv"
echo
echo "=== saved official full-run summary for the actual final system ==="
cat "$PACKAGE_ROOT/results/probe_reranker_v2_calibrated/A_cal_query_hardh2_accuracy/summary.csv"
