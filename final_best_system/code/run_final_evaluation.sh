#!/bin/bash
set -euo pipefail

PACKAGE_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPO_ROOT="$(cd "$PACKAGE_ROOT/.." && pwd)"
PYTHON="${PYTHON:-$REPO_ROOT/.venv/bin/python}"
DEVICE="${DEVICE:-auto}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PACKAGE_ROOT/results/recomputed_final_eval}"
if [ -d "$PACKAGE_ROOT/results/probe_embedding_v4_m04/probe" ]; then
  PROBE_RESULTS_DEFAULT="$PACKAGE_ROOT/results/probe_embedding_v4_m04"
else
  PROBE_RESULTS_DEFAULT="$PACKAGE_ROOT/results/probe_reranker_v2_calibrated"
fi
PROBE_RESULTS="${PROBE_RESULTS:-$PROBE_RESULTS_DEFAULT}"
PROBE_CHECKPOINT="${PROBE_CHECKPOINT:-$PROBE_RESULTS/probe/best_probe.pt}"

if [ ! -x "$PYTHON" ]; then
  echo "Python not found: $PYTHON" >&2
  echo "Set PYTHON=/path/to/python or create the project virtualenv." >&2
  exit 1
fi

export DL_PROJECT_ROOT="$PACKAGE_ROOT"

"$PYTHON" "$PACKAGE_ROOT/code/probe_filtering/evaluate_weighted_probe_reranker.py" \
  --device "$DEVICE" \
  --probe-results "$PROBE_RESULTS" \
  --checkpoint "$PACKAGE_ROOT/weights/best_val_official_like_at10.pt" \
  --probe-checkpoint "$PROBE_CHECKPOINT" \
  --source-batch-size 256 \
  --top-pool 500 \
  --top-k 10 \
  --threshold-objectives accuracy \
  --output-root "$OUTPUT_ROOT"

echo "Recomputed results saved in: $OUTPUT_ROOT"
