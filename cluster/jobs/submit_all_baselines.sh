#!/bin/bash
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

sbatch jobs/10_baseline_direct_sum.sh
sbatch jobs/11_baseline_direct_sequential.sh
sbatch jobs/12_baseline_contrastive_sum.sh
sbatch jobs/13_baseline_contrastive_sequential.sh
