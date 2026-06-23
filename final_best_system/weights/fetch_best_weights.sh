#!/bin/bash
set -euo pipefail

scp kuba.diquattro@baldo.disi.unitn.it:/mnt/meditech/group1/deep_learning/cluster/artifacts/training_runs/hpsearch_gate_v3_20260621_001921_long/gate_v3_sequentialgate_hybmp_l002_w050_balanced_length_b256_lr0.0001_src0.02_scale0.02_long/checkpoints/best_val_official_like_at10.pt /Users/kuba/deep_learning/final_best_system/weights/
echo "Copied best checkpoint into /Users/kuba/deep_learning/final_best_system/weights/"
