#!/usr/bin/env bash
# Run all five CPnav methods on one dataset, sequentially, over the paper's
# fixed 200-episode / 6-category set.
#
#   Usage:  bash run_all.sh <hm3d|mp3d>
#
# Ensemble runs LAST (it issues ~20 VLM calls/step, so it is the slowest and
# costliest) — you can Ctrl-C after the first four if you only want the cheap set.

set -u
DATASET=${1:-}
[[ -z "$DATASET" ]] && { echo "Usage: bash run_all.sh <hm3d|mp3d>"; exit 1; }
cd "$(dirname "$0")"

for METHOD in cpnav nocal simple prompt ensemble; do
  bash run.sh "$DATASET" "$METHOD" || echo ">>> $METHOD exited nonzero, continuing"
done
echo "=== ALL 5 METHODS DONE ($DATASET)   $(date '+%F %T') ==="
