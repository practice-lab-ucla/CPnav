#!/usr/bin/env bash
# CPnav — structured runner for the paper's ObjectNav experiments.
#
# Runs ONE method on ONE dataset over the paper's fixed 200-episode / 6-category
# evaluation set (chair, plant, sofa, bed, toilet, tv_monitor).
#
# CPnav's four internal baselines are the SAME agent under a different
# prediction-set mode — they share one config per dataset. You pick the method
# by name here; you do NOT edit configs or keep a config file per method.
#
#   Usage:  bash run.sh <hm3d|mp3d> <method> [batch_name]
#
#   method:
#     cpnav     CPnav (conformal)   --mode cp_nav       --threshold 0.064   (main method)
#     nocal     No Calibration      --mode cp_nav       --threshold 0
#     simple    Simple Set          --mode simple_set   --alpha 0.3
#     prompt    Prompt Set          --mode prompt_set
#     ensemble  Ensemble Set        --mode ensemble_set --alpha 0.3         (~20 VLM calls/step)
#
#   dataset:
#     hm3d   config/CPnav_5_nano_131.yaml       seed 43
#     mp3d   config/CPnav_5_nano_131_mp3d.yaml  seed 96
#
#   Examples:
#     bash run.sh hm3d cpnav
#     bash run.sh mp3d ensemble
#
# Requires OPENAI_API_KEY in the environment (or CPnav's .env). Results land in
# logs/<batch_name>/.  Run all five methods for a dataset with:  bash run_all.sh <hm3d|mp3d>

set -u

DATASET=${1:-}
METHOD=${2:-}
if [[ -z "$DATASET" || -z "$METHOD" ]]; then
  sed -n '2,30p' "$0"; exit 1
fi

# ---- dataset -> config + seed ------------------------------------------------
case "$DATASET" in
  hm3d) CONFIG=config/CPnav_5_nano_131.yaml;      SEED=43 ;;
  mp3d) CONFIG=config/CPnav_5_nano_131_mp3d.yaml; SEED=96 ;;
  *) echo "ERROR: dataset must be hm3d or mp3d (got '$DATASET')"; exit 1 ;;
esac

# ---- method -> mode flags + parallelism --------------------------------------
JOBS=25
case "$METHOD" in
  cpnav)    MODE_FLAGS="--mode cp_nav       --threshold 0.064" ;;
  nocal)    MODE_FLAGS="--mode cp_nav       --threshold 0"     ;;
  simple)   MODE_FLAGS="--mode simple_set   --alpha 0.3"       ;;
  prompt)   MODE_FLAGS="--mode prompt_set"                     ;;
  ensemble) MODE_FLAGS="--mode ensemble_set --alpha 0.3"; JOBS=8 ;;  # ~20 calls/step -> fewer jobs
  *) echo "ERROR: method must be cpnav|nocal|simple|prompt|ensemble (got '$METHOD')"; exit 1 ;;
esac

BATCH=${3:-${DATASET}_${METHOD}}

# ---- env (headless; CPnav conda) ---------------------------------------------
export MPLBACKEND=Agg
if [[ -z "${CONDA_DEFAULT_ENV:-}" || "${CONDA_DEFAULT_ENV}" != "CPnav" ]]; then
  source /home/qizhao.chen/miniconda3/etc/profile.d/conda.sh 2>/dev/null && conda activate CPnav 2>/dev/null || true
fi
cd "$(dirname "$0")"

echo "==================================================================="
echo ">>> $BATCH   ($DATASET / $METHOD)   $(date '+%F %T')"
echo ">>> config=$CONFIG  seed=$SEED  jobs=$JOBS  $MODE_FLAGS"
echo "==================================================================="
python scripts/batch_run_online.py \
  --config "$CONFIG" --n_exp 200 --seed "$SEED" --max_jobs "$JOBS" \
  --goal_categories chair plant sofa bed toilet tv_monitor \
  --batch_name "$BATCH" $MODE_FLAGS
rc=$?
echo ">>> DONE $BATCH   $(date '+%F %T')   exit=$rc"
exit $rc
