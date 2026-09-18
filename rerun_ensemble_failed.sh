#!/usr/bin/env bash
# Re-run ONLY the ensemble episodes that still lack a result, into the SAME
# mp3d_ensemble_notab dir. Completed episodes are left untouched.
#
# The canonical set of 200 episodes is taken from a COMPLETED sibling policy
# (mp3d_cpnav_notab; identical seed 96 -> identical episodes). This is what makes
# the script correct even after an interrupted re-run deleted a partial dir: the
# target no longer depends on the ensemble dir's own (possibly-deleted) folders.
#
# IDEMPOTENT: re-detects whatever is still missing each run. Re-run to mop up.
# Low concurrency (default 2 episodes; each fires 20 calls/step -> ~40 concurrent
# requests) to stay under the rate limit that broke the original batch.
#
# Usage:  bash rerun_ensemble_failed.sh        # 2 episodes at a time
#         bash rerun_ensemble_failed.sh 1      # safest (1 at a time)
set -u
source /home/qizhao.chen/miniconda3/etc/profile.d/conda.sh
conda activate CPnav
cd /mnt/extraSTORAGE/Project2/test_project/CPnav
export MPLBACKEND=Agg

D=logs/mp3d/mp3d_ensemble_notab
REF=logs/mp3d/mp3d_cpnav_notab      # completed sibling -> canonical 200 episode idx
CFG=config/CPnav_5_nano_131_mp3d.yaml
MAXJ=${1:-2}
mkdir -p "$D/episodes" "$D/terminal"

# Print the episode idx present in REF but not yet completed in D (one per line).
missing_idx() {
  python3 - "$REF" "$D" <<'PY'
import glob, os, re, sys
ref, d = sys.argv[1], sys.argv[2]
def idxset(base, require_summary=False):
    s = set()
    for e in glob.glob(os.path.join(base, 'episodes', 'episode_*')):
        m = re.match(r'episode_(\d+)_', os.path.basename(e))
        if not m:
            continue
        if require_summary and not (
            os.path.exists(e + '/terminal/terminal_summary.json') or
            os.path.exists(e + '/terminal/failure_summary.json')):
            continue
        s.add(int(m.group(1)))
    return s
target = idxset(ref) or idxset(d)        # fall back to ensemble's own dirs if REF absent
done   = idxset(d, require_summary=True)
for i in sorted(target - done):
    print(i)
PY
}

n_target=$(ls -d "$REF"/episodes/episode_* 2>/dev/null | wc -l)
[ "$n_target" -eq 0 ] && n_target=$(ls -d "$D"/episodes/episode_* 2>/dev/null | wc -l)

# 1. Detect missing episodes.
mapfile -t failed < <(missing_idx)
echo "$(date '+%F %T')  target=$n_target  to_rerun=${#failed[@]}  (max_jobs=$MAXJ)"
[ ${#failed[@]} -eq 0 ] && { echo "nothing to re-run -> ensemble complete ($n_target/$n_target)."; exit 0; }
echo "  missing idx: ${failed[*]}"

# 2. Re-run each missing idx (delete any leftover partial dir first), low concurrency.
running=0
for idx in "${failed[@]}"; do
  for pd in "$D"/episodes/episode_${idx}_*; do [ -d "$pd" ] && rm -rf "$pd"; done
  echo "  launching ensemble episode $idx"
  python scripts/run_online.py --config "$CFG" --log_dir "$D/episodes" \
    --episode "$idx" --threshold 0 --mode ensemble_set --alpha 0.3 \
    > "$D/terminal/rerun_episode_${idx}.log" 2>&1 &
  running=$((running+1))
  if [ "$running" -ge "$MAXJ" ]; then wait -n 2>/dev/null || wait; running=$((running-1)); fi
done
wait

# 3. Report completeness against the canonical target set.
mapfile -t still < <(missing_idx)
done_ct=$(( n_target - ${#still[@]} ))
echo "$(date '+%F %T')  done. completed: $done_ct/$n_target  | still missing: ${#still[@]}"
[ ${#still[@]} -gt 0 ] && echo ">>> still missing idx: ${still[*]}  — run this script AGAIN to mop up (or: bash rerun_ensemble_failed.sh 1)"
