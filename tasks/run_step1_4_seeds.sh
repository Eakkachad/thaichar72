#!/usr/bin/env bash
# Step 1.4 (HANDOFF §3.4): two extra seeds of the winner on the SAME split (seed 42 file) so the three runs can be ensembled,
# then the optional variance-across-splits run (split_seed0.csv, report only).
# Usage: bash tasks/run_step1_4_seeds.sh F10_effb0_trivial_synth_20 [--no-split-study]
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"
W="${1:?winner exp_id}"
CFG="configs/final_candidates/${W}.yaml"
COMMON=(--set device=cuda --set num_workers=4 --stop-on-error)
Q="uv run python scripts/run_local_queue.py"

echo "=== STEP 1.4a seeds 0 and 1 of ${W} on split_seed42 ($(date +%H:%M:%S))"
$Q "$CFG" --suffix _s0 --set seed=0 "${COMMON[@]}"
$Q "$CFG" --suffix _s1 --set seed=1 "${COMMON[@]}"

if [ "${2:-}" != "--no-split-study" ]; then
  echo "=== STEP 1.4b variance across splits: split_seed0.csv (report only) ($(date +%H:%M:%S))"
  $Q "$CFG" --suffix _sp0 --set split_file=data/splits/split_seed0.csv "${COMMON[@]}"
fi

echo "=== STEP 1.4 DONE ($(date +%H:%M:%S))"
