#!/usr/bin/env bash
# Step 1.3b: the strat top-3 (F16/F7/F10) all failed the robustness gate (mean non-clean top-1 >= F2's 0.9038 - 0.005),
# so extend the doc-split + robustness check to the next tier: trivial / randaug recipes that saw noise ops in training.
# Runs 3 doc trainings concurrently on the single GPU (each uses ~2 GB VRAM; the pipeline is CPU-bound), then sweeps.
# Usage: bash tasks/run_step1_3b_ext.sh
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"
COMMON=(--set device=cuda --set num_workers=4 --stop-on-error)
Q="uv run python scripts/run_local_queue.py"
LOG=tasks/logs/local_queue

echo "=== STEP 1.3b doc-split runs, 3 concurrent lanes ($(date +%H:%M:%S))"
$Q configs/final_candidates/F15_r18_b6ainit_trivial_synth_20.yaml configs/final_candidates/F2_r18_randaug_20.yaml \
   --suffix _doc --set split_kind=doc "${COMMON[@]}" > "$LOG/lane1.log" 2>&1 &
P1=$!
$Q configs/final_candidates/F12_r18_c1init_trivial_20.yaml configs/final_candidates/F14_r18_b6ainit_trivial_20.yaml \
   --suffix _doc --set split_kind=doc "${COMMON[@]}" > "$LOG/lane2.log" 2>&1 &
P2=$!
$Q configs/final_candidates/F8_r18_trivial_20.yaml \
   --suffix _doc --set split_kind=doc "${COMMON[@]}" > "$LOG/lane3.log" 2>&1 &
P3=$!
wait $P1; echo "lane1 rc=$?"
wait $P2; echo "lane2 rc=$?"
wait $P3; echo "lane3 rc=$?"

echo "=== STEP 1.3b robustness sweeps (full strat val, --binarize) ($(date +%H:%M:%S))"
for e in F15_r18_b6ainit_trivial_synth_20 F12_r18_c1init_trivial_20 F8_r18_trivial_20 F14_r18_b6ainit_trivial_20; do
  out="reports/robustness/${e}_full_bin"
  if [ -f "$out/results.csv" ]; then echo "skip $out (exists)"; continue; fi
  uv run python scripts/robustness.py --ckpt "runs/${e}/best.pt" --binarize --device cuda --batch-size 256 --out "$out"
done

uv run python scripts/collect_results.py
echo "=== STEP 1.3b DONE ($(date +%H:%M:%S))"
