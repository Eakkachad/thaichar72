#!/usr/bin/env bash
# Extras after the deliverable (HANDOFF "if time remains"): winner recipe on effb0 / convnext_tiny, KD into mobilenetv3
# large/small as "small models", plus a no-KD mnv3 control. 3 concurrent lanes, then robustness sweeps for all, then collect.
# Usage: bash tasks/run_extras.sh
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"
COMMON=(--set device=cuda --set num_workers=4 --stop-on-error)
Q="uv run python scripts/run_local_queue.py"
LOG=tasks/logs/local_queue

echo "=== EXTRAS: 3 lanes ($(date +%H:%M:%S))"
$Q configs/extras/X1_effb0_randaug_v2.yaml configs/extras/X2_effb0_kd_v2.yaml "${COMMON[@]}" > "$LOG/lane_xA.log" 2>&1 & PA=$!
$Q configs/extras/X4_mnv3l_kd_v2.yaml configs/extras/X5_mnv3s_kd_v2.yaml "${COMMON[@]}" > "$LOG/lane_xB.log" 2>&1 & PB=$!
$Q configs/extras/X3_convnext_tiny_randaug_v2.yaml configs/extras/X6_mnv3l_randaug_v2.yaml "${COMMON[@]}" > "$LOG/lane_xC.log" 2>&1 & PC=$!
wait $PA; echo "laneA rc=$?"; wait $PB; echo "laneB rc=$?"; wait $PC; echo "laneC rc=$?"

echo "=== EXTRAS robustness sweeps ($(date +%H:%M:%S))"
for e in X1_effb0_randaug_v2 X2_effb0_kd_v2 X3_convnext_tiny_randaug_v2 X4_mnv3l_kd_v2 X5_mnv3s_kd_v2 X6_mnv3l_randaug_v2; do
  out="reports/robustness/${e}_full_bin"
  [ -f "runs/${e}/best.pt" ] || { echo "skip $e (no checkpoint)"; continue; }
  [ -f "$out/results.csv" ] && { echo "skip $out (exists)"; continue; }
  uv run python scripts/robustness.py --ckpt "runs/${e}/best.pt" --binarize --device cuda --batch-size 256 --out "$out"
done
uv run python scripts/collect_results.py
echo "=== EXTRAS DONE ($(date +%H:%M:%S))"
