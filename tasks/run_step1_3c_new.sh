#!/usr/bin/env bash
# Step 1.3c: the two recipes added after the robustness gate (F18 = B6a init + full aug, F19 = B6a init + randaug).
# Waits for step 1.3b to finish, then: strat runs (2 lanes) -> doc runs (2 lanes) -> robustness sweeps -> collect.
# Usage: bash tasks/run_step1_3c_new.sh
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"
COMMON=(--set device=cuda --set num_workers=4 --stop-on-error)
Q="uv run python scripts/run_local_queue.py"
LOG=tasks/logs/local_queue
C18=configs/final_candidates/F18_r18_b6ainit_full_20.yaml
C19=configs/final_candidates/F19_r18_b6ainit_randaug_20.yaml

until grep -q "STEP 1.3b DONE" "$LOG/step1_3b_driver.log" 2>/dev/null; do sleep 20; done
echo "=== STEP 1.3c strat runs F18/F19, 2 lanes ($(date +%H:%M:%S))"
$Q "$C18" "${COMMON[@]}" > "$LOG/lane18.log" 2>&1 & P1=$!
$Q "$C19" "${COMMON[@]}" > "$LOG/lane19.log" 2>&1 & P2=$!
wait $P1; echo "lane18 rc=$?"; wait $P2; echo "lane19 rc=$?"

echo "=== STEP 1.3c doc runs F18/F19, 2 lanes ($(date +%H:%M:%S))"
$Q "$C18" --suffix _doc --set split_kind=doc "${COMMON[@]}" > "$LOG/lane18d.log" 2>&1 & P1=$!
$Q "$C19" --suffix _doc --set split_kind=doc "${COMMON[@]}" > "$LOG/lane19d.log" 2>&1 & P2=$!
wait $P1; echo "lane18d rc=$?"; wait $P2; echo "lane19d rc=$?"

echo "=== STEP 1.3c robustness sweeps ($(date +%H:%M:%S))"
for e in F18_r18_b6ainit_full_20 F19_r18_b6ainit_randaug_20; do
  out="reports/robustness/${e}_full_bin"
  if [ -f "$out/results.csv" ]; then echo "skip $out (exists)"; continue; fi
  uv run python scripts/robustness.py --ckpt "runs/${e}/best.pt" --binarize --device cuda --batch-size 256 --out "$out"
done
uv run python scripts/collect_results.py
echo "=== STEP 1.3c DONE ($(date +%H:%M:%S))"
