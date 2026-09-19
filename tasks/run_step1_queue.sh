#!/usr/bin/env bash
# Step-1 GPU queue on the RTX 4060 (HANDOFF §3.1 + §3.2), sequential, stop on first failure.
# Usage: bash tasks/run_step1_queue.sh  (logs: tasks/logs/local_queue/*.log, queue.log, step1_driver.log)
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"
COMMON=(--set device=cuda --set num_workers=4 --stop-on-error)
Q="uv run python scripts/run_local_queue.py"

echo "=== STEP 1.1 confound check: A1 base-aug on doc split, FULL data ($(date +%H:%M:%S))"
$Q configs/matrix/A1_resnet18_full_64.yaml --suffix _doc_full --set split_kind=doc --set subset_frac=1.0 "${COMMON[@]}"

echo "=== STEP 1.1b synthetic-pretrain init (B6a_ft) on doc split ($(date +%H:%M:%S))"
$Q configs/pretrain/B6a_ft_resnet18_64.yaml --suffix _doc --set split_kind=doc "${COMMON[@]}"

echo "=== STEP 1.2 final candidates F7-F17 ($(date +%H:%M:%S))"
$Q configs/final_candidates/F7_*.yaml configs/final_candidates/F8_*.yaml configs/final_candidates/F9_*.yaml \
   configs/final_candidates/F10_*.yaml configs/final_candidates/F11_*.yaml configs/final_candidates/F12_*.yaml \
   configs/final_candidates/F13_*.yaml configs/final_candidates/F14_*.yaml configs/final_candidates/F15_*.yaml \
   configs/final_candidates/F16_*.yaml configs/final_candidates/F17_*.yaml "${COMMON[@]}"

echo "=== STEP 1.2b negative controls F5 (geometry) / F6 (onoff) ($(date +%H:%M:%S))"
$Q configs/final_candidates/F5_*.yaml configs/final_candidates/F6_*.yaml "${COMMON[@]}"

echo "=== STEP 1 QUEUE DONE ($(date +%H:%M:%S))"
