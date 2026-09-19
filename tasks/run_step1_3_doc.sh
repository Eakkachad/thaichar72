#!/usr/bin/env bash
# Step 1.3 (HANDOFF §3.3): document-disjoint runs of the top-3 final candidates, then the full-val robustness sweep
# (Otsu on) of each candidate's STRAT checkpoint, to compare with F2's baseline 0.9038 (mean top-1 over non-clean rows).
# Usage: bash tasks/run_step1_3_doc.sh F16_r18_b6ainit_morph_20 F7_r18_morph_synth_20 F10_effb0_trivial_synth_20
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"
[ $# -ge 1 ] || { echo "usage: $0 <exp_id> [<exp_id> ...]"; exit 2; }

cfgs=()
for e in "$@"; do cfgs+=("configs/final_candidates/${e}.yaml"); done

echo "=== STEP 1.3a doc-split runs of: $* ($(date +%H:%M:%S))"
uv run python scripts/run_local_queue.py "${cfgs[@]}" --suffix _doc --set split_kind=doc \
    --set device=cuda --set num_workers=4 --stop-on-error

echo "=== STEP 1.3b robustness sweeps (full strat val, --binarize) ($(date +%H:%M:%S))"
for e in "$@"; do
  out="reports/robustness/${e}_full_bin"
  if [ -f "$out/results.csv" ]; then echo "skip $out (exists)"; continue; fi
  uv run python scripts/robustness.py --ckpt "runs/${e}/best.pt" --binarize --device cuda --batch-size 256 --out "$out"
done

echo "=== STEP 1.3 DONE ($(date +%H:%M:%S))"
