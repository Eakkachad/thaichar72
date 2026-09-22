#!/usr/bin/env bash
# Post-stage2 selection work (2026-09-22). Selection is on the DOC-DISJOINT split plus the
# robustness gate -- never on stratified val (CLAUDE.md section 2).
#
#   bash tasks/run_gen_followup.sh doc    G4_r18_g2init_randaug_20 G5_...   # doc-split re-runs
#   bash tasks/run_gen_followup.sh gate   G4_r18_g2init_randaug_20 ...      # robustness sweeps
#   bash tasks/run_gen_followup.sh probe  G4_r18_g2init_randaug_20 ...      # held-out tail probe
#   bash tasks/run_gen_followup.sh seeds  G4_r18_g2init_randaug_20          # 3 seeds of the winner
set -euo pipefail
cd "$(dirname "$0")/.."

DEV="--set device=cuda --set num_workers=8"
GATE_REF=0.9038          # F2, mean top-1 over non-clean rows (reports/robustness/F2_..._full_bin)
GATE_MIN=0.8988          # GATE_REF - 0.005

mode="${1:-}"; shift || true

case "$mode" in

doc)
  # A strat-trained model must never be scored on doc val (the partitions overlap), so the
  # doc number needs its own run with --suffix _doc.
  for c in "$@"; do
    uv run --no-sync python scripts/train.py --config "configs/gen/$c.yaml" \
      --exp-id "${c}_doc" --set split_kind=doc $DEV
  done
  ;;

gate)
  for c in "$@"; do
    uv run --no-sync python scripts/robustness.py --ckpt "runs/$c/best.pt" \
      --binarize --out "reports/robustness/${c}_full_bin"
    uv run --no-sync python - "$c" <<'PY'
import sys, pandas as pd
c = sys.argv[1]
d = pd.read_csv(f"reports/robustness/{c}_full_bin/results.csv")
m = d.loc[d.corruption != "clean", "top1"].mean()
clean = d.loc[d.corruption == "clean", "top1"].iloc[0]
print(f"{c}: clean {clean:.4f}  mean non-clean {m:.4f}  gate(>=0.8988) {'PASS' if m >= 0.8988 else 'FAIL'}")
PY
  done
  ;;

probe)
  for c in "$@"; do
    uv run --no-sync python scripts/eval_probe.py --ckpt "runs/$c/best.pt" \
      --out "reports/analysis/probe_${c}.json"
  done
  ;;

seeds)
  c="${1:?need one config}"
  for s in 0 1 2; do
    uv run --no-sync python scripts/train.py --config "configs/gen/$c.yaml" \
      --exp-id "${c}_s${s}" --set seed="$s" $DEV
  done
  ;;

*) echo "usage: $0 {doc|gate|probe|seeds} <exp_id...>" >&2; exit 2 ;;
esac
