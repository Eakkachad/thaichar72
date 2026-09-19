#!/usr/bin/env bash
# Label-corrected (DataV2) retrain of a recipe: strat + doc runs on data/splits/split_seed42_v2.csv (2 lanes),
# then cross-evaluation v1-model-on-v2-val and v2-model-on-v1-val for the report.
# Usage: bash tasks/run_step1_v2.sh F19_r18_b6ainit_randaug_20
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"
W="${1:?exp_id of the recipe (configs/final_candidates/<exp_id>.yaml)}"
CFG="configs/final_candidates/${W}.yaml"
V2=data/splits/split_seed42_v2.csv
V1=data/splits/split_seed42.csv
COMMON=(--set device=cuda --set num_workers=4 --stop-on-error --set split_file=$V2)
Q="uv run python scripts/run_local_queue.py"
LOG=tasks/logs/local_queue

echo "=== v2 retrain of ${W}: strat + doc lanes ($(date +%H:%M:%S))"
$Q "$CFG" --suffix _v2 "${COMMON[@]}" > "$LOG/lane_v2s.log" 2>&1 & P1=$!
$Q "$CFG" --suffix _v2_doc --set split_kind=doc "${COMMON[@]}" > "$LOG/lane_v2d.log" 2>&1 & P2=$!
wait $P1; echo "lane_v2s rc=$?"; wait $P2; echo "lane_v2d rc=$?"

echo "=== cross-evaluation ($(date +%H:%M:%S))"
mkdir -p reports/analysis/labels_v1_vs_v2
E="uv run python scripts/eval_split.py"
$E --ckpt "runs/${W}/best.pt"        --split-file $V1 --out "reports/analysis/labels_v1_vs_v2/${W}__v1model_v1val.json"
$E --ckpt "runs/${W}/best.pt"        --split-file $V2 --out "reports/analysis/labels_v1_vs_v2/${W}__v1model_v2val.json"
$E --ckpt "runs/${W}_v2/best.pt"     --split-file $V1 --out "reports/analysis/labels_v1_vs_v2/${W}__v2model_v1val.json"
$E --ckpt "runs/${W}_v2/best.pt"     --split-file $V2 --out "reports/analysis/labels_v1_vs_v2/${W}__v2model_v2val.json"
$E --ckpt "runs/${W}_doc/best.pt"    --split-file $V1 --split-kind doc --out "reports/analysis/labels_v1_vs_v2/${W}__v1model_v1docval.json"
$E --ckpt "runs/${W}_doc/best.pt"    --split-file $V2 --split-kind doc --out "reports/analysis/labels_v1_vs_v2/${W}__v1model_v2docval.json"
$E --ckpt "runs/${W}_v2_doc/best.pt" --split-file $V1 --split-kind doc --out "reports/analysis/labels_v1_vs_v2/${W}__v2model_v1docval.json"
$E --ckpt "runs/${W}_v2_doc/best.pt" --split-file $V2 --split-kind doc --out "reports/analysis/labels_v1_vs_v2/${W}__v2model_v2docval.json"
uv run python scripts/collect_results.py
echo "=== v2 DONE ($(date +%H:%M:%S))"
