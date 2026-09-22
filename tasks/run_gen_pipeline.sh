#!/usr/bin/env bash
# Generalisation study pipeline (2026-09-22, RTX 4070 / WSL2).
#
# Stages are independent and idempotent-ish; run one at a time:
#     bash tasks/run_gen_pipeline.sh prep        # clean index + splits + glyph cache + v2 split
#     bash tasks/run_gen_pipeline.sh verify      # re-evaluate the shipped K1 weight (integrity check)
#     bash tasks/run_gen_pipeline.sh synth       # render 203-font synthetic corpus, all 72 classes
#     bash tasks/run_gen_pipeline.sh extra       # import dataUpdate + merge sources
#     bash tasks/run_gen_pipeline.sh stage1      # G1/G2 pretraining
#     bash tasks/run_gen_pipeline.sh stage2      # F19 control + G3/G4/G5 fine-tunes
set -euo pipefail
cd "$(dirname "$0")/.."

DEV="--set device=cuda --set num_workers=8"
INCOMING="${INCOMING:-$HOME/work/incoming}"
PER_CLASS="${PER_CLASS:-1000}"

case "${1:-}" in

prep)
  echo "== clean index + splits (seeds 42/0/1) + glyph cache =="
  uv run --no-sync python scripts/prep_data.py
  echo "== label-corrected v2 split (replay of the tracked change list) =="
  uv run --no-sync python scripts/make_split_v2.py
  ls -la data/splits/
  ;;

verify)
  # The restored corpus must reproduce the shipped model's numbers, otherwise the
  # reconstruction (or the split regeneration) is wrong and nothing downstream is trustworthy.
  echo "== integrity check: shipped K1 weight on the regenerated v2 val split =="
  uv run --no-sync python scripts/eval_split.py \
    --ckpt weights/thaichar72_resnet18_64.pt \
    --split-file data/splits/split_seed42_v2.csv \
    --split-kind strat \
    --out reports/analysis/verify_K1_restored.json
  cat reports/analysis/verify_K1_restored.json
  echo
  echo "expected (reports/analysis/K1_r18_kd_20/summary.json): top1 0.9903  balanced 0.9875"
  ;;

synth)
  echo "== rendering $PER_CLASS glyphs/class x 72 classes from data/fonts_all =="
  ls data/fonts_all/*.ttf | wc -l
  uv run --no-sync python scripts/render_synth.py \
    --per-class "$PER_CLASS" --seed 0 \
    --fonts-dir data/fonts_all \
    --out-cache data/synth_big/glyphs.npz \
    --out-index data/synth_big/index.csv \
    --summary-out reports/data/synth_big_summary.md
  ;;

extra)
  echo "== import dataUpdate (in-universe 35 classes) =="
  uv run --no-sync python scripts/prep_dataupdate.py \
    --root "$INCOMING/synth/dataset" \
    --out-dir data/dataupdate
  # dataUpdate's shipped train/val split is per-image over renders that share a source,
  # so 99.2 % of its val side reuses a train source. Re-split it group-disjointly
  # (whole font families / whole writers) or the tail probe measures nothing.
  echo "== rebuild a group-disjoint train/probe split of dataUpdate =="
  uv run --no-sync python scripts/make_dataupdate_probe.py --dir data/dataupdate
  echo "== merge 203-font render + dataUpdate (probe-disjoint side) into one stage-1 source =="
  uv run --no-sync python scripts/merge_extra_sources.py \
    --inputs data/synth_big/index.csv data/dataupdate/index_train_dj.csv \
    --caches data/synth_big/glyphs.npz data/dataupdate/glyphs_train_dj.npz \
    --out-index data/synth_merged/index.csv \
    --out-cache data/synth_merged/glyphs.npz
  ;;

stage1)
  echo "== stage-1 pretraining (2 lanes) =="
  uv run --no-sync python scripts/train.py --config configs/gen/G1_synth203_pretrain_r18_64.yaml $DEV &
  uv run --no-sync python scripts/train.py --config configs/gen/G2_synth203du_pretrain_r18_64.yaml $DEV &
  wait
  ;;

stage2)
  echo "== control: F19 recipe re-run on THIS split (so G-runs compare like for like) =="
  uv run --no-sync python scripts/train.py --config configs/final_candidates/F19_r18_b6ainit_randaug_20.yaml \
    --exp-id F19_ctrl_v2_4070 \
    --set split_file=data/splits/split_seed42_v2.csv --set tta=false \
    --set init_from=weights/thaichar72_r18_synth_pretrain_init.pt $DEV
  echo "== G3 / G4 / G5 =="
  for c in G3_r18_g1init_randaug_20 G4_r18_g2init_randaug_20 G5_r18_g2init_randaug_fill500_20; do
    uv run --no-sync python scripts/train.py --config "configs/gen/$c.yaml" $DEV
  done
  uv run --no-sync python scripts/collect_results.py
  ;;

*)
  echo "usage: $0 {prep|verify|synth|extra|stage1|stage2}" >&2
  exit 2
  ;;
esac

# NOTE (2026-09-22): on this machine uv's HTTP client wedges against download.pytorch.org
# (0 B/s while curl pulls the same wheel at 15 MB/s), so `uv sync` never completes. The venv
# was built by fetching the two wheels with curl into ~/wheels and running
#   uv export --format requirements-txt --no-hashes --no-emit-project -o /tmp/reqs.txt
#   uv pip install --find-links ~/wheels -r /tmp/reqs.txt
#   uv pip install -e . --no-deps
# Because that is a manual install, every command here uses `uv run --no-sync` — a plain
# `uv run` would re-sync to the lock and undo it.
