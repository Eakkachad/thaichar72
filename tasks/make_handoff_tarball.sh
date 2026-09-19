#!/usr/bin/env bash
# Build a self-contained hand-off tarball of this project (repo + .git, dataset, data caches/splits/synth/external,
# deliverable weights, and the run checkpoints another machine needs to continue), plus sha256 + manifest.
#
#   bash tasks/make_handoff_tarball.sh [OUT_DIR]        # default OUT_DIR = parent of the repo (/home/CNN)
#
# Excluded on purpose: .venv (platform-specific; `uv sync` recreates it), __pycache__/.pytest_cache, *.executed.ipynb,
# last.pt of every run, best.pt of runs not listed in KEEP_CKPT, raw external downloads (only the npz + index are read),
# ThaiCharacter Dataset/__MACOSX, outputs/, dist/. Extract with: tar -xzf <file>  (Windows: built-in bsdtar `tar -xzf`).
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
NAME="$(basename "$REPO")"
OUT_DIR="${1:-$(dirname "$REPO")}"
STAMP="$(date +%F)"
OUT="$OUT_DIR/${NAME}_handoff_${STAMP}.tar.gz"
cd "$(dirname "$REPO")"

# checkpoints worth carrying (stage-1 inits, winner family, deliverable sources, references, extras)
KEEP_CKPT=(
  B6a_synth_pretrain_resnet18_64 C1_ext_pretrain_resnet18_64
  A1_resnet18_full_64_T4 B_trivial_resnet18_64_T4 D3_sampler_sqrt_resnet18_64_T4
  F1_r18_none_20 F2_r18_randaug_20 F3_r18_randaug_synth_20 F4_effb0_randaug_synth_20
  F10_effb0_trivial_synth_20 F16_r18_b6ainit_morph_20 F18_r18_b6ainit_full_20
  F19_r18_b6ainit_randaug_20 F19_r18_b6ainit_randaug_20_doc
  F19_r18_b6ainit_randaug_20_v2 F19_r18_b6ainit_randaug_20_v2_doc F19_r18_b6ainit_randaug_20_v2_s0 F19_r18_b6ainit_randaug_20_v2_s1
  K1_r18_kd_20 X1_effb0_randaug_v2 X5_mnv3s_kd_v2 X6_mnv3l_randaug_v2
)

LIST="$(mktemp)"
# everything tracked or untracked in the repo tree except the exclusions below
find "$NAME" \
  \( -path "$NAME/.venv" -o -path "$NAME/outputs" -o -path "$NAME/dist" -o -path "$NAME/ThaiCharacter Dataset/__MACOSX" \
     -o -name __pycache__ -o -name .pytest_cache -o -path "$NAME/data/external/alice" -o -path "$NAME/data/external/burapha" \
     -o -path "$NAME/data/external/downloads" \) -prune -o \
  -type f ! -name "*.executed.ipynb" ! -name "last.pt" ! -name ".DS_Store" ! -name "Thumbs.db" ! -name "*:Zone.Identifier" -print \
  | grep -v "/runs/.*/best\.pt$" > "$LIST"
for r in "${KEEP_CKPT[@]}"; do
  [ -f "$NAME/runs/$r/best.pt" ] && echo "$NAME/runs/$r/best.pt" >> "$LIST" || echo "note: no checkpoint for $r" >&2
done
sort -u "$LIST" -o "$LIST"

echo "files: $(wc -l < "$LIST")  ->  $OUT"
if command -v pigz >/dev/null; then tar -c -I pigz -f "$OUT" -T "$LIST"; else tar -czf "$OUT" -T "$LIST"; fi
( cd "$OUT_DIR" && sha256sum "$(basename "$OUT")" > "$OUT.sha256" )
cp "$LIST" "$OUT.manifest.txt"
cat > "$OUT_DIR/${NAME}_handoff_${STAMP}.README.txt" << EOF
$NAME hand-off package (built $STAMP on the RTX 4060 / WSL2 machine)
  archive : $(basename "$OUT")   (verify: sha256sum -c $(basename "$OUT").sha256  |  certutil -hashfile <file> SHA256)
  manifest: $(basename "$OUT").manifest.txt
  code    : https://github.com/Eakkachad/thaichar72 (private) — run 'git pull' inside $NAME after extracting
  read    : $NAME/tasks/HANDOFF.md section -1 (current state), $NAME/CLAUDE.md, $NAME/tasks/HANDOFF-OWNER-TH.md (Thai set-up)
  contents: repo + .git, dataset (ThaiCharacter Dataset/round2), data/{cache,splits (incl. v2),synth,external/*.npz+csv},
            weights/ (deliverables), ${#KEEP_CKPT[@]} run checkpoints (runs/<id>/best.pt), all metrics/logs/val_logits
  set-up  : uv sync  (torch cu126 auto-selected on Windows/WSL2, cpu wheels elsewhere); then see HANDOFF section 1
EOF
ls -la "$OUT" "$OUT.sha256" "$OUT.manifest.txt" "$OUT_DIR/${NAME}_handoff_${STAMP}.README.txt"
rm -f "$LIST"
