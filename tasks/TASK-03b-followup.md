# TASK-03b — Finish TASK-03 (follow-up round)

Read `tasks/TASK-03-augment-synth.md` fully — it is the spec. Round 1 delivered `src/thaichar/augment.py`,
`src/thaichar/synth.py`, `scripts/render_synth.py`, `tests/test_augment.py` (16 tests pass) and the
transforms/data.py changes, but stopped early. **Still missing — deliver all of these:**
1. `tests/test_synth.py` (spec D6) and make it pass.
2. `scripts/aug_examples.py` → `reports/figures/aug_examples.png` and `reports/figures/channel_modes.png` (spec D5);
   use the Thai font `assets/fonts/Sarabun-Regular.ttf` for any Thai text in matplotlib.
3. Full render: `uv run python scripts/render_synth.py --per-class 300 --seed 0` (must complete; < 5 min);
   verify `data/synth/index.csv` has ~300 per class × 72 and that every combining-mark class (ั ิ ี ึ ื ุ ู ็ ่ ้ ๊ ์)
   has ≥ 50 samples.
4. `reports/data/synth_summary.md`, `reports/figures/synth_montage.png`, `reports/figures/synth_vs_real.png` (spec D4).
5. Re-run the full acceptance block from the spec and paste its output.

WORKING RULES FOR THIS ROUND: run every command synchronously in the foreground and wait for it; do NOT
launch background tasks or subagents. Do not stop until every acceptance command in the spec succeeds. End
with the written report the spec asks for (files, commands, test output, fonts skipped, per-class counts
min/max, real-vs-synth table, surprises). Same scope restrictions as the spec; no git commit; do not touch
`models.py/losses.py/samplers.py/metrics.py/engine.py/train.py/external.py` (other agents own them).

ADDITIONAL DEFECT TO FIX: the round-1 render wrote `degraded=False` for every row and used only 10 fonts
round-robin. The spec (D4) requires the degradation pipeline (target-height resampling from the class's real
median, stroke-weight jitter, rotation/shear, random threshold, speckle) applied to the majority of samples
(default 80% degraded / 20% clean) and ALL usable fonts in `assets/fonts` sampled. Make `--degraded-frac`
a CLI flag (default 0.8) and report the real-vs-synth median height table to prove the sizes match.
