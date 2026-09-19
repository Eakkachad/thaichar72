# TASK-12 — Publication-quality result figures for the slides (from reports/experiments.csv + runs/)

## Context
Project root = this dir, `uv` Python 3.12 (pandas, matplotlib, numpy). Inputs:
- `reports/experiments.csv` — one row per run: `exp_id, model, img_size, mode, channel_mode, aug, loss, sampler, mixup,
  cutmix, extra, split_kind (strat|doc), subset_frac, epochs, seed, top1, balanced_acc, macro_f1, minority_acc, top5,
  tau_best, tau_best_bal, tta_top1, tta_bal, best_epoch, best_which, params_M, sec_per_epoch, latency_ms, device`.
  Naming: `A*` = backbone/mode study, `B_<aug>*` = augmentation ladder, `B6_synth_*` = synthetic data, `D*` = imbalance,
  `E*` = tricks, `S_resnet18_<size>` = input-size sweep, suffix `_T4` = stratified split, suffix `_doc` = document-disjoint
  split (same config, harder split). `runs/<exp_id>/log.csv` has per-epoch curves; `runs/<exp_id>/metrics.json` has
  `tau_sweep` and `per_class_recall`.
- Thai font for any Thai text: `assets/fonts/Sarabun-Regular.ttf` (register with matplotlib).
- Use a consistent, colour-blind-safe palette; value labels on bars; 150 dpi; titles in English; also save a `.svg` copy.
Re-running the script must regenerate everything (new rows will appear later — do not hard-code numbers).

## Scope — create/modify ONLY: `scripts/result_figures.py`, `tests/test_result_figures.py`, `reports/figures/results/**`.
No git commit. Commands are short — run them directly, no background tasks, no subagents.

## Figures (`uv run python scripts/result_figures.py [--csv reports/experiments.csv] [--out reports/figures/results]`)
1. `fig_A_transfer_modes.png` — grouped bars (top1, balanced_acc, minority_acc) for A* strat runs, grouped by model with
   mode (frozen/partial/full) on the x axis; horizontal dashed line at the old baseline 0.967 (label "previous Rust CNN").
2. `fig_B_aug_ladder.png` — for the augmentation ladder (aug ∈ none, base, morph, full, randaug, trivial + "full+synth"
   from B6_synth_all): paired bars **stratified vs document-disjoint** (balanced_acc) when both exist, top1 as markers;
   x order exactly none→base→morph→full→randaug→trivial→full+synth. Missing doc rows are simply omitted (script must not fail).
3. `fig_B_mix.png` — small bar chart: B_full vs B_full_mixup vs B_full_cutmix (top1, balanced_acc).
4. `fig_D_imbalance_tradeoff.png` — scatter of top1 (x) vs balanced_acc (y) for D* + B_full + B6_synth_* strat runs, each
   point labelled by method (short names), with an arrow/annotation for the D1 inverse-weights collapse if its top1 < 0.5
   (put it in an inset or clip the axis and annotate "off-chart: 0.116").
5. `fig_S_size_sweep.png` — two y-axes: balanced_acc & top1 vs img_size (32/64/96/128/224, S_* + B_full as 64) and
   sec_per_epoch + latency_ms as bars/secondary axis; annotate that real glyphs are ≤ 44 px.
6. `fig_E_tricks.png` — horizontal bars of Δbalanced_acc and Δtop1 relative to D0 for E1_llrd, E1_noema, E5_geometry,
   E6_onoff, E6_onoff_geometry (negative bars red, positive green).
7. `fig_gap_strat_vs_doc.png` — for every exp_id that exists with both `_T4` and `_doc`: dumbbell/slope chart of
   balanced_acc strat → doc, sorted by doc value; label the gap in points.
8. `fig_training_curves_best.png` — train loss + val balanced acc per epoch for the best strat run by balanced_acc and
   for B_none / B_full (three lines each panel).
9. `fig_tau_sweep_grid.png` — small multiples: balanced_acc vs tau for D0, D1_wce_sqrt, B_full_mixup, B_none.
Also write `reports/figures/results/README.md` listing each figure with a one-sentence caption that states the number it
shows (read from the data, e.g. "no-aug reaches 0.9815 balanced on stratified but ...").

## Tests — `tests/test_result_figures.py`: build a tiny fake CSV with a handful of rows (both suffixes) in tmp_path,
run `main()` with argv, assert every expected PNG exists and is > 5 kB; also assert the script tolerates missing groups.

## Acceptance (reviewer runs)
```bash
uv run pytest -q tests/test_result_figures.py
uv run python scripts/result_figures.py
ls -la reports/figures/results/ && cat reports/figures/results/README.md
```
## Report back
Files, commands, test output, and one surprising thing you saw in the data.
