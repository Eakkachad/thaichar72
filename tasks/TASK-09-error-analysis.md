# TASK-09 — Error-analysis + result-figure script (fast, no training)

## Context
Project root = this dir, `uv` Python 3.12 (numpy, pandas, matplotlib, pyyaml, tabulate installed). Each finished
experiment lives in `runs/<exp_id>/` with `metrics.json` (keys: `top1, top5, balanced_acc, macro_f1, minority_acc,
per_class_recall` (list 72, NaN if absent), `confusion` (72×72 nested list, rows=true, cols=pred), `class_counts_train`
(72), `tau_sweep` {tau: {...}}, `cfg`, `sec_per_epoch`, `params_total`, `latency_ms_bs1`), `log.csv` (per-epoch:
`epoch, lr, train_loss, train_acc, val_top1, val_bal_acc, val_macro_f1, val_minority_acc, [ema_*], epoch_seconds`)
and sometimes `val_logits.npy` (float16 N×72) + `val_labels.npy`. Class index i ↔ TIS code `CLASS_CODES[i]` ↔ char
`code_to_char(code)` (`src/thaichar/classes.py`). Use the Thai font `assets/fonts/Sarabun-Regular.ttf` for every
plot with Thai text (`matplotlib.font_manager.fontManager.addfont` + `plt.rcParams["font.family"]`), otherwise
labels render as boxes. Example run to test on: `runs/A1_resnet18_full_64_T4/`.

## Scope — create/modify ONLY: `scripts/error_analysis.py`, `scripts/plot_results.py`, `tests/test_error_analysis.py`,
`reports/figures/**`, `reports/analysis/**`. Do not touch anything else. No git commit. All commands here run in
a few seconds — run them in the foreground and wait for them.

## D1 — `scripts/error_analysis.py --run runs/<exp_id> [--topk 20]`
Writes into `reports/analysis/<exp_id>/`:
1. `confusion_matrix.png` — 72×72 row-normalised (recall) heatmap, Thai glyph tick labels on both axes (fontsize 6),
   log-ish colour scale so small off-diagonals are visible; title with exp_id and top1/bal acc.
2. `confused_pairs.md` — table of the top-k off-diagonal cells: rank, true glyph (code), pred glyph (code), count,
   % of the true class, and n_train of the true class. Also a symmetric "pair" table merging a→b and b→a.
3. `per_class_recall.png` — bar chart of per-class recall sorted ascending, bars coloured by category
   (consonant/vowel/tone_mark/digit), x tick = Thai glyph, second axis (or bar alpha) showing log n_train;
   horizontal line at overall balanced acc.
4. `worst_classes.md` — the 15 lowest-recall classes with n_train, n_val (from confusion row sums), recall, and the
   top-3 predicted labels for each.
5. `summary.json` — the numbers above in machine-readable form.

## D2 — `scripts/plot_results.py [--runs runs] [--out reports/figures]`
1. `training_curves_<exp_id>.png` for each run given by `--exp` (repeatable) — train loss + val top1/bal acc vs epoch
   (two panels), EMA curve dashed if present.
2. `matrix_A_bar.png` — from `reports/experiments.csv` (columns incl. `exp_id, model, mode, aug, top1, balanced_acc,
   minority_acc`): grouped bars of top1 / balanced_acc / minority_acc for all runs whose exp_id starts with a prefix
   given by `--prefix A` (default), sorted by balanced_acc, value labels on bars.
3. `tau_sweep_<exp_id>.png` — top1 and balanced acc vs tau from `metrics.json["tau_sweep"]`.

## D3 — `tests/test_error_analysis.py`
Build a fake `runs/_fake/metrics.json` + `log.csv` in `tmp_path` (random confusion, 72 classes) and assert both
scripts run (call their `main()` with argv) and produce the expected files with the Thai font registered.

## Acceptance (reviewer runs)
```bash
uv run pytest -q tests/test_error_analysis.py
uv run python scripts/error_analysis.py --run runs/A1_resnet18_full_64_T4
uv run python scripts/plot_results.py --exp A1_resnet18_full_64_T4 --exp A0_smallcnn_64_T4 --prefix A
ls reports/analysis/A1_resnet18_full_64_T4/ reports/figures/ | head -30
head -30 reports/analysis/A1_resnet18_full_64_T4/confused_pairs.md
```
## Report back
Files, commands, test output, and the top-10 confused pairs you observed.
