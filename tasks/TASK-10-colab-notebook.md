# TASK-10 — Deliverable notebook for Google Colab (Train + Inference), built from this repo

## Goal
`notebooks/ThaiChar72_Colab.ipynb` that a grader can open in Colab, run top-to-bottom, and (a) reproduce the final
training recipe or (b) skip training and run inference with the shipped weights. Must be **defensive**: every path
is a variable in ONE config cell; every optional step checks for files before acting; no cell assumes state from
a cell the user might skip.

## Context
Read `configs/README.md`, `configs/base.yaml`, `tasks/COLAB-USAGE.md`, `scripts/train.py`, `src/thaichar/engine.py`
(train_one(cfg) returns metrics; runs write `runs/<exp_id>/best.pt`), `src/thaichar/infer.py`
(load_checkpoint / preprocess_image / predict_topk), `scripts/prep_data.py` (builds clean index, splits, cache from
the raw dataset — needs `reports/eda/files.csv` from `scripts/eda.py`), `reports/02-EXPERIMENTS.md` (which recipe
won; use the config named there as FINAL_CONFIG, default `configs/final.yaml` if it exists, else
`configs/matrix/A1_resnet18_full_64.yaml`). Data on Colab arrives as the zip
`ThaiCharacter Dataset.zip` in Drive (path variable) or as an already-extracted folder.

## Scope — create/modify ONLY: `notebooks/ThaiChar72_Colab.ipynb`, `notebooks/README.md`, `scripts/build_notebook.py`
(generate the .ipynb programmatically with `nbformat` so it is reproducible — add `uv add --dev nbformat` and say
so), `tests/test_notebook.py` (checks the notebook JSON is valid, has the required section headings in order,
and every code cell compiles with `compile()`). No git commit. Commands here are short — run them directly.

## Notebook structure (markdown heading per section, in this order)
0. **Title + how to use** (2 modes: TRAIN / INFERENCE-ONLY), group info placeholder.
1. **Environment**: print Python/torch/CUDA/GPU name; `pip install -q timm opencv-python-headless pyyaml tabulate
   imagehash` (skip if importable); mount Drive **only if** `USE_DRIVE=True` (try/except so it also runs locally);
   clone/pull the project code: `REPO_URL` variable (GitHub, may be empty) else expect `CODE_ZIP` in Drive, else
   assume the notebook sits inside the repo; `sys.path.insert(0, "src")`.
2. **Config cell** (the ONLY place with paths): `PROJECT_DIR`, `DATA_ZIP`, `DATA_DIR`, `WEIGHTS_PATH`,
   `FINAL_CONFIG`, `OUTPUT_DIR`, `USE_DRIVE`, `MODE = "inference" | "train"`, `SEED`, `EPOCHS_OVERRIDE`.
3. **Data**: if `DATA_DIR/round2` missing and `DATA_ZIP` exists → unzip with Python `zipfile` (not the Windows
   extractor issue — mention it); count files per class into a 72-row table and assert 62,707 total (warn, don't
   crash, if different); run `scripts/eda.py` + `scripts/prep_data.py` only if `data/splits/split_seed42.csv` is
   missing; show class-distribution + montage images if present.
4. **Train** (skipped when MODE=="inference"): load FINAL_CONFIG yaml, apply overrides (seed, epochs, device),
   call `train_one(cfg)`, print the metrics summary, plot `log.csv` curves, copy `best.pt` to `WEIGHTS_PATH`.
   Optional cell: 3-seed loop (`RUN_SEEDS = [42, 0, 1]`) collecting mean ± std.
5. **Evaluate**: load `WEIGHTS_PATH`, evaluate on the val split (strat) and on the doc-disjoint split, print
   top-1 / balanced acc / macro-F1 / minority acc, tau sweep, confusion matrix figure (Thai font from
   `assets/fonts/Sarabun-Regular.ttf`), top-20 confused pairs table.
6. **Inference demo**: `predict_topk` on (a) 12 random val glyphs with true vs predicted, (b) uploaded files via
   `google.colab.files.upload()` guarded by try/except (fallback: a path list), (c) an **ipywidgets/Gradio** cell:
   try `import gradio`; if available build `gr.Interface(fn, gr.Image(type="pil"), gr.Label(num_top_classes=5))`
   with `share=False`; else fall back to `ipywidgets.FileUpload`. Print latency per image and model size (MB, params).
7. **Robustness quick check**: run `scripts/robustness.py` with `--limit 5` on the weights and show `curves.png`.
8. **Export**: save `metrics_summary.json` + figures to `OUTPUT_DIR` (Drive if mounted).

Notebook must run with MODE="inference" without touching the dataset zip at all if `WEIGHTS_PATH` exists and the
user supplies images. Put helper functions in the notebook itself only when they are notebook-specific; otherwise
import from `thaichar.*`.

## Acceptance (reviewer runs)
```bash
uv run python scripts/build_notebook.py && uv run pytest -q tests/test_notebook.py
uv run python -c "import json; nb=json.load(open('notebooks/ThaiChar72_Colab.ipynb')); print(len(nb['cells']), 'cells'); print([c['source'][0][:60] if isinstance(c['source'], list) else c['source'][:60] for c in nb['cells'] if c['cell_type']=='markdown'][:12])"
```
The reviewer will then execute it on Colab with `colab exec -f notebooks/ThaiChar72_Colab.ipynb`.

## Report back
Files, commands, test output, list of config variables, any Colab-specific assumption you made.
