# ThaiChar72 Google Colab Deliverable Notebook

This directory contains the Google Colab deliverable notebook for the 72-class Thai handwritten character recognition project: [`ThaiChar72_Colab.ipynb`](file:///home/eggchad/eakject/research/Deep_Man/Deep_CNN/notebooks/ThaiChar72_Colab.ipynb).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

---

## 1. Overview

`ThaiChar72_Colab.ipynb` is designed to be opened in Google Colab and executed cleanly from top to bottom by a reviewer or grader.

It operates in two defensive, self-contained modes:
1. **Inference-Only Mode (`MODE = "inference"`, Default)**:
   - Skips training entirely and evaluates existing/shipped weights (`WEIGHTS_PATH`).
   - Runs full evaluation on both the stratified validation split and the document-disjoint split.
   - Generates top-1 accuracy, balanced accuracy (mean recall per class), Macro-F1, and minority class accuracy (classes with $n_{\mathrm{train}} < 50$).
   - Performs a logit adjustment $\tau$ sweep.
   - Plots a 72×72 row-normalized confusion matrix with the Sarabun Thai font (`assets/fonts/Sarabun-Regular.ttf`) and lists top-20 confused pairs.
   - Provides interactive top-$k$ prediction on random validation samples, uploaded images, and an interactive Gradio / ipywidgets interface.
   - Runs a fast corruption robustness check (`scripts/robustness.py --limit 5`).
   - Exports all results and figures to `OUTPUT_DIR` (and Google Drive if mounted).
   - **Zero-touch**: Does not touch or require the raw dataset ZIP if pre-trained weights and cache exist.

2. **Training Mode (`MODE = "train"`)**:
   - Defensive unzipping of `DATA_ZIP` (`ThaiCharacter Dataset.zip`) using Python's `zipfile` module (avoids Windows path/encoding corruption).
   - Tallies per-class file counts across all 72 classes, asserting 62,707 total files (with non-crashing warnings if counts differ).
   - Executes EDA and data preparation (`scripts/eda.py` and `scripts/prep_data.py`) if split files are not already present.
   - Trains the winning model recipe (`FINAL_CONFIG`, ResNet-18 @ 64px, AdamW, Cosine Annealing, EMA) via `train_one(cfg)`.
   - Plots loss and accuracy curves from `log.csv` and copies the best checkpoint (`best.pt`) to `WEIGHTS_PATH`.
   - Includes an optional 3-seed training loop (`RUN_SEEDS = [42, 0, 1]`) to record mean ± std metrics.

---

## 2. Configuration Variables

All paths and global parameters are isolated in **Cell 2 (Configuration Cell)**. No other cell hardcodes file paths.

| Variable | Type | Default Value | Description |
|---|---|---|---|
| `MODE` | `str` | `"inference"` | Execution mode: `"inference"` (skip training) or `"train"`. |
| `USE_DRIVE` | `bool` | `False` | When `True`, mounts Google Drive at `/content/drive`. |
| `PROJECT_DIR` | `Path` | `Path(".").resolve()` | Root directory of the repository (`src/thaichar` parent). |
| `DATA_ZIP` | `Path` | `/content/drive/MyDrive/ThaiCharacter Dataset.zip` (Drive) or `data/ThaiCharacter Dataset.zip` (local) | Path to the raw dataset ZIP archive. |
| `DATA_DIR` | `Path` | `PROJECT_DIR / "data"` | Working directory for dataset splits and cache. |
| `FINAL_CONFIG` | `Path` | `configs/final.yaml` (if present) else `configs/matrix/A1_resnet18_full_64.yaml` | Training configuration YAML file. |
| `WEIGHTS_PATH` | `Path` | `runs/A1_resnet18_full_64_T4/best.pt` (or `weights/best.pt`) | Path to model weights checkpoint for evaluation/inference. |
| `OUTPUT_DIR` | `Path` | `PROJECT_DIR / "outputs"` | Target directory for exported metrics, logs, and plots. |
| `SEED` | `int` | `42` | Random seed for reproducibility across all libraries. |
| `EPOCHS_OVERRIDE`| `int` or `None` | `None` | Overrides configuration epochs (e.g., `1` for quick verification). |

---

## 3. Notebook Structure

The notebook strictly adheres to the section structure:

0. **`# 0. Title + How to Use: ThaiChar72 Character Recognition`**: Overview, instructions for the two execution modes, and group information placeholder.
1. **`## 1. Environment`**: Hardware check (Python, PyTorch, CUDA, GPU name), quiet dependency installation (`timm`, `opencv-python-headless`, `pyyaml`, `tabulate`, `imagehash`), optional Google Drive mounting, code setup/clone, and `sys.path.insert(0, "src")`.
2. **`## 2. Configuration`**: The single configuration cell defining all 10 control variables and paths.
3. **`## 3. Data Preparation and Inspection`**: Unzips `DATA_ZIP` via `zipfile`, tallies 72-row per-class file counts, verifies 62,707 total files (with warnings), conditionally runs `eda.py` and `prep_data.py`, and displays EDA montage/distribution images.
4. **`## 4. Model Training`**: Loads `FINAL_CONFIG`, applies overrides, executes `train_one(cfg)`, reports summary metrics, plots learning curves, copies `best.pt` to `WEIGHTS_PATH`, and provides an optional 3-seed loop.
5. **`## 5. Model Evaluation`**: Evaluates `WEIGHTS_PATH` on stratified and document-disjoint splits, computes tau sweep, renders 72×72 confusion matrix with Thai font (`Sarabun-Regular.ttf`), and produces the top-20 confused pairs table.
6. **`## 6. Inference Demonstration`**: Demonstrates `predict_topk` on 12 random validation glyphs, supports file upload via `google.colab.files.upload()`, provides an interactive Gradio/ipywidgets interface, and reports latency and parameter counts.
7. **`## 7. Robustness Quick Check`**: Executes `scripts/robustness.py --limit 5` on model weights and displays `curves.png`.
8. **`## 8. Export Metrics and Artifacts`**: Packages `metrics_summary.json` and figures to `OUTPUT_DIR` (and Google Drive if mounted).

---

## 4. Colab-Specific Assumptions & Defensive Design

1. **Defensive Path Resolution**: Paths dynamically adapt whether running from the repository root, inside `notebooks/`, or inside `/content/thaichar` on Google Colab.
2. **Native Python ZIP Extraction**: Python's `zipfile` module is used rather than OS extraction tools to prevent encoding corruption of Thai file paths on Windows or multi-byte filesystems.
3. **Graceful Drive Fallback**: Google Drive mounting is wrapped in `try/except`; if running outside Colab or with `USE_DRIVE=False`, execution continues without blocking or error.
4. **Pure Python Syntax**: All code cells use pure Python commands (e.g. `subprocess.check_call` rather than shell `!pip` magics) so every cell compiles under `compile()` without syntax errors.
5. **Hardware Expectations**:
   - Hardware: Google Colab Free Tier with T4 GPU (`--gpu T4`).
   - Training speed: ~28 s/epoch for ResNet-18 @ 64px on T4 GPU (AMP enabled).
   - CPU Fallback: Automatically falls back to CPU if CUDA is unavailable.

---

## 5. Programmatic Generation & Acceptance Testing

The notebook is generated programmatically using `nbformat` to guarantee 100% reproducibility.

### Build Notebook
```bash
uv run python scripts/build_notebook.py
```

### Run Notebook Acceptance Tests
```bash
uv run pytest -q tests/test_notebook.py
```

### Inspect Notebook Structure
```bash
uv run python -c "import json; nb=json.load(open('notebooks/ThaiChar72_Colab.ipynb')); print(len(nb['cells']), 'cells'); print([c['source'][0][:60] if isinstance(c['source'], list) else c['source'][:60] for c in nb['cells'] if c['cell_type']=='markdown'][:12])"
```
