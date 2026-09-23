# ThaiChar72 — 72-Class Thai Printed-Glyph Recognition

A course project implementing end-to-end recognition of all 72 Thai Unicode character classes (consonants, vowels, tone marks, numerals) from tiny binarised scans of printed Thai text (median 16×19 px).
Final model: **ResNet-18 @ 64 px**, two-stage transfer (ImageNet → synthetic Thai fonts → real scans), RandAugment (no flips),
AdamW + cosine + EMA, 20 epochs, then knowledge-distilled from 3 seeds into one network (`weights/thaichar72_resnet18_64.pt`, 44.9 MB).

---

## Results (2026-09-22, RTX 4070)

Raw numbers (no TTA) from [`reports/experiments.csv`](reports/experiments.csv); full story in
[`reports/FINAL-REPORT.md`](reports/FINAL-REPORT.md) and [`reports/02-EXPERIMENTS.md`](reports/02-EXPERIMENTS.md).

**Shipped model: `weights/thaichar72_r18_64_gen.pt`** — ImageNet → stage-1 pretrain on 203 typefaces
+ the `dataUpdate` corpus (126,114 synthetic glyphs) → fine-tune on real data with RandAugment.
Recipe: [`configs/final.yaml`](configs/final.yaml).

| Model | stratified | doc-disjoint | robustness* | unseen fonts** |
|---|---:|---:|---:|---:|
| **`thaichar72_r18_64_gen.pt`** (shipped) | **99.02 %** | **98.72 %** | **0.915** | **0.868** |
| `thaichar72_r18_64_gen_rotaug.pt` (wider rotation aug) | 99.00 % | – | – | – |
| `thaichar72_resnet18_64.pt` (K1, the previous shipped model) | 99.03 %† | 98.59 %† | 0.910† | 0.834 |
| `thaichar72_mnv3small_64_small.pt` (1.6 M params, 6.5 MB) | 99.00 %† | – | 0.905† | 0.774 |
| `thaichar72_resnet18_64_v1labels.pt` (original label convention) | 98.39 %† | 98.22 %† | 0.905† | 0.827 |

\*mean Top-1 over 33 corruption settings with Otsu binarisation (`scripts/robustness.py --binarize`).
\**retention on 62 typefaces the training render never used ([§N](reports/02-EXPERIMENTS.md)).
†measured on the earlier split; not directly comparable with the rows above (see FINAL-REPORT §13).

Selection was on the **document-disjoint split and the robustness gate**, never the stratified split,
which is saturated at ~99 % and separates nothing. 3 seeds of the shipped recipe: 98.99 ± 0.04 %.

### How it behaves on input it was not trained for

[`reports/stress/summary.md`](reports/stress/summary.md) measures all five weights on degraded,
invalid and flipped input. Headlines:

- **Low resolution is fine.** 0.974 on real glyphs under 9 px tall — the corpus already contains them.
- **JPEG, high resolution and aspect distortion cost nothing**; Otsu absorbs them.
- **Rotation was the real gap** (0.52 at 30°). `--rotation-tta` buys it back to 0.90 at zero cost on
  upright input.
- **Erosion is far more damaging than dilation** — never add a stroke-thinning step.
- **Invalid input (two glyphs touching, half a glyph, scribbles) is filterable by confidence alone**
  at AUROC ≥ 0.84, so no glyph splitter is needed.
- **Never flip.** 127 class/transform pairs become a *different real Thai character* confidently
  (บ↔ภ and ย↔ถ are exact reciprocals under vertical mirroring).

---

## Repository Layout

```
src/thaichar/           Core library
  classes.py            72-class codes, Unicode char map, category helper
  data.py               ThaiGlyphDataset, load_cache, build_datasets
  augment.py            Augmentation presets (none/trivial/full/randaug)
  synth.py              Synthetic Thai-font glyph generator
  models.py             SmallCNN + timm-backed ResNet/MNv3/EffNet factory
  engine.py             train_one(), predict(), pick_device(), merge_cfg()
  infer.py              load_checkpoint(), preprocess_image(), predict_topk()
  metrics.py            compute_metrics(), tau_sweep()
  losses.py             CE / weighted-CE / CB-focal loss
  samplers.py           Inverse-freq / sqrt-inv samplers
  transforms.py         Geometry encoding, on/off preprocessing

scripts/
  train.py              CLI training entry point (--config, --set key=value)
  run_local_queue.py    Sequential GPU queue of configs (skips finished runs)
  eval_split.py         Evaluate any checkpoint on the val rows of any split file (v1 vs v2 labels)
  make_split_v2.py      Apply the DataV2 label corrections to split_seed42.csv (replay from the tracked change list)
  export_weights.py     Package runs/<id>/best.pt as weights/<name>.pt + model card
  ensemble_eval.py      Offline ensemble / subset evaluation from saved val logits
  predict.py            Single-image or batch inference CLI
  prep_data.py          Build clean index + splits + glyph cache
  eda.py                EDA: per-class counts, dedup, distribution plots
  collect_results.py    Sweep runs/ → reports/experiments.csv
  robustness.py         Corruption sweep (rotation/blur/noise/occlusion)
  error_analysis.py     Per-class error breakdown, confused-pair table
  build_notebook.py     Programmatically generate notebooks/ThaiChar72_Colab.ipynb
  run_notebook_local.py Headless notebook execution harness (acceptance)
  colab_setup.sh        Bundle + upload code/data to Colab session
  colab_master_queue.sh Run full experiment matrix on Colab T4

configs/
  base.yaml             Default hyperparameter base
  matrix/               Per-experiment YAML overrides (A0…E6)
  pretrain/             Pre-training configs (synthetic data)
  final_candidates/     Candidate final configs F1–F19 + K1 (KD)
  extras/               Winner recipe on other backbones + small-model KD (X1–X6)
  final.yaml            The shipped recipe (F19 on the v2 split)

weights/                Deliverable weights + *.card.json (metrics, md5, cfg, load snippet); see weights/README.md

notebooks/
  ThaiChar72_Colab.ipynb    Colab deliverable (generated by build_notebook.py)
  README.md                 Config-variable reference and notebook guide

reports/
  experiments.csv       Machine-readable results table (42 experiments)
  01-EDA-REPORT.md      Dataset statistics (62,707 images, 72 classes)
  02-EXPERIMENTS.md     Narrative experiment log

tasks/                  Task specs and delegation log
  COLAB-USAGE.md        Verified colab-cli usage patterns
  DELEGATION-LOG.md     Per-task delegation history (research lead + agy)
```

---

## Quick Start (local, CPU)

Requires Python 3.12 and [`uv`](https://docs.astral.sh/uv/).

```bash
# 1. Install all dependencies
uv sync

# 2. EDA (requires raw dataset in 'ThaiCharacter Dataset/round2/')
uv run python scripts/eda.py --data "ThaiCharacter Dataset/round2" --out reports/eda

# 3. Build clean index, splits, and glyph cache
uv run python scripts/prep_data.py

# 4. Train (example: base ResNet-18 config)
uv run python scripts/train.py --config configs/matrix/A1_resnet18_full_64.yaml

# 5. Collect results from all runs
uv run python scripts/collect_results.py

# 6. Single-image prediction with the shipped model
uv run python scripts/predict.py --ckpt weights/thaichar72_r18_64_gen.pt path/to/glyph.png --topk 5

# 7. Corruption robustness sweep (full val, Otsu on)
uv run python scripts/robustness.py --ckpt weights/thaichar72_r18_64_gen.pt --binarize --out reports/robustness/my_sweep

# 8. Retrain the final recipe (needs data/ caches + weights/thaichar72_r18_synth_pretrain_init.pt or runs/B6a_*/best.pt)
uv run python scripts/make_split_v2.py                       # builds data/splits/split_seed42_v2.csv from the tracked change list
uv run python scripts/train.py --config configs/final.yaml --set device=cuda
```

### Scoring a folder of test images

Run the triage first: it reads the images with no labels and says how to run the classifier.

```bash
uv run python scripts/triage_input.py  --dir path/to/test_images
uv run python scripts/evaluate_folder.py --dir path/to/test_images [--rotation-tta]
```

`triage_input.py` reports resolution, ink weight, polarity and whether the batch is rotated, and
prints the `evaluate_folder.py` command to use. It detects rotation from *agreement on the winning
angle*, not from a confidence lift — a lift alone also appears on blurred or low-resolution batches
that are not rotated at all.

`evaluate_folder.py` accepts any folder layout, scores it when the filenames carry a class, and
tries both ink polarities by default.

### Inspecting the model

```bash
uv pip install jupyterlab ipykernel
uv run python -m ipykernel install --user --name thaichar --display-name "thaichar"
uv run jupyter lab notebooks/ThaiChar72_StressEval.ipynb
```

Two notebooks: `ThaiChar72_Colab.ipynb` is the deliverable (training + evaluation + an error
explorer showing every mistake as an image), and `ThaiChar72_StressEval.ipynb` is for checking the
robustness claims above rather than taking them — each table has a drill-down that shows the actual
images and what the model said about them. Both are generated; edit the builders in `scripts/`.

---

## Google Colab Usage

See [`tasks/COLAB-USAGE.md`](tasks/COLAB-USAGE.md) for verified colab-cli patterns and session management.
The Colab experiment scripts are in [`scripts/colab_setup.sh`](scripts/colab_setup.sh) and [`scripts/colab_master_queue.sh`](scripts/colab_master_queue.sh).

```bash
# Run the deliverable notebook locally (inference mode, < 15 min on CPU)
uv run python scripts/run_notebook_local.py --mode inference \
    --weights weights/thaichar72_resnet18_64.pt
```

The notebook is generated programmatically; never edit the `.ipynb` by hand:

```bash
uv run python scripts/build_notebook.py   # regenerate
uv run pytest -q tests/test_notebook.py   # validate structure
```

---

## Data Policy

- `ThaiCharacter Dataset/` (raw images) is **read-only and never committed**; it is listed in `.gitignore`.
- Deduplication: 2,566 exact-MD5 duplicates and 11 cross-class images removed → 60,117 clean rows.
- Processed artefacts committed: `data/splits/split_seed42.csv`, `data/cache/glyphs.npz` (~34 MB).
- External datasets (ALICE-THI, KVIS) are fetched on demand by `scripts/fetch_external.py` and stored in `data/external/` (also git-ignored).

---

## Loading Weights

```python
import sys; sys.path.insert(0, "src")
from thaichar.infer import load_checkpoint, preprocess_image, predict_topk

model, cfg = load_checkpoint("weights/thaichar72_resnet18_64.pt", device="cpu")   # or _fp16.pt / _v1labels.pt / mnv3small
# cfg contains model arch, img_size, channel_mode, etc.

import numpy as np
glyph = np.full((64, 64), 255, dtype=np.uint8)  # white canvas with black strokes
preds = predict_topk(model, cfg, glyph, k=5)
for char, code, prob in preds:
    print(f"{char} ({code}): {prob*100:.1f}%")
```

---

## Role-Split Note

Research lead designed experiments and reviewed deliverables.
Implementation tasks were delegated to the **Antigravity CLI (`agy`)** coding agent on the laptop; on the RTX 4060 machine (no `agy`)
the research lead wrote the code itself (`SELF` rows). See [`tasks/DELEGATION-LOG.md`](tasks/DELEGATION-LOG.md) for the per-task history,
[`tasks/HANDOFF.md`](tasks/HANDOFF.md) §−1 for the current state, and `bash tasks/make_handoff_tarball.sh` to package everything for another machine.
