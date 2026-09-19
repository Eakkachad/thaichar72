# TASK-02 — Clean index, stratified split, image cache, torch Dataset

## Context
Project root = this directory (`Deep_CNN`), `uv`-managed (Python 3.12). Torch (CPU), torchvision, timm,
numpy, pandas, pillow already installed — run with `uv run python ...`. No GPU on this machine.
Read `reports/eda/EDA.md` and `reports/eda/summary.json` first. `reports/eda/files.csv` already holds one
row per image with `path, code, char, category, prefix, doc_id, subset, line, idx, is_copy, group, width,
height, md5, ok, ink_frac, ...` — REUSE it, do not re-hash the dataset. `reports/eda/duplicates.json` holds
md5 duplicate groups with a `cross_class` flag. `src/thaichar/classes.py` has `CLASS_CODES`, `code_to_index`.

## Scope — you may create/modify ONLY
- `src/thaichar/data.py`, `src/thaichar/transforms.py`
- `scripts/prep_data.py`
- `data/**` (new; git-ignored outputs), `reports/data/**` (new)
- `tests/test_data.py` (new; add `pytest` with `uv add --dev pytest` if needed and say so)
HARD RULES: `ThaiCharacter Dataset/` is READ-ONLY. Do not touch `../CNN`, `../katgpt-rs`,
`../fly-connectome-template`, `tasks/`, `weights/`, `notebooks/`, `scripts/eda.py`, `reports/eda/`.
No `git commit`.

## Deliverable 1 — `src/thaichar/data.py`
`build_clean_index(files_csv="reports/eda/files.csv", duplicates_json="reports/eda/duplicates.json") -> pd.DataFrame`
Cleaning policy (apply in this order, log counts of each drop):
1. drop rows with `ok == False`
2. drop every file whose name starts with `Copy of` (they are mislabelled duplicates)
3. drop ALL members of md5 groups flagged `cross_class` (same bytes, two labels → label unknown)
4. within-class exact duplicates (same md5, same code): keep the lexicographically first path, drop the rest
Result columns: `path, code, label (0..71 via code_to_index), char, category, group, doc_id, prefix,
subset, width, height, ink_frac`. Expected size ≈ 60,1xx rows (report the exact number).

`make_stratified_split(df, seed=42, val_frac=0.2) -> pd.Series of {"train","val"}`
Per class, sort paths, shuffle with `np.random.default_rng(seed)`, then n_val = `round(val_frac*n)` for
n ≥ 5; n=4→1, n=3→1, n=2→1, n=1→0 (that class then has no val sample — record it). Deterministic.

`make_doc_split(df, seed=42, n_holdout_docs=4) -> pd.Series` — hold out `n_holdout_docs` of the 20
`doc_id`s (chosen with the seeded rng so that the held-out share of images is the closest to 20% among
200 random draws) → `"val"` for those docs, `"train"` otherwise. Secondary honesty split only.

`write_splits(df, out_dir="data/splits", seeds=(42, 0, 1))` writes `split_seed{S}.csv` with columns of the
clean index plus `split` (stratified) and `doc_split`. Assert train∩val = ∅ for both and print the overlap
count (must be 0).

## Deliverable 2 — image cache
`build_cache(df, out="data/cache/glyphs.npz")` decodes every image once with PIL (`convert("L")`),
binarises `img > 127` → uint8 {0,255} (0 = ink), and stores a flat concatenated uint8 array + `offsets,
widths, heights` (variable-size crops, NO resizing here) plus the path list, so the cache is independent of
the input size chosen later. Use multiprocessing (8 workers). `load_cache(path) -> GlyphCache` object with
`__getitem__(i) -> np.ndarray[H,W] uint8` and `.paths`. Whole cache should be ~20–30 MB.

## Deliverable 3 — `src/thaichar/transforms.py`
`fit_to_square(img_u8, size:int, margin:float=0.1, pad_value=255) -> np.ndarray[size,size]`:
aspect-preserving. Compute canvas side `c = ceil(max(h,w) * (1+2*margin))`, paste the glyph centred on a
white canvas of side c, then resize to `size` with PIL `Image.resize(..., Image.BILINEAR)`
(never stretch; never flip). `geometry_features(h, w, ink_frac) -> np.ndarray[4] float32` =
`[log(w), log(h), log(w/h), ink_frac]`.
Train-time augmentation is NOT part of this task (comes in TASK-03) — only provide the deterministic path.

## Deliverable 4 — torch Dataset
`class ThaiGlyphDataset(torch.utils.data.Dataset)` in `data.py`:
`__init__(self, split_df, cache, size=64, channels=3, transform=None, return_geometry=True)` where
`transform` is an optional callable applied to the uint8 square image (numpy HxW) BEFORE tensor conversion
(hook for augmentation later). `__getitem__` returns `(x, y, g)`: `x` float tensor `[channels,size,size]`
normalised with ImageNet mean/std when `channels==3` (replicate grey to 3 channels) or to [0,1] when
`channels==1`; `y` int64 label; `g` float32 tensor[4] geometry. Also `make_loader(ds, batch_size,
shuffle, num_workers, sampler=None)`.

## Deliverable 5 — `scripts/prep_data.py`
CLI that runs everything: clean index → splits (seeds 42, 0, 1) → cache → writes
`reports/data/split_summary.md` containing: drop counts per cleaning rule, final n, per-class table
(code, char, n_clean, n_train, n_val for seed 42, n_doc_train, n_doc_val), list of classes with 0 val
samples, the held-out doc ids, and the printed overlap assertions. Must finish in < 5 min.

## Deliverable 6 — `tests/test_data.py` (pytest)
- split is deterministic (same seed → identical assignment) and train∩val = ∅ for both split kinds
- every class with n ≥ 2 has ≥ 1 val sample in the stratified split
- `fit_to_square` preserves aspect ratio: a 10×30 glyph fills ≤ size/(1+2*margin) height and its
  width/height ratio is within 10% of 1/3 after resizing; output is `size×size`; no ink touches the border
- `ThaiGlyphDataset[0]` shapes: size 64 → x `[3,64,64]`, and with `channels=1,size=32` → `[1,32,32]`
- loading 2,000 random samples from the cache takes < 2 s

## Acceptance (reviewer runs)
```bash
uv run python scripts/prep_data.py
uv run pytest -q tests/test_data.py
uv run python -c "import pandas as pd; d=pd.read_csv('data/splits/split_seed42.csv'); print(len(d)); print(d.split.value_counts()); print(d.doc_split.value_counts()); assert d.label.nunique()==72"
ls -la data/cache/glyphs.npz reports/data/split_summary.md data/splits/
```

## Report back
Files created, commands run, drop counts per rule, final row count, train/val sizes, classes with 0 val
samples, held-out doc ids, cache size on disk, test output.
