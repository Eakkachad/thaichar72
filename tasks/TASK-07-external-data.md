# TASK-07 — Fetch public Thai character datasets and map them to our 72 classes

## Context
Project root = this dir (`Deep_CNN`), `uv` Python 3.12. Installed: numpy, pandas, pillow, torch, tqdm, requests?
You MAY `uv add datasets huggingface_hub requests beautifulsoup4` (say so). Read `src/thaichar/classes.py`
(`CLASS_CODES`, `code_to_char`, `char_to_code`, `code_to_index`) and `src/thaichar/data.py` (`build_cache` /
`load_cache` — the `.npz` glyph cache format: flat uint8 array + offsets/widths/heights + paths).
Our target classes: 72 TIS-620 codes = Thai consonants/vowels/marks/digits listed in `CLASS_CODES`.
Our data are **printed** binary glyphs; the public sets below are **handwritten** — that is expected; they are
for intermediate pretraining, not for mixing into validation.
Two other agents are concurrently editing `src/thaichar/{augment,synth,transforms,data,models,losses,
samplers,metrics,engine}.py` and `scripts/{train,render_synth,aug_examples,collect_results}.py` — do NOT touch
those. Network access is allowed. Keep total download < 3 GB; skip anything gated/login-only and say so.

## Scope — you may create/modify ONLY
`src/thaichar/external.py`, `scripts/fetch_external.py`, `data/external/**`, `reports/data/external_summary.md`,
`reports/figures/external_montage.png`, `tests/test_external.py`, `pyproject.toml` (deps only).
Dataset dir read-only; no `git commit`; do not touch `../CNN`, `../katgpt-rs`, `../fly-connectome-template`.

## Sources to try (in this order; record outcome for each)
1. **ALICE-THI** — HF `SEACrowd/alice_thi` (loader script `alice_thi.py`, `trust_remote_code=True`); 24,045
   handwritten images: THI-C68 (68 character classes) + THI-D10 (10 Thai digits). Inspect the loader to
   find the raw download URL and the label→character mapping (labels may be indices or folder names; map to
   Unicode characters explicitly and print the mapping table).
2. **KVIS Thai OCR** — HF `SEACrowd/kvis_th_ocr` (1,079 images, 44 consonants, CC-BY-4.0; loader may pull
   from Mendeley Data).
3. **Burapha-TH character + digit sets** — file server `https://services.informatics.buu.ac.th/datasets/Burapha-TH/`
   has `character/`, `digit/`, `syllable/` directories (skip `syllable/`). Crawl the listing with requests +
   BeautifulSoup, download archives/images for `character/` and `digit/` if total < 2 GB, and read any
   README/terms file present. Cite: Onuean et al., Applied Sciences 2022 (Burapha University).
4. (Optional, only if trivial) other HF datasets found by searching `huggingface_hub.list_datasets(search="thai")`
   whose content is isolated Thai characters or digits. Skip line/sentence-level OCR sets.

## D1 — `src/thaichar/external.py`
- `map_char_to_class(ch: str) -> int | None` — Unicode char → our label index via `char_to_code`; `None` if not
  one of the 72 (e.g. ฅ ฆ ฌ ฎ ฦ ะ ำ). Handle labels given as consonant names/transliterations if a dataset
  uses them (build an explicit dict, print it).
- Per-source `fetch_<name>(out_dir) -> pd.DataFrame` with columns `src_path, source, orig_label, char, code,
  label (or -1 if unmapped), writer_id (if available)`; images saved/kept as files under
  `data/external/<source>/...` (grayscale). 
- `binarize_for_cache(img) -> np.ndarray uint8 {0,255}`: grayscale → Otsu threshold (cv2 or numpy
  implementation) → ensure ink is dark (0) on white (255) (check mean; invert if needed) → crop to ink
  bounding box with 0 margin (like our data). Reject images whose ink fraction < 1% or > 90%.
- `build_external_cache(df, out_npz, index_csv)` writing the SAME `.npz` format as `data/cache/glyphs.npz`
  (must load with `load_cache`) + `index.csv` with `path, code, label, char, category, width, height,
  ink_frac, group="<source>_<writer or split>"`, mapped rows only.

## D2 — `scripts/fetch_external.py`
CLI `uv run python scripts/fetch_external.py --sources alice kvis burapha --out data/external` → downloads,
maps, builds `data/external/glyphs_external.npz` + `data/external/index.csv`, writes
`reports/data/external_summary.md` with: a table per source (name, URL, license/terms as found, citation,
n_images downloaded, n_mapped, n_unmapped and which characters were unmapped, image size stats), the combined
per-class count table for our 72 classes (which of our classes get **zero** external samples — expect the
tone marks / some vowels), and `reports/figures/external_montage.png` (one row per source, 12 samples each,
after binarisation, labelled with the mapped char).

## D3 — `tests/test_external.py` (no network; use 20 synthetic grey images)
`map_char_to_class('ก')==0`, `map_char_to_class('ฅ') is None`; `binarize_for_cache` returns {0,255}, ink dark,
tight-cropped; `build_external_cache` round-trips through `load_cache`.

## Acceptance (reviewer runs)
```bash
uv run pytest -q tests/test_external.py
uv run python scripts/fetch_external.py --sources alice kvis burapha --out data/external
uv run python -c "import pandas as pd; d=pd.read_csv('data/external/index.csv'); print(len(d)); print(d.groupby('group').size().head()); print(d.label.nunique(), 'classes covered')"
cat reports/data/external_summary.md | head -80
```
## Report back
Per source: success/failed and why, URL, license text found, counts; combined class coverage; disk usage; anything surprising.
