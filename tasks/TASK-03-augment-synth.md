# TASK-03 — Augmentation presets, channel encodings, synthetic Thai-font glyph renderer

## Context
Project root = this dir (`Deep_CNN`), `uv`-managed Python 3.12. Installed: numpy, pillow, pandas, torch (CPU),
torchvision, timm, matplotlib, scikit-learn, pytest. You MAY `uv add opencv-python-headless scipy` (say so).
Read first: `reports/01-EDA-REPORT.md` (Thai; key facts: glyphs are tiny 16×19 px median, strictly binary
0/255, tight-cropped, aspect ratio is a class cue, **never flip**), `src/thaichar/transforms.py`
(`fit_to_square`, `geometry_features`), `src/thaichar/data.py` (`GlyphCache`, `ThaiGlyphDataset`,
`load_cache`), `src/thaichar/classes.py` (`CLASS_CODES`, `code_to_char`, `category`),
`reports/eda/class_stats.csv` (per-class `med_w, med_h, mean_ink_frac`). Fonts: `assets/fonts/*.ttf` (27 Thai
Google Fonts, OFL).

## Scope — you may create/modify ONLY
`src/thaichar/augment.py`, `src/thaichar/synth.py`, `src/thaichar/transforms.py` (add functions; keep existing
signatures working), `src/thaichar/data.py` (ONLY the `ThaiGlyphDataset` class + a new `ConcatIndexDataset`
helper — see D3; do not change cleaning/split code), `scripts/render_synth.py`, `scripts/aug_examples.py`,
`tests/test_augment.py`, `tests/test_synth.py`, `data/synth/**`, `reports/figures/**`, `reports/data/synth_summary.md`,
`pyproject.toml` (deps only). HARD RULES: `ThaiCharacter Dataset/` read-only; do not touch `../CNN`,
`../katgpt-rs`, `../fly-connectome-template`, `tasks/`, `weights/`, `notebooks/`, `scripts/eda.py`,
`scripts/prep_data.py`, `reports/eda/`. No `git commit`. Another agent is concurrently writing
`src/thaichar/models.py`, `losses.py`, `samplers.py`, `engine.py`, `scripts/train.py` — do not create or edit those.

## D1 — `src/thaichar/augment.py`  (numpy uint8 in → numpy uint8 out; images are white bg 255, ink 0)
All ops work on a **square white canvas** (the padded canvas produced by `pad_to_square_canvas`, see D2), size
typically 20–60 px, BEFORE the final resize, so morphology acts at near-native stroke scale.
Ops (each a small class/func with `p` and magnitude; use cv2 or PIL):
- `RandomAffine(rot=8°, shear=10°, scale=(0.85,1.15), translate=0.08)` — border fill 255, no flip ever
- `MarginJitter(±15% canvas)` — re-pad / crop the canvas edges
- `StrokeWidth(p=0.4)` — dilate or erode ink with 3×3 (or 2×2) kernel, 1 iteration (ink is 0 → use erode on
  inverted mask or equivalently `cv2.erode` for thickening ink; get the polarity right and test it)
- `ResolutionJitter(p=0.4, scale=(0.5,1.0))` — downscale then upscale back (bilinear/nearest random), then
  re-binarise at 128 with p=0.5
- `Elastic(p=0.3, alpha=canvas*0.15, sigma=canvas*0.08)` — cv2.remap grid warp
- `Speckle(p=0.3, frac=0.01–0.03)` salt & pepper; `GaussianBlur(p=0.2, sigma≤0.6)`
- `RandomErasing(p=0.25, max side 20% of canvas, fill white or black)`
- `Rebinarize(p=0.5)` threshold 128 → keeps the "1-bit print" look
Presets via `get_transform(preset: str, seed: int|None=None) -> Callable[[np.ndarray], np.ndarray]`:
`none` (identity), `base` (affine+margin), `morph` (base+StrokeWidth+ResolutionJitter),
`full` (morph+Elastic+Speckle+Blur+Erasing+Rebinarize), `randaug` (RandAugment N=2, M∈[0,10] sampling from
the op list above with magnitude scaling; NO flips/colour ops), `trivial` (TrivialAugment: one op, uniform
magnitude). Every preset must guarantee the output still contains ink (≥ 1% ink pixels); if not, return the
input unchanged. Export `PRESETS: list[str]`.

## D2 — `src/thaichar/transforms.py` additions
- `pad_to_square_canvas(img_u8, margin=0.1, pad_value=255) -> np.ndarray[c,c]` (the first half of
  `fit_to_square`; refactor `fit_to_square` to call it then resize — behaviour must stay identical, existing
  tests in `tests/test_data.py` must still pass).
- `resize_square(canvas_u8, size) -> np.ndarray[size,size]` (bilinear).
- `encode_channels(square_u8, mode) -> np.ndarray[C,size,size] float32`, modes:
  `gray1` → [1,S,S] in [0,1]; `gray3` → replicate ×3 then ImageNet mean/std (identical to the current
  Dataset behaviour); **`onoff`** → 3 channels: (a) ink mask ∈{0,1}, (b) distance transform of the ink
  region (`cv2.distanceTransform` on ink), normalised by `S/16` and clipped to [0,1] (≈ stroke-thickness
  map), (c) Sobel edge magnitude of the ink mask normalised to [0,1]; then standardise each channel with
  mean 0.5/std 0.5. Document the biological motivation in the docstring in one sentence (ON/OFF pathway split
  in the fly optic lobe — separate channels for contrast polarity/structure).

## D3 — `ThaiGlyphDataset` changes (in `data.py`)
New signature: `ThaiGlyphDataset(split_df, cache, size=64, channel_mode="gray3", transform=None,
return_geometry=True, margin=0.1)`; keep `channels=` as a deprecated alias (3→gray3, 1→gray1).
Pipeline in `__getitem__`: crop → `pad_to_square_canvas(margin)` → `transform(canvas)` if given →
`resize_square(size)` → `encode_channels(mode)` → tensor. Geometry features unchanged.
Add `ConcatIndexDataset`-free approach instead: make the dataset accept a list of `(df, cache)` pairs
OR add a helper `merge_sources([(df, cache), ...]) -> (df_merged, MultiCache)` where `MultiCache` maps path →
(cache, idx). Purpose: real train split + synthetic glyphs in one loader. Choose one and document it.

## D4 — `src/thaichar/synth.py` + `scripts/render_synth.py`
Render all 72 classes from every font in `assets/fonts` with `PIL.ImageFont` (skip a font for a glyph if it
lacks the codepoint — detect via `font.getmask` being empty or `getbbox` None; log it).
- Standalone glyphs (consonants, เ แ โ ใ ไ ๅ ๆ ฯ, digits): render on white, crop to ink bbox.
- **Combining marks** (ั ิ ี ึ ื ุ ู ็ ่ ้ ๊ ์ — Unicode general category Mn): render `base+mark` with base
  `อ` (and also `ก`, `ป` for the upper marks to vary the vertical position), render the base alone at the
  same origin, subtract (pixels inked in base+mark but not in base) → crop to the mark's bbox. Reject if
  the result is empty or touches the base region.
- Degradation to match the real corpus, per sample: choose a target height by sampling around the class's
  real median height (`class_stats.csv` `med_h`, ×U(0.8,1.3)), render large (e.g. 128 px) then downscale
  to that height (aspect preserved), random stroke weight (dilate/erode 0–2 px at high res before
  downscale), small rotation ±3° / shear ±5°, binarise at a random threshold U(100,160), speckle p=0.3,
  optional 1-px erosion of the result p=0.2. Output must be uint8 {0,255}, tight-cropped like the real data.
- Also render a **clean, undegraded** variant flag for inspection.
- Output: `data/synth/glyphs_synth.npz` in the SAME format as `data/cache/glyphs.npz` (reuse `build_cache`'s
  writer or write a compatible one; `load_cache` must load it) and `data/synth/index.csv` with columns
  `path` (virtual id like `synth/<code>/<font>_<i>.png`), `code, label, char, category, width, height,
  ink_frac, font, degraded(bool), group="synth_<font>"`. CLI: `uv run python scripts/render_synth.py
  --per-class 300 --seed 0` (default 300/class → 21,600; must run < 5 min).
- `reports/data/synth_summary.md`: fonts used, glyphs skipped per font, per-class counts, comparison table of
  median w/h and mean ink_frac real vs synthetic; figure `reports/figures/synth_montage.png` (72 classes ×
  6 samples, degraded) and `reports/figures/synth_vs_real.png` (10 classes: 4 real + 4 synth each).

## D5 — `scripts/aug_examples.py`
Writes `reports/figures/aug_examples.png`: rows = 8 real glyphs (mixed classes), columns = original + 5
random draws for each preset base/morph/full/randaug/trivial (group columns by preset with titles). Also
`reports/figures/channel_modes.png` showing gray3 vs onoff channels for 6 glyphs.

## D6 — tests
`tests/test_augment.py`: every preset over 300 random real canvases → same dtype, square, ≥1% ink, values in
{0..255}; `none` is identity; `encode_channels` shapes for the 3 modes; `pad_to_square_canvas`+`resize_square`
equals old `fit_to_square` bit-exactly; Dataset with `channel_mode="onoff"` returns `[3,64,64]`.
`tests/test_synth.py`: index has 72 classes; every combining mark class has ≥ 50 samples and none is identical
to the rendered base; per-class synthetic median height within ±35% of real median; cache loads via
`load_cache` and `len == len(index)`.

## Acceptance (reviewer runs)
```bash
uv run pytest -q tests/test_data.py tests/test_augment.py tests/test_synth.py
uv run python scripts/render_synth.py --per-class 300 --seed 0
uv run python scripts/aug_examples.py
ls -la data/synth/glyphs_synth.npz data/synth/index.csv reports/figures/aug_examples.png reports/figures/synth_montage.png reports/figures/synth_vs_real.png reports/figures/channel_modes.png reports/data/synth_summary.md
```
## Report back
Files, commands, test output, fonts skipped, per-class synth counts (min/max), real-vs-synth size table, anything surprising.
