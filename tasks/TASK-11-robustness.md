# TASK-11 — Robustness sweep + inference helper (script only; full runs happen on Colab later)

## Context
Project root = this dir, `uv` Python 3.12, CPU only here (so keep every test command small with `--limit 200`).
Read: `src/thaichar/engine.py` (`predict`, `build_datasets`, `merge_cfg`, `load_compatible`; a checkpoint
`runs/<exp>/best.pt` holds `{"state_dict", "cfg", "class_codes", ...}`), `src/thaichar/models.py::build_model`,
`src/thaichar/data.py::ThaiGlyphDataset` (args: split_df, cache, size, channel_mode, transform (callable on the
padded uint8 canvas, white=255/ink=0), margin), `src/thaichar/transforms.py`, `src/thaichar/metrics.py`,
`src/thaichar/classes.py`. A trained example checkpoint: `runs/A1_resnet18_full_64_T4/best.pt` (cfg inside says
model=resnet18, img_size=64, channel_mode=gray3, geometry=false). Thai font for plots: `assets/fonts/Sarabun-Regular.ttf`.

## Scope — create/modify ONLY: `src/thaichar/infer.py`, `scripts/robustness.py`, `scripts/predict.py`,
`tests/test_infer.py`, `reports/robustness/**`. No git commit. Every test command below runs in < 60 s on CPU —
run them directly in the foreground and wait; no background tasks, no subagents.

## D1 — `src/thaichar/infer.py`
- `load_checkpoint(path, device="cpu") -> (model, cfg)`: rebuild with `build_model(**from cfg, pretrained=False)`,
  load state_dict, eval mode.
- `preprocess_image(img: np.ndarray | PIL.Image | path, cfg) -> (x tensor [1,C,S,S], g tensor [1,4])`:
  accept ANY input photo/scan: convert to grey → Otsu threshold (cv2) → make ink dark on white (invert if mean says
  otherwise) → crop to ink bbox (reject empty) → `pad_to_square_canvas(margin=cfg["margin"])` → `resize_square` →
  `encode_channels(cfg["channel_mode"])`; geometry from the crop's h, w, ink fraction.
- `predict_topk(model, cfg, img, k=5) -> list[(char, code, prob)]` (softmax; optional `tau` logit adjustment with
  a `log_prior` argument).
- `Corruptions`: deterministic canvas-level functions for the sweep, each `f(canvas_u8, severity) -> canvas_u8`:
  `rotate` (deg: 5,10,15,20), `thickness` (dilate/erode ink by 1,2,3 px — use both directions as `thicker`/`thinner`),
  `salt_pepper` (frac 0.01,0.05,0.10,0.20), `downscale` (scale 0.8,0.6,0.4 then back up, re-binarise),
  `occlusion` (random box covering 10,25,40 % of the canvas side, white fill, seeded), `blur` (sigma 0.5,1.0,1.5),
  `contrast` (grey-level: map ink 0→v and paper 255→255-v for v in 60,120,180, i.e. low-contrast scan — this
  tests images that are no longer binary), `translate` (shift 5,10,20 % of canvas with white fill),
  `background_noise` (add grey texture: gaussian noise sigma 20,40,60 on the paper only).
  Provide `CORRUPTIONS: dict[name, list[severity]]`.

## D2 — `scripts/robustness.py --ckpt runs/X/best.pt [--split-kind strat] [--limit N] [--out reports/robustness/X]`
Evaluate the checkpoint on the val split (first `--limit` samples per class-stratified subset if given) for the
clean case and for every (corruption, severity). Write `results.csv` (corruption, severity, top1, balanced_acc,
macro_f1, n) and `curves.png` (one panel per corruption: top1 and balanced acc vs severity, clean level as a
dashed line), plus `summary.md`. Reuse `ThaiGlyphDataset(..., transform=corruption_fn)` with a fixed-severity
lambda, and `engine.predict`. Must work on CPU and cuda (device auto).

## D3 — `scripts/predict.py --ckpt runs/X/best.pt image1.png [image2.jpg ...] [--topk 5] [--tau 0.0]`
Prints a table: file, top-k chars with probabilities. Also `--montage out.png` option drawing each input with its
top-3 (Thai font).

## D4 — `tests/test_infer.py` (CPU, < 60 s): use the checkpoint above if present else build an untrained
resnet18 cfg; assert `preprocess_image` handles (a) a white-on-black PIL image, (b) a grey JPEG-like array with
noise, (c) a real glyph from `data/cache/glyphs.npz` → tensor shapes correct, ink dark; `predict_topk` returns k
sorted probs summing ≤ 1; each corruption returns a same-size uint8 canvas with ink preserved (≥ 30 % of input
ink) for its smallest severity.

## Acceptance (reviewer runs)
```bash
uv run pytest -q tests/test_infer.py
uv run python scripts/robustness.py --ckpt runs/A1_resnet18_full_64_T4/best.pt --limit 3 --out reports/robustness/A1_smoke
uv run python scripts/predict.py --ckpt runs/A1_resnet18_full_64_T4/best.pt "ThaiCharacter Dataset/round2/161/bc_001sg_3_118.jpg" "ThaiCharacter Dataset/round2/240/bc_001sg_1_11.jpg" --topk 3
ls reports/robustness/A1_smoke/
```
(`--limit 3` = 3 images per class ≈ 210 images × ~30 corruption levels — must finish in ~1 min on CPU.)

## Report back
Files, commands, test output, the clean vs worst-corruption numbers from the smoke run.
