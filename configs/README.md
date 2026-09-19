# configs

- `base.yaml` — every key with defaults and comments (mirrors `thaichar/engine.py::DEFAULTS`).
- `smoke.yaml` — 1-epoch CPU smoke test (resnet18 @32 px, 3% of train).
- `matrix/A_*.yaml` — Experiment A (transfer-learning ranking at 64 px, 25% of train, 6 epochs, aug=base).

Run: `uv run python scripts/train.py --config configs/matrix/A1_resnet18_full_64.yaml [--set epochs=3]`
Aggregate: `uv run python scripts/collect_results.py` → `reports/experiments.{csv,md}`

Legal input sizes (timm 1.0.29, verified): `swin_tiny_patch4_window7_224` → 224 only;
`vit_tiny_patch16_224` / `vit_small_patch16_224` / `deit_small_patch16_224` → 32, 64, 96, 112, 128, 160, 192, 224
(multiples of 16, but 56 fails). CNNs (resnet, mobilenetv3, efficientnet, convnext) accept any size ≥ 32.
