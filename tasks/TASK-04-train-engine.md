# TASK-04 — Training engine (timm transfer learning, imbalance options, metrics, logging)

## Context
Project root = this dir (`Deep_CNN`), `uv`-managed Python 3.12, **CPU only here** (8 cores); the same code
must run unchanged on a Colab T4 GPU (`device="auto"` → cuda if available, AMP only on cuda). Installed:
torch 2.14 (CPU), torchvision, timm 1.0.29, numpy, pandas, pyyaml? (add `uv add pyyaml` if missing), pytest.
Read first: `docs/EXPERIMENT-PLAN.md` §1–2, `src/thaichar/data.py` (`load_cache`, `ThaiGlyphDataset`,
`make_loader`), `src/thaichar/transforms.py`, `src/thaichar/classes.py`, `data/splits/split_seed42.csv`
(columns incl. `path, code, label, split, doc_split, group, width, height, ink_frac`).
**Concurrency note:** another agent is editing `augment.py`, `synth.py`, `transforms.py` and the
`ThaiGlyphDataset` class right now. Do NOT edit those files. Import the augmentation lazily:
`from thaichar.augment import get_transform` inside a try/except; if unavailable or preset == "none", use
`transform=None`. Construct the dataset with keyword args `size=`, `transform=`, and `channel_mode=` if the
class accepts it (check `inspect.signature`), else `channels=3`.

## Scope — you may create/modify ONLY
`src/thaichar/models.py`, `src/thaichar/losses.py`, `src/thaichar/samplers.py`, `src/thaichar/metrics.py`,
`src/thaichar/engine.py`, `scripts/train.py`, `scripts/collect_results.py`, `configs/**`,
`tests/test_engine.py`, `runs/**` (outputs; git-ignored), `pyproject.toml` (deps only).
HARD RULES: dataset dir read-only; never touch `../CNN`, `../katgpt-rs`, `../fly-connectome-template`,
`tasks/`, `weights/`, `notebooks/`, `reports/eda/`, `scripts/eda.py`, `scripts/prep_data.py`. No `git commit`.

## D1 — `models.py`
`build_model(name, num_classes=72, pretrained=True, in_chans=3, img_size=64, mode="full", geometry=False,
drop_rate=0.1, partial_frac=0.35) -> nn.Module`
- `name="smallcnn"`: our own scratch CNN: 4 conv blocks (32-64-128-256, BN, ReLU, maxpool), GAP‖GMP, FC.
- otherwise timm: `timm.create_model(name, pretrained=pretrained, num_classes=0, in_chans=in_chans,
  **({"img_size": img_size} if the model family needs it — ViT/DeiT/Swin — else {}))`. For ViT/Swin with
  non-224 sizes rely on timm's pretrained pos-embed interpolation; for Swin ensure `img_size` is divisible
  by the window×patch requirement (document which sizes are legal, e.g. 96/128/224 for swin_tiny... test
  what works and record it in `configs/README.md`).
- Head: `GlyphHead(feat_dim, num_classes, geometry)`; if `geometry`, a tiny MLP 4→16 (ReLU) is concatenated
  to the pooled features before the final Linear. Forward signature always `model(x, g=None)`.
- `mode`: `frozen` → backbone `requires_grad=False` (train head only; backbone BN in eval mode);
  `partial` → unfreeze the last `partial_frac` fraction of backbone parameter tensors (by order) + head;
  `full` → everything. Provide `count_params(model)` (total, trainable).
- `param_groups(model, lr, weight_decay, llrd=None)`: if `llrd` (e.g. 0.8), layer-wise LR decay across
  backbone parameter-tensor order in ~6 buckets (head gets `lr`, earliest bucket `lr*llrd^6`); no weight
  decay on biases/norm params.

## D2 — `losses.py` and `samplers.py`
Losses (all take `(logits, targets)`): `ce` (with `label_smoothing`), `weighted_ce` (weights ∝ n^-power,
power ∈ {0.5, 1.0}, normalised to mean 1), `focal` (γ), `cb_focal` (class-balanced weights β, γ).
`logit_adjust(logits, log_prior, tau)` → `logits - tau*log_prior` (eval only).
`build_loss(cfg, class_counts)`.
Sampler: `build_sampler(labels, kind="none"|"sqrt_inv"|"inv", cap=10.0)` → `WeightedRandomSampler` with
per-sample weight ∝ n_c^-power capped at `cap × uniform`, `num_samples=len(labels)`.
Batch-level Mixup/CutMix via `timm.data.Mixup(mixup_alpha, cutmix_alpha, prob, label_smoothing, num_classes)`
producing soft targets → use `SoftTargetCrossEntropy` when active (only compatible with loss `ce`; error otherwise).

## D3 — `metrics.py`
`compute_metrics(y_true, y_pred_logits, class_counts_train, minority_thresh=50) -> dict` with: `top1, top5,
balanced_acc (mean per-class recall over classes present), macro_f1, minority_acc (recall on classes with
train n<50), n_classes_present, per_class_recall (list 72, NaN if absent), confusion (72×72 list)`.
Also `tau_sweep(logits, y, log_prior, taus=(0,0.25,0.5,0.75))` → dict tau→metrics summary.

## D4 — `engine.py`
`train_one(cfg: dict) -> dict` (returns final metrics). Behaviour:
- seed everything (`cfg.seed`); device auto; `torch.set_num_threads(cfg.get("threads", 8))` on CPU.
- data: read `cfg.split_file`, `cfg.split_kind ∈ {strat, doc}` selects which column defines train/val
  (train and val ALWAYS from the same partition — never mix); `cfg.subset_frac` (stratified per-class
  subsample of TRAIN only, min 1 per class, for fast ranking runs); optional `cfg.extra_train_index` +
  `cfg.extra_train_cache` (synthetic glyphs) appended to train (if the Dataset/merge helper from TASK-03
  exists; otherwise raise a clear NotImplementedError — do not hack it).
- model per D1; optimizer AdamW; schedule cosine with linear warmup (`warmup_epochs`), stepped per iteration;
  optional EMA (`timm.utils.ModelEmaV3`, decay cfg) — evaluate both raw and EMA, keep the better by
  selection metric; grad clip 1.0; AMP on cuda.
- per epoch: train loss/acc, val metrics (D3) on the val split, epoch seconds, lr → append to
  `runs/<exp_id>/log.csv`; save `best.pt` (state_dict + cfg + class list) by `cfg.select_metric`
  (default `balanced_acc`), and `last.pt`.
- at end: reload best, compute full metrics + `tau_sweep` (log_prior from train class counts) + optional
  TTA (`cfg.tta`: average logits over {identity, shift ±1px, scale 0.95/1.05, dilate, erode} of the canvas;
  implement TTA transforms locally with cv2/PIL on the square image — do not import augment.py for this),
  bs=1 CPU latency (median of 50 after 10 warmup, on a CPU copy of the model), params → `metrics.json`
  (include `cfg`, `n_train`, `n_val`, `epochs_done`, `total_seconds`, `best_epoch`, `device`, `torch/timm
  versions`, and a `caveats` list: e.g. "checkpoint selected on the same val split").
- `scripts/train.py --config configs/x.yaml [--set key=value ...]` (dotted keys allowed, YAML-typed values);
  `exp_id` defaults to the config stem; prints a one-line summary at the end.
- `scripts/collect_results.py` → scans `runs/*/metrics.json` → `reports/experiments.csv` and
  `reports/experiments.md` (columns: exp_id, model, img_size, mode, aug, loss, sampler, split_kind,
  subset_frac, epochs, top1, balanced_acc, macro_f1, minority_acc, tau_best_bal (tau, value), params_M,
  sec_per_epoch, latency_ms, seed).

## D5 — configs
`configs/base.yaml` (all keys with defaults + comments), `configs/smoke.yaml` (resnet18, img_size 32,
subset_frac 0.03, epochs 1, batch 64, split_kind strat, aug none, threads 8),
`configs/matrix/A_*.yaml` for: A0 smallcnn@64, A1 resnet18 frozen/partial/full @64, A2 mobilenetv3_large_100
frozen/partial/full @64, A3 efficientnet_b0 full @64 — all with `subset_frac: 0.25, epochs: 6, aug: base,
loss: ce (ls 0.1), sampler none, ema true`. `configs/README.md` documents keys and legal ViT/Swin sizes.

## D6 — tests (`tests/test_engine.py`, must run < 90 s on CPU)
- `build_model` for `smallcnn`, `resnet18`, `mobilenetv3_large_100` (pretrained=False to avoid downloads
  in tests), `vit_tiny_patch16_224` with `img_size=64` → forward `[2,3,S,S]` (+ g `[2,4]` when geometry)
  gives `[2,72]`; frozen mode → no backbone param has `requires_grad`; partial → some but not all.
- `compute_metrics` on a hand-made 5-class example gives the expected balanced_acc/top1;
  `logit_adjust` with tau=0 is identity; `build_sampler` weights respect the cap.
- `train_one` on the smoke config with `epochs=1, subset_frac=0.01, pretrained=False, img_size=32` runs end
  to end and writes `log.csv`, `metrics.json`, `best.pt`.

## Acceptance (reviewer runs)
```bash
uv run pytest -q tests/test_engine.py
uv run python scripts/train.py --config configs/smoke.yaml            # < 4 min on CPU, downloads resnet18 weights
uv run python scripts/collect_results.py && cat reports/experiments.md
uv run python -c "import json; m=json.load(open('runs/smoke/metrics.json')); print({k:m[k] for k in ['top1','balanced_acc','macro_f1','minority_acc','n_classes_present','params_total','latency_ms_bs1']}); print(m['tau_sweep'].keys())"
```
## Report back
Files, commands, test output, smoke run summary (time/epoch, metrics), which ViT/Swin sizes were legal, anything surprising.
