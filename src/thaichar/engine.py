"""Training / evaluation engine for the 72-class Thai glyph task.

`train_one(cfg)` runs one experiment end to end: data → model → optimiser/schedule →
epoch loop with per-epoch validation → best-checkpoint reload → final metrics
(tau sweep, optional TTA, bs=1 CPU latency) → `runs/<exp_id>/{log.csv, metrics.json, best.pt, last.pt}`.

Runs unchanged on CPU (this laptop) and on a Colab GPU (AMP only on cuda).
"""

from __future__ import annotations

import copy
import json
import math
import os
import platform
import random
import time
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from thaichar.classes import CLASS_CODES
from thaichar.data import ThaiGlyphDataset, load_cache, merge_sources
from thaichar.losses import build_loss, logit_adjust
from thaichar.metrics import compute_metrics, tau_sweep
from thaichar.models import build_model, count_params, param_groups
from thaichar.samplers import build_sampler

NUM_CLASSES = len(CLASS_CODES)

# ---------------------------------------------------------------------------
# Defaults (documented in configs/base.yaml)
# ---------------------------------------------------------------------------
DEFAULTS: dict[str, Any] = {
    "exp_id": None,
    "out_root": "runs",
    "seed": 42,
    "device": "auto",
    "threads": 4,
    "num_workers": 0 if os.name == "nt" else 2,  # spawn-based workers are slower than in-process on Windows
    "channels_last": True,
    # data
    "split_file": "data/splits/split_seed42.csv",
    "split_kind": "strat",  # strat | doc
    "cache": "data/cache/glyphs.npz",
    "subset_frac": 1.0,
    "img_size": 64,
    "channel_mode": "gray3",  # gray3 | gray1 | onoff
    "margin": 0.1,
    "aug": "none",  # none | base | morph | full | randaug | trivial
    "extra_train_index": None,  # e.g. data/synth/index.csv
    "extra_train_cache": None,  # e.g. data/synth/glyphs_synth.npz
    "extra_mode": "all",  # all | fill
    "extra_fill_to": 200,
    "extra_max_per_class": None,
    "use_real_train": True,  # False → train ONLY on extra_train_index (intermediate pretraining stage)
    "init_from": None,  # path to a previous run's best.pt; loads all shape-compatible tensors (backbone transfer)
    "kd_teachers": None,  # list of best.pt paths; soft-target knowledge distillation from their averaged softmax
    "kd_alpha": 0.7,  # weight of the KD term (rest = hard-label loss)
    "kd_T": 4.0,  # distillation temperature
    # model
    "model": "resnet18",
    "pretrained": True,
    "mode": "full",  # full | partial | frozen
    "partial_frac": 0.35,
    "geometry": False,
    "drop_rate": 0.1,
    # optimisation
    "epochs": 6,
    "batch_size": 128,
    "lr": 1e-3,
    "weight_decay": 0.05,
    "warmup_epochs": 1.0,
    "min_lr_ratio": 0.01,
    "llrd": None,
    "grad_clip": 1.0,
    "ema": True,
    "ema_decay": 0.999,
    "ema_eval": "last",  # last | every  (EMA validation each epoch doubles val cost)
    "amp": True,
    # loss / imbalance
    "loss": "ce",
    "label_smoothing": 0.1,
    "focal_gamma": 2.0,
    "ce_weight_power": 0.5,
    "cb_beta": 0.999,
    "sampler": "none",  # none | sqrt_inv | inv
    "sampler_cap": 10.0,
    "mixup": 0.0,
    "cutmix": 0.0,
    "mix_prob": 1.0,
    # evaluation
    "select_metric": "balanced_acc",
    "val_subset_frac": 1.0,  # per-epoch monitoring subset; final metrics always use the full val split
    "taus": [0.0, 0.25, 0.5, 0.75],
    "tta": False,
    "latency_iters": 50,
    "save_last": True,
}


def merge_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(DEFAULTS)
    out.update({k: v for k, v in cfg.items()})
    return out


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _subset_per_class(df: pd.DataFrame, frac: float, seed: int) -> pd.DataFrame:
    if frac >= 1.0:
        return df
    rng = np.random.default_rng(seed)
    parts = []
    for _, grp in df.groupby("label", sort=True):
        n = max(1, int(round(len(grp) * frac)))
        idx = rng.choice(len(grp), size=n, replace=False)
        parts.append(grp.iloc[np.sort(idx)])
    return pd.concat(parts, ignore_index=True)


def _select_extra(extra_df: pd.DataFrame, train_counts: np.ndarray, cfg: dict, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 1)
    parts = []
    for label, grp in extra_df.groupby("label", sort=True):
        label = int(label)
        if cfg["extra_mode"] == "fill":
            need = max(0, int(cfg["extra_fill_to"]) - int(train_counts[label]))
        else:
            need = len(grp)
        if cfg["extra_max_per_class"] is not None:
            need = min(need, int(cfg["extra_max_per_class"]))
        need = min(need, len(grp))
        if need <= 0:
            continue
        idx = rng.choice(len(grp), size=need, replace=False)
        parts.append(grp.iloc[np.sort(idx)])
    if not parts:
        return extra_df.iloc[0:0]
    return pd.concat(parts, ignore_index=True)


def build_datasets(cfg: dict[str, Any]) -> tuple[ThaiGlyphDataset, ThaiGlyphDataset, np.ndarray, dict]:
    """Return (train_ds, val_ds, class_counts_train[72], info)."""
    df = pd.read_csv(cfg["split_file"])
    col = {"strat": "split", "doc": "doc_split"}[cfg["split_kind"]]
    train_df = df[df[col] == "train"].reset_index(drop=True)
    val_df = df[df[col] == "val"].reset_index(drop=True)
    assert not set(train_df.path) & set(val_df.path), "train/val overlap!"
    train_df = _subset_per_class(train_df, float(cfg["subset_frac"]), int(cfg["seed"]))
    n_real = len(train_df)

    cache = load_cache(cfg["cache"])
    counts_real = np.bincount(train_df.label.values, minlength=NUM_CLASSES)

    transform = None
    if cfg["aug"] != "none":
        from thaichar.augment import get_transform  # lazy: written by another task

        transform = get_transform(cfg["aug"], seed=int(cfg["seed"]))

    n_extra = 0
    if cfg["extra_train_index"]:
        extra_df = pd.read_csv(cfg["extra_train_index"])
        extra_cache = load_cache(cfg["extra_train_cache"])
        extra_df = _select_extra(extra_df, counts_real, cfg, int(cfg["seed"]))
        n_extra = len(extra_df)
        keep = ["path", "label", "width", "height", "ink_frac"]
        sources = [(train_df[keep], cache), (extra_df[keep], extra_cache)] if cfg["use_real_train"] else [(extra_df[keep], extra_cache)]
        merged_df, multi = merge_sources(sources)
        train_ds = ThaiGlyphDataset(merged_df, multi, size=cfg["img_size"], channel_mode=cfg["channel_mode"],
                                    transform=transform, margin=cfg["margin"])
        class_counts = np.bincount(merged_df.label.values, minlength=NUM_CLASSES)
    else:
        train_ds = ThaiGlyphDataset(train_df, cache, size=cfg["img_size"], channel_mode=cfg["channel_mode"],
                                    transform=transform, margin=cfg["margin"])
        class_counts = counts_real
    val_ds = ThaiGlyphDataset(val_df, cache, size=cfg["img_size"], channel_mode=cfg["channel_mode"],
                              transform=None, margin=cfg["margin"])
    info = {"n_train_real": int(n_real), "n_train_extra": int(n_extra), "n_train": len(train_ds),
            "n_val": len(val_ds), "val_classes_present": int(val_df.label.nunique()),
            "class_counts_real": counts_real.tolist()}
    return train_ds, val_ds, class_counts, info


def _worker_init(worker_id: int) -> None:
    info = torch.utils.data.get_worker_info()
    if info is None:
        return
    ds = info.dataset
    t = getattr(ds, "transform", None)
    if t is not None and hasattr(t, "reseed"):
        t.reseed(int(info.seed) % (2**32))
    np.random.seed(int(info.seed) % (2**32))


def make_loaders(train_ds, val_ds, cfg) -> tuple[DataLoader, DataLoader]:
    labels = train_ds.df.label.values
    sampler = build_sampler(labels, kind=cfg["sampler"], cap=float(cfg["sampler_cap"]))
    if sampler is not None:
        # make the sampler deterministic per run
        sampler.generator = torch.Generator().manual_seed(int(cfg["seed"]))
    g = torch.Generator().manual_seed(int(cfg["seed"]))
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=(sampler is None), sampler=sampler,
                              num_workers=cfg["num_workers"], drop_last=False, generator=g,
                              worker_init_fn=_worker_init, persistent_workers=cfg["num_workers"] > 0)
    val_loader = DataLoader(val_ds, batch_size=max(cfg["batch_size"], 256), shuffle=False,
                            num_workers=cfg["num_workers"])
    return train_loader, val_loader


def make_val_monitor_loader(val_ds: ThaiGlyphDataset, cfg: dict) -> DataLoader:
    """Stratified per-class subset of val for cheap per-epoch monitoring."""
    frac = float(cfg.get("val_subset_frac", 1.0))
    if frac >= 1.0:
        return DataLoader(val_ds, batch_size=max(cfg["batch_size"], 256), shuffle=False, num_workers=cfg["num_workers"])
    sub_df = _subset_per_class(val_ds.df, frac, int(cfg["seed"]) + 7)
    ds = ThaiGlyphDataset(sub_df, val_ds.cache, size=cfg["img_size"], channel_mode=cfg["channel_mode"],
                          transform=None, margin=cfg["margin"])
    return DataLoader(ds, batch_size=max(cfg["batch_size"], 256), shuffle=False, num_workers=cfg["num_workers"])


def cosine_warmup_lambda(total_steps: int, warmup_steps: int, min_ratio: float) -> Callable[[int], float]:
    def f(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return (step + 1) / warmup_steps
        t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * min(1.0, t)))

    return f


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits_all, y_all = [], []
    for batch in loader:
        x, y, g = batch[0], batch[1], (batch[2] if len(batch) > 2 else None)
        x = x.to(device, non_blocking=True)
        if x.dim() == 4:
            x = x.contiguous(memory_format=torch.channels_last)
        g = g.to(device) if g is not None else None
        out = model(x, g)
        logits_all.append(out.float().cpu().numpy())
        y_all.append(y.numpy())
    return np.concatenate(logits_all), np.concatenate(y_all)


# ---------------------------------------------------------------------------
# TTA (canvas-level transforms; applied before the resize inside the Dataset)
# ---------------------------------------------------------------------------
def _shift(dx: int, dy: int) -> Callable[[np.ndarray], np.ndarray]:
    def f(c: np.ndarray) -> np.ndarray:
        out = np.full_like(c, 255)
        h, w = c.shape
        ys, yd = (slice(0, h - dy), slice(dy, h)) if dy >= 0 else (slice(-dy, h), slice(0, h + dy))
        xs, xd = (slice(0, w - dx), slice(dx, w)) if dx >= 0 else (slice(-dx, w), slice(0, w + dx))
        out[yd, xd] = c[ys, xs]
        return out

    return f


def _morph(kind: str) -> Callable[[np.ndarray], np.ndarray]:
    k = np.ones((2, 2), np.uint8)

    def f(c: np.ndarray) -> np.ndarray:
        # ink is 0 → thickening ink = erode on the grey image
        return cv2.erode(c, k) if kind == "dilate_ink" else cv2.dilate(c, k)

    return f


TTA_VIEWS: list[tuple[str, Callable | None, float | None]] = [
    ("identity", None, None),
    ("shift+1x", _shift(1, 0), None),
    ("shift-1x", _shift(-1, 0), None),
    ("shift+1y", _shift(0, 1), None),
    ("thicker", _morph("dilate_ink"), None),
    ("thinner", _morph("erode_ink"), None),
    ("margin0.05", None, 0.05),
    ("margin0.15", None, 0.15),
]


@torch.no_grad()
def predict_tta(model: nn.Module, val_ds: ThaiGlyphDataset, cfg: dict, device: torch.device) -> np.ndarray:
    probs = None
    for _, tf, margin in TTA_VIEWS:
        ds = ThaiGlyphDataset(val_ds.df, val_ds.cache, size=cfg["img_size"], channel_mode=cfg["channel_mode"],
                              transform=tf, margin=(margin if margin is not None else cfg["margin"]))
        loader = DataLoader(ds, batch_size=256, shuffle=False, num_workers=0)  # closures not picklable under spawn
        logits, _ = predict(model, loader, device)
        p = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
        probs = p if probs is None else probs + p
    return np.log(probs / len(TTA_VIEWS) + 1e-9)  # log-probs behave like logits for argmax / tau


def load_compatible(model: nn.Module, ckpt_path: str, device: torch.device) -> tuple[int, int]:
    """Load every tensor whose name and shape match (head is skipped when its shape differs)."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    own = model.state_dict()
    ok = {k: v for k, v in sd.items() if k in own and own[k].shape == v.shape}
    model.load_state_dict(ok, strict=False)
    return len(ok), len(sd) - len(ok)


# ---------------------------------------------------------------------------
# latency
# ---------------------------------------------------------------------------
@torch.no_grad()
def measure_latency_ms(model: nn.Module, cfg: dict, iters: int = 50) -> float:
    m = copy.deepcopy(model).cpu().eval()
    in_ch = 3 if cfg["channel_mode"] in ("gray3", "onoff") else 1
    x = torch.randn(1, in_ch, cfg["img_size"], cfg["img_size"])
    g = torch.zeros(1, 4) if cfg["geometry"] else None
    torch.set_num_threads(1)
    for _ in range(10):
        m(x, g)
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        m(x, g)
        ts.append((time.perf_counter() - t0) * 1000)
    torch.set_num_threads(int(cfg["threads"]))
    return float(np.median(ts))


# ---------------------------------------------------------------------------
# main entry
# ---------------------------------------------------------------------------
def train_one(cfg_in: dict[str, Any]) -> dict[str, Any]:
    cfg = merge_cfg(cfg_in)
    if not cfg["exp_id"]:
        cfg["exp_id"] = f"{cfg['model']}_{cfg['img_size']}_{cfg['mode']}_s{cfg['seed']}"
    out_dir = Path(cfg["out_root"]) / cfg["exp_id"]
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(int(cfg["seed"]))
    device = pick_device(cfg["device"])
    if device.type == "cpu":
        torch.set_num_threads(int(cfg["threads"]))
    use_amp = bool(cfg["amp"]) and device.type == "cuda"
    if device.type != "cuda":
        print(f"[{cfg['exp_id']}] WARNING: training on CPU (device={cfg['device']}, cuda_available={torch.cuda.is_available()}, "
              f"torch={torch.__version__}). Pass --set device=cuda to fail fast instead.", flush=True)
    t_start = time.time()

    # ---- data
    train_ds, val_ds, class_counts, info = build_datasets(cfg)
    counts_real = np.array(info["class_counts_real"])  # minority_acc is defined on REAL train counts (n<50)
    train_loader, val_loader = make_loaders(train_ds, val_ds, cfg)
    log_prior = np.log(np.clip(class_counts, 1, None) / class_counts.sum()).astype(np.float32)

    # ---- model
    in_ch = 3 if cfg["channel_mode"] in ("gray3", "onoff") else 1
    model = build_model(cfg["model"], num_classes=NUM_CLASSES, pretrained=bool(cfg["pretrained"]), in_chans=in_ch,
                        img_size=int(cfg["img_size"]), mode=cfg["mode"], geometry=bool(cfg["geometry"]),
                        drop_rate=float(cfg["drop_rate"]), partial_frac=float(cfg["partial_frac"])).to(device)
    if cfg["init_from"]:
        n_loaded, n_skipped = load_compatible(model, cfg["init_from"], device)
        print(f"[{cfg['exp_id']}] init_from {cfg['init_from']}: loaded {n_loaded} tensors, skipped {n_skipped}")
    if cfg["channels_last"]:
        model = model.to(memory_format=torch.channels_last)
    n_total, n_trainable = count_params(model)
    val_monitor = make_val_monitor_loader(val_ds, cfg)

    # ---- loss / mix
    mix = None
    if float(cfg["mixup"]) > 0 or float(cfg["cutmix"]) > 0:
        if cfg["loss"] != "ce":
            raise ValueError("mixup/cutmix only supported with loss=ce")
        from timm.data import Mixup
        from timm.loss import SoftTargetCrossEntropy

        mix = Mixup(mixup_alpha=float(cfg["mixup"]), cutmix_alpha=float(cfg["cutmix"]), prob=float(cfg["mix_prob"]),
                    label_smoothing=float(cfg["label_smoothing"]), num_classes=NUM_CLASSES)
        criterion: nn.Module = SoftTargetCrossEntropy()
    else:
        criterion = build_loss(cfg, torch.from_numpy(class_counts))
    criterion = criterion.to(device)
    teachers = []
    if cfg["kd_teachers"]:
        for tp in cfg["kd_teachers"]:
            tck = torch.load(tp, map_location=device, weights_only=False)
            tcfg = merge_cfg(tck["cfg"])
            t_in = 3 if tcfg["channel_mode"] in ("gray3", "onoff") else 1
            tm = build_model(tcfg["model"], num_classes=NUM_CLASSES, pretrained=False, in_chans=t_in,
                             img_size=int(tcfg["img_size"]), mode="full", geometry=bool(tcfg["geometry"]),
                             drop_rate=0.0).to(device)
            tm.load_state_dict(tck["state_dict"]); tm.eval()
            for prm in tm.parameters():
                prm.requires_grad = False
            assert tcfg["img_size"] == cfg["img_size"] and tcfg["channel_mode"] == cfg["channel_mode"], \
                "KD teachers must share img_size and channel_mode with the student (same input tensors)"
            teachers.append(tm)
        print(f"[{cfg['exp_id']}] KD from {len(teachers)} teacher(s), alpha={cfg['kd_alpha']}, T={cfg['kd_T']}")

    # ---- optimiser / schedule
    groups = param_groups(model, lr=float(cfg["lr"]), weight_decay=float(cfg["weight_decay"]), llrd=cfg["llrd"])
    optimizer = torch.optim.AdamW(groups)
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * int(cfg["epochs"])
    warmup_steps = int(float(cfg["warmup_epochs"]) * steps_per_epoch)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, cosine_warmup_lambda(total_steps, warmup_steps, float(cfg["min_lr_ratio"])))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    # use_warmup: effective decay ramps up with step count, so short runs are not stuck at the init weights
    ema = timm.utils.ModelEmaV3(model, decay=float(cfg["ema_decay"]), use_warmup=True) if cfg["ema"] else None

    # ---- loop
    log_rows: list[dict] = []
    best = {"metric": -1.0, "epoch": -1, "which": "raw"}
    sel = cfg["select_metric"]
    global_step = 0
    for epoch in range(int(cfg["epochs"])):
        model.train()
        t0 = time.time()
        loss_sum, correct, seen = 0.0, 0, 0
        for batch in train_loader:
            x, y, g = batch[0].to(device), batch[1].to(device), (batch[2].to(device) if len(batch) > 2 else None)
            if cfg["channels_last"]:
                x = x.contiguous(memory_format=torch.channels_last)
            if mix is not None:
                if x.size(0) % 2 == 1:  # timm Mixup needs even batch
                    x, y = x[:-1], y[:-1]
                    g = g[:-1] if g is not None else None
                x, y_soft = mix(x, y)
            with torch.autocast(device_type="cuda", enabled=use_amp):
                out = model(x, g)
                loss = criterion(out, y_soft if mix is not None else y)
                if teachers:
                    with torch.no_grad():
                        t_prob = torch.stack([torch.softmax(t(x, g).float() / float(cfg["kd_T"]), dim=1)
                                              for t in teachers]).mean(0)
                    log_s = torch.log_softmax(out.float() / float(cfg["kd_T"]), dim=1)
                    kd = torch.nn.functional.kl_div(log_s, t_prob, reduction="batchmean") * float(cfg["kd_T"]) ** 2
                    loss = (1 - float(cfg["kd_alpha"])) * loss + float(cfg["kd_alpha"]) * kd
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if cfg["grad_clip"]:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["grad_clip"]))
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            if ema is not None:
                ema.update(model, step=global_step)
            global_step += 1
            loss_sum += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            seen += x.size(0)
        train_time = time.time() - t0

        # ---- validate raw (+ EMA)
        last_epoch = epoch + 1 == int(cfg["epochs"])
        logits, y_val = predict(model, val_monitor, device)
        m_raw = compute_metrics(y_val, logits, counts_real)
        row = {"epoch": epoch + 1, "lr": optimizer.param_groups[0]["lr"], "train_loss": loss_sum / max(1, seen),
               "train_acc": correct / max(1, seen), "val_top1": m_raw["top1"], "val_bal_acc": m_raw["balanced_acc"],
               "val_macro_f1": m_raw["macro_f1"], "val_minority_acc": m_raw["minority_acc"]}
        cand = [("raw", m_raw, model)]
        if ema is not None and (cfg["ema_eval"] == "every" or last_epoch):
            logits_e, _ = predict(ema.module, val_monitor, device)
            m_ema = compute_metrics(y_val, logits_e, counts_real)
            row.update({"ema_val_top1": m_ema["top1"], "ema_val_bal_acc": m_ema["balanced_acc"]})
            cand.append(("ema", m_ema, ema.module))
        row["epoch_seconds"] = time.time() - t0
        row["train_seconds"] = train_time
        log_rows.append(row)
        pd.DataFrame(log_rows).to_csv(out_dir / "log.csv", index=False)
        print(f"[{cfg['exp_id']}] ep {epoch+1}/{cfg['epochs']} loss {row['train_loss']:.4f} "
              f"top1 {m_raw['top1']:.4f} bal {m_raw['balanced_acc']:.4f}"
              + (f" | ema top1 {row['ema_val_top1']:.4f} bal {row['ema_val_bal_acc']:.4f}" if "ema_val_top1" in row else "")
              + f" | {row['epoch_seconds']:.0f}s", flush=True)

        for which, m, mod in cand:
            if m[sel] > best["metric"]:
                best = {"metric": float(m[sel]), "epoch": epoch + 1, "which": which}
                torch.save({"state_dict": mod.state_dict(), "cfg": cfg, "class_codes": CLASS_CODES,
                            "epoch": epoch + 1, "which": which, "val_metrics": {k: m[k] for k in
                            ("top1", "top5", "balanced_acc", "macro_f1", "minority_acc")}}, out_dir / "best.pt")
        if cfg["save_last"]:
            torch.save({"state_dict": model.state_dict(), "cfg": cfg, "epoch": epoch + 1}, out_dir / "last.pt")

    # ---- final evaluation on the best checkpoint
    ckpt = torch.load(out_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    logits, y_val = predict(model, val_loader, device)
    final = compute_metrics(y_val, logits, counts_real)
    sweep = tau_sweep(logits, y_val, log_prior, counts_real, taus=tuple(float(t) for t in cfg["taus"]))
    tta_metrics = None
    if cfg["tta"]:
        logits_tta = predict_tta(model, val_ds, cfg, device)
        mt = compute_metrics(y_val, logits_tta, counts_real)
        tta_metrics = {k: mt[k] for k in ("top1", "top5", "balanced_acc", "macro_f1", "minority_acc")}
        tta_metrics["tau_sweep"] = tau_sweep(logits_tta, y_val, log_prior, counts_real,
                                             taus=tuple(float(t) for t in cfg["taus"]))
    latency = measure_latency_ms(model, cfg, iters=int(cfg["latency_iters"]))
    np.save(out_dir / "val_logits.npy", logits.astype(np.float16))
    np.save(out_dir / "val_labels.npy", y_val)

    metrics = {
        "exp_id": cfg["exp_id"], "cfg": cfg, **info,
        "class_counts_train": class_counts.tolist(),
        "epochs_done": int(cfg["epochs"]), "best_epoch": best["epoch"], "best_which": best["which"],
        "select_metric": sel,
        "top1": final["top1"], "top5": final["top5"], "balanced_acc": final["balanced_acc"],
        "macro_f1": final["macro_f1"], "minority_acc": final["minority_acc"],
        "n_classes_present": final["n_classes_present"],
        "per_class_recall": final["per_class_recall"], "confusion": final["confusion"],
        "tau_sweep": sweep, "tta": tta_metrics,
        "params_total": n_total, "params_trainable": n_trainable,
        "latency_ms_bs1": latency,
        "sec_per_epoch": float(np.mean([r["epoch_seconds"] for r in log_rows])) if log_rows else None,
        "total_seconds": time.time() - t_start, "device": str(device),
        "versions": {"torch": torch.__version__, "timm": timm.__version__, "python": platform.python_version()},
        "caveats": ["checkpoint selected by " + sel + " on the same validation split that is reported",
                    "tau_sweep is evaluated on the reported validation split (mildly optimistic)"],
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=1)
    return metrics
