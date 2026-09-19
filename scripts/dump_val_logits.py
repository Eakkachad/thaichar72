#!/usr/bin/env python
"""Recompute and save val_logits.npy / val_labels.npy for finished runs (e.g. runs whose Colab session died
before the logits were downloaded).   uv run python scripts/dump_val_logits.py --runs A1_resnet18_full_64_T4 ...
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from thaichar.engine import build_datasets, merge_cfg, pick_device, predict  # noqa: E402
from thaichar.infer import load_checkpoint  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--root", default="runs")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    device = pick_device("auto")
    for r in args.runs:
        d = Path(args.root) / r
        if (d / "val_logits.npy").exists() and not args.force:
            print("skip", r)
            continue
        t0 = time.time()
        model, cfg = load_checkpoint(str(d / "best.pt"), device=str(device))
        cfg = merge_cfg(cfg)
        _, val_ds, _, _ = build_datasets(cfg)
        loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=2)
        model = model.to(device).to(memory_format=torch.channels_last)
        logits, y = predict(model, loader, device)
        np.save(d / "val_logits.npy", logits.astype(np.float16))
        np.save(d / "val_labels.npy", y)
        acc = float((logits.argmax(1) == y).mean())
        print(f"{r}: top1 {acc:.4f}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
