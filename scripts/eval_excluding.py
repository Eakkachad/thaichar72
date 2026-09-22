#!/usr/bin/env python
"""Score checkpoints on a split's val rows while EXCLUDING a set of paths.

Relabelling makes the usual comparison dishonest: a model trained on the corrected labels is
scored against those same corrections, so it wins on the corrected images by construction.
Dropping those images from the metric leaves only the question worth asking -- did cleaner
training labels make the model better on everything ELSE?

    uv run --no-sync python scripts/eval_excluding.py \
        --split-file data/splits/split_seed42_v3.csv \
        --exclude data/splits/split_seed42_v3_changes.csv \
        --ckpt runs/G4_r18_g2init_randaug_20/best.pt runs/G8_r18_v3labels_randaug_20/best.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from thaichar.classes import CLASS_CODES  # noqa: E402
from thaichar.data import ThaiGlyphDataset, load_cache  # noqa: E402
from thaichar.engine import merge_cfg, pick_device, predict  # noqa: E402
from thaichar.models import build_model  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--split-file", required=True)
    ap.add_argument("--split-kind", choices=["strat", "doc"], default="strat")
    ap.add_argument("--exclude", default=None, help="CSV with a `path` column to leave out")
    ap.add_argument("--cache", default="data/cache/glyphs.npz")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = pick_device(args.device)
    col = {"strat": "split", "doc": "doc_split"}[args.split_kind]
    df = pd.read_csv(args.split_file)
    val = df[df[col] == "val"].reset_index(drop=True)
    n_all = len(val)

    excl: set[str] = set()
    if args.exclude:
        excl = set(pd.read_csv(args.exclude)["path"].astype(str))
        val = val[~val.path.isin(excl)].reset_index(drop=True)
    print(f"{args.split_file}  [{args.split_kind}]  val {n_all:,} rows"
          + (f", {n_all - len(val)} relabelled rows excluded -> {len(val):,} scored" if excl else ""))

    cache = load_cache(args.cache)
    counts = np.bincount(df[df[col] == "train"].label.values, minlength=len(CLASS_CODES))
    minority = counts < 50

    print(f"\n{'checkpoint':<46} {'top1':>8} {'balanced':>9} {'minority':>9}")
    for ck in args.ckpt:
        c = torch.load(ck, map_location=device, weights_only=False)
        cfg = merge_cfg(c["cfg"])
        in_ch = 3 if cfg["channel_mode"] in ("gray3", "onoff") else 1
        model = build_model(cfg["model"], num_classes=len(CLASS_CODES), pretrained=False,
                            in_chans=in_ch, img_size=int(cfg["img_size"]), mode="full",
                            geometry=bool(cfg["geometry"]), drop_rate=0.0).to(device)
        model.load_state_dict(c["state_dict"])
        model.eval()
        ds = ThaiGlyphDataset(val, cache, size=cfg["img_size"], channel_mode=cfg["channel_mode"],
                              transform=None, margin=cfg["margin"])
        logits, y = predict(model, DataLoader(ds, batch_size=256, shuffle=False, num_workers=0), device)
        pred = np.asarray(logits).argmax(1)
        y = np.asarray(y)
        ok = pred == y
        per = np.array([ok[y == c_].mean() if (y == c_).any() else np.nan
                        for c_ in range(len(CLASS_CODES))])
        mino = per[minority & ~np.isnan(per)]
        print(f"{Path(ck).parent.name:<46} {ok.mean():8.4f} {np.nanmean(per):9.4f} "
              f"{(mino.mean() if len(mino) else float('nan')):9.4f}")


if __name__ == "__main__":
    main()
