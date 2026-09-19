#!/usr/bin/env python
"""Evaluate a checkpoint on the validation rows of any split file (raw logits, no TTA) and print the standard metrics.

Used to compare models across label versions, e.g. a v1-trained model on the v2 (label-corrected) validation split:

    uv run python scripts/eval_split.py --ckpt runs/F2_r18_randaug_20/best.pt --split-file data/splits/split_seed42_v2.csv
    uv run python scripts/eval_split.py --ckpt runs/F2_r18_randaug_20_doc/best.pt --split-file data/splits/split_seed42_v2.csv --split-kind doc

The split kind defaults to the one the checkpoint was trained on (a strat model must only see split=="val", a doc model
only doc_split=="val"). minority_acc uses the REAL train counts of the same split file. Optional --out writes a JSON.
"""

from __future__ import annotations

import argparse
import json
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
from thaichar.metrics import compute_metrics  # noqa: E402
from thaichar.models import build_model  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split-file", required=True)
    ap.add_argument("--split-kind", choices=["strat", "doc"], default=None)
    ap.add_argument("--cache", default="data/cache/glyphs.npz")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = pick_device(args.device)
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = merge_cfg(ck["cfg"])
    kind = args.split_kind or cfg["split_kind"]
    if kind != cfg["split_kind"]:
        print(f"WARNING: checkpoint was trained on split_kind={cfg['split_kind']} but evaluating on {kind} "
              f"(partitions overlap -> optimistic)", file=sys.stderr)
    col = {"strat": "split", "doc": "doc_split"}[kind]
    df = pd.read_csv(args.split_file)
    train_df, val_df = df[df[col] == "train"], df[df[col] == "val"].reset_index(drop=True)
    counts_real = np.bincount(train_df.label.values, minlength=len(CLASS_CODES))

    in_ch = 3 if cfg["channel_mode"] in ("gray3", "onoff") else 1
    model = build_model(cfg["model"], num_classes=len(CLASS_CODES), pretrained=False, in_chans=in_ch,
                        img_size=int(cfg["img_size"]), mode="full", geometry=bool(cfg["geometry"]), drop_rate=0.0).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    ds = ThaiGlyphDataset(val_df, load_cache(args.cache), size=cfg["img_size"], channel_mode=cfg["channel_mode"],
                          transform=None, margin=cfg["margin"])
    logits, y = predict(model, DataLoader(ds, batch_size=256, shuffle=False, num_workers=0), device)
    m = compute_metrics(y, logits, counts_real)
    res = {"ckpt": args.ckpt, "split_file": args.split_file, "split_kind": kind, "n_val": int(len(val_df)),
           **{k: float(m[k]) for k in ("top1", "top5", "balanced_acc", "macro_f1", "minority_acc")}}
    print(json.dumps(res, ensure_ascii=False))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump({**res, "per_class_recall": m["per_class_recall"], "confusion": m["confusion"]},
                  open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
