#!/usr/bin/env python
"""Evaluate a checkpoint on the held-out dataUpdate tail probe.

The 35 classes dataUpdate covers hold ~3.4 % of the real corpus but 49 % of the class
list, and their real validation sets are 1-31 images (two classes have none). The probe
is the only way to say anything about whether those classes learned. It is SYNTHETIC, so
it is a domain proxy: report it, never select on it.

The probe is group-disjoint from training (see scripts/make_dataupdate_probe.py), but its
two halves do not carry the same weight and are reported separately:

  hw -- 171 writers/pages held out whole. Clean: no part of these hands is trained on,
        and no filename overlaps round2. This is the number to quote.
  ft -- 5 font families held out whole within dataUpdate, but those typefaces are still
        in the 203-font stage-1 render pool, so this measures robustness to unseen
        degradation of a seen typeface, not unseen-typeface generalisation.

    uv run --no-sync python scripts/eval_probe.py --ckpt runs/G4_.../best.pt
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
from thaichar.classes import CLASS_CODES, code_to_char  # noqa: E402
from thaichar.data import ThaiGlyphDataset, load_cache  # noqa: E402
from thaichar.engine import merge_cfg, pick_device, predict  # noqa: E402
from thaichar.models import build_model  # noqa: E402


def block(y: np.ndarray, pred: np.ndarray) -> dict:
    if len(y) == 0:
        return {"n": 0}
    correct = pred == y
    per_class = {}
    for c in np.unique(y):
        m = y == c
        per_class[int(c)] = float(correct[m].mean())
    return {
        "n": int(len(y)),
        "n_classes": int(len(per_class)),
        "top1": float(correct.mean()),
        "balanced_acc": float(np.mean(list(per_class.values()))),
        "worst_class_recall": float(min(per_class.values())),
        "per_class_recall": per_class,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--index", default="data/dataupdate/index_probe_dj.csv")
    ap.add_argument("--cache", default="data/dataupdate/glyphs_probe_dj.npz")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default=None)
    ap.add_argument("--top-confusions", type=int, default=10)
    args = ap.parse_args()

    device = pick_device(args.device)
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = merge_cfg(ck["cfg"])

    df = pd.read_csv(args.index, low_memory=False).reset_index(drop=True)
    in_ch = 3 if cfg["channel_mode"] in ("gray3", "onoff") else 1
    model = build_model(cfg["model"], num_classes=len(CLASS_CODES), pretrained=False, in_chans=in_ch,
                        img_size=int(cfg["img_size"]), mode="full", geometry=bool(cfg["geometry"]),
                        drop_rate=0.0).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()

    ds = ThaiGlyphDataset(df, load_cache(args.cache), size=cfg["img_size"],
                          channel_mode=cfg["channel_mode"], transform=None, margin=cfg["margin"])
    logits, y = predict(model, DataLoader(ds, batch_size=256, shuffle=False, num_workers=0), device)
    pred = np.asarray(logits).argmax(1)
    y = np.asarray(y)

    res = {
        "ckpt": args.ckpt,
        "index": args.index,
        "overall": block(y, pred),
        "hw_unseen_writers": block(y[(df.source == "hw").values], pred[(df.source == "hw").values]),
        "ft_heldout_families": block(y[(df.source == "ft").values], pred[(df.source == "ft").values]),
    }

    hw = res["hw_unseen_writers"]
    print(f"{args.ckpt}")
    for name in ("overall", "hw_unseen_writers", "ft_heldout_families"):
        b = res[name]
        if b["n"]:
            print(f"  {name:22s} n={b['n']:6d} cls={b['n_classes']:3d} "
                  f"top1={b['top1']:.4f} bal={b['balanced_acc']:.4f} worst={b['worst_class_recall']:.4f}")

    if hw["n"]:
        pc = sorted(hw["per_class_recall"].items(), key=lambda kv: kv[1])
        print("\n  weakest tail classes (hw, unseen writers):")
        for lab, r in pc[:12]:
            print(f"    {code_to_char(CLASS_CODES[lab])} (label {lab}): {r:.3f}")

    # what the tail is mistaken FOR -- drives the label-noise / specialist-head work
    wrong = pred != y
    if wrong.any():
        pairs = pd.Series([f"{code_to_char(CLASS_CODES[a])}->{code_to_char(CLASS_CODES[b])}"
                           for a, b in zip(y[wrong], pred[wrong])]).value_counts()
        print(f"\n  top confusions: {pairs.head(args.top_confusions).to_dict()}")
        res["top_confusions"] = pairs.head(args.top_confusions).to_dict()

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(res, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"\n  -> {args.out}")


if __name__ == "__main__":
    main()
