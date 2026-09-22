#!/usr/bin/env python
"""Assemble out-of-fold predictions and rank the corpus by how likely each label is wrong.

Every image is scored by a model that never trained on it (scripts/make_folds.py + one run per
fold), so a confident disagreement is evidence about the LABEL, not about memorisation.

Nothing here moves a file. It writes a ranked candidate list and montages; the project rule is
that a human confirms by eye before any label changes (CLAUDE.md, and the swap_verdict.csv pass
that found 6 confusable pairs still contaminated).

    uv run --no-sync python scripts/find_label_noise.py --k 5
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from thaichar.classes import CLASS_CODES, code_to_char  # noqa: E402
from thaichar.data import load_cache  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--runs-root", default="runs")
    ap.add_argument("--exp-prefix", default="OOF_f")
    ap.add_argument("--out", default="reports/analysis/label_noise_candidates.csv")
    ap.add_argument("--montage-dir", default="reports/analysis/label_noise")
    ap.add_argument("--min-conf", type=float, default=0.70,
                    help="only call it a candidate when the out-of-fold model is at least this sure")
    args = ap.parse_args()

    frames = []
    for f in range(args.k):
        sf = f"data/splits/fold{f}_of{args.k}.csv"
        rd = os.path.join(args.runs_root, f"{args.exp_prefix}{f}")
        lg, lb = os.path.join(rd, "val_logits.npy"), os.path.join(rd, "val_labels.npy")
        if not (os.path.exists(lg) and os.path.exists(lb)):
            raise SystemExit(f"missing out-of-fold predictions for fold {f}: {lg}")
        df = pd.read_csv(sf)
        val = df[df.split == "val"].reset_index(drop=True)
        logits = np.load(lg)
        labels = np.load(lb)
        if len(val) != len(logits):
            raise SystemExit(f"fold {f}: {len(val)} val rows but {len(logits)} logits")
        if not np.array_equal(val.label.to_numpy(), labels):
            raise SystemExit(f"fold {f}: label order does not match the split file")
        e = np.exp(logits - logits.max(1, keepdims=True))
        p = e / e.sum(1, keepdims=True)
        order = np.argsort(-p, axis=1)
        val = val.assign(
            fold=f,
            pred=order[:, 0],
            p_pred=p[np.arange(len(p)), order[:, 0]],
            p_true=p[np.arange(len(p)), labels],
            runner_up=order[:, 1],
        )
        frames.append(val)

    d = pd.concat(frames, ignore_index=True)
    d["true_char"] = d.label.map(lambda i: code_to_char(CLASS_CODES[i]))
    d["pred_char"] = d.pred.map(lambda i: code_to_char(CLASS_CODES[i]))
    d["disagree"] = d.pred != d.label
    # how much more the model believes its own answer than the filed label
    d["margin"] = d.p_pred - d.p_true

    print(f"out-of-fold coverage : {len(d):,} images (corpus is 59,942)")
    print(f"out-of-fold accuracy : {1 - d.disagree.mean():.4f}")
    print(f"disagreements        : {int(d.disagree.sum()):,}")

    cand = d[d.disagree & (d.p_pred >= args.min_conf)].sort_values("margin", ascending=False)
    print(f"candidates (model >= {args.min_conf:.2f} sure and disagreeing): {len(cand):,} "
          f"= {100*len(cand)/len(d):.2f}% of the corpus")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    cols = ["path", "label", "true_char", "pred", "pred_char", "p_true", "p_pred", "margin", "fold"]
    cand[cols].to_csv(args.out, index=False)
    print(f"-> {args.out}")

    pair = (cand.groupby(["true_char", "pred_char"]).size().reset_index(name="n")
                .sort_values("n", ascending=False))
    print("\ntop suspected label swaps (filed as -> model says):")
    print(pair.head(20).to_string(index=False))

    # montages, one per suspected pair, so a human can confirm before anything moves
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager as fmgr
    try:
        fmgr.fontManager.addfont("assets/fonts/Sarabun-Regular.ttf")
        plt.rcParams["font.family"] = fmgr.FontProperties(
            fname="assets/fonts/Sarabun-Regular.ttf").get_name()
    except Exception:  # noqa: BLE001
        pass

    cache = load_cache("data/cache/glyphs.npz")
    pidx = {p: i for i, p in enumerate(cache.paths)}
    os.makedirs(args.montage_dir, exist_ok=True)
    made = 0
    for _, row in pair.head(12).iterrows():
        sub = cand[(cand.true_char == row.true_char) & (cand.pred_char == row.pred_char)].head(24)
        n = len(sub)
        if n == 0:
            continue
        cols_n = min(8, n)
        rows_n = -(-n // cols_n)
        fig, ax = plt.subplots(rows_n, cols_n, figsize=(1.4 * cols_n, 1.8 * rows_n), squeeze=False)
        for a in np.ravel(ax):
            a.axis("off")
        for a, (_, r) in zip(np.ravel(ax), sub.iterrows()):
            i = pidx.get(r.path)
            if i is None:
                continue
            im = cache[i]
            h, w = im.shape[:2]
            c = max(h, w) + 4
            pad = np.full((c, c), 255, np.uint8)
            pad[(c - h) // 2:(c - h) // 2 + h, (c - w) // 2:(c - w) // 2 + w] = im
            a.imshow(pad, cmap="gray", vmin=0, vmax=255)
            a.set_title(f"{r.p_pred:.2f}", fontsize=8)
        fig.suptitle(f"filed as {row.true_char}  —  model says {row.pred_char}   ({row.n} images)",
                     fontsize=13, fontweight="bold")
        fig.tight_layout()
        out = os.path.join(args.montage_dir, f"{row.true_char}_to_{row.pred_char}.png")
        fig.savefig(out, dpi=130, bbox_inches="tight")
        plt.close(fig)
        made += 1
    print(f"\n{made} montage(s) -> {args.montage_dir}/  — confirm by eye before moving anything")


if __name__ == "__main__":
    main()
