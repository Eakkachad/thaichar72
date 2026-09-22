#!/usr/bin/env python
"""Put suspected mislabels next to confident examples of BOTH classes, so the eye has a reference.

A montage of suspects alone asks "is this an A or a B?" with nothing to compare against. This
renders three blocks -- images the out-of-fold model is confident are A, the disputed ones, and
images it is confident are B -- so the distinguishing stroke is visible rather than remembered.

    uv run --no-sync python scripts/compare_pair.py --pair "ด>ต"
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager as fmgr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from thaichar.classes import CLASS_CODES, code_to_char  # noqa: E402
from thaichar.data import load_cache  # noqa: E402

CHAR_TO_IDX = {code_to_char(c): i for i, c in enumerate(CLASS_CODES)}


def strip(ax, img, cache, pidx, title, colour="black"):
    i = pidx.get(img)
    if i is None:
        ax.axis("off")
        return
    im = cache[i]
    h, w = im.shape[:2]
    c = max(h, w) + 4
    pad = np.full((c, c), 255, np.uint8)
    pad[(c - h) // 2:(c - h) // 2 + h, (c - w) // 2:(c - w) // 2 + w] = im
    ax.imshow(pad, cmap="gray", vmin=0, vmax=255)
    ax.set_title(title, fontsize=8, color=colour)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(colour)
        s.set_linewidth(1.2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", required=True, help='"filed>model", e.g. "ด>ต"')
    ap.add_argument("--n", type=int, default=8, help="images per row")
    ap.add_argument("--candidates", default="reports/analysis/label_noise_candidates.csv")
    ap.add_argument("--split-file", default="data/splits/split_seed42_v2.csv")
    ap.add_argument("--out-dir", default="reports/analysis/label_noise")
    args = ap.parse_args()

    a, b = [s.strip() for s in args.pair.split(">", 1)]
    for ch in (a, b):
        if ch not in CHAR_TO_IDX:
            raise SystemExit(f"{ch!r} is not one of the 72 classes")

    try:
        fmgr.fontManager.addfont("assets/fonts/Sarabun-Regular.ttf")
        plt.rcParams["font.family"] = fmgr.FontProperties(
            fname="assets/fonts/Sarabun-Regular.ttf").get_name()
    except Exception:  # noqa: BLE001
        pass

    cand = pd.read_csv(args.candidates)
    split = pd.read_csv(args.split_file)
    cache = load_cache("data/cache/glyphs.npz")
    pidx = {p: i for i, p in enumerate(cache.paths)}

    disputed = cand[(cand.true_char == a) & (cand.pred_char == b)].sort_values(
        "p_pred", ascending=False)
    if disputed.empty:
        raise SystemExit(f"no disputed images for {a}>{b}")
    flagged = set(cand.path)

    # reference = images of each class the out-of-fold model did NOT dispute
    ref_a = split[(split.label == CHAR_TO_IDX[a]) & (~split.path.isin(flagged))].head(200)
    ref_b = split[(split.label == CHAR_TO_IDX[b]) & (~split.path.isin(flagged))].head(200)
    ref_a = ref_a.sample(min(args.n, len(ref_a)), random_state=0)
    ref_b = ref_b.sample(min(args.n, len(ref_b)), random_state=0)

    n = args.n
    rows = [("undisputed " + a, ref_a.path.tolist(), ["" for _ in range(len(ref_a))], "#1a7f37"),
            (f"DISPUTED: filed {a}, model says {b}", disputed.path.head(n).tolist(),
             [f"{v:.2f}" for v in disputed.p_pred.head(n)], "#c0392b"),
            ("undisputed " + b, ref_b.path.tolist(), ["" for _ in range(len(ref_b))], "#1a7f37")]
    if len(disputed) > n:
        rows.insert(2, ("DISPUTED (more)", disputed.path.iloc[n:2 * n].tolist(),
                        [f"{v:.2f}" for v in disputed.p_pred.iloc[n:2 * n]], "#c0392b"))

    fig, ax = plt.subplots(len(rows), n, figsize=(1.5 * n, 2.0 * len(rows)), squeeze=False)
    for r, (label, paths, titles, colour) in enumerate(rows):
        for c in range(n):
            ax[r][c].axis("off")
            if c < len(paths):
                ax[r][c].axis("on")
                strip(ax[r][c], paths[c], cache, pidx, titles[c] if c < len(titles) else "", colour)
        ax[r][0].set_ylabel(label, rotation=0, ha="right", va="center", fontsize=10,
                            fontweight="bold", color=colour, labelpad=8)
        ax[r][0].axis("on")
        ax[r][0].set_xticks([])
        ax[r][0].set_yticks([])

    fig.suptitle(f"{a}  vs  {b}   —   {len(disputed)} images filed as {a} that the model reads as {b}",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, f"compare_{a}_vs_{b}.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"{len(disputed)} disputed, mean confidence {disputed.p_pred.mean():.3f}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
