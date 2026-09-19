#!/usr/bin/env python
"""Figures: augmentation presets before/after, and channel encodings (gray3 vs onoff)."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from thaichar.augment import get_transform  # noqa: E402
from thaichar.classes import code_to_char  # noqa: E402
from thaichar.data import load_cache  # noqa: E402
from thaichar.transforms import encode_channels, pad_to_square_canvas, resize_square  # noqa: E402

FONT = "assets/fonts/Sarabun-Regular.ttf"
PRESETS = ["base", "morph", "full", "randaug", "trivial"]
N_DRAWS = 5


def setup_font() -> None:
    font_manager.fontManager.addfont(FONT)
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=FONT).get_name()


def pick_glyphs(n: int, seed: int = 0) -> list[tuple[int, np.ndarray]]:
    df = pd.read_csv("data/splits/split_seed42.csv")
    cache = load_cache("data/cache/glyphs.npz")
    idx = {p: i for i, p in enumerate(cache.paths)}
    rng = np.random.default_rng(seed)
    codes = rng.choice(sorted(df.code.unique()), size=n, replace=False)
    out = []
    for c in codes:
        row = df[df.code == c].sample(1, random_state=int(rng.integers(1e9))).iloc[0]
        out.append((int(c), cache[idx[row.path]]))
    return out


def fig_aug(glyphs, out: str) -> None:
    ncol = 1 + len(PRESETS) * N_DRAWS
    fig, axes = plt.subplots(len(glyphs), ncol, figsize=(ncol * 0.9, len(glyphs) * 0.95))
    tfs = {p: get_transform(p, seed=0) for p in PRESETS}
    for r, (code, img) in enumerate(glyphs):
        canvas = pad_to_square_canvas(img, margin=0.1)
        axes[r, 0].imshow(resize_square(canvas, 64), cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        axes[r, 0].set_ylabel(f"{code_to_char(code)} ({code})", rotation=0, labelpad=22, fontsize=11, va="center")
        for pi, p in enumerate(PRESETS):
            for d in range(N_DRAWS):
                ax = axes[r, 1 + pi * N_DRAWS + d]
                ax.imshow(resize_square(tfs[p](canvas), 64), cmap="gray", vmin=0, vmax=255, interpolation="nearest")
                if r == 0 and d == N_DRAWS // 2:
                    ax.set_title(p, fontsize=11, fontweight="bold")
    axes[0, 0].set_title("original", fontsize=11, fontweight="bold")
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Augmentation presets (canvas-level, no flips) — 5 random draws each", y=1.0)
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)


def fig_channels(glyphs, out: str) -> None:
    titles = ["gray3 (ch0)", "onoff ch0: ink", "onoff ch1: stroke thickness (DT)", "onoff ch2: edges"]
    fig, axes = plt.subplots(len(glyphs), 4, figsize=(4 * 1.7, len(glyphs) * 1.7))
    for r, (code, img) in enumerate(glyphs):
        square = resize_square(pad_to_square_canvas(img, 0.1), 64)
        g3 = encode_channels(square, "gray3")
        oo = encode_channels(square, "onoff")
        panels = [g3[0], oo[0], oo[1], oo[2]]
        for c, arr in enumerate(panels):
            axes[r, c].imshow(arr, cmap="gray" if c == 0 else "magma", interpolation="nearest")
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
            if r == 0:
                axes[r, c].set_title(titles[c], fontsize=9)
        axes[r, 0].set_ylabel(f"{code_to_char(code)} ({code})", rotation=0, labelpad=22, fontsize=11, va="center")
    fig.suptitle("Channel encodings: replicated grey vs ON/OFF-style [ink, distance transform, edges]")
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    setup_font()
    Path("reports/figures").mkdir(parents=True, exist_ok=True)
    fig_aug(pick_glyphs(8, seed=1), "reports/figures/aug_examples.png")
    fig_channels(pick_glyphs(6, seed=2), "reports/figures/channel_modes.png")
    print("wrote reports/figures/aug_examples.png and channel_modes.png")
