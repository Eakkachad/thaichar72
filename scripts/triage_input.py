#!/usr/bin/env python
"""Look at incoming test images and say which stress regime they are in, before classifying them.

The stress suite (reports/02-EXPERIMENTS.md section N) measured where each model breaks. This
script measures the same properties on whatever data actually arrives and turns the two into one
decision: which weight to use, whether to switch --rotation-tta on, and what confidence threshold
to trust.

It is deliberately label-free -- it only looks at the images -- so it can be run the moment the
data lands, before anything is known about it.

    uv run --no-sync python scripts/triage_input.py --dir <folder>
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from thaichar.data import load_cache  # noqa: E402
from thaichar.engine import pick_device  # noqa: E402
from thaichar.infer import (  # noqa: E402
    ROTATION_SEARCH_ANGLES,
    load_checkpoint,
    preprocess_image,
    rotate_canvas,
)

EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")

# What the corpus looks like, so "different from training" can be stated as a number.
# From reports/eda/class_stats.csv and the 493-glyph base sample used throughout section N.
REF = {"median_height": 24.0, "ink_frac": 0.44, "grey_levels": 2.0}


def collect(root: Path) -> list[Path]:
    out = []
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() in EXTS and p.is_file():
            out.append(p)
    return out


def describe(paths: list[Path], limit: int = 400) -> dict:
    """Label-free measurements of the incoming images."""
    sel = paths[:limit]
    hs, ws, inks, greys, dark_border, blur = [], [], [], [], 0, []
    for p in sel:
        a = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if a is None:
            continue
        hs.append(a.shape[0]); ws.append(a.shape[1])
        _, th = cv2.threshold(a, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        ys, xs = np.where(th == 0)
        if len(ys):
            crop = th[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
            inks.append(float((crop == 0).mean()))
        greys.append(len(np.unique(a)))
        ring = np.concatenate([a[0, :], a[-1, :], a[:, 0], a[:, -1]])
        if ring.mean() < 127.5:
            dark_border += 1
        blur.append(float(cv2.Laplacian(a, cv2.CV_64F).var()))
    return {
        "n_images": len(paths), "n_measured": len(hs),
        "median_height": float(np.median(hs)) if hs else float("nan"),
        "median_width": float(np.median(ws)) if ws else float("nan"),
        "median_ink_frac": float(np.median(inks)) if inks else float("nan"),
        "median_grey_levels": float(np.median(greys)) if greys else float("nan"),
        "share_dark_border": dark_border / max(1, len(sel)),
        "median_laplacian_var": float(np.median(blur)) if blur else float("nan"),
    }


def rotation_probe(ckpt: str, paths: list[Path], device, limit: int = 150) -> dict:
    """Does searching rotations raise mean confidence? If so the batch is rotated.

    This is the decision --rotation-tta needs and it needs no labels: on upright input the upright
    reading is already the most confident one, so the lift is ~0.
    """
    model, cfg = load_checkpoint(ckpt, device=device)
    model.eval()
    sel = paths[:limit]

    def conf_of(imgs):
        xs, gs = [], []
        for a in imgs:
            try:
                x, g = preprocess_image(a, cfg)
            except Exception:  # noqa: BLE001
                continue
            xs.append(x); gs.append(g)
        if not xs:
            return np.array([])
        with torch.no_grad():
            pr = torch.softmax(
                model(torch.cat(xs).to(device), torch.cat(gs).to(device)).float(), 1)
        return pr.max(1).values.cpu().numpy()

    raw = [cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) for p in sel]
    raw = [a for a in raw if a is not None]
    up = conf_of(raw)
    best = up.copy()
    best_angle = np.zeros(len(up), dtype=int)
    for deg in ROTATION_SEARCH_ANGLES:
        c = conf_of([rotate_canvas(a, deg) for a in raw])
        if len(c) != len(best):
            continue
        take = c > best
        best[take] = c[take]
        best_angle[take] = deg

    # Lift alone does NOT mean the batch is rotated: any degradation lowers the upright
    # confidence, after which some angle usually looks better by chance. Measured on probe
    # folders, the lift is 0.22 for a low-resolution batch and 0.23 for a thinned one, neither of
    # which is rotated at all. What separates a genuinely rotated batch is that the winning angle
    # is the SAME one across images, because the whole sheet was rotated by the same amount.
    nz = best_angle[best_angle != 0]
    if len(nz):
        vals, counts = np.unique(nz, return_counts=True)
        modal_angle = int(vals[counts.argmax()])
        # count the modal angle and its neighbour together: a 35 deg sheet splits between 30 and 45
        near = sum(int(c) for v, c in zip(vals, counts) if abs(int(v) - modal_angle) <= 15)
        modal_share = near / len(best_angle)
    else:
        modal_angle, modal_share = 0, 0.0

    return {"mean_conf_upright": float(up.mean()) if len(up) else float("nan"),
            "mean_conf_best_rotation": float(best.mean()) if len(best) else float("nan"),
            "rotation_lift": float((best - up).mean()) if len(up) else float("nan"),
            "share_unconfident": float((up < 0.85).mean()) if len(up) else float("nan"),
            "modal_angle": float(modal_angle),
            "modal_angle_share": float(modal_share),
            # "upright is fine" means upright is within noise of the best angle, not that it
            # strictly won: on clean input many angles score within a hair of each other.
            "share_upright_ok": float((up >= best - 0.05).mean()) if len(up) else float("nan")}


def verdict(d: dict, rot: dict) -> list[str]:
    """Turn the measurements into the decisions that have to be made before classifying."""
    out = []
    lift = rot.get("rotation_lift", 0.0)
    share = rot.get("modal_angle_share", 0.0)
    angle = rot.get("modal_angle", 0.0)
    if share >= 0.60 and lift >= 0.08:
        out.append(f"ROTATED by about {-angle:+.0f} deg: {share:.0%} of images are read best at "
                   f"the same angle and confidence lifts {lift:.3f}. Run with --rotation-tta -- "
                   f"section N measured that as worth up to +75 pt at 45 deg.")
    elif lift >= 0.08:
        out.append(f"DEGRADED BUT NOT CONSISTENTLY ROTATED: confidence lifts {lift:.3f} under "
                   f"rotation, but the winning angle is scattered ({share:.0%} agree), which is "
                   f"what a blurred or low-resolution batch looks like, not a rotated one. "
                   f"--rotation-tta is still worth trying (+11 pt on pixelated input in section N) "
                   f"but verify it on a few images rather than trusting it.")
    else:
        out.append(f"UPRIGHT (lift {lift:.3f}, {rot.get('share_upright_ok', 0):.0%} of images "
                   f"read fine upright). Leave --rotation-tta off; it only costs on heavy blur.")

    h = d["median_height"]
    if h == h:
        if h < 12:
            out.append(f"VERY LOW RESOLUTION (median height {h:.0f} px vs {REF['median_height']:.0f} "
                       f"in training). Section N: below 0.35 of the original scale nothing is "
                       f"recoverable -- expect a large drop and say so rather than trusting it.")
        elif h < 18:
            out.append(f"LOW RESOLUTION (median height {h:.0f} px vs {REF['median_height']:.0f}).")
        elif h > 120:
            out.append(f"HIGH RESOLUTION (median height {h:.0f} px). Harmless -- up to 8x cost "
                       f"nothing in section N.")

    if d["share_dark_border"] > 0.3:
        out.append(f"{d['share_dark_border']:.0%} of images have a dark border. Keep "
                   f"--polarity both (the default); it is what covers inverted input.")

    g = d["median_grey_levels"]
    if g == g and g > 32:
        out.append(f"GREYSCALE / PHOTO-LIKE input ({g:.0f} grey levels vs 2 in training). Otsu "
                   f"handles this and jpeg cost nothing in section N, but check a few by eye.")

    ink = d["median_ink_frac"]
    if ink == ink:
        if ink < REF["ink_frac"] - 0.12:
            out.append(f"THIN STROKES (ink {ink:.2f} vs {REF['ink_frac']:.2f}). This is the "
                       f"dangerous direction: section N found erosion costs far more than "
                       f"dilation. Do NOT add any thinning step.")
        elif ink > REF["ink_frac"] + 0.12:
            out.append(f"THICK STROKES (ink {ink:.2f} vs {REF['ink_frac']:.2f}). Tolerated well "
                       f"-- dilation by 5 px still retained 0.93.")

    unconf = rot.get("share_unconfident", 0.0)
    if unconf > 0.5:
        out.append(f"{unconf:.0%} of images are below 0.85 confidence. Either the data is far "
                   f"from training, or it contains multi-character / non-glyph crops. Section N: "
                   f"a confidence threshold separates those at AUROC >= 0.84.")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--ckpt", default="weights/thaichar72_r18_64_gen.pt")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default=None, help="write the measurements as JSON")
    args = ap.parse_args()

    root = Path(args.dir).expanduser()
    paths = collect(root)
    if not paths:
        raise SystemExit(f"no images under {root}")
    print(f"triage of {len(paths)} images under {root}")

    d = describe(paths)
    device = pick_device(args.device)
    rot = rotation_probe(args.ckpt, paths, device)

    print()
    print("--- what the images look like (no labels used) ---")
    for k in ("n_images", "median_height", "median_width", "median_ink_frac",
              "median_grey_levels", "share_dark_border", "median_laplacian_var"):
        v = d[k]
        print(f"  {k:<22} {v:.3f}" if isinstance(v, float) else f"  {k:<22} {v}")
    print()
    print("--- rotation probe ---")
    for k, v in rot.items():
        print(f"  {k:<26} {v:.4f}")
    print()
    print("--- what to do ---")
    for line in verdict(d, rot):
        print(f"  * {line}")
    print()
    print("Then classify with:")
    tta = (" --rotation-tta"
           if (rot.get("modal_angle_share", 0) >= 0.60 and rot.get("rotation_lift", 0) >= 0.08)
           else "")
    print(f"  uv run --no-sync python scripts/evaluate_folder.py --dir {root}{tta}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump({"images": d, "rotation": rot, "verdict": verdict(d, rot)},
                  open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
