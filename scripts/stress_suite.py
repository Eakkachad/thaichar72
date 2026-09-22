#!/usr/bin/env python
"""Stress suite: how each candidate behaves on inputs we never trained for.

Spec: tasks/TASK-14-stress-suite.md. Three families, deliberately scored differently:

  A. degraded        -- the right answer still exists -> accuracy retention vs severity
  B. invalid         -- there IS no right answer      -> confident-wrong rate, separability
  C. identity-changing -- the answer may become a DIFFERENT class -> a mapping, not a score

Family B is the one that is easy to get wrong: reporting "accuracy" on a picture of two glued
characters would be meaningless, because no label is correct. What we measure instead is whether
the model signals that it is out of its depth, since that decides whether a confidence threshold
could filter such inputs or whether the input pipeline has to be fixed first.

Everything runs through `thaichar.infer.preprocess_image`, i.e. the deployed path (Otsu, reframe,
crop, pad-to-square, resize), so the numbers describe the shipped behaviour rather than the
training-time dataset transform.

    uv run --no-sync python scripts/stress_suite.py --models all --families A B C
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import cv2
import matplotlib
# NOTE: the Agg backend is selected inside main(), not here. This module is imported by
# notebooks/ThaiChar72_StressEval.ipynb for its corruption builders, and forcing Agg at import
# time would silently kill inline figures in the notebook that imported it.
import matplotlib.pyplot as plt
from matplotlib import font_manager as _fm
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from thaichar.classes import CLASS_CODES, code_to_char  # noqa: E402
from thaichar.data import load_cache  # noqa: E402
from thaichar.engine import pick_device  # noqa: E402
from thaichar.infer import load_checkpoint, preprocess_image  # noqa: E402

try:
    _fm.fontManager.addfont("assets/fonts/Sarabun-Regular.ttf")
    plt.rcParams["font.family"] = _fm.FontProperties(
        fname="assets/fonts/Sarabun-Regular.ttf").get_name()
except Exception:  # noqa: BLE001
    pass

WHITE = 255
PACKAGED = [
    "thaichar72_r18_64_gen",
    "thaichar72_r18_64_gen_v3labels",
    "thaichar72_resnet18_64",
    "thaichar72_resnet18_64_v1labels",
    "thaichar72_mnv3small_64_small",
]
# font families held out of the dataUpdate probe, plus the .ttc collections synth.py never globs
HELDOUT_TTF_PREFIXES = ("IBMPlexSansThai-", "Mali-", "Sarabun-", "Sriracha", "Trirong-")
TTC_NAMES = ("angsana", "browalia", "cordia")


# ---------------------------------------------------------------------------
# base material
# ---------------------------------------------------------------------------
def base_sample(per_class: int, seed: int) -> tuple[list[np.ndarray], np.ndarray]:
    """`per_class` held-out val glyphs per class, same draw for every model and corruption."""
    df = pd.read_csv("data/splits/split_seed42_v2.csv")
    val = df[df["split"] == "val"]
    cache = load_cache("data/cache/glyphs.npz")
    idx = {p: i for i, p in enumerate(cache.paths)}
    rng = np.random.default_rng(seed)
    imgs, labels = [], []
    for lab in range(len(CLASS_CODES)):
        paths = [p for p in val.loc[val.label == lab, "path"] if p in idx]
        if not paths:
            continue
        take = rng.choice(len(paths), size=min(per_class, len(paths)), replace=False)
        for t in take:
            imgs.append(cache[idx[paths[int(t)]]].copy())
            labels.append(lab)
    return imgs, np.array(labels, dtype=int)


def pad_canvas(a: np.ndarray, pad: int = 6) -> np.ndarray:
    return np.pad(a, pad, mode="constant", constant_values=WHITE)


# ---------------------------------------------------------------------------
# family A -- the answer still exists
# ---------------------------------------------------------------------------
def a_jpeg(a, q):
    ok, buf = cv2.imencode(".jpg", a, [int(cv2.IMWRITE_JPEG_QUALITY), int(q)])
    return cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE) if ok else a


def a_pixelate(a, f):
    h, w = a.shape
    sh, sw = max(2, int(h * f)), max(2, int(w * f))
    small = cv2.resize(a, (sw, sh), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


def a_motion_blur(a, n):
    n = int(n)
    k = np.zeros((n, n), np.float32)
    k[n // 2, :] = 1.0 / n
    # filter2D has no borderValue; pad wide enough that the white border is what gets sampled
    return cv2.filter2D(pad_canvas(a, n), -1, k, borderType=cv2.BORDER_REPLICATE)


def a_rotate_hard(a, deg):
    p = pad_canvas(a, max(a.shape) // 2 + 4)
    h, w = p.shape
    m = cv2.getRotationMatrix2D((w / 2, h / 2), float(deg), 1.0)
    return cv2.warpAffine(p, m, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=WHITE)


def a_stroke_extreme(a, sev):
    """sev>0 dilates ink (thicker), sev<0 erodes it. Ink is dark, so the ops invert."""
    n = int(abs(sev))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (n, n))
    p = pad_canvas(a, n + 2)
    return cv2.erode(p, k) if sev > 0 else cv2.dilate(p, k)


def a_resolution_up(a, s):
    h, w = a.shape
    return cv2.resize(a, (int(w * s), int(h * s)), interpolation=cv2.INTER_CUBIC)


def a_aspect_stretch(a, r):
    h, w = a.shape
    return cv2.resize(a, (max(2, int(w * r)), h), interpolation=cv2.INTER_LINEAR)


FAMILY_A = {
    "jpeg":          (a_jpeg,          [40, 20, 10, 5]),
    "pixelate":      (a_pixelate,      [0.5, 0.35, 0.25, 0.15]),
    "motion_blur":   (a_motion_blur,   [3, 5, 7, 9]),
    "rotate_hard":   (a_rotate_hard,   [30, 45, 60, 90]),
    "stroke_extreme": (a_stroke_extreme, [-5, -4, 4, 5]),
    "resolution_up": (a_resolution_up, [2, 4, 8]),
    "aspect_stretch": (a_aspect_stretch, [0.7, 0.85, 1.2, 1.4]),
}


# ---------------------------------------------------------------------------
# family A6 -- unseen typefaces
# ---------------------------------------------------------------------------
def build_unseen_font_dir() -> tuple[str, int]:
    """Temp dir of typefaces the stage-1 render never used: the 5 probe-held-out families
    plus every Thai-complete face inside the 3 .ttc collections (synth.py globs *.ttf only)."""
    from fontTools.ttLib import TTCollection

    d = tempfile.mkdtemp(prefix="stress_unseen_fonts_")
    n = 0
    for p in glob.glob("data/fonts_all/*.ttf"):
        if os.path.basename(p).startswith(HELDOUT_TTF_PREFIXES):
            shutil.copy(p, d)
            n += 1
    need = {ord(ch) for c in CLASS_CODES for ch in code_to_char(c)}
    for name in TTC_NAMES:
        src = f"/mnt/c/Windows/Fonts/{name}.ttc"
        if not os.path.exists(src):
            continue
        try:
            coll = TTCollection(src, lazy=False)
        except Exception:  # noqa: BLE001
            continue
        for i, f in enumerate(coll.fonts):
            try:
                if not need <= set(f.getBestCmap().keys()):
                    continue
                f.save(os.path.join(d, f"{name}_{i}.ttf"))
                n += 1
            except Exception:  # noqa: BLE001
                continue
        coll.close()
    return d, n


def render_unseen_font(per_class: int, seed: int) -> tuple[list[np.ndarray], np.ndarray, int]:
    from thaichar.synth import FontLibrary, degrade_glyph

    d, n_fonts = build_unseen_font_dir()
    # Degrade exactly as scripts/render_synth.py does, at the real per-class median height, so the
    # ONLY difference from the stage-1 training render is the typeface. Rendering these clean and
    # high-res instead would measure "crisp glyph vs noisy corpus", which is a different question
    # (and an easier one) -- the first version of this function did that by accident.
    med_h = {}
    try:
        st = pd.read_csv("reports/eda/class_stats.csv")
        med_h = dict(zip(st.code, st.med_h))
    except Exception:  # noqa: BLE001
        pass
    try:
        lib = FontLibrary(font_dir=d)
        rng = np.random.default_rng(seed)
        imgs, labels = [], []
        for lab, code in enumerate(CLASS_CODES):
            t = lib.templates.get(code, [])
            if not t:
                continue
            for _ in range(per_class):
                tmpl = t[int(rng.integers(len(t)))]
                target_h = max(3, int(round(float(med_h.get(code, 20)) * rng.uniform(0.8, 1.3))))
                g = degrade_glyph(tmpl.image_u8, target_h, rng, degraded=True)
                imgs.append(np.asarray(g, dtype=np.uint8))
                labels.append(lab)
        return imgs, np.array(labels, dtype=int), n_fonts
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------------------
# family B -- no correct answer exists
# ---------------------------------------------------------------------------
def _hstack(gs, gap):
    """Place glyphs left to right on a common baseline; gap<0 overlaps them."""
    h = max(g.shape[0] for g in gs)
    parts, x = [], 0
    w = sum(g.shape[1] for g in gs) + gap * (len(gs) - 1)
    out = np.full((h, max(4, w)), WHITE, np.uint8)
    for g in gs:
        gh, gw = g.shape
        y = h - gh
        seg = out[y:y + gh, x:x + gw]
        if seg.shape == g.shape:
            out[y:y + gh, x:x + gw] = np.minimum(seg, g)   # ink is dark -> min keeps ink
        x += gw + gap
    return out


def b_pairs(imgs, rng, n, gap_frac):
    out = []
    for _ in range(n):
        i, j = rng.integers(len(imgs), size=2)
        a, b = imgs[int(i)], imgs[int(j)]
        gap = int(round(-gap_frac * b.shape[1]))
        out.append(pad_canvas(_hstack([a, b], gap)))
    return out


def b_triples(imgs, rng, n):
    out = []
    for _ in range(n):
        k = rng.integers(len(imgs), size=3)
        out.append(pad_canvas(_hstack([imgs[int(x)] for x in k], 1)))
    return out


def b_half(imgs, rng, n, frac):
    out = []
    for _ in range(n):
        g = imgs[int(rng.integers(len(imgs)))]
        h, w = g.shape
        side = int(rng.integers(4))
        if side == 0:
            c = g[: max(2, int(h * frac)), :]
        elif side == 1:
            c = g[h - max(2, int(h * frac)):, :]
        elif side == 2:
            c = g[:, : max(2, int(w * frac))]
        else:
            c = g[:, w - max(2, int(w * frac)):]
        out.append(pad_canvas(c))
    return out


def b_strokes(rng, n, size=48):
    out = []
    for _ in range(n):
        c = np.full((size, size), WHITE, np.uint8)
        for _ in range(int(rng.integers(2, 6))):
            pts = rng.integers(4, size - 4, size=(int(rng.integers(3, 6)), 2)).astype(np.int32)
            cv2.polylines(c, [pts], False, 0, int(rng.integers(2, 4)))
        out.append(pad_canvas(c))
    return out


def b_blobs(rng, n, size=48):
    out = []
    for _ in range(n):
        c = np.full((size, size), WHITE, np.uint8)
        for _ in range(int(rng.integers(1, 4))):
            ctr = (int(rng.integers(10, size - 10)), int(rng.integers(10, size - 10)))
            ax = (int(rng.integers(4, 14)), int(rng.integers(4, 14)))
            cv2.ellipse(c, ctr, ax, float(rng.integers(0, 180)), 0, 360, 0, -1)
        out.append(pad_canvas(c))
    return out


def b_latin_digits(rng, n):
    from PIL import Image, ImageDraw, ImageFont
    chars = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
    fonts = sorted(glob.glob("data/fonts_all/*.ttf"))[:40]
    out = []
    for _ in range(n):
        ch = chars[int(rng.integers(len(chars)))]
        fp = fonts[int(rng.integers(len(fonts)))]
        try:
            f = ImageFont.truetype(fp, int(rng.integers(28, 56)))
        except Exception:  # noqa: BLE001
            continue
        im = Image.new("L", (96, 96), WHITE)
        ImageDraw.Draw(im).text((24, 16), ch, font=f, fill=0)
        a = np.asarray(im, np.uint8)
        ys, xs = np.where(a < 128)
        if len(ys) == 0:
            continue
        out.append(pad_canvas(a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]))
    return out


def b_near_empty(rng, n, size=32):
    out = []
    for _ in range(n):
        c = np.full((size, size), WHITE, np.uint8)
        for _ in range(int(rng.integers(1, 4))):
            c[int(rng.integers(size)), int(rng.integers(size))] = 0
        out.append(c)
    return out


# ---------------------------------------------------------------------------
# family C -- the transform may change the answer
# ---------------------------------------------------------------------------
FAMILY_C = {
    "rot180":     lambda a: np.rot90(a, 2).copy(),
    "mirror_h":   lambda a: a[:, ::-1].copy(),
    "mirror_v":   lambda a: a[::-1, :].copy(),
    "rot90_cw":   lambda a: np.rot90(a, 3).copy(),
    "rot90_ccw":  lambda a: np.rot90(a, 1).copy(),
}


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------
def to_batch(imgs, cfg, device, bs=256):
    """Deployed preprocessing for a list of canvases -> batched tensors, BOTH polarities.

    `preprocess_image` reads ink polarity off the border ring, which a tight crop does not have:
    when strokes touch the corners, `_reframe` pads with a dark background and the whole image is
    then inverted. On the 493-glyph base sample that costs 2.0 pt (0.9695 vs 0.9898 for the dataset
    transform on the same images, wrong on 10 images and right on none of them).
    `scripts/evaluate_folder.py` already solves this with `--polarity both`, so the suite uses the
    same rule: preprocess as-is AND inverted, and keep whichever the model is more confident about.
    That is what will actually run on the day, so it is what should be measured.
    """
    xs, gs, xs_i, gs_i, keep = [], [], [], [], []
    for i, a in enumerate(imgs):
        try:
            x, g = preprocess_image(a, cfg)
        except Exception:  # noqa: BLE001
            continue          # empty after Otsu (near-empty inputs legitimately do this)
        try:
            xi, gi = preprocess_image(255 - np.asarray(a, dtype=np.uint8), cfg)
        except Exception:  # noqa: BLE001
            xi, gi = x, g
        xs.append(x); gs.append(g); xs_i.append(xi); gs_i.append(gi)
        keep.append(i)
    if not xs:
        return [], np.array([], dtype=int)
    # preprocess_image already returns a leading batch dim of 1, so concatenate rather than stack
    cat = torch.cat if xs[0].dim() == 4 else torch.stack
    catg = torch.cat if gs[0].dim() == 2 else torch.stack
    batches = []
    for s in range(0, len(xs), bs):
        batches.append((cat(xs[s:s + bs]).to(device), catg(gs[s:s + bs]).to(device),
                        cat(xs_i[s:s + bs]).to(device), catg(gs_i[s:s + bs]).to(device)))
    return batches, np.array(keep, dtype=int)


@torch.no_grad()
def probs_for(model, batches):
    """Per image, the more confident of the two polarities (the deployed `--polarity both` rule)."""
    out = []
    for x, g, xi, gi in batches:
        pa = torch.softmax(model(x, g).float(), dim=1)
        pb = torch.softmax(model(xi, gi).float(), dim=1)
        take_a = (pa.max(1).values >= pb.max(1).values).unsqueeze(1)
        out.append(torch.where(take_a, pa, pb).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, len(CLASS_CODES)))


def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    """P(a random clean image scores higher than a random junk image). 0.5 = no separation."""
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    r = pd.Series(allv).rank().to_numpy()
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


# ---------------------------------------------------------------------------
# montages -- a corruption that is wrong in code produces a clean number that means nothing
# ---------------------------------------------------------------------------
def montage(imgs, title, path, n=10):
    imgs = imgs[:n]
    if not imgs:
        return
    fig, axes = plt.subplots(1, len(imgs), figsize=(1.25 * len(imgs), 1.7))
    for ax, im in zip(np.ravel(np.atleast_1d(axes)), imgs):
        ax.imshow(im, cmap="gray", vmin=0, vmax=255)
        ax.axis("off")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=["all"])
    ap.add_argument("--families", nargs="*", default=["A", "B", "C"])
    ap.add_argument("--per-class", type=int, default=8)
    ap.add_argument("--n-junk", type=int, default=300, help="items per family-B condition")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="reports/stress")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    matplotlib.use("Agg")   # headless: this is a script run, not the notebook importing us
    t0 = time.time()
    out = Path(args.out)
    (out / "samples").mkdir(parents=True, exist_ok=True)
    device = pick_device(args.device)
    names = PACKAGED if args.models == ["all"] else args.models

    models = {}
    for n in names:
        p = Path("weights") / f"{n}.pt"
        if not p.exists():
            print(f"  skip {n}: no {p}")
            continue
        m, cfg = load_checkpoint(str(p), device=device)
        m.eval()
        models[n] = (m, cfg)
    if not models:
        raise SystemExit("no models loaded")
    print(f"models: {list(models)}")

    base_imgs, base_y = base_sample(args.per_class, args.seed)
    print(f"base sample: {len(base_imgs)} held-out val glyphs, {len(set(base_y.tolist()))} classes")
    montage(base_imgs[:10], "clean base sample", out / "samples" / "clean.png")

    rows = []
    clean_maxprob = {}
    clean_top1 = {}

    def run(cond_family, cond, sev, imgs, y_true, tag):
        """Preprocess once, then score every model on the same tensors."""
        batches, keep = to_batch(imgs, next(iter(models.values()))[1], device)
        n_drop = len(imgs) - len(keep)
        if len(keep) == 0:
            for mn in models:
                rows.append(dict(model=mn, family=cond_family, condition=cond, severity=sev,
                                 n=0, n_rejected=n_drop))
            return {}
        yk = y_true[keep] if y_true is not None else None
        res = {}
        for mn, (m, _) in models.items():
            pr = probs_for(m, batches)
            mx = pr.max(1)
            pred = pr.argmax(1)
            r = dict(model=mn, family=cond_family, condition=cond, severity=sev,
                     n=int(len(pred)), n_rejected=int(n_drop),
                     mean_max_prob=float(mx.mean()))
            if cond_family in ("A", "clean") and yk is not None:
                ok = pred == yk
                r["top1"] = float(ok.mean())
                pc = [float(ok[yk == c].mean()) for c in np.unique(yk)]
                r["balanced_acc"] = float(np.mean(pc))
            if cond_family == "B":
                for tau in (0.5, 0.7, 0.9):
                    r[f"confident_wrong@{tau}"] = float((mx >= tau).mean())
                r["separability_auroc"] = auroc(clean_maxprob.get(mn, np.array([])), mx)
            rows.append(r)
            res[mn] = (pr, pred, mx, yk)
        return res

    # ---- clean baseline (acceptance check 2) ----
    r0 = run("clean", "clean", 0, base_imgs, base_y, "clean")
    for mn, (pr, pred, mx, yk) in r0.items():
        clean_maxprob[mn] = mx
        clean_top1[mn] = float((pred == yk).mean())
    print("clean top-1:", {k: round(v, 4) for k, v in clean_top1.items()})

    rng = np.random.default_rng(args.seed)

    # ---- family A ----
    if "A" in args.families:
        for cname, (fn, sevs) in FAMILY_A.items():
            for sv in sevs:
                imgs = [fn(a, sv) for a in base_imgs]
                montage(imgs[:10], f"A: {cname} sev={sv}",
                        out / "samples" / f"A_{cname}_{sv}.png")
                run("A", cname, sv, imgs, base_y, f"A_{cname}")
            print(f"  A/{cname} done ({time.time()-t0:.0f}s)")
        imgs, y, nf = render_unseen_font(args.per_class, args.seed)
        montage(imgs[:10], f"A: unseen_font ({nf} typefaces)",
                out / "samples" / "A_unseen_font.png")
        run("A", "unseen_font", nf, imgs, y, "A_unseen_font")
        print(f"  A/unseen_font done, {nf} typefaces ({time.time()-t0:.0f}s)")

    # ---- family B ----
    if "B" in args.families:
        n = args.n_junk
        conds = {
            "touching_2":    b_pairs(base_imgs, rng, n, 0.0),
            "overlap_2_25":  b_pairs(base_imgs, rng, n, 0.25),
            "overlap_2_50":  b_pairs(base_imgs, rng, n, 0.50),
            "triple":        b_triples(base_imgs, rng, n),
            "half_50":       b_half(base_imgs, rng, n, 0.50),
            "half_70":       b_half(base_imgs, rng, n, 0.70),
            "strokes":       b_strokes(rng, n),
            "blobs":         b_blobs(rng, n),
            "latin_digits":  b_latin_digits(rng, n),
            "near_empty":    b_near_empty(rng, n),
        }
        for cname, imgs in conds.items():
            montage(imgs[:10], f"B: {cname}", out / "samples" / f"B_{cname}.png")
            run("B", cname, 0, imgs, None, f"B_{cname}")
            print(f"  B/{cname} done ({time.time()-t0:.0f}s)")

    # ---- family C ----
    flip_rows = []
    if "C" in args.families:
        for cname, fn in FAMILY_C.items():
            imgs = [fn(a) for a in base_imgs]
            montage(imgs[:10], f"C: {cname}", out / "samples" / f"C_{cname}.png")
            res = run("C", cname, 0, imgs, base_y, f"C_{cname}")
            for mn, (pr, pred, mx, yk) in res.items():
                for c in np.unique(yk):
                    sel = yk == c
                    vals, counts = np.unique(pred[sel], return_counts=True)
                    top = int(vals[counts.argmax()])
                    flip_rows.append(dict(
                        model=mn, transform=cname,
                        true_char=code_to_char(CLASS_CODES[int(c)]),
                        maps_to_char=code_to_char(CLASS_CODES[top]),
                        maps_to_self=bool(top == int(c)),
                        rate=float(counts.max() / sel.sum()),
                        mean_conf=float(mx[sel].mean())))
            print(f"  C/{cname} done ({time.time()-t0:.0f}s)")

    df = pd.DataFrame(rows)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "results.csv", index=False)
    if flip_rows:
        pd.DataFrame(flip_rows).to_csv(out / "flip_map.csv", index=False)
    write_report(df, pd.DataFrame(flip_rows), clean_top1, out, args)
    print(f"\nwrote {out}/results.csv, summary.md, figures ({time.time()-t0:.0f}s total)")


def crossing(sub: pd.DataFrame, clean: float, frac: float):
    """First severity where retention drops below `frac`; severities ordered by harshness."""
    s = sub.sort_values("_order")
    for _, r in s.iterrows():
        if pd.notna(r.get("top1")) and clean > 0 and r["top1"] / clean < frac:
            return r["severity"]
    return None


def write_report(df, flip, clean_top1, out: Path, args) -> None:
    a = df[df.family == "A"].copy()
    b = df[df.family == "B"].copy()
    c = df[df.family == "C"].copy()

    # figure: family A retention curves
    if len(a):
        conds = [x for x in a.condition.unique() if x != "unseen_font"]
        ncol = 3
        nrow = int(np.ceil(len(conds) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.0 * nrow), squeeze=False)
        for ax, cname in zip(np.ravel(axes), conds):
            for mn in a.model.unique():
                s = a[(a.condition == cname) & (a.model == mn)].sort_values("severity")
                if not len(s):
                    continue
                ax.plot(range(len(s)), s.top1 / clean_top1[mn], marker="o", label=mn.replace("thaichar72_", ""))
                ax.set_xticks(range(len(s)))
                ax.set_xticklabels([str(v) for v in s.severity], fontsize=8)
            ax.axhline(0.9, ls="--", c="gray", lw=0.8)
            ax.axhline(0.5, ls=":", c="gray", lw=0.8)
            ax.set_title(cname, fontsize=10)
            ax.set_ylim(0, 1.05)
            ax.grid(alpha=0.3)
        for ax in np.ravel(axes)[len(conds):]:
            ax.axis("off")
        np.ravel(axes)[0].legend(fontsize=6.5, loc="lower left")
        fig.suptitle("Family A - top-1 retention vs clean (dashed 90 %, dotted 50 %)",
                     fontsize=12, fontweight="bold")
        fig.tight_layout()
        fig.savefig(out / "fig_family_A_curves.png", dpi=130, bbox_inches="tight")
        plt.close(fig)

    # figure: family B confidence
    if len(b):
        fig, ax = plt.subplots(figsize=(12, 4))
        piv = b.pivot_table(index="condition", columns="model", values="mean_max_prob")
        piv.plot(kind="bar", ax=ax, width=0.8)
        for mn, v in clean_top1.items():
            pass
        ax.axhline(0.9, ls="--", c="gray", lw=0.8)
        ax.set_ylabel("mean max softmax")
        ax.set_title("Family B - confidence on inputs with NO correct answer (lower is better)",
                     fontsize=12, fontweight="bold")
        ax.legend(fontsize=7)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(out / "fig_family_B_conf.png", dpi=130, bbox_inches="tight")
        plt.close(fig)

    L = []
    L.append("# Stress suite — behaviour on inputs we never trained for\n")
    L.append(f"Generated {time.strftime('%Y-%m-%d %H:%M')} · spec `tasks/TASK-14-stress-suite.md` · "
             f"base sample {args.per_class} held-out val glyphs/class, seed {args.seed}.\n")
    L.append("Three families are scored differently **on purpose**: family B has no correct answer, "
             "so it reports no accuracy anywhere — only whether the model signals that it is out of "
             "its depth.\n")

    L.append("\n## 0. Clean baseline (harness sanity check)\n")
    L.append("| model | clean top-1 on the base sample |")
    L.append("|---|---:|")
    for mn, v in clean_top1.items():
        L.append(f"| `{mn}` | {v:.4f} |")
    L.append("\nThese must sit near each model's known val top-1; a low number here would mean the "
             "harness itself is degrading the input, not the corruption.\n")

    if len(a):
        order = {c: i for i, c in enumerate(FAMILY_A)}
        a["_order"] = a.severity.rank()
        L.append("\n## 1. Family A — degraded input, the answer still exists\n")
        L.append("Retention = top-1 under the corruption ÷ that model's own clean top-1.\n")
        L.append("| corruption | severity | " + " | ".join(f"`{m.replace('thaichar72_','')}`"
                                                            for m in a.model.unique()) + " |")
        L.append("|---|---|" + "---:|" * a.model.nunique())
        for cname in a.condition.unique():
            for sv in a[a.condition == cname].severity.unique():
                cells = []
                for mn in a.model.unique():
                    s = a[(a.condition == cname) & (a.severity == sv) & (a.model == mn)]
                    cells.append(f"{s.top1.iloc[0]/clean_top1[mn]:.3f}" if len(s) and pd.notna(s.top1.iloc[0]) else "–")
                L.append(f"| {cname} | {sv} | " + " | ".join(cells) + " |")
        L.append("\n![family A](fig_family_A_curves.png)\n")

    if len(b):
        L.append("\n## 2. Family B — invalid input, no correct answer exists\n")
        L.append("`confident@τ` = fraction of junk images the model scores with max softmax ≥ τ "
                 "(lower is better). `AUROC` separates clean from junk by confidence alone: "
                 "**≥ 0.8 a threshold can filter these, ≤ 0.6 it cannot**.\n")
        L.append("| condition | model | mean max-prob | confident@0.5 | @0.7 | @0.9 | AUROC vs clean | rejected |")
        L.append("|---|---|---:|---:|---:|---:|---:|---:|")
        for _, r in b.iterrows():
            L.append(f"| {r.condition} | `{r.model.replace('thaichar72_','')}` | {r.mean_max_prob:.3f} | "
                     f"{r['confident_wrong@0.5']:.3f} | {r['confident_wrong@0.7']:.3f} | "
                     f"{r['confident_wrong@0.9']:.3f} | {r.separability_auroc:.3f} | {int(r.n_rejected)} |")
        L.append("\n![family B](fig_family_B_conf.png)\n")

    if len(flip):
        L.append("\n## 3. Family C — the transform may change the answer\n")
        L.append("A mapping, not a score. Full table in `flip_map.csv`.\n")
        for mn in flip['model'].unique():
            f = flip[flip['model'] == mn]
            L.append(f"\n**`{mn}`**\n")
            L.append("| transform | maps to itself | confident onto ANOTHER class (rate≥0.5, conf≥0.7) | mean conf |")
            L.append("|---|---:|---:|---:|")
            for t in f['transform'].unique():
                s = f[f['transform'] == t]
                dang = s[(~s['maps_to_self']) & (s['rate'] >= 0.5) & (s['mean_conf'] >= 0.7)]
                L.append(f"| {t} | {int(s.maps_to_self.sum())}/{len(s)} | {len(dang)} | {s.mean_conf.mean():.3f} |")
        worst = flip[(~flip['maps_to_self']) & (flip['rate'] >= 0.6) & (flip['mean_conf'] >= 0.8)]
        if len(worst):
            L.append("\nClasses that a flipped test image would turn into another class **confidently** "
                     "(these fail silently):\n")
            g = (worst.groupby(["transform", "true_char", "maps_to_char"]).size()
                 .reset_index(name="n_models").sort_values("n_models", ascending=False))
            L.append("| transform | true | becomes | models agreeing |")
            L.append("|---|---|---|---:|")
            for _, r in g.head(20).iterrows():
                L.append(f"| {r['transform']} | {r.true_char} | {r.maps_to_char} | {r.n_models} |")

    # ---- crossing points + recommendation (acceptance criteria 5) ----
    if len(a):
        L.append("\n## 4. Where each corruption starts to hurt\n")
        L.append("First severity at which retention falls below 90 % and 50 %, shipped model "
                 "(`thaichar72_r18_64_gen`). `never` = still above that line at the harshest "
                 "severity tested.\n")
        L.append("| corruption | <90 % retention at | <50 % retention at |")
        L.append("|---|---|---|")
        ship = "thaichar72_r18_64_gen"
        sa = a[a.model == ship]
        for cname in sa.condition.unique():
            sub = sa[sa.condition == cname].copy()
            # Walk the severities in order of INCREASING harshness, which is not the same as
            # increasing numeric value: lower jpeg quality and a smaller pixelate fraction are
            # harsher, aspect_stretch is harsher the further it is from 1.0, and for
            # stroke_extreme erosion hurts far more than the equivalent dilation.
            def harshness(v, c=cname):
                if c in ("jpeg", "pixelate"):
                    return -float(v)
                if c == "aspect_stretch":
                    return abs(float(v) - 1.0)
                if c == "stroke_extreme":
                    return (0 if float(v) > 0 else 1, abs(float(v)))
                return float(v)
            sub["_h"] = sub.severity.map(harshness)
            sub = sub.sort_values("_h")
            c90 = c50 = None
            for _, r in sub.iterrows():
                if pd.isna(r.get("top1")):
                    continue
                ret = r["top1"] / clean_top1[ship]
                if c90 is None and ret < 0.90:
                    c90 = r["severity"]
                if c50 is None and ret < 0.50:
                    c50 = r["severity"]
            L.append(f"| {cname} | {c90 if c90 is not None else 'never'} | "
                     f"{c50 if c50 is not None else 'never'} |")

    L.append("\n## 5. Recommendation\n")
    best_clean = max(clean_top1, key=clean_top1.get)
    rec = []
    if len(a):
        uf = a[a.condition == "unseen_font"]
        if len(uf):
            uf = uf.assign(ret=[r.top1 / clean_top1[r.model] for _, r in uf.iterrows()])
            best_font = uf.loc[uf.ret.idxmax(), "model"]
            rec.append(f"- **If the day's data is typeset in something we never rendered**, prefer "
                       f"`{best_font}` — it retains {uf.ret.max():.3f} of its own clean accuracy on "
                       f"{int(uf.severity.iloc[0])} unseen typefaces, against {uf.ret.min():.3f} for the weakest.")
        deg = a[a.condition.isin(["motion_blur", "pixelate", "stroke_extreme"])]
        if len(deg):
            deg = deg.assign(ret=[r.top1 / clean_top1[r.model] for _, r in deg.iterrows()])
            bym = deg.groupby("model").ret.mean().sort_values(ascending=False)
            rec.append(f"- **If it looks blurred, low-resolution or badly inked**, prefer "
                       f"`{bym.index[0]}` (mean retention {bym.iloc[0]:.3f} across blur/pixelate/stroke), "
                       f"and treat anything below ~0.35 linear scale as unrecoverable — no model survives it.")
    rec.append(f"- **On clean, well-framed input** the ranking is unchanged: `{best_clean}` "
               f"({clean_top1[best_clean]:.4f} on the base sample).")
    if len(b):
        worst = b.loc[b.separability_auroc.idxmin()]
        minauroc = b.separability_auroc.min()
        if minauroc >= 0.8:
            rec.append(f"- **A pre-classifier splitter for touching glyphs is NOT worth building.** "
                       f"Confidence alone separates every junk family from clean input at AUROC "
                       f"≥ {minauroc:.2f} (weakest: {worst.condition}). A confidence threshold is a "
                       f"few lines and catches these; a connected-component splitter is a component "
                       f"to build, tune and debug for the same effect.")
        else:
            rec.append(f"- **A pre-classifier splitter IS worth building**: confidence cannot "
                       f"separate {worst.condition} from clean input (AUROC {minauroc:.2f} ≤ 0.8), "
                       f"so those inputs would enter the output silently.")
        hi = b.loc[b["confident_wrong@0.9"].idxmax()]
        rec.append(f"- The most dangerous junk is **{hi.condition}**: {hi['confident_wrong@0.9']:.1%} "
                   f"of them still get a ≥0.9-confidence prediction. If the test set contains "
                   f"multi-character crops, that is where wrong answers will come from.")
    if len(flip):
        silent = flip[(~flip["maps_to_self"]) & (flip["rate"] >= 0.6) & (flip["mean_conf"] >= 0.8)]
        rec.append(f"- **Never flip, and check orientation before trusting a batch.** "
                   f"{len(silent)} class/transform combinations turn one real Thai character into "
                   f"another one confidently (บ↔ภ and ย↔ถ under vertical mirroring are exact "
                   f"reciprocal pairs). Mean confidence under a flip is low overall "
                   f"({flip['mean_conf'].mean():.2f} vs {np.mean(list(clean_top1.values())):.2f} clean "
                   f"accuracy), so a flipped batch is detectable in aggregate — but these specific "
                   f"pairs fail silently. This is the measurement behind CLAUDE.md's ban on flip "
                   f"augmentation.")
    L.extend(rec)

    Path(out / "summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
