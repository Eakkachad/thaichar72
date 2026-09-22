#!/usr/bin/env python
"""On-site evaluation harness: point it at a folder of images, get every candidate model ranked.

Built for the hand-in day, which runs like a hackathon: unseen data, a one-hour window, repeated
runs allowed, best result counts. The job here is to remove every avoidable source of delay and
of silent failure in that hour.

    # labels taken from the folder names, every packaged weight scored and ranked
    uv run --no-sync python scripts/evaluate_folder.py --dir /path/to/test_data

    # no labels available -- just write predictions from the default model
    uv run --no-sync python scripts/evaluate_folder.py --dir /path/to/test_data --no-labels

    # specific checkpoints, in the order you want them tried
    uv run --no-sync python scripts/evaluate_folder.py --dir DIR --ckpt weights/a.pt runs/b/best.pt

Label discovery accepts the shapes this corpus and its relatives actually use: a folder named
by TIS-620 code ("161"), code+char ("161ก"), the bare Thai character, or a manifest CSV with
path/label-ish columns (--labels-csv). Anything it cannot resolve is reported, not guessed.

Polarity is the one input property that cannot be read off the image reliably. `preprocess_image`
decides it from the 1-px border ring, which is right for a glyph with background around it but wrong
for a TIGHT crop where the strokes touch all four edges -- and class mai-ek is a near-solid blob, so
"ink is the minority" does not rescue it either. Measured on 493 held-out glyphs with the shipped
model: correctly bordered input 0.9939, the same glyphs tight-cropped 0.9229, and a dark border
around dark ink 0.4949. So this script does not guess: --polarity both (the default) preprocesses
each image as-is and inverted, runs both, and keeps whichever the model is more confident about.
Costs one extra forward pass and removes the failure mode.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from thaichar.classes import CLASS_CODES, code_to_char, code_to_index  # noqa: E402
from thaichar.engine import pick_device  # noqa: E402
from thaichar.infer import load_checkpoint, preprocess_image  # noqa: E402

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def _load_gray(path: str) -> np.ndarray:
    from PIL import Image
    return np.asarray(Image.open(path).convert("L"), dtype=np.uint8)


def _bordered(a: np.ndarray) -> np.ndarray:
    """Surround the glyph with its own background so the border-ring polarity test is valid."""
    # lightest corner, not the median -- see thaichar.infer._reframe for why the median inverts
    # tight crops and why that also defeated --polarity both
    v = int(np.max([a[0, 0], a[0, -1], a[-1, 0], a[-1, -1]]))
    pad = max(4, min(a.shape[:2]) // 8)
    return np.pad(a, pad, constant_values=v)
CHAR_TO_IDX = {code_to_char(c): i for i, c in enumerate(CLASS_CODES)}


def label_from_name(name: str) -> int | None:
    """'161', '161ก', 'ก', 'class_161' -> class index, else None."""
    m = re.search(r"(\d{2,3})", name)
    if m:
        code = int(m.group(1))
        if code in CLASS_CODES:
            return code_to_index(code)
    for ch in name:
        if ch in CHAR_TO_IDX:
            return CHAR_TO_IDX[ch]
    return None


def discover(root: Path, use_labels: bool) -> pd.DataFrame:
    rows = []
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMG_EXT:
            lab = None
            if use_labels:
                # nearest labelled ancestor, innermost first
                for part in reversed(p.relative_to(root).parts[:-1]):
                    lab = label_from_name(part)
                    if lab is not None:
                        break
                if lab is None:
                    lab = label_from_name(p.stem)
            rows.append({"path": str(p), "rel": str(p.relative_to(root)), "label": lab})
    return pd.DataFrame(rows)


def default_ckpts(root: Path) -> list[str]:
    """Every packaged weight, best first, so the strongest is tried while time remains."""
    # Order matters: the hour is finite, so the strongest candidate is scored first and the
    # label-convention hedges follow. gen = the current recipe on v2 labels; gen_v3labels = the
    # same recipe after 146 human-confirmed label fixes; v1labels = before the DataV2 corrections.
    # Which convention the hidden key follows is unknowable in advance -- so all three go, and the
    # data decides.
    order = ["thaichar72_r18_64_gen.pt", "thaichar72_r18_64_gen_v3labels.pt",
             "thaichar72_resnet18_64.pt", "thaichar72_mnv3small_64_small.pt",
             "thaichar72_resnet18_64_v1labels.pt"]
    out = [str(root / "weights" / n) for n in order if (root / "weights" / n).exists()]
    return out or [str(root / "weights" / "thaichar72_r18_64_gen.pt")]


def _forward(model, X, G, device, batch=256) -> np.ndarray:
    outs = []
    with torch.no_grad():
        for s in range(0, len(X), batch):
            outs.append(model(X[s:s + batch].to(device), G[s:s + batch].to(device)).float().cpu())
    return torch.cat(outs).numpy()


def _softmax_max(logits: np.ndarray) -> np.ndarray:
    e = np.exp(logits - logits.max(1, keepdims=True))
    return (e / e.sum(1, keepdims=True)).max(1)


def _rotation_pass(model, cfg, paths, la: np.ndarray, device, tau: float, margin: float,
                   batch: int) -> tuple[np.ndarray, int]:
    """Re-read only the images the model is unsure about, from a set of rotated copies.

    Rotation is the sharpest fragility the stress suite found (retention 0.52 at 30 deg, 0.10 at
    45 deg -- reports/02-EXPERIMENTS.md section N). Searching rotations recovers 38-75 pt of it.
    The tau/margin guards are what keep it from costing anything on upright input; see
    thaichar.infer.predict_with_rotation_search for the measurements behind both numbers.
    """
    from thaichar.infer import ROTATION_SEARCH_ANGLES, rotate_canvas

    up = _softmax_max(la)
    need = [i for i in range(len(la)) if up[i] < tau]
    if not need:
        return la, 0
    best = up.copy()
    changed = 0
    for deg in ROTATION_SEARCH_ANGLES:
        xs, gs, idx = [], [], []
        for i in need:
            try:
                a = _bordered(rotate_canvas(_load_gray(paths[i]), deg))
                x, g = preprocess_image(a, cfg)
            except Exception:  # noqa: BLE001
                continue
            xs.append(x); gs.append(g); idx.append(i)
        if not xs:
            continue
        lr = _forward(model, torch.cat(xs), torch.cat(gs), device, batch)
        cr = _softmax_max(lr)
        for k, i in enumerate(idx):
            if cr[k] > max(best[i], up[i] + margin):
                la[i] = lr[k]
                best[i] = cr[k]
                changed += 1
    return la, changed


def run_one(ckpt: str, df: pd.DataFrame, device, polarity: str = "both",
            batch: int = 256, rotation_tta: bool = False,
            rot_tau: float = 0.85, rot_margin: float = 0.30) -> tuple[np.ndarray, np.ndarray, dict]:
    model, cfg = load_checkpoint(ckpt, device=device)
    model.eval()
    xs_a, gs_a, xs_b, gs_b, ok_idx, failed = [], [], [], [], [], []
    for i, p in enumerate(df["path"]):
        try:
            a = _bordered(_load_gray(p))
            x, g = preprocess_image(a, cfg)
            if polarity == "both":
                xb, gb = preprocess_image(_bordered(255 - _load_gray(p)), cfg)
                xs_b.append(xb)
                gs_b.append(gb)
            xs_a.append(x)
            gs_a.append(g)
            ok_idx.append(i)
        except Exception as e:  # noqa: BLE001
            failed.append((p, str(e)))
    if not xs_a:
        raise SystemExit("no image could be preprocessed -- check --dir")

    la = _forward(model, torch.cat(xs_a), torch.cat(gs_a), device, batch)
    n_flip = 0
    if polarity == "both":
        lb = _forward(model, torch.cat(xs_b), torch.cat(gs_b), device, batch)
        take_b = _softmax_max(lb) > _softmax_max(la)
        n_flip = int(take_b.sum())
        la[take_b] = lb[take_b]
    n_rot = 0
    if rotation_tta:
        ok_paths = [df["path"].iloc[i] for i in ok_idx]
        la, n_rot = _rotation_pass(model, cfg, ok_paths, la, device, rot_tau, rot_margin, batch)
    return la, np.asarray(ok_idx), {"cfg": cfg, "failed": failed, "n_flipped": n_flip,
                                    "n_rotation_rereads": n_rot}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="folder of test images (searched recursively)")
    ap.add_argument("--ckpt", nargs="*", default=None, help="checkpoints to try (default: all packaged weights)")
    ap.add_argument("--no-labels", action="store_true", help="predictions only, no scoring")
    ap.add_argument("--labels-csv", default=None, help="CSV with a path column and a label/code/char column")
    ap.add_argument("--out-dir", default="outputs/onsite")
    ap.add_argument("--polarity", choices=["both", "asis"], default="both",
                    help="both (default): try the image and its inverse, keep the more confident")
    ap.add_argument("--rotation-tta", action="store_true",
                    help="re-read low-confidence images from rotated copies; buys back 38-75 pt "
                         "when the input is rotated and costs ~0 on upright input "
                         "(reports/02-EXPERIMENTS.md section N). scripts/triage_input.py says "
                         "whether this batch needs it")
    ap.add_argument("--rot-tau", type=float, default=0.85,
                    help="only search rotations below this confidence (default 0.85)")
    ap.add_argument("--rot-margin", type=float, default=0.30,
                    help="only accept a rotated reading if it beats upright by this (default 0.30)")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    device = pick_device(args.device)
    t0 = time.time()

    df = discover(Path(args.dir).expanduser(), not args.no_labels)
    if df.empty:
        raise SystemExit(f"no images found under {args.dir}")

    if args.labels_csv:
        man = pd.read_csv(args.labels_csv)
        keycol = next((c for c in ("path", "file", "filename", "image") if c in man.columns), man.columns[0])
        labcol = next((c for c in ("label", "code", "char", "class") if c in man.columns), None)
        if labcol is None:
            raise SystemExit(f"--labels-csv has no label/code/char/class column: {list(man.columns)}")
        man["_k"] = man[keycol].astype(str).map(lambda s: os.path.basename(str(s)))
        if labcol == "label":
            lut = dict(zip(man["_k"], man[labcol].astype(int)))
        elif labcol == "code":
            lut = {k: code_to_index(int(v)) for k, v in zip(man["_k"], man[labcol]) if int(v) in CLASS_CODES}
        else:
            lut = {k: CHAR_TO_IDX[v] for k, v in zip(man["_k"], man[labcol].astype(str)) if v in CHAR_TO_IDX}
        df["label"] = df["path"].map(lambda p: lut.get(os.path.basename(p)))

    n_lab = int(df["label"].notna().sum()) if "label" in df else 0
    scored = (not args.no_labels) and n_lab > 0
    print(f"images found      : {len(df):,}")
    print(f"labels resolved   : {n_lab:,}" + ("" if scored else "   -> running in prediction-only mode"))
    if scored and n_lab < len(df):
        miss = df[df["label"].isna()]
        print(f"  WARNING: {len(miss)} image(s) have no label and are excluded from scoring, e.g.:")
        for r in miss["rel"].head(3):
            print(f"    {r}")
    if scored:
        present = sorted(df.loc[df["label"].notna(), "label"].astype(int).unique())
        print(f"classes present   : {len(present)} of 72")
        if len(present) < 72:
            absent = [code_to_char(CLASS_CODES[i]) for i in range(72) if i not in present]
            print(f"  not in this set : {' '.join(absent)}")
        vc = df.loc[df["label"].notna(), "label"].value_counts()
        print(f"per-class count   : min {vc.min()}, median {int(vc.median())}, max {vc.max()}"
              f"  -> {'roughly balanced' if vc.max() <= 3 * vc.min() else 'imbalanced'}")

    ckpts = args.ckpt or default_ckpts(root)
    outdir = Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    y = df["label"].to_numpy() if scored else None
    results = []
    for ck in ckpts:
        if not os.path.exists(ck):
            print(f"\n{ck}: MISSING, skipped")
            continue
        t = time.time()
        logits, ok, info = run_one(ck, df, device, polarity=args.polarity,
                                   rotation_tta=args.rotation_tta,
                                   rot_tau=args.rot_tau, rot_margin=args.rot_margin)
        pred = logits.argmax(1)
        name = Path(ck).stem
        sub = df.iloc[ok].copy()
        sub["pred_label"] = pred
        sub["pred_char"] = [code_to_char(CLASS_CODES[i]) for i in pred]
        e = np.exp(logits - logits.max(1, keepdims=True))
        sub["confidence"] = (e / e.sum(1, keepdims=True)).max(1)
        row = {"ckpt": ck, "model": name, "n": len(sub), "seconds": time.time() - t,
               "n_failed": len(info["failed"])}
        if scored:
            yy = sub["label"].to_numpy()
            m = ~pd.isna(yy)
            correct = pred[m] == yy[m].astype(int)
            row["top1"] = float(correct.mean())
            per = {int(c): float(correct[yy[m].astype(int) == c].mean())
                   for c in np.unique(yy[m].astype(int))}
            row["balanced"] = float(np.mean(list(per.values())))
            sub["correct"] = np.where(m, pred == np.where(m, yy, 0), np.nan)
            row["_per_class"] = per
        sub.drop(columns=["_k"], errors="ignore").to_csv(outdir / f"predictions_{name}.csv", index=False)
        results.append(row)
        if info.get("n_flipped"):
            pct = 100 * info["n_flipped"] / max(1, len(sub))
            print(f"  {name}: polarity inverted for {info['n_flipped']} image(s) ({pct:.0f}%)"
                  + ("  <- the whole set looks inverted vs our training data" if pct > 60 else ""))
        if info["failed"]:
            print(f"  {name}: {len(info['failed'])} image(s) failed preprocessing, e.g. {info['failed'][0][1][:70]}")
        if scored:
            print(f"  {name:<44} top1 {row['top1']:.4f}  balanced {row['balanced']:.4f}  ({row['seconds']:.1f}s)")
        else:
            print(f"  {name:<44} {len(sub)} predictions written ({row['seconds']:.1f}s)")

    if not results:
        raise SystemExit("no checkpoint produced a result")

    if scored:
        rank = sorted(results, key=lambda r: r["top1"], reverse=True)
        print("\n=== RANKING (pick the top one) ===")
        for i, r in enumerate(rank, 1):
            print(f"{i}. {r['model']:<44} top1 {r['top1']:.4f}  balanced {r['balanced']:.4f}")
        best = rank[0]
        per = best.pop("_per_class")
        weak = sorted(per.items(), key=lambda kv: kv[1])[:10]
        print(f"\nweakest classes for {best['model']}:")
        for lab, acc in weak:
            print(f"  {code_to_char(CLASS_CODES[lab])}: {acc:.3f}")
        for r in rank[1:]:
            r.pop("_per_class", None)

    summary = {"dir": args.dir, "n_images": len(df), "n_labelled": n_lab,
               "scored": scored, "total_seconds": time.time() - t0, "results": results}
    with open(outdir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    print(f"\nper-image predictions + summary -> {outdir}/   (total {time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
