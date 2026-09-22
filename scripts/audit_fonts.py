"""Audit a font directory for real glyph coverage of the 72-class universe.

`thaichar.synth.FontLibrary` decides a font supports a character by asking PIL for a
non-empty mask. A missing glyph rendered as the `.notdef` box is a non-empty mask, so a
font can silently contribute boxes for rare glyphs. This script checks two things per font:

1. cmap coverage  -- is the codepoint mapped at all (fontTools)?
2. .notdef shape  -- does the rendered bitmap equal the bitmap of a codepoint that is
                     certainly absent (U+FFFF)? Catches fonts mapping to a box glyph.

Writes reports/data/font_audit.csv (one row per font x char) plus a per-font summary, and
prints the fonts that should be dropped from the render pool.

    uv run --no-sync python scripts/audit_fonts.py --fonts-dir data/fonts_all
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from fontTools.ttLib import TTFont

from thaichar.classes import CLASS_CODES, code_to_char
from thaichar.synth import COMBINING_MARK_CODES, UPPER_MARK_CODES

SENTINEL = "￿"  # never mapped -> always renders .notdef
BASES = ["อ", "ก", "ป"]


def _bitmap(font: ImageFont.FreeTypeFont, text: str, size: int = 200) -> np.ndarray | None:
    im = Image.new("L", (size, size), 255)
    ImageDraw.Draw(im).text((size // 4, size // 4), text, font=font, fill=0)
    arr = np.asarray(im, dtype=np.uint8)
    rows, cols = np.where(arr < 128)
    if len(rows) == 0:
        return None
    return arr[rows.min(): rows.max() + 1, cols.min(): cols.max() + 1].copy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fonts-dir", default="data/fonts_all")
    ap.add_argument("--out-csv", default="reports/data/font_audit.csv")
    ap.add_argument("--out-summary", default="reports/data/font_audit_summary.csv")
    ap.add_argument("--max-missing", type=int, default=6,
                    help="Fonts missing more than this many of the 72 glyphs are flagged for dropping.")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.fonts_dir, "*.ttf")))
    rows: list[dict] = []

    for p in paths:
        stem = os.path.splitext(os.path.basename(p))[0]
        try:
            tt = TTFont(p, fontNumber=0, lazy=True)
            cmap = set(tt.getBestCmap().keys())
            tt.close()
        except Exception as e:  # noqa: BLE001
            print(f"[warn] cannot parse {stem}: {e}")
            cmap = set()
        try:
            pil = ImageFont.truetype(p, 100)
        except Exception as e:  # noqa: BLE001
            print(f"[warn] cannot load {stem}: {e}")
            continue

        notdef = _bitmap(pil, SENTINEL)
        base_ok = all(ord(b) in cmap for b in BASES)

        for code in CLASS_CODES:
            ch = code_to_char(code)
            in_cmap = all(ord(c) in cmap for c in ch)
            bm = _bitmap(pil, ch)
            looks_notdef = (
                bm is not None
                and notdef is not None
                and bm.shape == notdef.shape
                and np.array_equal(bm, notdef)
            )
            rows.append(dict(
                font=stem, code=code, char=ch,
                is_mark=code in COMBINING_MARK_CODES,
                is_upper_mark=code in UPPER_MARK_CODES,
                in_cmap=in_cmap, blank=bm is None, looks_notdef=bool(looks_notdef),
                ok=bool(in_cmap and bm is not None and not looks_notdef),
            ))
        if not base_ok:
            print(f"[warn] {stem}: missing one of the mark bases {BASES}")

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    df.to_csv(args.out_csv, index=False)

    summary = (
        df.groupby("font")
          .agg(n_ok=("ok", "sum"),
               n_missing_cmap=("in_cmap", lambda s: int((~s).sum())),
               n_notdef=("looks_notdef", "sum"),
               n_blank=("blank", "sum"))
          .reset_index()
    )
    summary["n_bad"] = len(CLASS_CODES) - summary["n_ok"]
    summary = summary.sort_values("n_bad", ascending=False)
    summary.to_csv(args.out_summary, index=False)

    drop = summary[summary["n_bad"] > args.max_missing]
    print(f"fonts audited : {len(summary)}")
    print(f"fully covering 72 classes : {int((summary['n_bad'] == 0).sum())}")
    print(f"flagged for dropping (> {args.max_missing} bad glyphs) : {len(drop)}")
    if len(drop):
        print(drop.to_string(index=False))
    per_class = df.groupby(["code", "char"])["ok"].sum().reset_index().sort_values("ok")
    print("\nleast-supported classes (n fonts that can render them):")
    print(per_class.head(12).to_string(index=False))
    print(f"\nwrote {args.out_csv} and {args.out_summary}")


if __name__ == "__main__":
    main()
