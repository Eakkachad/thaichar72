"""Import the `dataUpdate` synthetic corpus into the project's extra-data format.

`dataUpdate` ships 98,071 synthetic Thai glyphs over 50 class folders, but only
35 of those classes exist in this project's 72-class universe (see
`reports/03-DATA-AUDIT-2026-09-22.md` §2.1). This script keeps the in-universe
ones and emits the `(index.csv, glyphs.npz)` pair the engine expects in
`extra_train_index` / `extra_train_cache`.

The out-of-universe classes are written to a *separate* index (`--out-index-extra`)
so they can still be used as additional pretraining classes later without ever
leaking into a 72-way run.

Usage:
    uv run python scripts/prep_dataupdate.py \
        --root ~/work/incoming/synth/dataset \
        --out-dir data/dataupdate
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

from thaichar.classes import CLASS_CODES, code_to_char, code_to_index
from thaichar.synth import build_synth_cache

UNIVERSE = set(CLASS_CODES)


def _load_one(full: str) -> tuple[np.ndarray, int, int, float]:
    """Same binarisation as thaichar.data._process_one, plus ink fraction."""
    from PIL import Image

    img = Image.open(full).convert("L")
    arr = np.asarray(img, dtype=np.uint8)
    binarised = np.where(arr > 127, np.uint8(255), np.uint8(0))
    ink = float((binarised == 0).mean())
    return binarised, binarised.shape[0], binarised.shape[1], ink


def code_of(folder: str) -> int | None:
    m = re.match(r"^(\d+)", folder)
    return int(m.group(1)) if m else None


def collect(root: str) -> pd.DataFrame:
    """Walk dataset/{train,val}/<class folder>/*.jpg into a flat frame."""
    rows = []
    for split in ("train", "val"):
        sub = os.path.join(root, split)
        if not os.path.isdir(sub):
            continue
        for folder in sorted(os.listdir(sub)):
            code = code_of(folder)
            if code is None:
                continue
            d = os.path.join(sub, folder)
            for fn in sorted(os.listdir(d)):
                if not fn.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue
                rows.append(
                    {
                        "abs_path": os.path.join(d, fn),
                        "path": f"dataupdate/{split}/{folder}/{fn}",
                        "code": code,
                        "src_split": split,
                        # hw_ = real handwriting pushed into the domain, ft_ = font render
                        "source": "hw" if fn.startswith("hw_") else "ft",
                    }
                )
    return pd.DataFrame(rows)


def attach_font_family(df: pd.DataFrame, root: str) -> pd.DataFrame:
    """Merge the shipped manifest so each row carries its source font / writer."""
    man_path = os.path.join(root, "manifest.csv")
    if not os.path.exists(man_path):
        df["source_name"] = ""
        return df
    man = pd.read_csv(man_path)
    man["key"] = man["path"].astype(str).str.replace("\\", "/", regex=False)
    df["key"] = df["path"].str.replace("dataupdate/", "", regex=False)
    merged = df.merge(man[["key", "source_name"]], on="key", how="left")
    merged["source_name"] = merged["source_name"].fillna("")
    return merged.drop(columns=["key"])


def family_of(source_name: str) -> str:
    """Google-Fonts file name -> family, so a family never straddles a split."""
    stem = os.path.splitext(str(source_name))[0]
    stem = re.sub(r"\[.*?\]", "", stem)
    return re.split(r"[-_]", stem)[0] if stem else ""


def build(df: pd.DataFrame, out_index: str, out_cache: str, workers: int) -> None:
    if df.empty:
        print(f"  (nothing to write for {out_index})")
        return
    with Pool(processes=workers) as pool:
        results = pool.map(_load_one, df["abs_path"].tolist())

    images = [r[0] for r in results]
    df = df.copy()
    df["height"] = [r[1] for r in results]
    df["width"] = [r[2] for r in results]
    df["ink_frac"] = [r[3] for r in results]
    df["char"] = df["code"].map(code_to_char)
    df["family"] = df["source_name"].map(family_of)

    os.makedirs(os.path.dirname(out_index) or ".", exist_ok=True)
    cols = ["path", "label", "code", "char", "width", "height", "ink_frac",
            "source", "source_name", "family", "src_split"]
    df[[c for c in cols if c in df.columns]].to_csv(out_index, index=False)
    build_synth_cache(images, df["path"].tolist(), out_cache)
    print(f"  index -> {out_index}  ({len(df)} rows)")
    print(f"  cache -> {out_cache}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="extracted dataUpdate dataset/ dir")
    ap.add_argument("--out-dir", default="data/dataupdate")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    root = os.path.expanduser(args.root)
    df = collect(root)
    print(f"found {len(df)} images across {df['code'].nunique()} class folders")

    df = attach_font_family(df, root)

    in_uni = df[df["code"].isin(UNIVERSE)].copy()
    out_uni = df[~df["code"].isin(UNIVERSE)].copy()
    in_uni["label"] = in_uni["code"].map(code_to_index)
    # out-of-universe rows get labels beyond NUM_CLASSES, assigned densely
    extra_codes = sorted(out_uni["code"].unique())
    base = len(CLASS_CODES)
    out_uni["label"] = out_uni["code"].map(
        {c: base + i for i, c in enumerate(extra_codes)}
    )

    print(f"in-universe : {len(in_uni)} images / {in_uni['code'].nunique()} classes")
    print(f"out-of-univ : {len(out_uni)} images / {out_uni['code'].nunique()} classes "
          f"(codes {extra_codes})")
    print("source mix  :", in_uni["source"].value_counts().to_dict())
    print("font families:", in_uni.loc[in_uni['source'] == 'ft', 'family'].nunique())

    os.makedirs(args.out_dir, exist_ok=True)

    # dataUpdate ships its own train/val split. Only its `train` side is ever used for
    # training; its `val` side is kept aside untouched as a probe for the 35 starved
    # classes, whose real validation sets are 1-31 images and therefore uninformative.
    # The probe is synthetic, so it is a domain proxy, not a substitute for real val --
    # it is reported separately and never used to select a model.
    uni_train = in_uni[in_uni["src_split"] == "train"]
    uni_probe = in_uni[in_uni["src_split"] == "val"]
    print(f"\nin-universe train : {len(uni_train)}")
    print(f"in-universe probe : {len(uni_probe)} (held out, never trained on)")

    print("\n[in-universe / train]")
    build(uni_train, os.path.join(args.out_dir, "index.csv"),
          os.path.join(args.out_dir, "glyphs.npz"), args.workers)
    print("\n[in-universe / held-out probe]")
    build(uni_probe, os.path.join(args.out_dir, "index_probe.csv"),
          os.path.join(args.out_dir, "glyphs_probe.npz"), args.workers)
    print("\n[out-of-universe]")
    build(out_uni, os.path.join(args.out_dir, "index_oov.csv"),
          os.path.join(args.out_dir, "glyphs_oov.npz"), args.workers)


if __name__ == "__main__":
    sys.exit(main())
