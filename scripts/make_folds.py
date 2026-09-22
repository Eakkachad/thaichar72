#!/usr/bin/env python
"""Write k stratified fold split files so every image can get an out-of-fold prediction.

Label-noise hunting needs a prediction for EVERY image from a model that never trained on
it. The strat and doc partitions only cover ~20% each and overlap, so neither can score the
whole corpus honestly. k folds give 100% coverage.

Emits data/splits/fold{k}_of{K}.csv, each a copy of the base split file whose `split`
column is train everywhere except the held-out fold.

    uv run --no-sync python scripts/make_folds.py --k 5
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-file", default="data/splits/split_seed42_v2.csv")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="data/splits")
    args = ap.parse_args()

    df = pd.read_csv(args.split_file)
    rng = np.random.default_rng(args.seed)

    # stratified by label so every fold sees every class it can
    fold = np.empty(len(df), dtype=int)
    for lab, idx in df.groupby("label").groups.items():
        idx = np.asarray(idx)
        perm = rng.permutation(len(idx))
        fold[idx[perm]] = np.arange(len(idx)) % args.k

    df = df.assign(fold=fold)
    base = os.path.join(args.out_dir, f"folds_k{args.k}.csv")
    df.to_csv(base, index=False)
    print(f"fold assignment -> {base}")

    for f in range(args.k):
        sub = df.copy()
        sub["split"] = np.where(sub.fold == f, "val", "train")
        p = os.path.join(args.out_dir, f"fold{f}_of{args.k}.csv")
        sub.to_csv(p, index=False)
        n_val = int((sub.split == "val").sum())
        n_cls = int(sub.loc[sub.split == "val", "label"].nunique())
        print(f"  fold {f}: val {n_val:,} rows, {n_cls} classes -> {p}")

    cover = df.groupby("label").size()
    thin = cover[cover < args.k]
    if len(thin):
        print(f"\nnote: {len(thin)} class(es) have fewer than {args.k} images, so they are "
              f"absent from some folds' val side (unavoidable): "
              f"{ {int(k): int(v) for k, v in thin.items()} }")


if __name__ == "__main__":
    main()
