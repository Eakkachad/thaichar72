"""Merge several (index.csv, glyphs.npz) extra-data pairs into one pair.

The engine accepts a single `extra_train_index` / `extra_train_cache`, but the
generalisation study wants stage-1 pretraining on *several* synthetic sources at
once (our own 206-font render plus the `dataUpdate` corpus). This concatenates
them, keeping paths unique so the cache lookup stays 1:1.

Usage:
    uv run python scripts/merge_extra_sources.py \
        --inputs data/synth_big/index.csv data/dataupdate/index.csv \
        --out-index data/synth_merged/index.csv \
        --out-cache data/synth_merged/glyphs.npz
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from thaichar.data import load_cache
from thaichar.synth import build_synth_cache


def cache_path_for(index_path: str) -> str:
    """Sibling .npz for an index.csv (index.csv -> glyphs.npz, else <stem>.npz)."""
    d = os.path.dirname(index_path)
    for cand in ("glyphs.npz", "glyphs_synth.npz"):
        p = os.path.join(d, cand)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"no glyph cache found next to {index_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True, help="index.csv paths")
    ap.add_argument("--caches", nargs="*", default=None,
                    help="matching .npz paths (default: sibling of each index)")
    ap.add_argument("--out-index", required=True)
    ap.add_argument("--out-cache", required=True)
    args = ap.parse_args()

    caches = args.caches or [cache_path_for(p) for p in args.inputs]
    if len(caches) != len(args.inputs):
        raise SystemExit("--caches must have the same length as --inputs")

    frames: list[pd.DataFrame] = []
    images: list[np.ndarray] = []
    paths: list[str] = []

    for idx_path, cache_path in zip(args.inputs, caches):
        df = pd.read_csv(idx_path)
        cache = load_cache(cache_path)
        by_path = {p: i for i, p in enumerate(cache.paths)}
        tag = os.path.basename(os.path.dirname(idx_path)) or "src"

        kept = 0
        new_paths = []
        for p in df["path"].astype(str):
            i = by_path.get(p)
            if i is None:
                new_paths.append(None)
                continue
            # prefix keeps paths unique across sources
            np_ = f"{tag}::{p}"
            images.append(cache[i])
            paths.append(np_)
            new_paths.append(np_)
            kept += 1
        df["path"] = new_paths
        df = df[df["path"].notna()].copy()
        df["source_set"] = tag
        frames.append(df)
        print(f"{idx_path}: {kept}/{len(by_path)} glyphs taken  (tag={tag})")

    merged = pd.concat(frames, ignore_index=True)
    os.makedirs(os.path.dirname(args.out_index) or ".", exist_ok=True)
    merged.to_csv(args.out_index, index=False)
    build_synth_cache(images, paths, args.out_cache)

    print(f"\nmerged rows   : {len(merged)}")
    print(f"classes       : {merged['label'].nunique()}")
    print(f"index -> {args.out_index}")
    print(f"cache -> {args.out_cache}")


if __name__ == "__main__":
    main()
