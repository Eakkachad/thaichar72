"""Data cleaning, splitting, caching, and torch Dataset for Thai glyphs."""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Sampler

from thaichar.classes import CLASS_CODES, code_to_index, code_to_char, category
from thaichar.transforms import (
    encode_channels,
    fit_to_square,
    geometry_features,
    pad_to_square_canvas,
    resize_square,
)

logger = logging.getLogger(__name__)

# ImageNet stats (for 3-channel normalisation)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ---------------------------------------------------------------------------
# Deliverable 1: Clean Index
# ---------------------------------------------------------------------------

def build_clean_index(
    files_csv: str = "reports/eda/files.csv",
    duplicates_json: str = "reports/eda/duplicates.json",
) -> pd.DataFrame:
    """Build a cleaned index from the EDA outputs.

    Cleaning policy (applied in order, each logged):
      1. Drop rows where ``ok == False``.
      2. Drop files whose basename starts with ``Copy of``.
      3. Drop ALL members of md5 groups flagged ``cross_class``.
      4. Within-class exact duplicates (same md5 + same code): keep the
         lexicographically-first path, drop the rest.

    Returns
    -------
    pd.DataFrame
        Columns: path, code, label, char, category, group, doc_id, prefix,
        subset, width, height, ink_frac
    """
    df = pd.read_csv(files_csv)
    n0 = len(df)

    # --- Rule 1: ok == False ---
    mask_ok = df["ok"].astype(bool)
    n_bad = int((~mask_ok).sum())
    df = df[mask_ok].copy()
    logger.info("Rule 1 (ok==False):  dropped %d rows", n_bad)

    # --- Rule 2: filename starts with "Copy of" ---
    is_copy = df["path"].apply(lambda p: os.path.basename(p).startswith("Copy of"))
    n_copy = int(is_copy.sum())
    df = df[~is_copy].copy()
    logger.info("Rule 2 (Copy of):    dropped %d rows", n_copy)

    # --- Rule 3: cross-class md5 groups ---
    with open(duplicates_json) as f:
        dup_data = json.load(f)
    md5_groups = dup_data["md5_groups"]
    cross_md5s = {g["md5"] for g in md5_groups if g["cross_class"]}
    mask_cross = df["md5"].isin(cross_md5s)
    n_cross = int(mask_cross.sum())
    df = df[~mask_cross].copy()
    logger.info("Rule 3 (cross-class): dropped %d rows", n_cross)

    # --- Rule 4: within-class exact duplicates (same md5 + code) ---
    df = df.sort_values("path")
    before_dedup = len(df)
    df = df.drop_duplicates(subset=["md5", "code"], keep="first")
    n_within = before_dedup - len(df)
    logger.info("Rule 4 (within-dup): dropped %d rows", n_within)

    total_dropped = n0 - len(df)
    logger.info(
        "Total dropped: %d  |  Remaining: %d (from %d)", total_dropped, len(df), n0
    )

    # Build label column
    df = df.copy()
    df["label"] = df["code"].map(code_to_index)
    df["char"] = df["code"].map(code_to_char)
    df["category"] = df["code"].map(category)

    keep_cols = [
        "path", "code", "label", "char", "category",
        "group", "doc_id", "prefix", "subset",
        "width", "height", "ink_frac",
    ]
    df = df[keep_cols].reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Deliverable 1: Splits
# ---------------------------------------------------------------------------

def make_stratified_split(
    df: pd.DataFrame,
    seed: int = 42,
    val_frac: float = 0.2,
) -> pd.Series:
    """Per-class stratified split (deterministic).

    For each class (sorted by code), sort paths, shuffle with seeded rng,
    then assign the first ``n_val`` to ``"val"``, rest to ``"train"``.

    ``n_val`` rules: n>=5 → round(val_frac*n); n=4→1; n=3→1; n=2→1; n=1→0.

    Returns
    -------
    pd.Series
        Index aligned with *df*, values ``"train"`` or ``"val"``.
    """
    rng = np.random.default_rng(seed)
    split = pd.Series("train", index=df.index)

    zero_val_classes: list[int] = []

    for code in sorted(df["code"].unique()):
        mask = df["code"] == code
        idx = df.loc[mask].sort_values("path").index.tolist()
        rng.shuffle(idx)
        n = len(idx)

        if n >= 5:
            n_val = round(val_frac * n)
        elif n == 4:
            n_val = 1
        elif n == 3:
            n_val = 1
        elif n == 2:
            n_val = 1
        else:  # n == 1
            n_val = 0
            zero_val_classes.append(code)

        split.iloc[split.index.get_indexer(idx[:n_val])] = "val"

    if zero_val_classes:
        logger.info(
            "Classes with 0 val samples (stratified): %s",
            zero_val_classes,
        )

    return split


def make_doc_split(
    df: pd.DataFrame,
    seed: int = 42,
    n_holdout_docs: int = 4,
) -> pd.Series:
    """Document-level split: hold out *n_holdout_docs* doc_ids for validation.

    Picks the combination (from 200 random draws) whose held-out share of
    images is closest to 20%.

    Returns
    -------
    pd.Series
        Index aligned with *df*, values ``"train"`` or ``"val"``.
    """
    rng = np.random.default_rng(seed)
    all_docs = sorted(df["doc_id"].unique())
    n_total = len(df)
    target = 0.20

    best_docs = None
    best_diff = float("inf")

    for _ in range(200):
        chosen = rng.choice(all_docs, size=n_holdout_docs, replace=False)
        n_held = int(df["doc_id"].isin(chosen).sum())
        diff = abs(n_held / n_total - target)
        if diff < best_diff:
            best_diff = diff
            best_docs = list(chosen)

    logger.info("Doc-split holdout doc_ids: %s  (%.2f%% of data)",
                sorted(best_docs), 100 * df["doc_id"].isin(best_docs).sum() / n_total)

    split = pd.Series("train", index=df.index)
    split[df["doc_id"].isin(best_docs)] = "val"
    return split


def write_splits(
    df: pd.DataFrame,
    out_dir: str = "data/splits",
    seeds: tuple[int, ...] = (42, 0, 1),
) -> None:
    """Write split CSV files for each seed.

    Each file has all clean-index columns plus ``split`` (stratified) and
    ``doc_split``.
    """
    os.makedirs(out_dir, exist_ok=True)
    for seed in seeds:
        out = df.copy()
        out["split"] = make_stratified_split(df, seed=seed)
        out["doc_split"] = make_doc_split(df, seed=seed)

        # Overlap assertions
        train_strat = set(out.index[out["split"] == "train"])
        val_strat = set(out.index[out["split"] == "val"])
        overlap_strat = len(train_strat & val_strat)

        train_doc = set(out.index[out["doc_split"] == "train"])
        val_doc = set(out.index[out["doc_split"] == "val"])
        overlap_doc = len(train_doc & val_doc)

        assert overlap_strat == 0, f"Stratified overlap != 0 for seed {seed}"
        assert overlap_doc == 0, f"Doc-split overlap != 0 for seed {seed}"
        print(f"  seed {seed}: stratified overlap = {overlap_strat}, "
              f"doc overlap = {overlap_doc}")

        path = os.path.join(out_dir, f"split_seed{seed}.csv")
        out.to_csv(path, index=False)
        logger.info("Wrote %s (%d rows)", path, len(out))


# ---------------------------------------------------------------------------
# Deliverable 2: Image Cache
# ---------------------------------------------------------------------------

def _process_one(args: tuple[str, str]) -> tuple[np.ndarray, int, int]:
    """Load, greyscale, binarise one image. Called by pool workers."""
    path, base_dir = args
    full = os.path.join(base_dir, path) if base_dir else path
    img = Image.open(full).convert("L")
    arr = np.asarray(img, dtype=np.uint8)
    # binarise: > 127 → 255 (background), else → 0 (ink)
    binarised = np.where(arr > 127, np.uint8(255), np.uint8(0))
    return binarised, binarised.shape[0], binarised.shape[1]


def build_cache(
    df: pd.DataFrame,
    out: str = "data/cache/glyphs.npz",
    n_workers: int = 8,
    base_dir: str = "",
) -> None:
    """Build a flat .npz cache of binarised glyph images.

    Stores: ``data`` (flat uint8), ``offsets``, ``widths``, ``heights``,
    ``paths``.
    """
    os.makedirs(os.path.dirname(out), exist_ok=True)
    paths = df["path"].tolist()

    args_list = [(p, base_dir) for p in paths]

    with Pool(processes=n_workers) as pool:
        results = pool.map(_process_one, args_list)

    flat_parts: list[np.ndarray] = []
    offsets = np.zeros(len(results), dtype=np.int64)
    widths = np.zeros(len(results), dtype=np.int32)
    heights = np.zeros(len(results), dtype=np.int32)

    offset = 0
    for i, (arr, h, w) in enumerate(results):
        flat_parts.append(arr.ravel())
        offsets[i] = offset
        widths[i] = w
        heights[i] = h
        offset += h * w

    data = np.concatenate(flat_parts)
    path_arr = np.array(paths, dtype=str)

    np.savez(
        out,
        data=data,
        offsets=offsets,
        widths=widths,
        heights=heights,
        paths=path_arr,
    )
    mb = os.path.getsize(out) / (1024 * 1024)
    logger.info("Cache written to %s  (%.1f MB, %d images)", out, mb, len(paths))


@dataclass
class GlyphCache:
    """Fast in-memory glyph cache loaded from an .npz file."""

    data: np.ndarray      # flat uint8
    offsets: np.ndarray    # int64
    widths: np.ndarray     # int32
    heights: np.ndarray    # int32
    paths: list[str]

    def __len__(self) -> int:
        return len(self.offsets)

    def __getitem__(self, i: int) -> np.ndarray:
        """Return the i-th glyph as an ndarray[H, W] uint8."""
        h = int(self.heights[i])
        w = int(self.widths[i])
        start = int(self.offsets[i])
        end = start + h * w
        return self.data[start:end].reshape(h, w)


def load_cache(path: str = "data/cache/glyphs.npz") -> GlyphCache:
    """Load a glyph cache from disk."""
    npz = np.load(path, allow_pickle=False)
    return GlyphCache(
        data=npz["data"],
        offsets=npz["offsets"],
        widths=npz["widths"],
        heights=npz["heights"],
        paths=npz["paths"].tolist(),
    )


@dataclass
class MultiCache:
    """Multi-source glyph cache mapping paths across multiple GlyphCache instances."""

    caches: list[GlyphCache]
    path_to_entry: dict[str, tuple[int, int]]  # path -> (cache_idx, item_idx)
    paths: list[str]

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> np.ndarray:
        p = self.paths[i]
        c_idx, item_idx = self.path_to_entry[p]
        return self.caches[c_idx][item_idx]

    def get_by_path(self, path: str) -> np.ndarray:
        c_idx, item_idx = self.path_to_entry[path]
        return self.caches[c_idx][item_idx]


def merge_sources(
    sources: Sequence[tuple[pd.DataFrame, GlyphCache]],
) -> tuple[pd.DataFrame, MultiCache]:
    """Merge multiple (df, cache) sources into a unified DataFrame and MultiCache.

    Enables real train split + synthetic glyphs in a single DataLoader without
    re-concatenating cache arrays on disk.

    Parameters
    ----------
    sources : Sequence[tuple[pd.DataFrame, GlyphCache]]
        List of (split_df, cache) pairs.

    Returns
    -------
    tuple[pd.DataFrame, MultiCache]
        Merged DataFrame and unified MultiCache.
    """
    dfs: list[pd.DataFrame] = []
    caches: list[GlyphCache] = []
    paths: list[str] = []
    path_to_entry: dict[str, tuple[int, int]] = {}

    for cache_idx, (df, cache) in enumerate(sources):
        dfs.append(df)
        caches.append(cache)
        for item_idx, path in enumerate(cache.paths):
            paths.append(path)
            path_to_entry[path] = (cache_idx, item_idx)

    merged_df = pd.concat(dfs, ignore_index=True)
    multi_cache = MultiCache(caches=caches, path_to_entry=path_to_entry, paths=paths)
    return merged_df, multi_cache


# ---------------------------------------------------------------------------
# Deliverable 4: Torch Dataset
# ---------------------------------------------------------------------------

class ThaiGlyphDataset(Dataset):
    """PyTorch dataset backed by split DataFrame + GlyphCache (or MultiCache).

    Parameters
    ----------
    split_df : pd.DataFrame | Sequence[tuple[pd.DataFrame, GlyphCache]]
        Subset of the clean index (e.g. train rows only), or list of (df, cache) pairs.
    cache : GlyphCache | MultiCache | None
        Pre-loaded glyph cache (optional if split_df is a list of sources).
    size : int
        Square output size.
    channel_mode : str
        'gray3' (replicate grey x3 + ImageNet norm), 'gray1' (1-channel [0, 1]),
        or 'onoff' (3 channels: ink mask, distance transform, Sobel edges).
    transform : callable | None
        Optional augmentation applied to the square uint8 canvas *before* resizing.
    return_geometry : bool
        Whether to return geometry features.
    margin : float
        Fractional padding margin around the glyph canvas.
    channels : int | None
        Deprecated alias for channel_mode (3 -> 'gray3', 1 -> 'gray1').
    """

    def __init__(
        self,
        split_df: pd.DataFrame | Sequence[tuple[pd.DataFrame, GlyphCache]],
        cache: GlyphCache | MultiCache | None = None,
        size: int = 64,
        channel_mode: str = "gray3",
        transform: Callable | None = None,
        return_geometry: bool = True,
        margin: float = 0.1,
        channels: int | None = None,
    ) -> None:
        if cache is None and isinstance(split_df, (list, tuple)):
            split_df, cache = merge_sources(split_df)

        if channels is not None:
            channel_mode = "gray3" if channels == 3 else "gray1"

        self.df = split_df.reset_index(drop=True)
        self.cache = cache
        self.size = size
        self.channel_mode = channel_mode
        self.channels = 3 if channel_mode in ("gray3", "onoff") else 1
        self.transform = transform
        self.return_geometry = return_geometry
        self.margin = margin

        # Build path -> cache-index map
        self._cache_idx = {p: i for i, p in enumerate(cache.paths)}

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        cache_i = self._cache_idx[row["path"]]
        img_u8 = self.cache[cache_i]  # HxW uint8

        # 1. Pad to square canvas
        canvas = pad_to_square_canvas(img_u8, margin=self.margin)

        # 2. Transform canvas if given
        if self.transform is not None:
            canvas = self.transform(canvas)

        # 3. Resize to square
        square = resize_square(canvas, self.size)

        # 4. Multi-channel encoding
        img_encoded = encode_channels(square, self.channel_mode)
        x = torch.from_numpy(img_encoded)

        y = torch.tensor(int(row["label"]), dtype=torch.int64)

        if self.return_geometry:
            g = torch.from_numpy(
                geometry_features(int(row["height"]), int(row["width"]), float(row["ink_frac"]))
            )
            return x, y, g

        return x, y


def make_loader(
    ds: ThaiGlyphDataset,
    batch_size: int,
    shuffle: bool = False,
    num_workers: int = 0,
    sampler: Sampler | None = None,
) -> DataLoader:
    """Create a DataLoader for a ThaiGlyphDataset."""
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=(shuffle and sampler is None),
        num_workers=num_workers,
        sampler=sampler,
        pin_memory=False,
    )
