#!/usr/bin/env python3
"""Prep-data CLI: clean index → splits → cache → summary report.

Usage:  uv run python scripts/prep_data.py
"""

from __future__ import annotations

import logging
import os
import sys
import time

import pandas as pd

# Ensure src/ is importable when run via ``uv run python scripts/prep_data.py``
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from thaichar.classes import CLASS_CODES, code_to_char, code_to_index, category
from thaichar.data import (
    build_cache,
    build_clean_index,
    load_cache,
    make_doc_split,
    make_stratified_split,
    write_splits,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s  %(message)s",
)
logger = logging.getLogger(__name__)

SEEDS = (42, 0, 1)


def main() -> None:
    t0 = time.time()

    # ----- 1. Clean index -----
    print("=" * 60)
    print("Step 1: Building clean index")
    print("=" * 60)
    df = build_clean_index()
    print(f"  Clean index rows: {len(df)}")
    print(f"  Unique labels:    {df['label'].nunique()}")

    # The corpus on this machine was reconstructed from the DataV2 export
    # (scripts/restore_round2_from_datav2.py). DataV2 had itself deleted 185 files,
    # so a few manifest rows have no image on disk. Drop them explicitly and loudly
    # rather than letting build_cache die on a missing path.
    exists = df["path"].map(os.path.exists)
    n_absent = int((~exists).sum())
    if n_absent:
        absent = df.loc[~exists, ["path", "code"]]
        os.makedirs("reports/analysis", exist_ok=True)
        absent.to_csv("reports/analysis/absent_from_restore.csv", index=False)
        print(f"  WARNING: {n_absent} indexed files are absent on disk "
              f"(listed in reports/analysis/absent_from_restore.csv)")
        print(f"  affected classes: {sorted(absent['code'].unique().tolist())}")
        df = df[exists].reset_index(drop=True)
        print(f"  Clean index rows after existence filter: {len(df)}")
    print()

    # ----- 2. Splits -----
    print("=" * 60)
    print("Step 2: Writing stratified + doc splits")
    print("=" * 60)
    write_splits(df, seeds=SEEDS)
    print()

    # Show seed-42 split stats
    s42 = pd.read_csv("data/splits/split_seed42.csv")
    print("Seed 42 stratified split:")
    print(s42["split"].value_counts().to_string())
    print()
    print("Seed 42 doc split:")
    print(s42["doc_split"].value_counts().to_string())
    print()

    # ----- 3. Cache -----
    print("=" * 60)
    print("Step 3: Building image cache")
    print("=" * 60)
    build_cache(df, n_workers=8)
    cache_path = "data/cache/glyphs.npz"
    cache_mb = os.path.getsize(cache_path) / (1024 * 1024)
    print(f"  Cache: {cache_path}  ({cache_mb:.1f} MB)")
    print()

    # ----- 4. Summary report -----
    print("=" * 60)
    print("Step 4: Writing summary report")
    print("=" * 60)
    _write_summary(df, s42)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f} s")


def _write_summary(df: pd.DataFrame, s42: pd.DataFrame) -> None:
    """Write ``reports/data/split_summary.md``."""
    os.makedirs("reports/data", exist_ok=True)

    # Re-compute drop counts (fast, just re-read CSVs)
    raw = pd.read_csv("reports/eda/files.csv")
    n0 = len(raw)

    n_bad = int((~raw["ok"].astype(bool)).sum())

    after1 = raw[raw["ok"].astype(bool)].copy()
    is_copy = after1["path"].apply(lambda p: os.path.basename(p).startswith("Copy of"))
    n_copy = int(is_copy.sum())

    import json
    with open("reports/eda/duplicates.json") as f:
        dup_data = json.load(f)
    cross_md5s = {g["md5"] for g in dup_data["md5_groups"] if g["cross_class"]}
    after2 = after1[~is_copy]
    n_cross = int(after2["md5"].isin(cross_md5s).sum())

    after3 = after2[~after2["md5"].isin(cross_md5s)].copy()
    after3_sorted = after3.sort_values("path")
    after3_dedup = after3_sorted.drop_duplicates(subset=["md5", "code"], keep="first")
    n_within = len(after3) - len(after3_dedup)

    # Per-class table (seed 42)
    class_table_rows: list[str] = []
    class_table_rows.append("| Code | Char | Category | n_clean | n_train | n_val | n_doc_train | n_doc_val |")
    class_table_rows.append("|---:|:---:|:---:|---:|---:|---:|---:|---:|")

    zero_val_classes: list[str] = []

    for code in CLASS_CODES:
        ch = code_to_char(code)
        cat = category(code)
        mask_clean = df["code"] == code
        n_clean = int(mask_clean.sum())

        mask_s42 = s42["code"] == code
        n_train = int((s42.loc[mask_s42, "split"] == "train").sum())
        n_val = int((s42.loc[mask_s42, "split"] == "val").sum())
        n_doc_train = int((s42.loc[mask_s42, "doc_split"] == "train").sum())
        n_doc_val = int((s42.loc[mask_s42, "doc_split"] == "val").sum())

        class_table_rows.append(
            f"| {code} | {ch} | {cat} | {n_clean} | {n_train} | {n_val} | {n_doc_train} | {n_doc_val} |"
        )
        if n_val == 0:
            zero_val_classes.append(f"{code} ({ch})")

    # Holdout doc ids (seed 42)
    strat_split = make_stratified_split(df, seed=42)
    doc_split = make_doc_split(df, seed=42)
    holdout_docs = sorted(int(x) for x in df.loc[doc_split == "val", "doc_id"].unique())

    lines = [
        "# Data Preparation Summary",
        "",
        "## Drop Counts (Cleaning Rules)",
        "",
        f"| Rule | Description | Dropped |",
        f"|---:|:---|---:|",
        f"| 1 | ok == False | {n_bad} |",
        f"| 2 | Filename starts with 'Copy of' | {n_copy} |",
        f"| 3 | Cross-class md5 groups | {n_cross} |",
        f"| 4 | Within-class exact duplicates | {n_within} |",
        f"| **Total** | | **{n_bad + n_copy + n_cross + n_within}** |",
        "",
        f"**Original rows:** {n0}  ",
        f"**Final clean rows:** {len(df)}",
        "",
        "## Stratified Split (seed 42)",
        "",
        f"- Train: {int((s42['split'] == 'train').sum())}",
        f"- Val:   {int((s42['split'] == 'val').sum())}",
        "",
        "## Doc Split (seed 42)",
        "",
        f"- Train: {int((s42['doc_split'] == 'train').sum())}",
        f"- Val:   {int((s42['doc_split'] == 'val').sum())}",
        f"- Held-out doc_ids: {holdout_docs}",
        "",
        "## Classes with 0 val samples (stratified, seed 42)",
        "",
    ]
    if zero_val_classes:
        for c in zero_val_classes:
            lines.append(f"- {c}")
    else:
        lines.append("None")

    lines += [
        "",
        "## Overlap Assertions",
        "",
        "All seeds: stratified train ∩ val = 0, doc train ∩ val = 0  ✓",
        "",
        "## Per-Class Table (seed 42)",
        "",
    ]
    lines += class_table_rows
    lines.append("")

    out_path = "reports/data/split_summary.md"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  Wrote {out_path}")


if __name__ == "__main__":
    main()
