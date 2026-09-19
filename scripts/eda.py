#!/usr/bin/env python3
"""EDA script for the Thai Character 72-class corpus.

Usage:
    uv run python scripts/eda.py --data "ThaiCharacter Dataset/round2" --out reports/eda
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import statistics
import sys
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

import imagehash
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Import our own package
# ---------------------------------------------------------------------------
from thaichar.classes import (
    CLASS_CODES,
    CLASS_CHARS,
    category,
    code_to_char,
    code_to_index,
    unicode_name,
)
from thaichar.filenames import parse_filename

# ---------------------------------------------------------------------------
# Per-file worker
# ---------------------------------------------------------------------------
IGNORE_FILES = {".DS_Store", "Thumbs.db"}
IGNORE_DIRS = {"__MACOSX"}


def _process_file(args: tuple) -> dict | None:
    """Process a single image file. Runs in a worker process."""
    fpath, code = args
    fname = os.path.basename(fpath)

    # Parse filename
    info = parse_filename(fname)

    # Image stats
    ok = True
    width = height = 0
    mode = ""
    file_bytes = 0
    ink_frac = 0.0
    mid_grey_frac = 0.0
    border_mean = 0.0
    phash_hex = ""
    md5 = ""

    try:
        file_bytes = os.path.getsize(fpath)
        with open(fpath, "rb") as f:
            data = f.read()
        md5 = hashlib.md5(data).hexdigest()

        img = Image.open(fpath)
        img.load()
        width, height = img.size
        mode = img.mode

        grey = img.convert("L")
        arr = np.asarray(grey, dtype=np.uint8)

        n_pixels = arr.size
        if n_pixels > 0:
            ink_frac = float(np.sum(arr < 128)) / n_pixels
            mid_grey_frac = float(np.sum((arr >= 32) & (arr < 224))) / n_pixels

            # Border: top row, bottom row, left col, right col (avoiding double-count corners)
            if arr.shape[0] >= 2 and arr.shape[1] >= 2:
                border_pixels = np.concatenate([
                    arr[0, :], arr[-1, :], arr[1:-1, 0], arr[1:-1, -1]
                ])
            elif arr.shape[0] == 1:
                border_pixels = arr[0, :]
            elif arr.shape[1] == 1:
                border_pixels = arr[:, 0]
            else:
                border_pixels = arr.ravel()
            border_mean = float(border_pixels.mean())

        phash_hex = str(imagehash.phash(img, hash_size=8))

    except Exception:
        ok = False

    return {
        "path": fpath,
        "code": code,
        "char": code_to_char(code),
        "category": category(code),
        # Filename fields
        "prefix": info["prefix"],
        "doc_id": info["doc_id"],
        "subset": info["subset"],
        "line": info["line"],
        "idx": info["idx"],
        "is_copy": info["is_copy"],
        "group": info["group"],
        "parsed": info["parsed"],
        # Image fields
        "width": width,
        "height": height,
        "mode": mode,
        "bytes": file_bytes,
        "md5": md5,
        "ok": ok,
        "ink_frac": round(ink_frac, 6),
        "mid_grey_frac": round(mid_grey_frac, 6),
        "border_mean": round(border_mean, 2),
        "phash": phash_hex,
    }


# ---------------------------------------------------------------------------
# Collect file list
# ---------------------------------------------------------------------------
def collect_files(data_dir: str) -> list[tuple[str, int]]:
    """Walk the dataset directory and return list of (filepath, code) tuples."""
    files = []
    skipped_non_jpg = []
    for entry in sorted(os.listdir(data_dir)):
        if entry in IGNORE_DIRS or entry.startswith("."):
            continue
        subdir = os.path.join(data_dir, entry)
        if not os.path.isdir(subdir):
            continue
        try:
            code = int(entry)
        except ValueError:
            continue
        if code not in CLASS_CODES:
            continue
        for fname in sorted(os.listdir(subdir)):
            if fname in IGNORE_FILES or fname.startswith("."):
                continue
            if not fname.lower().endswith(".jpg"):
                skipped_non_jpg.append(os.path.join(subdir, fname))
                continue
            files.append((os.path.join(subdir, fname), code))
    return files, skipped_non_jpg


# ---------------------------------------------------------------------------
# EDA analysis & output
# ---------------------------------------------------------------------------
def gini(values: list[int]) -> float:
    """Compute the Gini coefficient of a list of positive integers."""
    arr = np.array(sorted(values), dtype=float)
    n = len(arr)
    if n == 0 or arr.sum() == 0:
        return 0.0
    index = np.arange(1, n + 1)
    return float((2.0 * np.sum(index * arr) / (n * arr.sum())) - (n + 1) / n)


def main():
    parser = argparse.ArgumentParser(description="Thai Character EDA")
    parser.add_argument("--data", required=True, help="Path to round2 directory")
    parser.add_argument("--out", required=True, help="Output directory for reports")
    parser.add_argument("--workers", type=int, default=8, help="Number of workers")
    args = parser.parse_args()

    data_dir = args.data
    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)

    # Register Thai font
    font_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "assets", "fonts", "Sarabun-Regular.ttf"
    )
    if os.path.exists(font_path):
        font_manager.fontManager.addfont(font_path)
        plt.rcParams["font.family"] = "Sarabun"
    else:
        print(f"WARNING: Thai font not found at {font_path}", file=sys.stderr)

    print("Collecting files...")
    file_list, skipped_non_jpg = collect_files(data_dir)
    total_files = len(file_list)
    print(f"Found {total_files} .jpg files, {len(skipped_non_jpg)} non-jpg skipped")

    print(f"Processing {total_files} files with {args.workers} workers...")
    with Pool(args.workers) as pool:
        rows = list(tqdm(
            pool.imap(_process_file, file_list, chunksize=200),
            total=total_files,
            desc="Processing images",
        ))

    # Filter out None (shouldn't happen, but be safe)
    rows = [r for r in rows if r is not None]

    # -----------------------------------------------------------------------
    # 1. files.csv
    # -----------------------------------------------------------------------
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "files.csv"), index=False)
    print(f"Wrote files.csv ({len(df)} rows)")

    # -----------------------------------------------------------------------
    # 2. class_stats.csv
    # -----------------------------------------------------------------------
    stats_rows = []
    for code in CLASS_CODES:
        sub = df[df["code"] == code]
        n = len(sub)
        n_ok = int(sub["ok"].sum())
        n_unique_md5 = sub["md5"].nunique()
        n_groups = sub["group"].nunique()
        med_w = float(sub["width"].median()) if n > 0 else 0
        med_h = float(sub["height"].median()) if n > 0 else 0
        med_aspect = round(med_w / med_h, 4) if med_h > 0 else 0
        min_w = int(sub["width"].min()) if n > 0 else 0
        max_w = int(sub["width"].max()) if n > 0 else 0
        min_h = int(sub["height"].min()) if n > 0 else 0
        max_h = int(sub["height"].max()) if n > 0 else 0
        mean_ink = round(float(sub["ink_frac"].mean()), 6) if n > 0 else 0
        share = round(100.0 * n / total_files, 4) if total_files > 0 else 0
        n_train = round(0.8 * n)
        n_val = n - n_train

        stats_rows.append({
            "code": code,
            "char": code_to_char(code),
            "unicode_name": unicode_name(code),
            "category": category(code),
            "n_files": n,
            "n_ok": n_ok,
            "n_unique_md5": n_unique_md5,
            "n_groups": n_groups,
            "med_w": med_w,
            "med_h": med_h,
            "med_aspect": med_aspect,
            "min_w": min_w,
            "max_w": max_w,
            "min_h": min_h,
            "max_h": max_h,
            "mean_ink_frac": mean_ink,
            "share_pct": share,
            "n_train_80": n_train,
            "n_val_20": n_val,
        })

    class_df = pd.DataFrame(stats_rows)
    class_df.to_csv(os.path.join(out_dir, "class_stats.csv"), index=False)
    print(f"Wrote class_stats.csv ({len(class_df)} rows)")

    # -----------------------------------------------------------------------
    # 3. duplicates.json
    # -----------------------------------------------------------------------
    md5_groups_raw = df.groupby("md5").agg(
        n=("path", "size"),
        paths=("path", list),
        codes=("code", lambda x: sorted(set(x))),
    ).reset_index()
    md5_dups = md5_groups_raw[md5_groups_raw["n"] > 1]

    md5_group_list = []
    for _, row in md5_dups.iterrows():
        md5_group_list.append({
            "md5": row["md5"],
            "n": int(row["n"]),
            "paths": row["paths"],
            "codes": [int(c) for c in row["codes"]],
            "cross_class": len(row["codes"]) > 1,
        })

    n_redundant = sum(g["n"] - 1 for g in md5_group_list)
    n_cross_class = sum(1 for g in md5_group_list if g["cross_class"])
    n_unique_after_dedup = len(md5_groups_raw)
    copy_of_files = sorted(df[df["is_copy"] == True]["path"].tolist())

    duplicates = {
        "md5_groups": md5_group_list,
        "n_redundant_files": n_redundant,
        "n_cross_class_groups": n_cross_class,
        "n_unique_after_dedup": n_unique_after_dedup,
        "copy_of_files": copy_of_files,
    }
    with open(os.path.join(out_dir, "duplicates.json"), "w", encoding="utf-8") as f:
        json.dump(duplicates, f, indent=2, ensure_ascii=False)
    print(f"Wrote duplicates.json ({len(md5_group_list)} duplicate groups)")

    # -----------------------------------------------------------------------
    # 4. summary.json
    # -----------------------------------------------------------------------
    n_ok_total = int(df["ok"].sum())
    corrupted = sorted(df[df["ok"] == False]["path"].tolist())
    class_sizes = class_df["n_files"].tolist()
    imbalance_ratio = round(max(class_sizes) / min(class_sizes), 2) if min(class_sizes) > 0 else float("inf")
    gini_val = round(gini(class_sizes), 4)
    top10_share = round(float(class_df.nlargest(10, "n_files")["share_pct"].sum()), 2)
    classes_lt50 = int((class_df["n_files"] < 50).sum())
    classes_lt10 = int((class_df["n_files"] < 10).sum())
    classes_lt5 = int((class_df["n_files"] < 5).sum())
    median_w = float(df["width"].median())
    median_h = float(df["height"].median())

    # Grey level check
    overall_mid_grey = round(float(df["mid_grey_frac"].mean()), 4)

    # Group counts
    prefix_counts = df["prefix"].value_counts().to_dict()
    # Convert keys to str to be safe
    prefix_counts = {str(k): int(v) for k, v in prefix_counts.items()}
    distinct_doc_ids = int(df["doc_id"].nunique())
    distinct_groups = int(df["group"].nunique())

    # Per-category totals
    cat_totals = df.groupby("category").size().to_dict()
    cat_totals = {str(k): int(v) for k, v in cat_totals.items()}

    summary = {
        "total_files": total_files,
        "total_ok": n_ok_total,
        "corrupted_files": corrupted,
        "non_jpg_skipped": skipped_non_jpg,
        "n_classes": len(CLASS_CODES),
        "imbalance_ratio_max_min": imbalance_ratio,
        "gini_class_sizes": gini_val,
        "top10_share_pct": top10_share,
        "classes_with_n_lt_50": classes_lt50,
        "classes_with_n_lt_10": classes_lt10,
        "classes_with_n_lt_5": classes_lt5,
        "median_width": median_w,
        "median_height": median_h,
        "overall_mid_grey_frac": overall_mid_grey,
        "group_counts": {
            "per_prefix": prefix_counts,
            "distinct_doc_ids": distinct_doc_ids,
            "distinct_groups": distinct_groups,
        },
        "duplicate_summary": {
            "n_duplicate_md5_groups": len(md5_group_list),
            "n_redundant_files": n_redundant,
            "n_cross_class_groups": n_cross_class,
            "n_unique_after_dedup": n_unique_after_dedup,
            "n_copy_of_files": len(copy_of_files),
        },
        "per_category_totals": cat_totals,
    }
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print("Wrote summary.json")

    # -----------------------------------------------------------------------
    # 5. Figures
    # -----------------------------------------------------------------------
    DPI = 150
    category_colors = {
        "consonant": "#2196F3",
        "vowel": "#4CAF50",
        "tone_mark": "#FF9800",
        "digit": "#E91E63",
    }

    # --- class_distribution.png ---
    sorted_class = class_df.sort_values("n_files", ascending=False).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(18, 6))
    colors = [category_colors.get(cat, "grey") for cat in sorted_class["category"]]
    ax.bar(range(len(sorted_class)), sorted_class["n_files"], color=colors)
    ax.set_yscale("log")
    ax.set_xticks(range(len(sorted_class)))
    ax.set_xticklabels(sorted_class["char"], fontsize=7)
    ax.axhline(y=50, color="red", linestyle="--", linewidth=0.8, label="n=50")
    ax.axhline(y=10, color="orange", linestyle="--", linewidth=0.8, label="n=10")
    ax.set_xlabel("Thai Character")
    ax.set_ylabel("Count (log scale)")
    ax.set_title("Class Distribution (72 classes, sorted descending)")
    ax.legend(loc="upper right")
    # Add category legend
    from matplotlib.patches import Patch
    legend_patches = [Patch(color=c, label=l) for l, c in category_colors.items()]
    ax.legend(handles=legend_patches + [
        plt.Line2D([0], [0], color="red", linestyle="--", label="n=50"),
        plt.Line2D([0], [0], color="orange", linestyle="--", label="n=10"),
    ], loc="upper right", fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "class_distribution.png"), dpi=DPI)
    plt.close(fig)
    print("Wrote class_distribution.png")

    # --- class_montage.png ---
    rng = random.Random(0)
    n_cols = 6  # samples per class
    n_rows = 72
    cell_w, cell_h = 24, 24  # pixel size per cell in the figure
    fig, axes = plt.subplots(n_rows, n_cols + 1, figsize=(n_cols * 0.7 + 2, n_rows * 0.35))
    for row_i, code in enumerate(CLASS_CODES):
        sub = df[(df["code"] == code) & (df["ok"] == True)]
        char = code_to_char(code)
        n = len(sub)

        # Label column
        axes[row_i, 0].text(0.5, 0.5, f"{char} ({code}) n={n}",
                            fontsize=4, ha="center", va="center",
                            transform=axes[row_i, 0].transAxes)
        axes[row_i, 0].axis("off")

        samples = sub.sample(n=min(n_cols, n), random_state=0)["path"].tolist()
        for col_i in range(n_cols):
            ax = axes[row_i, col_i + 1]
            ax.axis("off")
            if col_i < len(samples):
                try:
                    img = Image.open(samples[col_i]).convert("L")
                    ax.imshow(np.asarray(img), cmap="gray", interpolation="nearest")
                except Exception:
                    pass

    fig.suptitle("Class Montage: 72 classes × 6 samples", fontsize=8, y=1.0)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "class_montage.png"), dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("Wrote class_montage.png")

    # --- size_scatter.png ---
    fig, ax = plt.subplots(figsize=(8, 6))
    ok_df = df[df["ok"] == True]
    ax.scatter(ok_df["width"], ok_df["height"], alpha=0.05, s=5, color="#2196F3")
    ax.axvline(x=median_w, color="red", linestyle="--", linewidth=1, label=f"median w={median_w:.0f}")
    ax.axhline(y=median_h, color="green", linestyle="--", linewidth=1, label=f"median h={median_h:.0f}")
    ax.set_xlabel("Width (px)")
    ax.set_ylabel("Height (px)")
    ax.set_title("Image Size Distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "size_scatter.png"), dpi=DPI)
    plt.close(fig)
    print("Wrote size_scatter.png")

    # --- grey_histogram.png ---
    rng2 = random.Random(0)
    sample_paths = rng2.sample(ok_df["path"].tolist(), min(2000, len(ok_df)))
    all_pixels = []
    for p in sample_paths:
        try:
            img = Image.open(p).convert("L")
            all_pixels.append(np.asarray(img).ravel())
        except Exception:
            pass
    if all_pixels:
        all_pixels = np.concatenate(all_pixels)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(all_pixels, bins=256, range=(0, 255), color="#607D8B", log=True)
        ax.set_xlabel("Pixel Intensity")
        ax.set_ylabel("Count (log scale)")
        ax.set_title("Pixel Intensity Distribution (2000-image sample)")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "grey_histogram.png"), dpi=DPI)
        plt.close(fig)
        print("Wrote grey_histogram.png")

    # --- per_group_counts.png ---
    group_counts = df.groupby("group").size().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(range(len(group_counts)), group_counts.values, color="#795548", width=1.0)
    ax.set_xlabel("Group (prefix_doc)")
    ax.set_ylabel("Number of files")
    ax.set_title(f"Files per Group ({len(group_counts)} groups)")
    ax.set_xlim(-0.5, len(group_counts) - 0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "per_group_counts.png"), dpi=DPI)
    plt.close(fig)
    print("Wrote per_group_counts.png")

    # -----------------------------------------------------------------------
    # 6. EDA.md
    # -----------------------------------------------------------------------
    # Build cross-class collision examples
    cross_class_examples = [g for g in md5_group_list if g["cross_class"]][:5]
    cross_class_section = ""
    if cross_class_examples:
        cross_class_section = "### Cross-class collision examples\n\n"
        for i, g in enumerate(cross_class_examples, 1):
            chars = ", ".join(f"{code_to_char(c)} ({c})" for c in g["codes"])
            cross_class_section += f"**{i}.** MD5 `{g['md5']}` — {g['n']} files across classes: {chars}\n"
            for p in g["paths"][:4]:
                cross_class_section += f"  - `{p}`\n"
            cross_class_section += "\n"

    # Build class table
    class_table_rows = []
    for _, r in class_df.iterrows():
        class_table_rows.append(
            f"| {int(r['code'])} | {r['char']} | {r['category']} | {int(r['n_files'])} | "
            f"{r['share_pct']:.2f}% | {r['med_w']:.0f}×{r['med_h']:.0f} | "
            f"{int(r['n_groups'])} | {int(r['n_train_80'])}/{int(r['n_val_20'])} |"
        )
    class_table = "\n".join(class_table_rows)

    eda_md = f"""# EDA Report — Thai Character 72-Class Corpus

## Headline Numbers

| Metric | Value |
|--------|-------|
| Total files | {total_files} |
| Total OK (loadable) | {n_ok_total} |
| Corrupted | {len(corrupted)} |
| Classes | {len(CLASS_CODES)} |
| Imbalance ratio (max/min) | {imbalance_ratio} |
| Gini coefficient | {gini_val} |
| Top-10 class share | {top10_share}% |
| Classes with n<50 | {classes_lt50} |
| Classes with n<10 | {classes_lt10} |
| Classes with n<5 | {classes_lt5} |
| Median image size | {median_w:.0f}×{median_h:.0f} px |
| Overall mid-grey fraction | {overall_mid_grey} |
| Duplicate MD5 groups | {len(md5_group_list)} |
| Redundant files | {n_redundant} |
| Cross-class collisions | {n_cross_class} |
| Unique after dedup | {n_unique_after_dedup} |
| "Copy of" files | {len(copy_of_files)} |
| Distinct groups | {distinct_groups} |
| Distinct doc IDs | {distinct_doc_ids} |

## Per-Category Totals

| Category | Count |
|----------|-------|
{chr(10).join(f"| {k} | {v} |" for k, v in sorted(cat_totals.items()))}

## Full 72-Class Table

| Code | Char | Category | n | Share | Med WxH | Groups | Train/Val |
|------|------|----------|---|-------|---------|--------|-----------|
{class_table}

## Imbalance Analysis

The dataset is **highly imbalanced** with an imbalance ratio of **{imbalance_ratio}** (max class
has {max(class_sizes)} files vs min class with {min(class_sizes)} files). The Gini coefficient
is {gini_val}, confirming significant inequality. The top 10 classes account for {top10_share}%
of all files. {classes_lt50} classes have fewer than 50 samples, {classes_lt10} have fewer than
10, and {classes_lt5} have fewer than 5.

## Duplicates & Hygiene

- **{len(md5_group_list)}** duplicate MD5 groups found, covering **{n_redundant}** redundant files.
- **{n_cross_class}** cross-class collision groups (same image in different class folders).
- **{len(copy_of_files)}** files with "Copy of" prefix detected.
- After deduplication: **{n_unique_after_dedup}** unique images.

{cross_class_section}

## Group Structure

Files are organised by `prefix_docid` groups. There are **{distinct_groups}** distinct groups
across **{distinct_doc_ids}** unique document IDs.

**Per-prefix counts:**

| Prefix | Files |
|--------|-------|
{chr(10).join(f"| {k} | {v} |" for k, v in sorted(prefix_counts.items()))}

## Figures

- ![Class Distribution](class_distribution.png)
- ![Class Montage](class_montage.png)
- ![Size Scatter](size_scatter.png)
- ![Grey Histogram](grey_histogram.png)
- ![Per-Group Counts](per_group_counts.png)
"""

    with open(os.path.join(out_dir, "EDA.md"), "w", encoding="utf-8") as f:
        f.write(eda_md)
    print("Wrote EDA.md")

    # -----------------------------------------------------------------------
    # Final summary printout
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("EDA COMPLETE")
    print("=" * 60)
    print(f"Total files:             {total_files}")
    print(f"Total OK:                {n_ok_total}")
    print(f"Corrupted:               {len(corrupted)}")
    print(f"Classes:                 {len(CLASS_CODES)}")
    print(f"Imbalance ratio:         {imbalance_ratio}")
    print(f"Gini:                    {gini_val}")
    print(f"Duplicate MD5 groups:    {len(md5_group_list)}")
    print(f"Redundant files:         {n_redundant}")
    print(f"Cross-class collisions:  {n_cross_class}")
    print(f"Copy-of files:           {len(copy_of_files)}")
    print(f"Median size:             {median_w:.0f}×{median_h:.0f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
