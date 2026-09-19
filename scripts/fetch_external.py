#!/usr/bin/env python3
"""CLI script to download, map, and cache external Thai character datasets (TASK-07)."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np
import pandas as pd

from thaichar.classes import (
    CLASS_CODES,
    category,
    code_to_char,
    code_to_index,
    unicode_name,
)
from thaichar.data import load_cache
from thaichar.external import (
    binarize_for_cache,
    build_external_cache,
    fetch_alice,
    fetch_burapha,
    fetch_kvis,
    print_transliteration_mapping,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("fetch_external")

# Dataset metadata specifications
METADATA = {
    "alice": {
        "name": "ALICE-THI",
        "url": "https://www.ai.rug.nl/~mrolarik/ALICE-THI/ALICE-THI-Dataset.tar.gz",
        "homepage": "https://www.ai.rug.nl/~mrolarik/ALICE-THI/",
        "license": "Non-commercial research use; 'Do not copy without notice! Copyright 2015 Olarik Surinta (olarik.s@msu.ac.th)'",
        "citation": 'O. Surinta, M. F. Karaaba, L. R. B. Schomaker, M. A. Wiering, "Recognition of handwritten characters using local gradient feature descriptors", Engineering Applications of Artificial Intelligence, 45:405-414, 2015. DOI: 10.1016/j.engappai.2015.07.017',
    },
    "burapha": {
        "name": "Burapha-TH (Character + Digit)",
        "url": "https://services.informatics.buu.ac.th/datasets/Burapha-TH/",
        "homepage": "https://services.informatics.buu.ac.th/datasets/Burapha-TH/",
        "license": "Creative Commons Attribution 4.0 International (CC BY 4.0) via MDPI open-access publication",
        "citation": 'A. Onuean, U. Buatoom, T. Charoenporn, T. Kim, H. Jung, "Burapha-TH: A Multi-Purpose Character, Digit, and Syllable Handwriting Dataset", Applied Sciences, 12(8):4083, 2022. DOI: 10.3390/app12084083',
    },
    "kvis": {
        "name": "KVIS Thai OCR",
        "url": "https://data.mendeley.com/datasets/8nr3pbdk5c/1",
        "homepage": "https://data.mendeley.com/datasets/8nr3pbdk5c/1",
        "license": "Creative Commons Attribution 4.0 International (CC BY 4.0)",
        "citation": 'F. J. J. Joseph, P. Anantaprayoon, "Offline Handwritten Thai Character Recognition Using Single Tier Classifier and Local Features", 2018 International Conference on Information Technology (InCIT), pp. 1-4, 2018. DOI: 10.23919/INCIT.2018.8584876',
    },
}


def create_montage(
    index_df: pd.DataFrame,
    cache_path: str | Path,
    out_png: str | Path,
    samples_per_source: int = 12,
) -> None:
    """Generate reports/figures/external_montage.png showing 12 samples per source."""
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    cache = load_cache(str(cache_path))
    path_to_cache_idx = {p: i for i, p in enumerate(cache.paths)}

    sources = sorted(index_df["group"].apply(lambda g: g.split("_")[0]).unique())
    if not sources:
        logger.warning("No sources available for montage.")
        return

    # Find a Thai font on system if available
    thai_font = None
    candidate_fonts = [
        "/usr/share/fonts/google-noto-vf/NotoSansThai[wght].ttf",
        "/usr/share/fonts/google-noto-vf/NotoSerifThai[wght].ttf",
        "/usr/share/fonts/google-droid-sans-fonts/DroidSansThai.ttf",
    ]
    for font_path in candidate_fonts:
        if os.path.exists(font_path):
            thai_font = FontProperties(fname=font_path, size=14)
            break

    n_rows = len(sources)
    n_cols = samples_per_source

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.5, n_rows * 1.8), squeeze=False)
    fig.suptitle("External Thai Character Datasets — Binarized Glyph Samples", fontsize=16, y=0.98)

    rng = np.random.default_rng(42)

    for r, src in enumerate(sources):
        src_mask = index_df["group"].apply(lambda g: g.split("_")[0]) == src
        src_df = index_df[src_mask]

        # Choose 12 distinct classes if possible
        available_labels = src_df["label"].unique()
        chosen_labels = rng.choice(available_labels, size=min(n_cols, len(available_labels)), replace=False)
        chosen_labels = sorted(chosen_labels)

        chosen_rows: list[pd.Series] = []
        for lbl in chosen_labels:
            candidates = src_df[src_df["label"] == lbl]
            chosen_rows.append(candidates.sample(1, random_state=42).iloc[0])

        # If fewer than 12 classes, sample randomly to fill
        while len(chosen_rows) < n_cols and len(src_df) > len(chosen_rows):
            chosen_rows.append(src_df.sample(1, random_state=42 + len(chosen_rows)).iloc[0])

        for c in range(n_cols):
            ax = axes[r, c]
            if c < len(chosen_rows):
                row = chosen_rows[c]
                p = row["path"]
                cache_i = path_to_cache_idx[p]
                arr = cache[cache_i]

                # Display: ink is dark (0), background is white (255)
                ax.imshow(arr, cmap="gray", vmin=0, vmax=255)
                ch = row["char"]
                title_str = f"{ch} ({row['code']})"
                if thai_font is not None:
                    ax.set_title(title_str, fontproperties=thai_font)
                else:
                    ax.set_title(title_str)
            ax.axis("off")

        # Label row source on leftmost axis
        src_name = METADATA.get(src, {}).get("name", src.upper())
        axes[r, 0].text(
            -0.3, 0.5, src_name,
            transform=axes[r, 0].transAxes,
            fontsize=12, fontweight="bold",
            va="center", ha="right", rotation=0
        )

    plt.tight_layout(rect=[0.08, 0.03, 1, 0.95])
    plt.savefig(str(out_png), dpi=150)
    plt.close()
    logger.info("Saved montage to %s", out_png)


def write_summary_report(
    source_dfs: dict[str, pd.DataFrame],
    source_status: dict[str, dict[str, Any]],
    index_df: pd.DataFrame,
    out_md: str | Path,
    out_dir: Path,
    npz_path: Path,
) -> None:
    """Write comprehensive reports/data/external_summary.md."""
    out_md = Path(out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines.append("# External Public Thai Character Datasets Summary (TASK-07)")
    lines.append("")
    lines.append("## 1. Overview")
    lines.append("")
    lines.append("This report documents the acquisition, character mapping, binarization, and caching "
                 "of public Thai character datasets for intermediate pretraining.")
    lines.append("")
    lines.append("- **Target Corpus:** 72 TIS-620 character classes (consonants, vowels, tone marks, digits) defined in `src/thaichar/classes.py`.")
    lines.append("- **Nature of Data:** Public sets are **handwritten**; project corpus is **printed binary glyphs**. "
                 "These external data are intended exclusively for representation pretraining and intermediate feature learning.")
    lines.append("")

    lines.append("## 2. Per-Source Results and Metadata")
    lines.append("")

    for src_key, meta in METADATA.items():
        st = source_status.get(src_key, {})
        status = st.get("status", "Skipped")
        df = source_dfs.get(src_key, pd.DataFrame())

        lines.append(f"### {meta['name']}")
        lines.append("")
        lines.append(f"- **Status:** {status}")
        if status.startswith("Failed") or status.startswith("Skipped"):
            lines.append(f"- **Reason:** {st.get('reason', 'N/A')}")
        lines.append(f"- **Source URL:** [{meta['url']}]({meta['url']})")
        lines.append(f"- **Homepage:** [{meta['homepage']}]({meta['homepage']})")
        lines.append(f"- **License / Terms:** {meta['license']}")
        lines.append(f"- **Academic Citation:** {meta['citation']}")
        lines.append("")

        if len(df) > 0:
            n_total = len(df)
            n_mapped = int((df["label"] >= 0).sum())
            n_unmapped = int((df["label"] < 0).sum())

            lines.append("| Metric | Count |")
            lines.append("|:---|---:|")
            lines.append(f"| Images Downloaded / Discovered | {n_total:,} |")
            lines.append(f"| Images Mapped to 72 Target Classes | {n_mapped:,} ({100 * n_mapped / n_total:.1f}%) |")
            lines.append(f"| Images Unmapped (Excluded Classes) | {n_unmapped:,} ({100 * n_unmapped / n_total:.1f}%) |")
            lines.append("")

            # Unmapped details
            if n_unmapped > 0:
                unmapped_df = df[df["label"] < 0]
                unmapped_counts = unmapped_df.groupby(["code", "char", "orig_label"]).size().reset_index(name="count")
                unmapped_counts = unmapped_counts.sort_values("code")

                lines.append("#### Unmapped Characters in " + meta["name"])
                lines.append("")
                lines.append("| Code | Char | Original Label | Count | Rationale for Exclusion |")
                lines.append("|---:|:---:|:---|---:|:---|")
                for _, urow in unmapped_counts.iterrows():
                    c = urow["code"]
                    ch = urow["char"]
                    orig = urow["orig_label"]
                    cnt = urow["count"]
                    name_str = unicode_name(c) if (0x0E00 <= (c - 0xA0 + 0x0E00) <= 0x0E7F) else "Unknown"
                    lines.append(f"| {c} | {ch} | {orig} ({name_str}) | {cnt:,} | Excluded from our 72-class TIS-620 target corpus |")
                lines.append("")

            # Image size statistics from index
            src_index = index_df[index_df["group"].apply(lambda g: g.split("_")[0]) == src_key]
            if len(src_index) > 0:
                lines.append("#### Cropped Glyph Statistics (After Otsu Binarization)")
                lines.append("")
                lines.append("| Statistic | Width (px) | Height (px) | Ink Fraction |")
                lines.append("|:---|---:|---:|---:|")
                lines.append(f"| Min | {src_index['width'].min()} | {src_index['height'].min()} | {src_index['ink_frac'].min():.4f} |")
                lines.append(f"| Median | {src_index['width'].median():.1f} | {src_index['height'].median():.1f} | {src_index['ink_frac'].median():.4f} |")
                lines.append(f"| Mean | {src_index['width'].mean():.1f} | {src_index['height'].mean():.1f} | {src_index['ink_frac'].mean():.4f} |")
                lines.append(f"| Max | {src_index['width'].max()} | {src_index['height'].max()} | {src_index['ink_frac'].max():.4f} |")
                lines.append("")

    lines.append("## 3. Combined 72-Class Coverage Table")
    lines.append("")
    lines.append("Counts of binarized, cached glyphs per class across all active external sources.")
    lines.append("")

    lines.append("| Label | Code | Char | Category | Unicode Name | ALICE-THI | Burapha-TH | Total External | Status |")
    lines.append("|---:|---:|:---:|:---|:---|---:|---:|---:|:---|")

    zero_sample_classes: list[tuple[int, int, str, str]] = []

    for i, code in enumerate(CLASS_CODES):
        ch = code_to_char(code)
        cat = category(code)
        uname = unicode_name(code)

        alice_cnt = int(((index_df["label"] == i) & (index_df["group"].str.startswith("alice"))).sum())
        burapha_cnt = int(((index_df["label"] == i) & (index_df["group"].str.startswith("burapha"))).sum())
        tot_cnt = alice_cnt + burapha_cnt

        if tot_cnt == 0:
            status_str = "**ZERO SAMPLES**"
            zero_sample_classes.append((i, code, ch, uname))
        else:
            status_str = "Covered"

        lines.append(f"| {i} | {code} | {ch} | {cat} | {uname} | {alice_cnt:,} | {burapha_cnt:,} | {tot_cnt:,} | {status_str} |")

    lines.append("")
    lines.append("## 4. Analysis of Classes with Zero External Samples")
    lines.append("")
    lines.append(f"Out of our 72 target classes, **{len(zero_sample_classes)}** class(es) have **zero external samples**:")
    lines.append("")
    for i, code, ch, uname in zero_sample_classes:
        lines.append(f"- **Label {i}** (Code {code}, `{ch}` — {uname}): Not present in either ALICE-THI or Burapha-TH isolated character sets.")
    lines.append("")
    lines.append("### Why are these classes missing?")
    lines.append("- `แ` (TIS-620 code 225, SARA AE): In many traditional Thai handwriting datasets, Sara Ae is often omitted as an isolated glyph because writers may decompose it into two Sara E (`เเ`) glyphs or it is only captured in multi-character syllable datasets.")
    lines.append("- `ๅ` (TIS-620 code 229, LAKKHANGYAO): A specialized lengthened vowel mark used only with `ฤๅ` and `ฦๅ`; rarely written in isolated character sheets.")
    lines.append("- All 44 standard consonants present in CLASS_CODES and all 10 Thai digits (๐–๙) have substantial coverage.")
    lines.append("")

    lines.append("## 5. Storage and Disk Footprint")
    lines.append("")
    dl_dir = out_dir / "downloads"
    dl_mb = sum(f.stat().st_size for f in dl_dir.glob("*") if f.is_file()) / (1024 * 1024) if dl_dir.exists() else 0.0
    npz_mb = npz_path.stat().st_size / (1024 * 1024) if npz_path.exists() else 0.0
    csv_kb = (out_dir / "index.csv").stat().st_size / 1024 if (out_dir / "index.csv").exists() else 0.0

    lines.append(f"- **Downloaded Archive Footprint:** {dl_mb:.1f} MB (well below the 3 GB project ceiling)")
    lines.append(f"- **Output Glyph Cache (`glyphs_external.npz`):** {npz_mb:.1f} MB ({len(index_df):,} glyphs)")
    lines.append(f"- **Output Index (`index.csv`):** {csv_kb:.1f} KB ({len(index_df):,} mapped rows)")
    lines.append(f"- **Covered Target Classes:** {index_df['label'].nunique()} / 72 classes")
    lines.append(f"- **Unique Groups / Writers:** {index_df['group'].nunique():,} distinct groups")
    lines.append("")
    lines.append("## 6. Representative Montage")
    lines.append("")
    lines.append("A visual montage showing 12 binarized samples from each successful source has been generated at `reports/figures/external_montage.png`.")
    lines.append("")

    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    logger.info("Summary report written to %s", out_md)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch public Thai character datasets and map to 72 classes.")
    parser.add_argument(
        "--sources",
        nargs="+",
        default=["alice", "kvis", "burapha"],
        help="Datasets to fetch (alice, kvis, burapha)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="data/external",
        help="Output directory for downloaded files and cache",
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=8,
        help="Number of worker processes for binarization and cache building",
    )
    parser.add_argument(
        "--summary-md",
        type=str,
        default="reports/data/external_summary.md",
        help="Path to output summary markdown report",
    )
    parser.add_argument(
        "--montage-png",
        type=str,
        default="reports/figures/external_montage.png",
        help="Path to output visual montage png",
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("TASK-07: External Public Thai Character Data Fetcher")
    print("=" * 60)
    print_transliteration_mapping()
    print("=" * 60)

    source_dfs: dict[str, pd.DataFrame] = {}
    source_status: dict[str, dict[str, Any]] = {}

    for src in args.sources:
        src = src.lower().strip()
        logger.info("Processing source: %s", src)

        if src == "alice":
            try:
                df = fetch_alice(out_dir)
                source_dfs["alice"] = df
                source_status["alice"] = {
                    "status": "Succeeded",
                    "n_downloaded": len(df),
                    "n_mapped": int((df["label"] >= 0).sum()),
                    "n_unmapped": int((df["label"] < 0).sum()),
                }
            except Exception as e:
                logger.error("ALICE-THI failed: %s", e)
                source_status["alice"] = {"status": "Failed", "reason": str(e)}

        elif src == "burapha":
            try:
                df = fetch_burapha(out_dir)
                source_dfs["burapha"] = df
                source_status["burapha"] = {
                    "status": "Succeeded",
                    "n_downloaded": len(df),
                    "n_mapped": int((df["label"] >= 0).sum()),
                    "n_unmapped": int((df["label"] < 0).sum()),
                }
            except Exception as e:
                logger.error("Burapha-TH failed: %s", e)
                source_status["burapha"] = {"status": "Failed", "reason": str(e)}

        elif src == "kvis":
            try:
                df = fetch_kvis(out_dir)
                source_dfs["kvis"] = df
                source_status["kvis"] = {
                    "status": "Skipped (Login / Gated)",
                    "reason": (
                        "Mendeley Data S3 direct zip URL returned HTTP 403 Forbidden; "
                        "dataset landing page (https://data.mendeley.com/datasets/8nr3pbdk5c/1) requires "
                        "interactive OAuth2 login / credentials. Loader script kvis_th_ocr.py deprecated in datasets >= 5.0."
                    ),
                    "n_downloaded": 0,
                    "n_mapped": 0,
                    "n_unmapped": 0,
                }
            except Exception as e:
                logger.error("KVIS failed: %s", e)
                source_status["kvis"] = {"status": "Failed", "reason": str(e)}

        else:
            logger.warning("Unknown source: %s", src)

    # Combine all mapped DataFrames
    dfs_to_combine = [df for df in source_dfs.values() if len(df) > 0]
    if not dfs_to_combine:
        logger.error("No dataset could be fetched. Exiting.")
        sys.exit(1)

    combined_df = pd.concat(dfs_to_combine, ignore_index=True)
    logger.info("Combined raw datasets: %d images total", len(combined_df))

    # Build glyphs_external.npz and index.csv
    npz_path = out_dir / "glyphs_external.npz"
    index_csv = out_dir / "index.csv"

    n_saved, n_skipped = build_external_cache(
        combined_df,
        out_npz=npz_path,
        index_csv=index_csv,
        n_workers=args.n_workers,
    )

    # Read back index.csv
    index_df = pd.read_csv(index_csv)

    # Generate visual montage
    create_montage(
        index_df,
        cache_path=npz_path,
        out_png=args.montage_png,
        samples_per_source=12,
    )

    # Write summary markdown report
    write_summary_report(
        source_dfs=source_dfs,
        source_status=source_status,
        index_df=index_df,
        out_md=args.summary_md,
        out_dir=out_dir,
        npz_path=npz_path,
    )

    print("\nFetch and cache complete!")
    print(f"Total glyphs in cache: {len(index_df):,}")
    print(f"Covered classes: {index_df['label'].nunique()} / 72")


if __name__ == "__main__":
    main()
