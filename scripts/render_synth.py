"""Render synthetic Thai glyphs dataset from OFL fonts with realistic degradation.

Usage:
    uv run python scripts/render_synth.py --per-class 300 --seed 0
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as _fm
try:
    _fm.fontManager.addfont('assets/fonts/Sarabun-Regular.ttf')
    plt.rcParams['font.family'] = _fm.FontProperties(fname='assets/fonts/Sarabun-Regular.ttf').get_name()
except Exception:  # noqa: BLE001
    pass
import numpy as np
import pandas as pd
from PIL import Image

from thaichar.classes import CLASS_CODES, category, code_to_char, code_to_index
from thaichar.data import load_cache
from thaichar.synth import FontLibrary, build_synth_cache, degrade_glyph

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render synthetic Thai glyphs from fonts.")
    parser.add_argument("--per-class", type=int, default=300, help="Number of samples per class.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--out-cache", type=str, default="data/synth/glyphs_synth.npz", help="Cache output path.")
    parser.add_argument("--out-index", type=str, default="data/synth/index.csv", help="Index CSV output path.")
    parser.add_argument("--summary-out", type=str, default="reports/data/synth_summary.md", help="Summary report.")
    parser.add_argument("--figures-dir", type=str, default="reports/figures", help="Output directory for figures.")
    parser.add_argument("--fonts-dir", type=str, default="assets/fonts", help="Fonts directory.")
    parser.add_argument("--class-stats", type=str, default="reports/eda/class_stats.csv", help="EDA class stats CSV.")
    parser.add_argument("--clean", action="store_true", help="Render only clean undegraded glyphs.")
    return parser.parse_args()


def render_montage_figure(
    images_by_code: dict[int, list[np.ndarray]],
    out_path: str,
) -> None:
    """Render 72 classes x 6 samples degraded montage."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    n_classes = len(CLASS_CODES)
    n_samples = 6

    # Create figure: 72 rows x 6 columns
    fig, axes = plt.subplots(n_classes, n_samples, figsize=(6, 48), squeeze=False)
    fig.subplots_adjust(wspace=0.05, hspace=0.1, left=0.15, right=0.98, top=0.99, bottom=0.01)

    for row_i, code in enumerate(CLASS_CODES):
        char = code_to_char(code)
        samples = images_by_code.get(code, [])
        for col_j in range(n_samples):
            ax = axes[row_i, col_j]
            if col_j < len(samples):
                img = samples[col_j]
                # Pad to square canvas for neat display
                h, w = img.shape[:2]
                c = max(h, w) + 4
                pad_img = np.full((c, c), 255, dtype=np.uint8)
                y0 = (c - h) // 2
                x0 = (c - w) // 2
                pad_img[y0 : y0 + h, x0 : x0 + w] = img
                ax.imshow(pad_img, cmap="gray", vmin=0, vmax=255)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color("#cccccc")
                spine.set_linewidth(0.5)

            if col_j == 0:
                ax.set_ylabel(f"{char} ({code})", rotation=0, labelpad=25, va="center", fontsize=8)

    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved synth montage to %s", out_path)


def render_synth_vs_real_figure(
    synth_by_code: dict[int, list[np.ndarray]],
    real_cache_path: str,
    real_df_path: str,
    out_path: str,
) -> None:
    """Render 10 classes: 4 real + 4 synth each."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # 10 representative classes: consonants, confusable pairs, vowels, marks, minority digits
    # 161: ก, 162: ข, 210: า, 229: ๅ, 209: ั, 232: ่, 240: ๐, 241: ๑, 163: ฃ (n=1), 177: ฑ (n=1)
    rep_classes = [161, 162, 210, 229, 209, 232, 240, 241, 163, 177]

    real_cache = None
    real_df = None
    if os.path.exists(real_cache_path) and os.path.exists(real_df_path):
        try:
            real_cache = load_cache(real_cache_path)
            real_df = pd.read_csv(real_df_path)
            real_path_map = {p: i for i, p in enumerate(real_cache.paths)}
        except Exception as e:
            logger.warning("Could not load real data for comparison: %s", e)

    fig, axes = plt.subplots(len(rep_classes), 8, figsize=(10, 12), squeeze=False)
    col_titles = ["Real 1", "Real 2", "Real 3", "Real 4", "Synth 1", "Synth 2", "Synth 3", "Synth 4"]

    for row_i, code in enumerate(rep_classes):
        char = code_to_char(code)
        # Fetch up to 4 real images
        real_imgs: list[np.ndarray] = []
        if real_df is not None and real_cache is not None:
            c_paths = real_df.loc[real_df["code"] == code, "path"].tolist()
            for p in c_paths[:4]:
                if p in real_path_map:
                    real_imgs.append(real_cache[real_path_map[p]])

        synth_imgs = synth_by_code.get(code, [])[:4]

        for col_j in range(8):
            ax = axes[row_i, col_j]
            if row_i == 0:
                ax.set_title(col_titles[col_j], fontsize=9, fontweight="bold", pad=4)

            img = None
            if col_j < 4:
                if col_j < len(real_imgs):
                    img = real_imgs[col_j]
            else:
                s_idx = col_j - 4
                if s_idx < len(synth_imgs):
                    img = synth_imgs[s_idx]

            if img is not None:
                h, w = img.shape[:2]
                c = max(h, w) + 4
                pad_img = np.full((c, c), 255, dtype=np.uint8)
                y0 = (c - h) // 2
                x0 = (c - w) // 2
                pad_img[y0 : y0 + h, x0 : x0 + w] = img
                ax.imshow(pad_img, cmap="gray", vmin=0, vmax=255)
            else:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center", fontsize=8, color="gray")

            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color("#444444" if col_j == 4 else "#dddddd")
                spine.set_linewidth(1.5 if col_j == 4 else 0.5)

            if col_j == 0:
                ax.set_ylabel(f"{char} ({code})", rotation=0, labelpad=30, va="center", fontsize=10, fontweight="bold")

    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved synth vs real figure to %s", out_path)


def write_summary_report(
    synth_df: pd.DataFrame,
    class_stats_path: str,
    font_lib: FontLibrary,
    out_path: str,
) -> None:
    """Generate reports/data/synth_summary.md."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    real_stats = pd.read_csv(class_stats_path).set_index("code")

    lines: list[str] = [
        "# Synthetic Thai Glyph Dataset Summary",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Total synthetic glyphs: **{len(synth_df):,}** ({len(CLASS_CODES)} classes)",
        "",
        "## 1. Fonts Used",
        "",
        f"Total fonts loaded: **{len(font_lib.fonts)}**",
        "",
        "| Font | Status | Glyphs Skipped |",
        "|---|---|---|",
    ]

    for font_stem in sorted(font_lib.fonts.keys()):
        sk = font_lib.skipped.get(font_stem, [])
        sk_str = ", ".join(f"{ch} ({c})" for c, ch in sk) if sk else "None (all 72 supported)"
        status = "Full support" if not sk else f"Skipped {len(sk)}"
        lines.append(f"| `{font_stem}` | {status} | {sk_str} |")

    lines.extend([
        "",
        "## 2. Real vs Synthetic Glyph Size Comparison",
        "",
        "| Code | Char | Category | N Synth | Real Med W | Synth Med W | Real Med H | Synth Med H | Real Mean Ink | Synth Mean Ink |",
        "|---:|:---:|:---:|---:|---:|---:|---:|---:|---:|---:|",
    ])

    for code in CLASS_CODES:
        sub = synth_df[synth_df["code"] == code]
        char = code_to_char(code)
        cat = category(code)
        n_synth = len(sub)
        synth_med_w = float(sub["width"].median()) if n_synth > 0 else 0.0
        synth_med_h = float(sub["height"].median()) if n_synth > 0 else 0.0
        synth_mean_ink = float(sub["ink_frac"].mean()) if n_synth > 0 else 0.0

        real_med_w = float(real_stats.loc[code, "med_w"]) if code in real_stats.index else 0.0
        real_med_h = float(real_stats.loc[code, "med_h"]) if code in real_stats.index else 0.0
        real_mean_ink = float(real_stats.loc[code, "mean_ink_frac"]) if code in real_stats.index else 0.0

        lines.append(
            f"| {code} | {char} | {cat} | {n_synth} | {real_med_w:.1f} | {synth_med_w:.1f} | "
            f"{real_med_h:.1f} | {synth_med_h:.1f} | {real_mean_ink:.3f} | {synth_mean_ink:.3f} |"
        )

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    logger.info("Wrote summary report to %s", out_path)


def main() -> None:
    args = parse_args()
    t0 = time.time()

    logger.info("Initializing FontLibrary from %s...", args.fonts_dir)
    font_lib = FontLibrary(font_dir=args.fonts_dir)

    logger.info("Loading class statistics from %s...", args.class_stats)
    class_stats = pd.read_csv(args.class_stats).set_index("code")

    rng = np.random.default_rng(args.seed)

    all_images: list[np.ndarray] = []
    all_paths: list[str] = []
    index_records: list[dict] = []
    montage_images: dict[int, list[np.ndarray]] = {c: [] for c in CLASS_CODES}

    logger.info("Rendering %d samples per class across 72 classes (total %d)...",
                args.per_class, args.per_class * len(CLASS_CODES))

    for code in CLASS_CODES:
        char = code_to_char(code)
        cat = category(code)
        label = code_to_index(code)

        templates = font_lib.templates.get(code, [])
        if not templates:
            logger.error("No templates available for code %d (%s)!", code, char)
            continue

        real_med_h = float(class_stats.loc[code, "med_h"])

        n_templates = len(templates)
        for i in range(args.per_class):
            tmpl = templates[i % n_templates]
            # First sample of each template is clean undegraded if not overridden
            if args.clean:
                is_degraded = False
            else:
                is_degraded = (i >= n_templates)  # clean variant for first sample of each font

            # Sample target height around real median height
            target_h = max(3, int(round(real_med_h * rng.uniform(0.8, 1.3))))

            img = degrade_glyph(tmpl.image_u8, target_h, rng, degraded=is_degraded)
            h, w = img.shape[:2]
            ink_frac = float(np.mean(img == 0))

            virtual_path = f"synth/{code}/{tmpl.font_name}_{i:04d}.png"

            all_images.append(img)
            all_paths.append(virtual_path)

            if len(montage_images[code]) < 6 and is_degraded:
                montage_images[code].append(img)

            index_records.append({
                "path": virtual_path,
                "code": code,
                "label": label,
                "char": char,
                "category": cat,
                "width": w,
                "height": h,
                "ink_frac": ink_frac,
                "font": tmpl.font_name,
                "degraded": is_degraded,
                "group": f"synth_{tmpl.font_name}",
            })

    synth_df = pd.DataFrame(index_records)

    # Save index CSV
    os.makedirs(os.path.dirname(args.out_index), exist_ok=True)
    synth_df.to_csv(args.out_index, index=False)
    logger.info("Saved synthetic index to %s (%d rows)", args.out_index, len(synth_df))

    # Save cache NPZ
    build_synth_cache(all_images, all_paths, args.out_cache)

    # Write summary report
    write_summary_report(synth_df, args.class_stats, font_lib, args.summary_out)

    # Render figures
    montage_path = os.path.join(args.figures_dir, "synth_montage.png")
    render_montage_figure(montage_images, montage_path)

    vs_real_path = os.path.join(args.figures_dir, "synth_vs_real.png")
    render_synth_vs_real_figure(
        montage_images,
        real_cache_path="data/cache/glyphs.npz",
        real_df_path="reports/eda/files.csv",
        out_path=vs_real_path,
    )

    elapsed = time.time() - t0
    logger.info("Rendering completed in %.2f s (%.1f glyphs/s)", elapsed, len(all_images) / max(elapsed, 0.001))


if __name__ == "__main__":
    main()
