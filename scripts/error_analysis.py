#!/usr/bin/env python
"""Error analysis script for 72-class Thai character classification.

Produces:
1. confusion_matrix.png — 72x72 row-normalised (recall) heatmap with Thai glyph ticks,
   log-ish colour scale, exp_id and top1/bal acc in title.
2. confused_pairs.md — table of top-k off-diagonal cells and symmetric merged pairs.
3. per_class_recall.png — bar chart of per-class recall sorted ascending, coloured by category,
   Thai glyph ticks, second axis showing log n_train, horizontal line at balanced acc.
4. worst_classes.md — 15 lowest-recall classes with n_train, n_val, recall, top-3 predictions.
5. summary.json — machine-readable metrics and rankings.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch
import numpy as np
from tabulate import tabulate

# Ensure src/ is on sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from thaichar.classes import CLASS_CODES, CLASS_CHARS, code_to_char, category


CATEGORY_COLORS = {
    "consonant": "#1f77b4",  # blue
    "vowel": "#ff7f0e",      # orange
    "tone_mark": "#2ca02c",  # green
    "digit": "#d62728",      # red
}


def register_thai_font(font_path: Path | str | None = None) -> str:
    """Register Sarabun font with matplotlib fontManager and set font.family."""
    if font_path is None:
        font_path = _REPO_ROOT / "assets" / "fonts" / "Sarabun-Regular.ttf"
    font_path = Path(font_path)
    if font_path.exists():
        font_manager.fontManager.addfont(str(font_path))
        bold_path = font_path.parent / "Sarabun-Bold.ttf"
        if bold_path.exists():
            font_manager.fontManager.addfont(str(bold_path))
        prop = font_manager.FontProperties(fname=str(font_path))
        family = prop.get_name()
        plt.rcParams["font.family"] = family
        plt.rcParams["axes.unicode_minus"] = False
        return family
    return plt.rcParams.get("font.family", ["sans-serif"])[0]


def plot_confusion_matrix(
    cm: np.ndarray,
    exp_id: str,
    top1: float,
    bal_acc: float,
    out_path: Path,
) -> None:
    """Plot 72x72 row-normalised confusion matrix heatmap with log-ish colour scale."""
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.zeros_like(cm, dtype=float)
    np.divide(cm, row_sums, out=cm_norm, where=row_sums > 0)

    fig, ax = plt.subplots(figsize=(14, 12))
    # Log-ish colour scale so small off-diagonals (e.g. 0.01 - 0.05) are clearly visible
    norm = mcolors.PowerNorm(gamma=0.35, vmin=0.0, vmax=1.0)
    im = ax.imshow(cm_norm, cmap="Blues", norm=norm, interpolation="nearest")

    ax.set_xticks(range(72))
    ax.set_yticks(range(72))
    ax.set_xticklabels(CLASS_CHARS, fontsize=6, rotation=90)
    ax.set_yticklabels(CLASS_CHARS, fontsize=6)
    ax.set_xlabel("Predicted Class", fontsize=10, labelpad=6)
    ax.set_ylabel("True Class", fontsize=10, labelpad=6)

    title_text = f"Confusion Matrix: {exp_id} | Top-1 Acc: {top1*100:.2f}% | Bal Acc: {bal_acc*100:.2f}%"
    ax.set_title(title_text, fontsize=12, pad=12, fontweight="bold")

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Recall (Row-Normalized)", fontsize=9)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_per_class_recall(
    per_class_items: list[dict],
    exp_id: str,
    bal_acc: float,
    out_path: Path,
) -> None:
    """Bar chart of per-class recall sorted ascending with category colors and log n_train."""
    # Sort ascending by recall, using secondary key n_train
    sorted_items = sorted(
        per_class_items,
        key=lambda item: (
            item["recall"] if (item["recall"] is not None and not np.isnan(item["recall"])) else -1.0,
            item["n_train"],
        ),
    )

    n_classes = len(sorted_items)
    x = np.arange(n_classes)
    recalls = [
        item["recall"] if (item["recall"] is not None and not np.isnan(item["recall"])) else 0.0
        for item in sorted_items
    ]
    bar_colors = [CATEGORY_COLORS.get(item["category"], "#888888") for item in sorted_items]

    fig, ax1 = plt.subplots(figsize=(16, 6))

    # Recall bars
    bars = ax1.bar(x, recalls, color=bar_colors, width=0.75, alpha=0.88, zorder=3)
    ax1.set_xticks(x)
    ax1.set_xticklabels([item["char"] for item in sorted_items], fontsize=6, rotation=90)
    ax1.set_xlim(-0.8, n_classes - 0.2)
    ax1.set_ylim(0, 1.05)
    ax1.set_xlabel("Thai Character (TIS-620)", fontsize=10, labelpad=6)
    ax1.set_ylabel("Recall", fontsize=10, labelpad=6)
    ax1.grid(axis="y", linestyle=":", alpha=0.5, zorder=0)

    # Horizontal line at balanced accuracy
    line_bal = ax1.axhline(
        bal_acc,
        color="crimson",
        linestyle="--",
        linewidth=1.8,
        label=f"Balanced Acc ({bal_acc*100:.2f}%)",
        zorder=4,
    )

    # Secondary axis for log10(n_train)
    ax2 = ax1.twinx()
    log_train = [math.log10(max(1, item["n_train"])) for item in sorted_items]
    (line_train,) = ax2.plot(
        x,
        log_train,
        color="#444444",
        linestyle=":",
        marker="o",
        markersize=3,
        alpha=0.75,
        label=r"$\log_{10}(n_{\mathrm{train}})$",
        zorder=5,
    )
    ax2.set_ylabel(r"$\log_{10}(n_{\mathrm{train}})$", fontsize=10, color="#444444", labelpad=6)
    ax2.tick_params(axis="y", labelcolor="#444444")
    ax2.set_ylim(0, max(log_train) * 1.15 if log_train else 5)

    # Legend handles
    category_patches = [
        Patch(facecolor=CATEGORY_COLORS["consonant"], label="Consonant"),
        Patch(facecolor=CATEGORY_COLORS["vowel"], label="Vowel"),
        Patch(facecolor=CATEGORY_COLORS["tone_mark"], label="Tone / Mark"),
        Patch(facecolor=CATEGORY_COLORS["digit"], label="Digit"),
    ]
    all_handles = category_patches + [line_bal, line_train]
    ax1.legend(handles=all_handles, loc="upper left", framealpha=0.9, fontsize=9)

    ax1.set_title(
        f"Per-Class Recall (Sorted Ascending): {exp_id} (Bal Acc: {bal_acc*100:.2f}%)",
        fontsize=12,
        pad=10,
        fontweight="bold",
    )

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def build_confused_pairs_tables(
    cm: np.ndarray,
    n_train: list[int],
    topk: int,
) -> tuple[list[dict], list[dict], str]:
    """Compute directed and symmetric confused pairs and return markdown string."""
    row_sums = cm.sum(axis=1)

    # Directed off-diagonal cells
    directed = []
    for i in range(72):
        for j in range(72):
            if i != j and cm[i, j] > 0:
                count = int(cm[i, j])
                n_v = int(row_sums[i])
                pct = (count / n_v * 100.0) if n_v > 0 else 0.0
                directed.append({
                    "true_idx": i,
                    "pred_idx": j,
                    "true_code": CLASS_CODES[i],
                    "pred_code": CLASS_CODES[j],
                    "true_char": CLASS_CHARS[i],
                    "pred_char": CLASS_CHARS[j],
                    "count": count,
                    "pct_of_true": pct,
                    "n_train": int(n_train[i]),
                })

    directed.sort(key=lambda x: (x["count"], x["pct_of_true"]), reverse=True)

    directed_topk = directed[:topk]
    directed_rows = []
    for rank, d in enumerate(directed_topk, 1):
        d["rank"] = rank
        directed_rows.append([
            rank,
            f"{d['true_char']} ({d['true_code']})",
            f"{d['pred_char']} ({d['pred_code']})",
            d["count"],
            f"{d['pct_of_true']:.2f}%",
            d["n_train"],
        ])

    directed_headers = [
        "Rank", "True Glyph (Code)", "Pred Glyph (Code)", "Count", "% of True Class", "n_train (True)"
    ]
    directed_md = tabulate(directed_rows, headers=directed_headers, tablefmt="github")

    # Symmetric pairs merging a -> b and b -> a
    symmetric = []
    for i in range(72):
        for j in range(i + 1, 72):
            c_ij = int(cm[i, j])
            c_ji = int(cm[j, i])
            tot = c_ij + c_ji
            if tot > 0:
                symmetric.append({
                    "code_a": CLASS_CODES[i],
                    "char_a": CLASS_CHARS[i],
                    "code_b": CLASS_CODES[j],
                    "char_b": CLASS_CHARS[j],
                    "total_count": tot,
                    "count_a_to_b": c_ij,
                    "count_b_to_a": c_ji,
                    "n_train_a": int(n_train[i]),
                    "n_train_b": int(n_train[j]),
                })

    symmetric.sort(key=lambda x: (x["total_count"], max(x["count_a_to_b"], x["count_b_to_a"])), reverse=True)

    symmetric_topk = symmetric[:topk]
    symmetric_rows = []
    for rank, s in enumerate(symmetric_topk, 1):
        s["rank"] = rank
        symmetric_rows.append([
            rank,
            f"{s['char_a']} ({s['code_a']})",
            f"{s['char_b']} ({s['code_b']})",
            s["total_count"],
            s["count_a_to_b"],
            s["count_b_to_a"],
            s["n_train_a"],
            s["n_train_b"],
        ])

    symmetric_headers = [
        "Rank", "Glyph A (Code)", "Glyph B (Code)", "Total Count", "A → B", "B → A", "n_train (A)", "n_train (B)"
    ]
    symmetric_md = tabulate(symmetric_rows, headers=symmetric_headers, tablefmt="github")

    md_content = (
        f"## Top-{topk} Directed Off-Diagonal Confusions\n\n"
        f"{directed_md}\n\n"
        f"## Top-{topk} Symmetric Confused Pairs (Merged A ↔ B)\n\n"
        f"{symmetric_md}\n"
    )

    return directed_topk, symmetric_topk, md_content


def build_worst_classes_table(
    cm: np.ndarray,
    n_train: list[int],
    n_worst: int = 15,
) -> tuple[list[dict], str]:
    """Find lowest-recall classes with n_train, n_val, recall, and top-3 predicted labels."""
    row_sums = cm.sum(axis=1)

    evaluated = []
    unevaluated = []

    for i in range(72):
        n_v = int(row_sums[i])
        n_tr = int(n_train[i])
        code = CLASS_CODES[i]
        char = CLASS_CHARS[i]
        cat = category(code)

        if n_v > 0:
            rec = float(cm[i, i]) / n_v
            # Get top predictions with count > 0
            preds = [(j, int(cm[i, j])) for j in range(72) if cm[i, j] > 0]
            preds.sort(key=lambda x: x[1], reverse=True)
            top3 = [
                {
                    "code": CLASS_CODES[j],
                    "char": CLASS_CHARS[j],
                    "count": cnt,
                }
                for j, cnt in preds[:3]
            ]
            evaluated.append({
                "idx": i,
                "code": code,
                "char": char,
                "category": cat,
                "recall": rec,
                "n_val": n_v,
                "n_train": n_tr,
                "top3_pred": top3,
            })
        else:
            unevaluated.append({
                "idx": i,
                "code": code,
                "char": char,
                "category": cat,
                "n_train": n_tr,
            })

    evaluated.sort(key=lambda x: (x["recall"], -x["n_val"], -x["n_train"]))
    worst = evaluated[:n_worst]

    rows = []
    for rank, w in enumerate(worst, 1):
        w["rank"] = rank
        top3_str = ", ".join(f"{p['char']} ({p['code']}): {p['count']}" for p in w["top3_pred"])
        rows.append([
            rank,
            f"{w['char']} ({w['code']})",
            w["category"],
            f"{w['recall']*100:.2f}%",
            w["n_val"],
            w["n_train"],
            top3_str,
        ])

    headers = [
        "Rank", "Glyph (Code)", "Category", "Recall", "n_val", "n_train", "Top-3 Predicted Labels"
    ]
    md = f"## 15 Lowest-Recall Classes\n\n{tabulate(rows, headers=headers, tablefmt='github')}\n"

    if unevaluated:
        un_rows = [
            [f"{u['char']} ({u['code']})", u["category"], u["n_train"]]
            for u in unevaluated
        ]
        un_headers = ["Glyph (Code)", "Category", "n_train"]
        md += f"\n### Unevaluated Classes (n_val = 0)\n\n{tabulate(un_rows, headers=un_headers, tablefmt='github')}\n"

    return worst, md


def run_error_analysis(
    run_dir: str | Path,
    topk: int = 20,
    out_dir: str | Path | None = None,
) -> dict:
    """Execute complete error analysis pipeline for a given run directory."""
    run_path = Path(run_dir)
    if run_path.is_file():
        metrics_file = run_path
        run_path = run_path.parent
    else:
        metrics_file = run_path / "metrics.json"

    if not metrics_file.exists():
        raise FileNotFoundError(f"metrics.json not found in {run_path}")

    with open(metrics_file, encoding="utf-8") as f:
        metrics = json.load(f)

    exp_id = metrics.get("exp_id") or run_path.name
    top1 = float(metrics.get("top1", 0.0))
    bal_acc = float(metrics.get("balanced_acc", 0.0))

    cm = np.array(metrics["confusion"], dtype=float)
    if cm.shape != (72, 72):
        raise ValueError(f"Expected 72x72 confusion matrix, got {cm.shape}")

    n_train = metrics.get("class_counts_train")
    if not n_train or len(n_train) != 72:
        n_train = [0] * 72

    if out_dir is None:
        out_path = _REPO_ROOT / "reports" / "analysis" / exp_id
    else:
        out_path = Path(out_dir)

    out_path.mkdir(parents=True, exist_ok=True)

    # Register Thai font
    register_thai_font()

    # 1. confusion_matrix.png
    cm_png = out_path / "confusion_matrix.png"
    plot_confusion_matrix(cm, exp_id, top1, bal_acc, cm_png)

    # 2. confused_pairs.md
    directed_topk, symmetric_topk, pairs_md = build_confused_pairs_tables(cm, n_train, topk=topk)
    pairs_file = out_path / "confused_pairs.md"
    pairs_file.write_text(f"# Confused Pairs Analysis: {exp_id}\n\n{pairs_md}", encoding="utf-8")

    # 3. per_class_recall.png
    row_sums = cm.sum(axis=1)
    per_class_items = []
    for i in range(72):
        n_v = int(row_sums[i])
        rec = float(cm[i, i]) / n_v if n_v > 0 else np.nan
        per_class_items.append({
            "index": i,
            "code": CLASS_CODES[i],
            "char": CLASS_CHARS[i],
            "category": category(CLASS_CODES[i]),
            "recall": rec,
            "n_val": n_v,
            "n_train": int(n_train[i]),
        })

    recall_png = out_path / "per_class_recall.png"
    plot_per_class_recall(per_class_items, exp_id, bal_acc, recall_png)

    # 4. worst_classes.md
    worst_list, worst_md = build_worst_classes_table(cm, n_train, n_worst=15)
    worst_file = out_path / "worst_classes.md"
    worst_file.write_text(f"# Lowest-Recall Classes: {exp_id}\n\n{worst_md}", encoding="utf-8")

    # 5. summary.json
    summary = {
        "exp_id": exp_id,
        "top1": top1,
        "balanced_acc": bal_acc,
        "topk": topk,
        "top_confused_directed": directed_topk,
        "top_confused_symmetric": symmetric_topk,
        "worst_classes": worst_list,
        "per_class": per_class_items,
    }
    summary_file = out_path / "summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Error analysis complete for {exp_id}. Output written to {out_path}")
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Error analysis for Thai character model run.")
    parser.add_argument("--run", type=str, required=True, help="Path to run directory (e.g. runs/A1_resnet18_full_64_T4)")
    parser.add_argument("--topk", type=int, default=20, help="Number of top confused pairs to report (default: 20)")
    parser.add_argument("--out", type=str, default=None, help="Output directory (default: reports/analysis/<exp_id>)")
    args = parser.parse_args(argv)

    run_error_analysis(run_dir=args.run, topk=args.topk, out_dir=args.out)


if __name__ == "__main__":
    main()
