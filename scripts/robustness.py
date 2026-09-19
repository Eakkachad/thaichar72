"""Evaluate model robustness under deterministic canvas-level corruptions."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from thaichar.data import ThaiGlyphDataset, load_cache
from thaichar.engine import pick_device, predict
from thaichar.infer import CORRUPTIONS, apply_corruption, load_checkpoint
from thaichar.metrics import compute_metrics

NUM_CLASSES = 72


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate checkpoint robustness under corruptions.")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint (best.pt)")
    parser.add_argument("--split-kind", type=str, default=None, choices=["strat", "doc"], help="Validation split kind")
    parser.add_argument("--limit", type=int, default=None, help="Limit samples per class-stratified subset")
    parser.add_argument("--out", type=str, default=None, help="Output directory for reports")
    parser.add_argument("--device", type=str, default="auto", help="Device (cpu, cuda, auto)")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for evaluation")
    parser.add_argument("--num-workers", type=int, default=0, help="Dataloader workers")
    parser.add_argument("--binarize", action="store_true", help="Apply Otsu binarisation after the corruption (simulates the deployed preprocess_image pipeline)")
    return parser.parse_args()


def plot_curves(results_df: pd.DataFrame, clean_metrics: dict[str, float], out_path: Path) -> None:
    """Plot one panel per corruption comparing top1 and balanced_acc vs severity."""
    corruptions = [c for c in CORRUPTIONS.keys() if c in results_df["corruption"].values]
    n_panels = len(corruptions)
    if n_panels == 0:
        return

    n_cols = 5
    n_rows = math.ceil(n_panels / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows), sharey=True)
    axes_flat = axes.flatten() if n_panels > 1 else [axes]

    clean_top1 = clean_metrics["top1"]
    clean_bacc = clean_metrics["balanced_acc"]

    for idx, name in enumerate(corruptions):
        ax = axes_flat[idx]
        sub = results_df[results_df["corruption"] == name]
        sevs = sub["severity"].values
        top1_vals = sub["top1"].values
        bacc_vals = sub["balanced_acc"].values

        x_labels = [str(s) for s in sevs]
        x_indices = np.arange(len(x_labels))

        ax.plot(x_indices, top1_vals, "o-", color="#1f77b4", label="Top-1")
        ax.plot(x_indices, bacc_vals, "s--", color="#ff7f0e", label="Balanced Acc")

        ax.axhline(clean_top1, color="#1f77b4", linestyle=":", alpha=0.7, label="Clean Top-1" if idx == 0 else "")
        ax.axhline(clean_bacc, color="#ff7f0e", linestyle=":", alpha=0.7, label="Clean Bal-Acc" if idx == 0 else "")

        ax.set_title(name, fontsize=11, fontweight="bold")
        ax.set_xticks(x_indices)
        ax.set_xticklabels(x_labels, rotation=30, fontsize=8)
        ax.set_xlabel("Severity", fontsize=9)
        if idx % n_cols == 0:
            ax.set_ylabel("Accuracy", fontsize=9)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.3, linestyle="--")

    # Hide extra unused subplots
    for idx in range(n_panels, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    axes_flat[0].legend(loc="lower left", fontsize=8)
    fig.suptitle("Model Robustness Under Canvas Corruptions", fontsize=14, y=1.01)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_summary(
    results_df: pd.DataFrame,
    clean_row: dict[str, Any],
    ckpt_path: str,
    n_samples: int,
    out_path: Path,
) -> None:
    """Write comprehensive markdown summary report."""
    clean_top1 = clean_row["top1"]
    clean_bacc = clean_row["balanced_acc"]
    clean_mf1 = clean_row["macro_f1"]

    corrupt_rows = results_df[results_df["corruption"] != "clean"]

    # Calculate worst performance per corruption
    summary_per_corr = []
    for name, grp in corrupt_rows.groupby("corruption", sort=False):
        worst_row = grp.loc[grp["balanced_acc"].idxmin()]
        summary_per_corr.append({
            "corruption": name,
            "worst_severity": worst_row["severity"],
            "worst_top1": worst_row["top1"],
            "worst_balanced_acc": worst_row["balanced_acc"],
            "worst_macro_f1": worst_row["macro_f1"],
            "drop_top1": clean_top1 - worst_row["top1"],
            "drop_bacc": clean_bacc - worst_row["balanced_acc"],
        })

    sum_df = pd.DataFrame(summary_per_corr).sort_values("worst_balanced_acc", ascending=True)
    worst_overall = sum_df.iloc[0] if len(sum_df) > 0 else None

    lines = [
        "# Robustness Evaluation Summary",
        "",
        f"- **Checkpoint**: `{ckpt_path}`",
        f"- **Evaluation samples (n)**: {n_samples}",
        f"- **Clean Top-1**: {clean_top1:.4f} ({clean_top1 * 100:.2f}%)",
        f"- **Clean Balanced Accuracy**: {clean_bacc:.4f} ({clean_bacc * 100:.2f}%)",
        f"- **Clean Macro F1**: {clean_mf1:.4f}",
        "",
    ]

    if worst_overall is not None:
        lines.extend([
            "## Most Damaging Corruption",
            f"- **Corruption**: `{worst_overall['corruption']}` (severity: `{worst_overall['worst_severity']}`)",
            f"- **Worst Top-1**: {worst_overall['worst_top1']:.4f} (drop: -{worst_overall['drop_top1']:.4f})",
            f"- **Worst Balanced Accuracy**: {worst_overall['worst_balanced_acc']:.4f} (drop: -{worst_overall['drop_bacc']:.4f})",
            "",
        ])

    lines.extend([
        "## Worst-Severity Summary by Corruption",
        "",
        "| Corruption | Worst Sev | Top-1 | Bal Acc | Macro F1 | Top-1 Drop | Bal Acc Drop |",
        "|:-----------|:---------:|------:|--------:|---------:|-----------:|-------------:|",
    ])

    for _, r in sum_df.iterrows():
        lines.append(
            f"| {r['corruption']} | {r['worst_severity']} | "
            f"{r['worst_top1']:.4f} | {r['worst_balanced_acc']:.4f} | {r['worst_macro_f1']:.4f} | "
            f"-{r['drop_top1']:.4f} | -{r['drop_bacc']:.4f} |"
        )

    lines.extend([
        "",
        "## Detailed Results",
        "",
        "| Corruption | Severity | Top-1 | Balanced Acc | Macro F1 | N |",
        "|:-----------|:---------|------:|-------------:|---------:|--:|",
    ])

    for _, r in results_df.iterrows():
        lines.append(
            f"| {r['corruption']} | {r['severity']} | "
            f"{r['top1']:.4f} | {r['balanced_acc']:.4f} | {r['macro_f1']:.4f} | {r['n']} |"
        )

    lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def _otsu(canvas):
    """Otsu threshold → {0,255} uint8, ink dark on white (border ring decides polarity)."""
    import cv2, numpy as np
    _, t = cv2.threshold(canvas.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    border = np.concatenate([t[0, :], t[-1, :], t[:, 0], t[:, -1]])
    return (255 - t) if border.mean() < 127.5 else t


def main() -> None:
    args = parse_args()
    device = pick_device(args.device)

    model, cfg = load_checkpoint(args.ckpt, device=device)
    model = model.to(device)

    split_kind = args.split_kind or cfg.get("split_kind", "strat")
    split_col = {"strat": "split", "doc": "doc_split"}[split_kind]
    df = pd.read_csv(cfg["split_file"])
    val_df = df[df[split_col] == "val"].reset_index(drop=True)
    train_df = df[df[split_col] == "train"].reset_index(drop=True)
    class_counts_train = np.bincount(train_df["label"].values, minlength=NUM_CLASSES)

    cache = load_cache(cfg["cache"])

    if args.limit is not None and args.limit > 0:
        val_df = val_df.groupby("label", sort=True).head(args.limit).reset_index(drop=True)

    batch_size = args.batch_size
    img_size = int(cfg["img_size"])
    channel_mode = cfg["channel_mode"]
    margin = float(cfg["margin"])

    out_dir = Path(args.out) if args.out else Path("reports/robustness") / Path(args.ckpt).parent.name
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []

    # 1. Clean evaluation
    print(f"Evaluating clean validation set ({len(val_df)} samples)...")
    clean_ds = ThaiGlyphDataset(
        val_df,
        cache,
        size=img_size,
        channel_mode=channel_mode,
        transform=None,
        margin=margin,
    )
    clean_loader = DataLoader(clean_ds, batch_size=batch_size, shuffle=False, num_workers=args.num_workers)
    logits, y = predict(model, clean_loader, device)
    clean_m = compute_metrics(y, logits, class_counts_train)
    clean_row = {
        "corruption": "clean",
        "severity": 0,
        "top1": float(clean_m["top1"]),
        "balanced_acc": float(clean_m["balanced_acc"]),
        "macro_f1": float(clean_m["macro_f1"]),
        "n": int(len(y)),
    }
    rows.append(clean_row)
    print(f"Clean: Top-1={clean_m['top1']:.4f}, BalAcc={clean_m['balanced_acc']:.4f}, MacroF1={clean_m['macro_f1']:.4f}")

    # 2. Corruptions evaluation
    for name, severities in CORRUPTIONS.items():
        for sev in severities:
            print(f"Evaluating {name} (severity={sev})...")
            corrupt_ds = ThaiGlyphDataset(
                val_df,
                cache,
                size=img_size,
                channel_mode=channel_mode,
                transform=(lambda c, n=name, s=sev: _otsu(apply_corruption(c, n, s))) if args.binarize else (lambda c, n=name, s=sev: apply_corruption(c, n, s)),
                margin=margin,
            )
            loader = DataLoader(corrupt_ds, batch_size=batch_size, shuffle=False, num_workers=args.num_workers)
            c_logits, c_y = predict(model, loader, device)
            m = compute_metrics(c_y, c_logits, class_counts_train)
            row = {
                "corruption": name,
                "severity": sev,
                "top1": float(m["top1"]),
                "balanced_acc": float(m["balanced_acc"]),
                "macro_f1": float(m["macro_f1"]),
                "n": int(len(c_y)),
            }
            rows.append(row)
            print(f"  -> Top-1={m['top1']:.4f}, BalAcc={m['balanced_acc']:.4f}, MacroF1={m['macro_f1']:.4f}")

    # 3. Output files
    results_df = pd.DataFrame(rows)
    csv_path = out_dir / "results.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"Saved results table to {csv_path}")

    plot_path = out_dir / "curves.png"
    plot_curves(results_df, clean_row, plot_path)
    print(f"Saved curves plot to {plot_path}")

    summary_path = out_dir / "summary.md"
    write_summary(results_df, clean_row, args.ckpt, len(val_df), summary_path)
    print(f"Saved summary report to {summary_path}")


if __name__ == "__main__":
    main()
