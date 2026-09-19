#!/usr/bin/env python
"""Plotting script for experiment results and comparisons.

Produces:
1. training_curves_<exp_id>.png — train loss + val top1/bal acc vs epoch (two panels),
   with dashed EMA curves if present.
2. matrix_<prefix>_bar.png — grouped bars of top1 / balanced_acc / minority_acc from
   reports/experiments.csv for all runs matching prefix, sorted by balanced_acc, with value labels.
3. tau_sweep_<exp_id>.png — top1 and balanced acc vs tau from metrics.json["tau_sweep"].
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd

# Ensure src/ is on sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))


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


def plot_training_curves(log_csv_path: Path, exp_id: str, out_path: Path) -> None:
    """Plot training curves: train loss (panel 1) and val accuracies with EMA (panel 2)."""
    df = pd.read_csv(log_csv_path)

    epochs = pd.to_numeric(df["epoch"], errors="coerce")
    train_loss = pd.to_numeric(df["train_loss"], errors="coerce")
    val_top1 = pd.to_numeric(df.get("val_top1"), errors="coerce")
    val_bal = pd.to_numeric(df.get("val_bal_acc"), errors="coerce")

    ema_top1 = pd.to_numeric(df.get("ema_val_top1"), errors="coerce") if "ema_val_top1" in df else None
    ema_bal = pd.to_numeric(df.get("ema_val_bal_acc"), errors="coerce") if "ema_val_bal_acc" in df else None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))

    # Panel 1: Train Loss vs Epoch
    ax1.plot(epochs, train_loss, color="#1f77b4", marker="o", markersize=4, linewidth=1.8, label="Train Loss")
    ax1.set_xlabel("Epoch", fontsize=10)
    ax1.set_ylabel("Loss", fontsize=10)
    ax1.set_title(f"Train Loss: {exp_id}", fontsize=11, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper right", fontsize=9)

    # Panel 2: Val Accuracies vs Epoch
    if val_top1 is not None and val_top1.notna().any():
        ax2.plot(epochs, val_top1, color="#2ca02c", marker="o", markersize=4, linewidth=1.8, label="Val Top-1")
    if val_bal is not None and val_bal.notna().any():
        ax2.plot(epochs, val_bal, color="#d62728", marker="s", markersize=4, linewidth=1.8, label="Val Balanced Acc")

    # Dashed EMA curves if present
    if ema_top1 is not None:
        valid_mask = ema_top1.notna()
        if valid_mask.any():
            ax2.plot(
                epochs[valid_mask],
                ema_top1[valid_mask],
                color="#2ca02c",
                linestyle="--",
                marker="^",
                markersize=5,
                linewidth=1.8,
                label="EMA Val Top-1",
            )
    if ema_bal is not None:
        valid_mask = ema_bal.notna()
        if valid_mask.any():
            ax2.plot(
                epochs[valid_mask],
                ema_bal[valid_mask],
                color="#d62728",
                linestyle="--",
                marker="v",
                markersize=5,
                linewidth=1.8,
                label="EMA Val Balanced Acc",
            )

    ax2.set_xlabel("Epoch", fontsize=10)
    ax2.set_ylabel("Accuracy", fontsize=10)
    ax2.set_title(f"Val Accuracy: {exp_id}", fontsize=11, fontweight="bold")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="lower right", fontsize=9)

    fig.suptitle(f"Training Dynamics: {exp_id}", fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_matrix_bar(csv_path: Path, prefix: str, out_path: Path) -> None:
    """Grouped bar chart of top1, balanced_acc, minority_acc for runs matching prefix, sorted by balanced_acc."""
    df = pd.read_csv(csv_path)

    # Filter by prefix
    df_match = df[df["exp_id"].astype(str).str.startswith(prefix)].copy()
    if df_match.empty:
        print(f"Warning: No runs found starting with prefix '{prefix}' in {csv_path}")
        return

    # Ensure required columns are float
    for col in ["top1", "balanced_acc", "minority_acc"]:
        if col in df_match.columns:
            df_match[col] = pd.to_numeric(df_match[col], errors="coerce").fillna(0.0)
        else:
            df_match[col] = 0.0

    # Sort by balanced_acc descending (standard leaderboard ranking)
    df_sorted = df_match.sort_values(by="balanced_acc", ascending=False).reset_index(drop=True)

    n_runs = len(df_sorted)
    x = np.arange(n_runs)
    width = 0.26

    fig, ax = plt.subplots(figsize=(max(8, n_runs * 1.5), 6))

    rects1 = ax.bar(x - width, df_sorted["top1"], width, label="Top-1 Acc", color="#1f77b4", alpha=0.9)
    rects2 = ax.bar(x, df_sorted["balanced_acc"], width, label="Balanced Acc", color="#2ca02c", alpha=0.9)
    rects3 = ax.bar(x + width, df_sorted["minority_acc"], width, label="Minority Acc", color="#ff7f0e", alpha=0.9)

    # Value labels on bars
    ax.bar_label(rects1, fmt="%.3f", rotation=90, padding=4, fontsize=7)
    ax.bar_label(rects2, fmt="%.3f", rotation=90, padding=4, fontsize=7)
    ax.bar_label(rects3, fmt="%.3f", rotation=90, padding=4, fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(df_sorted["exp_id"], rotation=30, ha="right", fontsize=8)
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("Accuracy", fontsize=10)
    ax.set_title(f"Experiment Comparison (Prefix: {prefix}*, Sorted by Balanced Acc)", fontsize=12, fontweight="bold", pad=12)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_tau_sweep(metrics_json_path: Path, exp_id: str, out_path: Path) -> None:
    """Plot top1 and balanced acc vs tau from metrics.json['tau_sweep']."""
    with open(metrics_json_path, encoding="utf-8") as f:
        metrics = json.load(f)

    tau_sweep = metrics.get("tau_sweep")
    if not tau_sweep:
        print(f"Notice: No tau_sweep data found for {exp_id}, skipping tau_sweep plot.")
        return

    # Parse and sort taus
    sorted_taus_str = sorted(tau_sweep.keys(), key=lambda t: float(t))
    taus = [float(t) for t in sorted_taus_str]
    top1s = [tau_sweep[t].get("top1", 0.0) for t in sorted_taus_str]
    bal_accs = [tau_sweep[t].get("balanced_acc", 0.0) for t in sorted_taus_str]
    minority_accs = [tau_sweep[t].get("minority_acc") for t in sorted_taus_str]

    fig, ax = plt.subplots(figsize=(7, 4.8))

    ax.plot(taus, top1s, marker="o", markersize=6, linewidth=2, color="#1f77b4", label="Top-1 Acc")
    ax.plot(taus, bal_accs, marker="s", markersize=6, linewidth=2, color="#2ca02c", label="Balanced Acc")

    if all(m is not None for m in minority_accs):
        ax.plot(taus, minority_accs, marker="^", markersize=5, linewidth=1.5, linestyle=":", color="#ff7f0e", label="Minority Acc")

    # Mark best tau for balanced acc
    best_idx = int(np.argmax(bal_accs))
    best_tau = taus[best_idx]
    best_bal = bal_accs[best_idx]
    ax.axvline(best_tau, color="gray", linestyle="--", alpha=0.6, label=f"Best tau={best_tau} ({best_bal*100:.2f}%)")

    ax.set_xlabel("Tau (Logit Adjustment Temperature)", fontsize=10)
    ax.set_ylabel("Accuracy", fontsize=10)
    ax.set_title(f"Post-Hoc Prior Adjustment: {exp_id}", fontsize=11, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="best", fontsize=9)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Plot experiment results and comparisons.")
    parser.add_argument("--runs", default="runs", help="Runs directory (default: runs)")
    parser.add_argument("--exp", action="append", default=[], help="Experiment IDs to plot (repeatable)")
    parser.add_argument("--prefix", default="A", help="Prefix for matrix comparison bar chart (default: A)")
    parser.add_argument("--out", default="reports/figures", help="Output directory for figures (default: reports/figures)")
    parser.add_argument("--csv", default="reports/experiments.csv", help="Path to experiments.csv (default: reports/experiments.csv)")
    args = parser.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = Path(args.runs)

    # Register Thai font
    register_thai_font()

    # Determine which experiments to process
    exp_ids = args.exp
    if not exp_ids:
        # If none specified, find all subdirectories in runs_dir with metrics.json or log.csv
        if runs_dir.exists():
            exp_ids = [
                p.name for p in sorted(runs_dir.iterdir())
                if p.is_dir() and ((p / "metrics.json").exists() or (p / "log.csv").exists())
            ]

    # 1 & 3: For each experiment, plot training curves and tau sweep
    for exp_id in exp_ids:
        run_path = runs_dir / exp_id

        # 1. Training curves
        log_csv = run_path / "log.csv"
        if log_csv.exists():
            curves_path = out_dir / f"training_curves_{exp_id}.png"
            plot_training_curves(log_csv, exp_id, curves_path)
            print(f"Generated {curves_path}")
        else:
            print(f"Notice: No log.csv found for {exp_id} in {run_path}")

        # 3. Tau sweep
        metrics_json = run_path / "metrics.json"
        if metrics_json.exists():
            tau_path = out_dir / f"tau_sweep_{exp_id}.png"
            plot_tau_sweep(metrics_json, exp_id, tau_path)
            print(f"Generated {tau_path}")
        else:
            print(f"Notice: No metrics.json found for {exp_id} in {run_path}")

    # 2. Matrix bar comparison
    csv_path = Path(args.csv)
    if csv_path.exists():
        bar_path = out_dir / f"matrix_{args.prefix}_bar.png"
        plot_matrix_bar(csv_path, args.prefix, bar_path)
        print(f"Generated {bar_path}")
    else:
        print(f"Notice: {csv_path} not found, skipping matrix bar chart.")


if __name__ == "__main__":
    main()
