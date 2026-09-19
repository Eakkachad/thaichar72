"""CLI helper for single or multi-image Thai glyph prediction with optional montage."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image

from thaichar.engine import pick_device
from thaichar.infer import load_checkpoint, predict_topk

NUM_CLASSES = 72
FONT_PATH = "assets/fonts/Sarabun-Regular.ttf"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict Thai character classes for input images.")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to trained checkpoint (best.pt)")
    parser.add_argument("images", nargs="+", type=str, help="Paths to input images")
    parser.add_argument("--topk", type=int, default=5, help="Number of top predictions to display")
    parser.add_argument("--tau", type=float, default=0.0, help="Logit adjustment temperature tau")
    parser.add_argument("--montage", type=str, default=None, help="Output image file to save montage visualization")
    parser.add_argument("--device", type=str, default="auto", help="Device to use (cpu, cuda, auto)")
    return parser.parse_args()


def create_montage(
    image_paths: list[str],
    predictions: list[list[tuple[str, int, float]]],
    out_path: str,
    font_path: str = FONT_PATH,
) -> None:
    """Create a montage grid of input images and their top-3 predicted characters."""
    n = len(image_paths)
    if n == 0:
        return

    n_cols = min(n, 5)
    n_rows = math.ceil(n / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.2 * n_cols, 3.8 * n_rows))
    axes_flat = axes.flatten() if n > 1 else [axes]

    font_prop = fm.FontProperties(fname=font_path) if Path(font_path).exists() else None

    for idx, (img_path, topk) in enumerate(zip(image_paths, predictions)):
        ax = axes_flat[idx]
        try:
            pil_img = Image.open(img_path)
            ax.imshow(pil_img, cmap="gray" if pil_img.mode == "L" else None)
        except Exception as e:
            ax.text(0.5, 0.5, f"Error:\n{e}", ha="center", va="center", fontsize=8)

        ax.axis("off")

        # Display top 3
        top3 = topk[:3]
        title_lines = [f"{char} ({code}): {prob:.1%}" for char, code, prob in top3]
        title_text = f"{Path(img_path).name}\n" + "\n".join(title_lines)
        ax.set_title(title_text, fontproperties=font_prop, fontsize=10)

    # Hide extra unused subplots
    for idx in range(n, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved montage visualization to {out_path}")


def main() -> None:
    args = parse_args()
    device = pick_device(args.device)

    model, cfg = load_checkpoint(args.ckpt, device=device)
    model = model.to(device)

    # Prepare log_prior if tau != 0.0
    log_prior = None
    if float(args.tau) != 0.0:
        split_kind = cfg.get("split_kind", "strat")
        split_col = {"strat": "split", "doc": "doc_split"}[split_kind]
        split_file = cfg.get("split_file", "data/splits/split_seed42.csv")
        df = pd.read_csv(split_file)
        train_df = df[df[split_col] == "train"]
        counts = np.bincount(train_df["label"].values, minlength=NUM_CLASSES)
        log_prior = np.log(np.clip(counts, 1, None) / counts.sum()).astype(np.float32)

    all_preds: list[list[tuple[str, int, float]]] = []

    # Header and formatting
    max_path_len = max(len(str(p)) for p in args.images)
    max_path_len = max(max_path_len, 10)

    print(f"{'File':<{max_path_len}}  Top-{args.topk} Predictions (Char [Code]: Prob)")
    print("-" * (max_path_len + 55))

    for img_path in args.images:
        preds = predict_topk(
            model=model,
            cfg=cfg,
            img=img_path,
            k=args.topk,
            tau=args.tau,
            log_prior=log_prior,
        )
        all_preds.append(preds)

        pred_strs = [f"{char} ({code}): {prob:.4f}" for char, code, prob in preds]
        row_str = " | ".join(pred_strs)
        print(f"{img_path:<{max_path_len}}  {row_str}")

    if args.montage:
        create_montage(args.images, all_preds, args.montage)


if __name__ == "__main__":
    main()
