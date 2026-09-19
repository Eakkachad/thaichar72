"""Tests for scripts/result_figures.py (TASK-12)."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import List

import pandas as pd
import pytest

# Ensure scripts/ and src/ are importable
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "scripts"))
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

import result_figures

EXPECTED_FIGURE_NAMES = [
    "fig_A_transfer_modes",
    "fig_B_aug_ladder",
    "fig_B_mix",
    "fig_D_imbalance_tradeoff",
    "fig_S_size_sweep",
    "fig_E_tricks",
    "fig_gap_strat_vs_doc",
    "fig_training_curves_best",
    "fig_tau_sweep_grid",
]


@pytest.fixture
def fake_experiments_and_runs(tmp_path: Path):
    """Build a minimal fake CSV with a handful of rows (both suffixes) and minimal runs."""
    csv_path = tmp_path / "experiments.csv"
    runs_dir = tmp_path / "runs"
    out_dir = tmp_path / "results"
    runs_dir.mkdir(parents=True)
    out_dir.mkdir(parents=True)

    # Handful of representative rows covering both _T4 and _doc suffixes
    rows = [
        # A* transfer modes
        {
            "exp_id": "A0_smallcnn_64_T4", "model": "smallcnn", "img_size": 64, "mode": "full",
            "aug": "base", "loss": "ce", "sampler": "none", "mixup": 0.0, "cutmix": 0.0, "extra": "-",
            "split_kind": "strat", "top1": 0.974, "balanced_acc": 0.918, "minority_acc": 0.890,
            "sec_per_epoch": 20.0, "latency_ms": 10.0,
        },
        {
            "exp_id": "A0_smallcnn_64_doc", "model": "smallcnn", "img_size": 64, "mode": "full",
            "aug": "base", "loss": "ce", "sampler": "none", "mixup": 0.0, "cutmix": 0.0, "extra": "-",
            "split_kind": "doc", "top1": 0.956, "balanced_acc": 0.828, "minority_acc": 0.800,
            "sec_per_epoch": 20.0, "latency_ms": 10.0,
        },
        {
            "exp_id": "A1_resnet18_full_64_T4", "model": "resnet18", "img_size": 64, "mode": "full",
            "aug": "base", "loss": "ce", "sampler": "none", "mixup": 0.0, "cutmix": 0.0, "extra": "-",
            "split_kind": "strat", "top1": 0.975, "balanced_acc": 0.979, "minority_acc": 0.966,
            "sec_per_epoch": 34.0, "latency_ms": 12.0,
        },
        {
            "exp_id": "A1_resnet18_full_64_doc", "model": "resnet18", "img_size": 64, "mode": "full",
            "aug": "base", "loss": "ce", "sampler": "none", "mixup": 0.0, "cutmix": 0.0, "extra": "-",
            "split_kind": "doc", "top1": 0.963, "balanced_acc": 0.936, "minority_acc": 0.920,
            "sec_per_epoch": 34.0, "latency_ms": 12.0,
        },
        {
            "exp_id": "A1_resnet18_frozen_64_T4", "model": "resnet18", "img_size": 64, "mode": "frozen",
            "aug": "base", "loss": "ce", "sampler": "none", "mixup": 0.0, "cutmix": 0.0, "extra": "-",
            "split_kind": "strat", "top1": 0.856, "balanced_acc": 0.563, "minority_acc": 0.216,
            "sec_per_epoch": 18.0, "latency_ms": 11.0,
        },
        {
            "exp_id": "A1_resnet18_partial_64_T4", "model": "resnet18", "img_size": 64, "mode": "partial",
            "aug": "base", "loss": "ce", "sampler": "none", "mixup": 0.0, "cutmix": 0.0, "extra": "-",
            "split_kind": "strat", "top1": 0.962, "balanced_acc": 0.906, "minority_acc": 0.875,
            "sec_per_epoch": 25.0, "latency_ms": 11.5,
        },
        # Augmentation ladder rows
        {
            "exp_id": "B_none_resnet18_64_T4", "model": "resnet18", "img_size": 64, "mode": "full",
            "aug": "none", "loss": "ce", "sampler": "none", "mixup": 0.0, "cutmix": 0.0, "extra": "-",
            "split_kind": "strat", "top1": 0.984, "balanced_acc": 0.981, "minority_acc": 0.966,
            "sec_per_epoch": 19.0, "latency_ms": 14.0,
        },
        {
            "exp_id": "D0_ce_ls_resnet18_64_T4", "model": "resnet18", "img_size": 64, "mode": "full",
            "aug": "full", "loss": "ce", "sampler": "none", "mixup": 0.0, "cutmix": 0.0, "extra": "-",
            "split_kind": "strat", "top1": 0.978, "balanced_acc": 0.964, "minority_acc": 0.958,
            "sec_per_epoch": 34.2, "latency_ms": 15.9,
        },
        {
            "exp_id": "B_full_resnet18_64_doc", "model": "resnet18", "img_size": 64, "mode": "full",
            "aug": "full", "loss": "ce", "sampler": "none", "mixup": 0.0, "cutmix": 0.0, "extra": "-",
            "split_kind": "doc", "top1": 0.972, "balanced_acc": 0.961, "minority_acc": 0.948,
            "sec_per_epoch": 34.4, "latency_ms": 12.0,
        },
        {
            "exp_id": "B_full_mixup_resnet18_64_T4", "model": "resnet18", "img_size": 64, "mode": "full",
            "aug": "full", "loss": "ce", "sampler": "none", "mixup": 0.2, "cutmix": 0.0, "extra": "-",
            "split_kind": "strat", "top1": 0.976, "balanced_acc": 0.949, "minority_acc": 0.908,
            "sec_per_epoch": 33.9, "latency_ms": 15.6,
        },
        {
            "exp_id": "B_full_cutmix_resnet18_64_T4", "model": "resnet18", "img_size": 64, "mode": "full",
            "aug": "full", "loss": "ce", "sampler": "none", "mixup": 0.0, "cutmix": 1.0, "extra": "-",
            "split_kind": "strat", "top1": 0.973, "balanced_acc": 0.940, "minority_acc": 0.908,
            "sec_per_epoch": 34.7, "latency_ms": 14.0,
        },
        # Imbalance rows including collapse
        {
            "exp_id": "D1_wce_inv_resnet18_64_T4", "model": "resnet18", "img_size": 64, "mode": "full",
            "aug": "full", "loss": "weighted_ce", "sampler": "none", "mixup": 0.0, "cutmix": 0.0, "extra": "-",
            "split_kind": "strat", "top1": 0.116, "balanced_acc": 0.604, "minority_acc": 0.975,
            "sec_per_epoch": 34.8, "latency_ms": 11.3,
        },
        {
            "exp_id": "D1_wce_sqrt_resnet18_64_T4", "model": "resnet18", "img_size": 64, "mode": "full",
            "aug": "full", "loss": "weighted_ce", "sampler": "none", "mixup": 0.0, "cutmix": 0.0, "extra": "-",
            "split_kind": "strat", "top1": 0.972, "balanced_acc": 0.984, "minority_acc": 0.975,
            "sec_per_epoch": 34.8, "latency_ms": 11.7,
        },
        {
            "exp_id": "D3_sampler_inv_cap10_resnet18_64_T4", "model": "resnet18", "img_size": 64, "mode": "full",
            "aug": "full", "loss": "ce", "sampler": "inv", "mixup": 0.0, "cutmix": 0.0, "extra": "-",
            "split_kind": "strat", "top1": 0.969, "balanced_acc": 0.984, "minority_acc": 0.975,
            "sec_per_epoch": 34.8, "latency_ms": 11.3,
        },
        # Tricks
        {
            "exp_id": "E5_geometry_resnet18_64_T4", "model": "resnet18", "img_size": 64, "mode": "full",
            "aug": "full", "loss": "ce", "sampler": "none", "mixup": 0.0, "cutmix": 0.0, "extra": "-",
            "split_kind": "strat", "top1": 0.977, "balanced_acc": 0.969, "minority_acc": 0.950,
            "sec_per_epoch": 35.0, "latency_ms": 12.0,
        },
        # Size sweep
        {
            "exp_id": "S_resnet18_32_T4", "model": "resnet18", "img_size": 32, "mode": "full",
            "aug": "full", "loss": "ce", "sampler": "none", "mixup": 0.0, "cutmix": 0.0, "extra": "-",
            "split_kind": "strat", "top1": 0.975, "balanced_acc": 0.958, "minority_acc": 0.940,
            "sec_per_epoch": 29.5, "latency_ms": 9.0,
        },
        {
            "exp_id": "S_resnet18_224_T4", "model": "resnet18", "img_size": 224, "mode": "full",
            "aug": "full", "loss": "ce", "sampler": "none", "mixup": 0.0, "cutmix": 0.0, "extra": "-",
            "split_kind": "strat", "top1": 0.977, "balanced_acc": 0.967, "minority_acc": 0.950,
            "sec_per_epoch": 116.5, "latency_ms": 71.1,
        },
    ]
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    # Minimal run folders with log.csv and metrics.json for figure 8 & 9
    sample_runs = [
        "D3_sampler_inv_cap10_resnet18_64_T4",
        "B_none_resnet18_64_T4",
        "D0_ce_ls_resnet18_64_T4",
        "D1_wce_sqrt_resnet18_64_T4",
        "B_full_mixup_resnet18_64_T4",
    ]
    for r in sample_runs:
        rd = runs_dir / r
        rd.mkdir(parents=True, exist_ok=True)
        # log.csv
        log_df = pd.DataFrame({
            "epoch": [1, 2, 3],
            "train_loss": [1.5, 0.9, 0.8],
            "val_bal_acc": [0.85, 0.94, 0.97],
            "val_top1": [0.90, 0.95, 0.97],
        })
        log_df.to_csv(rd / "log.csv", index=False)

        # metrics.json
        metrics_data = {
            "exp_id": r,
            "tau_sweep": {
                "0.0": {"top1": 0.975, "balanced_acc": 0.964},
                "0.5": {"top1": 0.970, "balanced_acc": 0.978},
                "0.75": {"top1": 0.965, "balanced_acc": 0.980},
            },
        }
        with open(rd / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics_data, f)

    return csv_path, runs_dir, out_dir


def test_result_figures_all_figures_created(fake_experiments_and_runs):
    """Assert all 9 PNG figures exist, are > 5 kB, and have matching SVG copies and README."""
    csv_path, runs_dir, out_dir = fake_experiments_and_runs

    argv = [
        "--csv", str(csv_path),
        "--runs-dir", str(runs_dir),
        "--out", str(out_dir),
    ]
    ret = result_figures.main(argv)
    assert ret == 0

    for fig_name in EXPECTED_FIGURE_NAMES:
        png_file = out_dir / f"{fig_name}.png"
        svg_file = out_dir / f"{fig_name}.svg"

        assert png_file.exists(), f"Expected PNG missing: {png_file}"
        assert svg_file.exists(), f"Expected SVG missing: {svg_file}"

        png_size = png_file.stat().st_size
        svg_size = svg_file.stat().st_size

        assert png_size > 5000, f"PNG {fig_name}.png is too small ({png_size} bytes, expected > 5 kB)"
        assert svg_size > 1000, f"SVG {fig_name}.svg is too small ({svg_size} bytes)"

    readme_file = out_dir / "README.md"
    assert readme_file.exists()
    assert readme_file.stat().st_size > 200


def test_result_figures_tolerates_missing_groups(tmp_path: Path):
    """Assert script tolerates completely missing groups without crashing."""
    csv_path = tmp_path / "sparse.csv"
    empty_runs = tmp_path / "empty_runs"
    out_dir = tmp_path / "sparse_out"
    empty_runs.mkdir(parents=True)
    out_dir.mkdir(parents=True)

    # Bare minimum: 2 rows, no doc rows, no E tricks, no sizes
    sparse_rows = [
        {
            "exp_id": "A0_smallcnn_64_T4", "model": "smallcnn", "img_size": 64, "mode": "full",
            "aug": "base", "split_kind": "strat", "top1": 0.95, "balanced_acc": 0.90,
        },
        {
            "exp_id": "B_none_resnet18_64_T4", "model": "resnet18", "img_size": 64, "mode": "full",
            "aug": "none", "split_kind": "strat", "top1": 0.96, "balanced_acc": 0.92,
        },
    ]
    pd.DataFrame(sparse_rows).to_csv(csv_path, index=False)

    argv = [
        "--csv", str(csv_path),
        "--runs-dir", str(empty_runs),
        "--out", str(out_dir),
    ]
    ret = result_figures.main(argv)
    assert ret == 0

    # All 9 expected PNG files must still exist and be > 5 kB
    for fig_name in EXPECTED_FIGURE_NAMES:
        png_file = out_dir / f"{fig_name}.png"
        assert png_file.exists(), f"Missing {png_file} on sparse data"
        assert png_file.stat().st_size > 5000, f"{fig_name}.png is <= 5 kB on sparse data"

    assert (out_dir / "README.md").exists()
