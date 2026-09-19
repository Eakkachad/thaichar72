"""Tests for error analysis and result plotting scripts (TASK-09 D3)."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

# Ensure scripts/ and src/ are importable
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "scripts"))
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

import error_analysis
import plot_results


@pytest.fixture
def fake_experiment_data(tmp_path: Path):
    """Create fake run data with 72 classes in tmp_path."""
    runs_dir = tmp_path / "runs"
    fake_run = runs_dir / "_fake"
    fake_run.mkdir(parents=True)

    # 72-class random confusion matrix
    rng = np.random.default_rng(1234)
    conf = rng.integers(0, 4, size=(72, 72))
    # Emphasize diagonal so recall is reasonable
    np.fill_diagonal(conf, rng.integers(40, 80, size=72))

    row_sums = conf.sum(axis=1)
    class_counts_train = rng.integers(100, 1000, size=72).tolist()
    per_class_recall = (np.diag(conf) / row_sums).tolist()
    top1 = float(np.trace(conf) / conf.sum())
    bal_acc = float(np.mean(per_class_recall))

    metrics_data = {
        "exp_id": "_fake",
        "top1": top1,
        "balanced_acc": bal_acc,
        "confusion": conf.tolist(),
        "class_counts_train": class_counts_train,
        "per_class_recall": per_class_recall,
        "tau_sweep": {
            "0.0": {"top1": top1, "balanced_acc": bal_acc, "minority_acc": 0.85},
            "0.5": {"top1": top1 - 0.01, "balanced_acc": bal_acc + 0.01, "minority_acc": 0.88},
        },
    }
    with open(fake_run / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_data, f)

    # log.csv with 3 epochs and EMA on epoch 3
    log_rows = [
        {"epoch": 1, "lr": 1e-3, "train_loss": 2.5, "train_acc": 0.5, "val_top1": 0.7, "val_bal_acc": 0.65,
         "val_macro_f1": 0.65, "val_minority_acc": 0.5, "epoch_seconds": 10.0, "train_seconds": 8.0,
         "ema_val_top1": "", "ema_val_bal_acc": ""},
        {"epoch": 2, "lr": 5e-4, "train_loss": 1.2, "train_acc": 0.8, "val_top1": 0.85, "val_bal_acc": 0.82,
         "val_macro_f1": 0.82, "val_minority_acc": 0.75, "epoch_seconds": 10.0, "train_seconds": 8.0,
         "ema_val_top1": "", "ema_val_bal_acc": ""},
        {"epoch": 3, "lr": 1e-4, "train_loss": 0.6, "train_acc": 0.92, "val_top1": top1, "val_bal_acc": bal_acc,
         "val_macro_f1": 0.88, "val_minority_acc": 0.85, "epoch_seconds": 10.0, "train_seconds": 8.0,
         "ema_val_top1": top1 + 0.005, "ema_val_bal_acc": bal_acc + 0.005},
    ]
    pd.DataFrame(log_rows).to_csv(fake_run / "log.csv", index=False)

    # experiments.csv with prefix _fake
    exp_csv_path = tmp_path / "reports" / "experiments.csv"
    exp_csv_path.parent.mkdir(parents=True, exist_ok=True)
    exp_rows = [
        {"exp_id": "_fake_model_A", "model": "resnet18", "mode": "full", "aug": "base",
         "top1": 0.95, "balanced_acc": 0.94, "minority_acc": 0.90},
        {"exp_id": "_fake_model_B", "model": "smallcnn", "mode": "full", "aug": "base",
         "top1": 0.91, "balanced_acc": 0.89, "minority_acc": 0.84},
    ]
    pd.DataFrame(exp_rows).to_csv(exp_csv_path, index=False)

    return {
        "fake_run": fake_run,
        "runs_dir": runs_dir,
        "exp_csv": exp_csv_path,
        "tmp_path": tmp_path,
    }


def test_error_analysis_script(fake_experiment_data):
    """Test error_analysis.main() on fake run and verify generated files and Thai font."""
    fake_run = fake_experiment_data["fake_run"]
    out_dir = fake_experiment_data["tmp_path"] / "reports" / "analysis" / "_fake"

    # Run error analysis
    error_analysis.main(["--run", str(fake_run), "--topk", "15", "--out", str(out_dir)])

    # Assert all 5 expected files exist and are non-empty
    expected_files = [
        out_dir / "confusion_matrix.png",
        out_dir / "confused_pairs.md",
        out_dir / "per_class_recall.png",
        out_dir / "worst_classes.md",
        out_dir / "summary.json",
    ]
    for p in expected_files:
        assert p.exists(), f"Missing expected file: {p}"
        assert p.stat().st_size > 0, f"File is empty: {p}"

    # Verify summary.json content
    with open(out_dir / "summary.json", encoding="utf-8") as f:
        summary = json.load(f)
    assert summary["exp_id"] == "_fake"
    assert "top1" in summary
    assert "balanced_acc" in summary
    assert len(summary["top_confused_directed"]) <= 15
    assert len(summary["worst_classes"]) <= 15
    assert len(summary["per_class"]) == 72

    # Assert Thai font is registered
    registered_fonts = {f.name for f in matplotlib.font_manager.fontManager.ttflist}
    assert "Sarabun" in registered_fonts
    assert "Sarabun" in plt.rcParams["font.family"]


def test_plot_results_script(fake_experiment_data):
    """Test plot_results.main() on fake run and verify generated figures."""
    runs_dir = fake_experiment_data["runs_dir"]
    exp_csv = fake_experiment_data["exp_csv"]
    out_figures = fake_experiment_data["tmp_path"] / "reports" / "figures"

    # Run plotting
    plot_results.main([
        "--runs", str(runs_dir),
        "--exp", "_fake",
        "--prefix", "_fake",
        "--out", str(out_figures),
        "--csv", str(exp_csv),
    ])

    # Assert expected files exist and are non-empty
    expected_figures = [
        out_figures / "training_curves__fake.png",
        out_figures / "tau_sweep__fake.png",
        out_figures / "matrix__fake_bar.png",
    ]
    for p in expected_figures:
        assert p.exists(), f"Missing expected figure: {p}"
        assert p.stat().st_size > 0, f"Figure is empty: {p}"

    # Assert Thai font is registered
    registered_fonts = {f.name for f in matplotlib.font_manager.fontManager.ttflist}
    assert "Sarabun" in registered_fonts
    assert "Sarabun" in plt.rcParams["font.family"]
