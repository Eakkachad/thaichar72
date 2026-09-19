"""Engine / model / metric tests (CPU, no pretrained downloads, < 90 s)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from thaichar.engine import cosine_warmup_lambda, train_one
from thaichar.losses import logit_adjust
from thaichar.metrics import compute_metrics
from thaichar.models import build_model
from thaichar.samplers import build_sampler


@pytest.mark.parametrize("name,size", [("smallcnn", 64), ("resnet18", 64), ("mobilenetv3_large_100", 64),
                                       ("vit_tiny_patch16_224", 64)])
def test_build_model_forward(name, size):
    m = build_model(name, pretrained=False, img_size=size, geometry=True, mode="full")
    y = m(torch.randn(2, 3, size, size), torch.randn(2, 4))
    assert y.shape == (2, 72)


def test_modes_freeze():
    frozen = build_model("resnet18", pretrained=False, mode="frozen")
    assert not any(p.requires_grad for p in frozen.backbone.parameters())
    assert all(p.requires_grad for p in frozen.head.parameters())
    partial = build_model("resnet18", pretrained=False, mode="partial")
    flags = [p.requires_grad for p in partial.backbone.parameters()]
    assert any(flags) and not all(flags)


def test_compute_metrics_small():
    y = np.array([0, 0, 1, 1, 2, 3, 4])
    logits = np.zeros((7, 5))
    pred = [0, 0, 1, 0, 2, 3, 0]  # class 1 recall .5, class 4 recall 0
    logits[np.arange(7), pred] = 1.0
    m = compute_metrics(y, logits, np.array([100, 100, 10, 10, 10]))
    assert abs(m["top1"] - 5 / 7) < 1e-9
    assert abs(m["balanced_acc"] - np.mean([1, 0.5, 1, 1, 0])) < 1e-9
    assert m["n_classes_present"] == 5
    assert abs(m["minority_acc"] - 2 / 3) < 1e-9  # classes 2,3,4 (n<50): 2 of 3 correct


def test_logit_adjust_identity_and_sampler_cap():
    x = torch.randn(4, 72)
    assert torch.allclose(logit_adjust(x, torch.randn(72), 0.0), x)
    labels = np.array([0] * 100 + [1] * 1)
    s = build_sampler(labels, kind="inv", cap=3.0)
    w = s.weights.numpy()
    assert w.max() <= 3.0 + 1e-6 and w.min() > 0


def test_cosine_schedule():
    f = cosine_warmup_lambda(total_steps=100, warmup_steps=10, min_ratio=0.1)
    assert f(0) < f(9) <= 1.0 and abs(f(10) - 1.0) < 1e-9 and abs(f(100) - 0.1) < 1e-6


def test_train_one_smoke(tmp_path):
    cfg = {"exp_id": "pytest_smoke", "out_root": str(tmp_path), "model": "resnet18", "pretrained": False,
           "img_size": 32, "subset_frac": 0.01, "epochs": 1, "batch_size": 64, "ema": True, "latency_iters": 5,
           "threads": 4}
    m = train_one(cfg)
    assert (tmp_path / "pytest_smoke" / "metrics.json").exists()
    assert (tmp_path / "pytest_smoke" / "best.pt").exists()
    assert (tmp_path / "pytest_smoke" / "log.csv").exists()
    for k in ("top1", "balanced_acc", "macro_f1", "minority_acc", "tau_sweep", "latency_ms_bs1"):
        assert k in m
    assert set(m["tau_sweep"].keys()) == {"0.0", "0.25", "0.5", "0.75"}
