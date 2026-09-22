"""Inference and corruption unit tests (< 60 s on CPU)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from thaichar.classes import CLASS_CODES
from thaichar.data import load_cache
from thaichar.infer import (
    CORRUPTIONS,
    apply_corruption,
    load_checkpoint,
    predict_topk,
    predict_with_rotation_search,
    preprocess_image,
    rotate_canvas,
)
from thaichar.models import build_model
from thaichar.transforms import pad_to_square_canvas

CKPT_PATH = Path("runs/A1_resnet18_full_64_T4/best.pt")


@pytest.fixture(scope="module")
def model_and_cfg():
    """Load pretrained checkpoint if present, else construct untrained resnet18."""
    if CKPT_PATH.exists():
        return load_checkpoint(CKPT_PATH, device="cpu")
    cfg = {
        "model": "resnet18",
        "img_size": 64,
        "channel_mode": "gray3",
        "geometry": False,
        "mode": "full",
        "margin": 0.1,
    }
    model = build_model(
        name="resnet18",
        num_classes=len(CLASS_CODES),
        pretrained=False,
        in_chans=3,
        img_size=64,
        geometry=False,
    )
    model.eval()
    return model, cfg


def test_preprocess_image_white_on_black_pil(model_and_cfg):
    """(a) Preprocess handles a white-on-black PIL image."""
    _, cfg = model_and_cfg
    # Black background (0), white ink glyph (255)
    arr = np.zeros((50, 50), dtype=np.uint8)
    arr[15:35, 20:30] = 255
    pil_img = Image.fromarray(arr, mode="L")

    x, g = preprocess_image(pil_img, cfg)

    assert isinstance(x, torch.Tensor)
    assert isinstance(g, torch.Tensor)
    assert x.shape == (1, 3, cfg["img_size"], cfg["img_size"])
    assert g.shape == (1, 4)
    assert x.dtype == torch.float32
    assert g.dtype == torch.float32

    # Ink should be dark on white canvas: corner background > center glyph
    bg_val = float(x[0, 0, 0, 0])
    ink_val = float(x[0, 0, cfg["img_size"] // 2, cfg["img_size"] // 2])
    assert bg_val > ink_val, f"Expected ink to be dark (bg={bg_val} > ink={ink_val})"


def test_preprocess_image_noisy_jpeg_array(model_and_cfg):
    """(b) Preprocess handles a grey JPEG-like array with noise."""
    _, cfg = model_and_cfg
    rng = np.random.default_rng(42)
    # White/grey background ~220, dark ink ~40, plus noise ~10
    arr = np.full((60, 60), 220, dtype=np.uint8)
    arr[20:40, 25:35] = 40
    noise = rng.normal(0, 10, (60, 60)).astype(np.int32)
    noisy = np.clip(arr.astype(np.int32) + noise, 0, 255).astype(np.uint8)

    x, g = preprocess_image(noisy, cfg)

    assert x.shape == (1, 3, cfg["img_size"], cfg["img_size"])
    assert g.shape == (1, 4)
    # Ink dark
    bg_val = float(x[0, 0, 0, 0])
    ink_val = float(x[0, 0, cfg["img_size"] // 2, cfg["img_size"] // 2])
    assert bg_val > ink_val, f"Expected ink to be dark (bg={bg_val} > ink={ink_val})"


def test_preprocess_image_real_glyph(model_and_cfg):
    """(c) Preprocess handles a real glyph from data/cache/glyphs.npz."""
    _, cfg = model_and_cfg
    cache = load_cache("data/cache/glyphs.npz")
    real_glyph = cache[0]

    x, g = preprocess_image(real_glyph, cfg)

    assert x.shape == (1, 3, cfg["img_size"], cfg["img_size"])
    assert g.shape == (1, 4)
    # Ink dark: corner background > minimum value (ink)
    bg_val = float(x[0, 0, 0, 0])
    min_val = float(x[0, 0].min())
    assert bg_val > min_val, f"Expected ink to be dark (bg={bg_val} > min={min_val})"


def test_preprocess_image_rejects_empty(model_and_cfg):
    """Preprocess rejects empty images where no ink is detected."""
    _, cfg = model_and_cfg
    all_white = np.full((50, 50), 255, dtype=np.uint8)
    with pytest.raises(ValueError, match="no ink detected"):
        preprocess_image(all_white, cfg)


def test_predict_topk(model_and_cfg):
    """predict_topk returns k sorted probs summing <= 1."""
    model, cfg = model_and_cfg
    cache = load_cache("data/cache/glyphs.npz")
    real_glyph = cache[0]

    k = 5
    preds = predict_topk(model, cfg, real_glyph, k=k)

    assert len(preds) == k
    prob_sum = 0.0
    for i, (ch, code, prob) in enumerate(preds):
        assert isinstance(ch, str) and len(ch) == 1
        assert isinstance(code, int) and code in CLASS_CODES
        assert 0.0 <= prob <= 1.0
        prob_sum += prob
        if i > 0:
            # Sorted descending
            assert preds[i - 1][2] >= prob, f"Predictions not sorted: {preds[i-1][2]} < {prob}"

    assert prob_sum <= 1.0 + 1e-5


def test_predict_topk_with_tau(model_and_cfg):
    """predict_topk with tau adjustment returns valid sorted probabilities."""
    model, cfg = model_and_cfg
    cache = load_cache("data/cache/glyphs.npz")
    real_glyph = cache[0]

    log_prior = np.zeros(len(CLASS_CODES), dtype=np.float32)
    preds = predict_topk(model, cfg, real_glyph, k=5, tau=0.5, log_prior=log_prior)

    assert len(preds) == 5
    prob_sum = sum(p for _, _, p in preds)
    assert prob_sum <= 1.0 + 1e-5
    for i in range(len(preds) - 1):
        assert preds[i][2] >= preds[i + 1][2]


def test_corruptions_preserve_ink():
    """Each corruption returns same-size uint8 canvas with ink preserved (>= 30% input ink)."""
    cache = load_cache("data/cache/glyphs.npz")
    canvas = pad_to_square_canvas(cache[0], margin=0.1)
    input_ink = int((canvas < 128).sum())
    assert input_ink > 0

    for name, severities in CORRUPTIONS.items():
        smallest_sev = severities[0]
        corrupted = apply_corruption(canvas, name, smallest_sev)

        assert corrupted.shape == canvas.shape, f"{name}: expected shape {canvas.shape}, got {corrupted.shape}"
        assert corrupted.dtype == np.uint8, f"{name}: expected uint8, got {corrupted.dtype}"

        output_ink = int((corrupted < 128).sum())
        ratio = output_ink / float(input_ink)
        assert ratio >= 0.30, f"{name} (sev={smallest_sev}) failed ink preservation: ratio {ratio:.3f} < 0.30"


def test_rotate_canvas_keeps_the_glyph_and_the_background():
    a = np.full((20, 14), 255, np.uint8)
    a[4:16, 5:9] = 0
    for deg in (10, 45, 90, -30):
        r = rotate_canvas(a, deg)
        assert r.dtype == np.uint8
        assert (r < 128).sum() > 0, f"ink disappeared at {deg} deg"
        # padding must be background, not ink -- preprocess_image reads polarity off the border
        assert r[0, 0] > 200 and r[-1, -1] > 200


def test_rotation_search_leaves_confident_images_alone(model_and_cfg):
    """The guard that matters: an upright glyph the model is already sure about must not be
    re-read from a rotated copy, or the feature costs accuracy on clean input."""
    model, cfg = model_and_cfg
    a = np.full((28, 20), 255, np.uint8)
    a[6:22, 7:13] = 0
    canvases = [a] * 4
    p_plain, c_plain, _ = predict_with_rotation_search(
        model, cfg, canvases, tau=0.0, margin=0.0)          # tau=0 -> never search
    p_srch, c_srch, n = predict_with_rotation_search(
        model, cfg, canvases, tau=0.0, margin=0.0)
    assert n == 0
    assert np.array_equal(p_plain, p_srch)
    assert len(p_srch) == len(canvases)


def test_rotation_search_reports_rejected_images(model_and_cfg):
    model, cfg = model_and_cfg
    blank = np.full((16, 16), 255, np.uint8)               # no ink -> preprocess rejects it
    preds, confs, _ = predict_with_rotation_search(model, cfg, [blank], tau=1.0)
    assert preds[0] == -1 and confs[0] == -1.0
