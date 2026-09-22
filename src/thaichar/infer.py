"""Inference helpers, preprocessing, and deterministic canvas corruptions."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from thaichar.classes import CLASS_CODES, CLASS_CHARS, code_to_char, index_to_code
from thaichar.engine import merge_cfg
from thaichar.models import build_model
from thaichar.transforms import (
    encode_channels,
    geometry_features,
    pad_to_square_canvas,
    resize_square,
)

NUM_CLASSES = len(CLASS_CODES)


# ---------------------------------------------------------------------------
# Checkpoint Loading
# ---------------------------------------------------------------------------

def load_checkpoint(
    path: str | Path,
    device: str | torch.device = "cpu",
) -> tuple[nn.Module, dict[str, Any]]:
    """Rebuild model from checkpoint configuration and load weights.

    Parameters
    ----------
    path : str | Path
        Path to checkpoint (best.pt or last.pt).
    device : str | torch.device
        Device on which to load the model.

    Returns
    -------
    model : nn.Module
        Instantiated model in eval mode.
    cfg : dict[str, Any]
        Merged configuration dictionary.
    """
    dev = torch.device(device) if isinstance(device, str) else device
    ckpt = torch.load(str(path), map_location=dev, weights_only=False)
    raw_cfg = ckpt.get("cfg", {})
    cfg = merge_cfg(raw_cfg)

    in_ch = 1 if cfg.get("channel_mode") == "gray1" else 3
    model = build_model(
        name=cfg["model"],
        num_classes=NUM_CLASSES,
        pretrained=False,
        in_chans=in_ch,
        img_size=int(cfg["img_size"]),
        mode=cfg["mode"],
        geometry=bool(cfg.get("geometry", False)),
        drop_rate=float(cfg.get("drop_rate", 0.0)),
        partial_frac=float(cfg.get("partial_frac", 0.35)),
    ).to(dev)

    state_dict = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state_dict)

    if cfg.get("channels_last", False):
        model = model.to(memory_format=torch.channels_last)

    model.eval()
    return model, cfg


# ---------------------------------------------------------------------------
# Image Preprocessing
# ---------------------------------------------------------------------------

def _reframe(arr: np.ndarray, tol: int = 8) -> np.ndarray:
    """Strip constant-valued border rings, then re-pad with the background colour they revealed.

    `tol` allows for JPEG ringing on a nominally flat frame. If the image is uniform end to end
    it is returned untouched so the caller's "no ink" check still fires.
    """
    a = arr
    while min(a.shape[:2]) > 2:
        ring = np.concatenate([a[0, :], a[-1, :], a[:, 0], a[:, -1]])
        if int(ring.max()) - int(ring.min()) > tol:
            break
        a = a[1:-1, 1:-1]
    if min(a.shape[:2]) < 1:
        return arr
    # Background comes from the CORNERS of what is left, not from the frames we peeled (a frame
    # someone padded on says nothing about the paper) and not from the whole ring (on a tight crop
    # the ring is full of edge-touching strokes -- using it costs 7 pt on well-framed input).
    # Corners are the pixels a glyph is least likely to occupy.
    bg = int(np.median([a[0, 0], a[0, -1], a[-1, 0], a[-1, -1]]))
    pad_px = max(4, min(a.shape[:2]) // 8)
    return np.pad(a, pad_px, mode="constant", constant_values=bg)


def preprocess_image(
    img: np.ndarray | Image.Image | str | Path,
    cfg: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Preprocess arbitrary input image/photo/scan for inference.

    Pipeline:
      1. Convert to greyscale uint8 array.
      2. Otsu threshold (cv2).
      3. Make ink dark on white (invert if mean < 127.5).
      4. Crop to ink bounding box (reject empty).
      5. Pad to square canvas with margin (pad_to_square_canvas).
      6. Resize to square (resize_square).
      7. Multi-channel encoding (encode_channels).
      8. Geometry features from crop height, width, and ink fraction.

    Parameters
    ----------
    img : np.ndarray | Image.Image | str | Path
        Input image in any standard format.
    cfg : dict[str, Any]
        Config dict containing margin, img_size, and channel_mode.

    Returns
    -------
    x : torch.Tensor
        Batch tensor of shape [1, C, S, S] float32.
    g : torch.Tensor
        Batch geometry tensor of shape [1, 4] float32.
    """
    # 1. Convert to 2D greyscale uint8 array
    if isinstance(img, (str, Path)):
        pil_img = Image.open(str(img)).convert("L")
        arr = np.asarray(pil_img, dtype=np.uint8)
    elif isinstance(img, Image.Image):
        arr = np.asarray(img.convert("L"), dtype=np.uint8)
    elif isinstance(img, np.ndarray):
        arr = img
        if arr.dtype in (np.float32, np.float64):
            if arr.max() <= 1.0 and arr.min() >= 0.0:
                arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
            else:
                arr = arr.clip(0, 255).astype(np.uint8)
        if arr.ndim == 3:
            if arr.shape[2] == 4:
                arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2GRAY)
            elif arr.shape[2] == 3:
                arr = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
            elif arr.shape[2] == 1:
                arr = arr[:, :, 0]
            elif arr.shape[0] == 1:
                arr = arr[0]
            else:
                arr = arr[:, :, 0]
        arr = arr.astype(np.uint8)
    else:
        raise TypeError(f"Unsupported image type: {type(img)}")

    # 2a. Normalise the framing before thresholding, because step 3 reads polarity off the border
    #     ring and two common input shapes make that ring lie:
    #       * a TIGHT crop -- strokes touch all four edges, so the ring is mostly ink and the glyph
    #         gets inverted (measured on 493 held-out glyphs: 0.9229 vs 0.9939 correctly framed);
    #       * a crop someone padded with zeros -- a dark frame around dark ink on light paper, which
    #         inverts the image and destroys it (0.4949).
    #     Peeling every constant-valued ring answers both: whatever colour the frame is, the last
    #     constant ring before real structure begins is the true background, and re-padding with it
    #     gives step 3 a ring that means what it is supposed to mean.
    arr = _reframe(arr)

    # 2b. Otsu threshold (cv2)
    _, thresh = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 3. Make ink dark on white. Decide polarity from the 1-px border ring (background), not the global
    #    mean: tight-cropped glyphs can be >50 % ink, so the mean would wrongly invert them.
    border = np.concatenate([thresh[0, :], thresh[-1, :], thresh[:, 0], thresh[:, -1]])
    if border.mean() < 127.5:
        thresh = 255 - thresh

    # 4. Crop to ink bounding box (reject empty)
    ink_mask = (thresh == 0)
    rows = np.where(ink_mask.any(axis=1))[0]
    cols = np.where(ink_mask.any(axis=0))[0]

    if len(rows) == 0 or len(cols) == 0:
        raise ValueError("Empty image: no ink detected after thresholding")

    r_min, r_max = int(rows[0]), int(rows[-1])
    c_min, c_max = int(cols[0]), int(cols[-1])

    cropped = thresh[r_min : r_max + 1, c_min : c_max + 1]
    h, w = cropped.shape[:2]
    ink_frac = float(np.mean(cropped == 0))

    # 5. Pad to square canvas
    margin = float(cfg.get("margin", 0.1))
    canvas = pad_to_square_canvas(cropped, margin=margin, pad_value=255)

    # 6. Resize square
    img_size = int(cfg.get("img_size", 64))
    square = resize_square(canvas, size=img_size)

    # 7. Multi-channel encoding
    channel_mode = str(cfg.get("channel_mode", "gray3"))
    encoded = encode_channels(square, mode=channel_mode)

    # 8. Geometry features
    geo = geometry_features(h, w, ink_frac)

    x = torch.from_numpy(encoded).unsqueeze(0)
    g = torch.from_numpy(geo).unsqueeze(0)
    return x, g


# ---------------------------------------------------------------------------
# Top-K Prediction
# ---------------------------------------------------------------------------

def predict_topk(
    model: nn.Module,
    cfg: dict[str, Any],
    img: np.ndarray | Image.Image | str | Path | torch.Tensor,
    k: int = 5,
    tau: float = 0.0,
    log_prior: np.ndarray | torch.Tensor | None = None,
) -> list[tuple[str, int, float]]:
    """Predict top-k Thai character classes for an image.

    Parameters
    ----------
    model : nn.Module
        Loaded model in eval mode.
    cfg : dict[str, Any]
        Config dict.
    img : np.ndarray | Image.Image | str | Path | torch.Tensor
        Input image or preprocessed tensor.
    k : int
        Number of top predictions to return.
    tau : float
        Logit adjustment parameter (0.0 = none).
    log_prior : np.ndarray | torch.Tensor | None
        Training log class priors for tau adjustment.

    Returns
    -------
    list of (char, code, prob) sorted descending by probability.
    """
    device = next(model.parameters()).device
    model.eval()

    if isinstance(img, torch.Tensor):
        x = img.to(device)
        if x.dim() == 3:
            x = x.unsqueeze(0)
        g = None
    else:
        x, g = preprocess_image(img, cfg)
        x = x.to(device)
        if g is not None:
            g = g.to(device)

    if cfg.get("channels_last", False) and x.dim() == 4:
        x = x.contiguous(memory_format=torch.channels_last)

    with torch.no_grad():
        logits = model(x, g if bool(cfg.get("geometry", False)) else None)

    if float(tau) != 0.0 and log_prior is not None:
        lp = torch.as_tensor(log_prior, dtype=logits.dtype, device=logits.device)
        if lp.dim() == 1:
            lp = lp.unsqueeze(0)
        logits = logits - float(tau) * lp

    probs = F.softmax(logits, dim=-1)[0]
    num_k = min(k, len(probs))
    topk_vals, topk_inds = torch.topk(probs, k=num_k)

    results = []
    for p_val, idx_val in zip(topk_vals, topk_inds):
        i = int(idx_val.item())
        code = index_to_code(i)
        char = code_to_char(code)
        prob = float(p_val.item())
        results.append((char, code, prob))

    return results


# ---------------------------------------------------------------------------
# Canvas Corruptions (Deterministic, f(canvas_u8, severity) -> canvas_u8)
# ---------------------------------------------------------------------------

def corrupt_rotate(canvas_u8: np.ndarray, severity: float | int) -> np.ndarray:
    """Rotate canvas around center by severity degrees."""
    deg = float(severity)
    h, w = canvas_u8.shape[:2]
    cx, cy = (w / 2.0, h / 2.0)
    m = cv2.getRotationMatrix2D((cx, cy), deg, 1.0)
    return cv2.warpAffine(
        canvas_u8,
        m,
        (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,
    )


def corrupt_thicker(canvas_u8: np.ndarray, severity: int) -> np.ndarray:
    """Thicken ink by dilating ink (eroding canvas) by severity px."""
    ink_mask = (canvas_u8 < 128).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    dilated = cv2.dilate(ink_mask, k, iterations=int(severity))
    return np.where(dilated > 0, np.uint8(0), np.uint8(255))


def corrupt_thinner(canvas_u8: np.ndarray, severity: int) -> np.ndarray:
    """Thin ink by eroding ink (dilating canvas) by severity px."""
    ink_mask = (canvas_u8 < 128).astype(np.uint8)
    # Erode ink by severity px
    if int(severity) == 1:
        k = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        eroded = cv2.erode(ink_mask, k, iterations=1)
        # If erosion removes > 70% of ink on a thin glyph, fall back to 2x2 kernel
        if ink_mask.sum() > 0 and eroded.sum() < 0.30 * ink_mask.sum():
            k2 = np.ones((2, 2), dtype=np.uint8)
            eroded = cv2.erode(ink_mask, k2, iterations=1)
    else:
        k = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        eroded = cv2.erode(ink_mask, k, iterations=int(severity))
    return np.where(eroded > 0, np.uint8(0), np.uint8(255))


def thickness(canvas_u8: np.ndarray, severity: int) -> np.ndarray:
    """Adjust stroke thickness: positive = thicker, negative = thinner."""
    if severity >= 0:
        return corrupt_thicker(canvas_u8, severity)
    return corrupt_thinner(canvas_u8, abs(severity))


def corrupt_salt_pepper(canvas_u8: np.ndarray, severity: float) -> np.ndarray:
    """Add deterministic salt and pepper noise covering fraction of pixels."""
    frac = float(severity)
    h, w = canvas_u8.shape[:2]
    n_noisy = int(round(h * w * frac))
    if n_noisy == 0:
        return canvas_u8.copy()
    seed = int((42 + int(frac * 10000) + int(np.sum(canvas_u8[:2, :2])) + h * 31) % (2**31 - 1))
    rng = np.random.default_rng(seed)
    out = canvas_u8.copy()
    ys = rng.integers(0, h, size=n_noisy)
    xs = rng.integers(0, w, size=n_noisy)
    vals = rng.choice(np.array([0, 255], dtype=np.uint8), size=n_noisy)
    out[ys, xs] = vals
    return out


def corrupt_downscale(canvas_u8: np.ndarray, severity: float) -> np.ndarray:
    """Downscale canvas by scale factor then upscale back and re-binarise."""
    scale = float(severity)
    h, w = canvas_u8.shape[:2]
    low_h = max(1, int(round(h * scale)))
    low_w = max(1, int(round(w * scale)))
    down = cv2.resize(canvas_u8, (low_w, low_h), interpolation=cv2.INTER_AREA)
    up = cv2.resize(down, (w, h), interpolation=cv2.INTER_LINEAR)
    return np.where(up > 127, np.uint8(255), np.uint8(0))


def corrupt_occlusion(canvas_u8: np.ndarray, severity: float | int) -> np.ndarray:
    """Cover random square of side frac with white fill (seeded)."""
    frac = float(severity) / 100.0 if float(severity) > 1.0 else float(severity)
    h, w = canvas_u8.shape[:2]
    box_h = max(1, int(round(h * frac)))
    box_w = max(1, int(round(w * frac)))
    seed = int((123 + int(frac * 1000) + int(np.sum(canvas_u8[:2, :2])) + w * 17) % (2**31 - 1))
    rng = np.random.default_rng(seed)
    y0 = int(rng.integers(0, max(1, h - box_h + 1)))
    x0 = int(rng.integers(0, max(1, w - box_w + 1)))
    out = canvas_u8.copy()
    out[y0 : y0 + box_h, x0 : x0 + box_w] = 255
    return out


def corrupt_blur(canvas_u8: np.ndarray, severity: float) -> np.ndarray:
    """Apply Gaussian blur with given sigma."""
    sigma = float(severity)
    ksize = int(math.ceil(sigma * 3)) * 2 + 1
    return cv2.GaussianBlur(canvas_u8, (ksize, ksize), sigmaX=sigma, borderType=cv2.BORDER_REPLICATE)


def corrupt_contrast(canvas_u8: np.ndarray, severity: int | float) -> np.ndarray:
    """Map ink 0 -> v and paper 255 -> 255 - v for low-contrast scan simulation."""
    v = float(severity)
    p_f = canvas_u8.astype(np.float32)
    out_f = v + (p_f / 255.0) * (255.0 - 2.0 * v)
    return np.clip(np.round(out_f), 0, 255).astype(np.uint8)


def corrupt_translate(canvas_u8: np.ndarray, severity: float | int) -> np.ndarray:
    """Shift canvas by fraction with white fill."""
    frac = float(severity) / 100.0 if float(severity) > 1.0 else float(severity)
    h, w = canvas_u8.shape[:2]
    dx = int(round(w * frac))
    out = np.full_like(canvas_u8, 255)
    if dx < w:
        out[:, dx:] = canvas_u8[:, :w - dx]
    return out


def corrupt_background_noise(canvas_u8: np.ndarray, severity: int | float) -> np.ndarray:
    """Add Gaussian noise with sigma on paper pixels only (ink preserved)."""
    sigma = float(severity)
    h, w = canvas_u8.shape[:2]
    seed = int((777 + int(sigma * 100) + int(np.sum(canvas_u8[:2, :2])) + h * 13) % (2**31 - 1))
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, sigma, size=(h, w)).astype(np.float32)
    paper_mask = (canvas_u8 >= 128)
    out = canvas_u8.copy().astype(np.float32)
    out[paper_mask] = np.clip(out[paper_mask] + noise[paper_mask], 0, 255)
    return out.astype(np.uint8)


# Aliases matching corruption names
rotate = corrupt_rotate
thicker = corrupt_thicker
thinner = corrupt_thinner
salt_pepper = corrupt_salt_pepper
downscale = corrupt_downscale
occlusion = corrupt_occlusion
blur = corrupt_blur
contrast = corrupt_contrast
translate = corrupt_translate
background_noise = corrupt_background_noise

CORRUPTIONS: dict[str, list[Any]] = {
    "rotate": [5, 10, 15, 20],
    "thicker": [1, 2, 3],
    "thinner": [1, 2, 3],
    "salt_pepper": [0.01, 0.05, 0.10, 0.20],
    "downscale": [0.8, 0.6, 0.4],
    "occlusion": [0.10, 0.25, 0.40],
    "blur": [0.5, 1.0, 1.5],
    "contrast": [60, 120, 180],
    "translate": [0.05, 0.10, 0.20],
    "background_noise": [20, 40, 60],
}

CORRUPTION_FNS: dict[str, Callable[[np.ndarray, Any], np.ndarray]] = {
    "rotate": corrupt_rotate,
    "thicker": corrupt_thicker,
    "thinner": corrupt_thinner,
    "salt_pepper": corrupt_salt_pepper,
    "downscale": corrupt_downscale,
    "occlusion": corrupt_occlusion,
    "blur": corrupt_blur,
    "contrast": corrupt_contrast,
    "translate": corrupt_translate,
    "background_noise": corrupt_background_noise,
}


def apply_corruption(canvas_u8: np.ndarray, name: str, severity: Any) -> np.ndarray:
    """Apply a named corruption with severity to uint8 canvas."""
    if name not in CORRUPTION_FNS:
        raise ValueError(f"Unknown corruption {name!r}. Available: {list(CORRUPTION_FNS.keys())}")
    return CORRUPTION_FNS[name](canvas_u8, severity)
