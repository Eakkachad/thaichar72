"""Deterministic transforms for Thai glyph images."""

import math

import numpy as np
from PIL import Image


def pad_to_square_canvas(
    img_u8: np.ndarray,
    margin: float = 0.1,
    pad_value: int = 255,
) -> np.ndarray:
    """Pad a glyph into a square canvas with fractional margin.

    Parameters
    ----------
    img_u8 : ndarray[H, W] uint8
        Input glyph image (greyscale).
    margin : float
        Fractional margin around the glyph.
    pad_value : int
        Fill value for the canvas (255 = white).

    Returns
    -------
    ndarray[c, c] uint8
        Square canvas containing the glyph centred.
    """
    h, w = img_u8.shape[:2]
    c = math.ceil(max(h, w) * (1 + 2 * margin))
    canvas = np.full((c, c), pad_value, dtype=np.uint8)

    y0 = (c - h) // 2
    x0 = (c - w) // 2
    canvas[y0 : y0 + h, x0 : x0 + w] = img_u8
    return canvas


def resize_square(canvas_u8: np.ndarray, size: int) -> np.ndarray:
    """Resize a square canvas to size × size using bilinear interpolation (PIL).

    Parameters
    ----------
    canvas_u8 : ndarray[C, C] uint8
        Input square canvas.
    size : int
        Target square side length.

    Returns
    -------
    ndarray[size, size] uint8
    """
    pil_img = Image.fromarray(canvas_u8, mode="L")
    pil_img = pil_img.resize((size, size), Image.BILINEAR)
    return np.asarray(pil_img, dtype=np.uint8)


def fit_to_square(
    img_u8: np.ndarray,
    size: int,
    margin: float = 0.1,
    pad_value: int = 255,
) -> np.ndarray:
    """Aspect-preserving resize of a glyph into a square canvas.

    1. Compute canvas side ``c = ceil(max(h, w) * (1 + 2 * margin))``.
    2. Paste the glyph centred on a white canvas of side *c*.
    3. Resize to *size × size* with bilinear interpolation (PIL).

    Parameters
    ----------
    img_u8 : ndarray[H, W] uint8
        Input glyph image (greyscale).
    size : int
        Target square side length.
    margin : float
        Fractional margin around the glyph.
    pad_value : int
        Fill value for the canvas (255 = white).

    Returns
    -------
    ndarray[size, size] uint8
    """
    canvas = pad_to_square_canvas(img_u8, margin=margin, pad_value=pad_value)
    return resize_square(canvas, size)


def encode_channels(square_u8: np.ndarray, mode: str) -> np.ndarray:
    """Encode a square greyscale glyph into multi-channel float32 tensor array.

    Modes:
      - 'gray1': 1 channel [1, S, S] in [0, 1].
      - 'gray3': 3 channels [3, S, S] replicating greyscale, ImageNet normalised.
      - 'onoff': 3 channels [3, S, S] motivated by the ON/OFF pathway split in the
        fly optic lobe that separates contrast polarity and structural features:
        (a) ink mask in {0, 1}, (b) distance transform of ink region normalised by
        S/16 and clipped to [0, 1], (c) Sobel edge magnitude normalised to [0, 1];
        then each channel standardised with mean 0.5 / std 0.5.

    Parameters
    ----------
    square_u8 : ndarray[S, S] uint8
        Input square glyph image (white background 255, dark ink 0).
    mode : str
        One of 'gray1', 'gray3', 'onoff'.

    Returns
    -------
    ndarray[C, S, S] float32
    """
    if mode == "gray1":
        img_f = square_u8.astype(np.float32) / 255.0
        return img_f[np.newaxis, :, :].astype(np.float32)

    elif mode == "gray3":
        img_f = square_u8.astype(np.float32) / 255.0
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        ch0 = (img_f - mean[0]) / std[0]
        ch1 = (img_f - mean[1]) / std[1]
        ch2 = (img_f - mean[2]) / std[2]
        return np.stack([ch0, ch1, ch2], axis=0).astype(np.float32)

    elif mode == "onoff":
        import cv2

        s = square_u8.shape[0]
        # (a) ink mask in {0, 1}: ink is < 128
        ink_mask = (square_u8 < 128).astype(np.float32)

        # (b) distance transform of ink region, normalised by S/16, clipped to [0, 1]
        ink_u8 = (ink_mask > 0).astype(np.uint8)
        dist = cv2.distanceTransform(ink_u8, cv2.DIST_L2, 3)
        scale = max(s / 16.0, 1e-6)
        ch_dist = np.clip(dist / scale, 0.0, 1.0).astype(np.float32)

        # (c) Sobel edge magnitude of ink mask normalised to [0, 1]
        gx = cv2.Sobel(ink_mask, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(ink_mask, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.hypot(gx, gy)
        mag_max = float(mag.max())
        if mag_max > 0:
            ch_edge = (mag / mag_max).astype(np.float32)
        else:
            ch_edge = np.zeros_like(mag, dtype=np.float32)

        # Standardise each channel with mean 0.5 / std 0.5 -> [-1, 1]
        ch_a = (ink_mask - 0.5) / 0.5
        ch_b = (ch_dist - 0.5) / 0.5
        ch_c = (ch_edge - 0.5) / 0.5

        return np.stack([ch_a, ch_b, ch_c], axis=0).astype(np.float32)

    else:
        raise ValueError(f"Unknown channel mode: {mode!r}. Expected 'gray1', 'gray3', or 'onoff'.")


def geometry_features(h: int, w: int, ink_frac: float) -> np.ndarray:
    """Compute geometry feature vector ``[log(w), log(h), log(w/h), ink_frac]``.

    Parameters
    ----------
    h, w : int
        Original glyph height and width.
    ink_frac : float
        Fraction of ink pixels (from EDA metadata).

    Returns
    -------
    ndarray[4] float32
    """
    return np.array(
        [math.log(w), math.log(h), math.log(w / h), ink_frac],
        dtype=np.float32,
    )

