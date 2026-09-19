"""Deterministic transforms for Thai glyph images."""

import math

import numpy as np
from PIL import Image


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
    h, w = img_u8.shape[:2]
    c = math.ceil(max(h, w) * (1 + 2 * margin))
    canvas = np.full((c, c), pad_value, dtype=np.uint8)

    y0 = (c - h) // 2
    x0 = (c - w) // 2
    canvas[y0 : y0 + h, x0 : x0 + w] = img_u8

    pil_img = Image.fromarray(canvas, mode="L")
    pil_img = pil_img.resize((size, size), Image.BILINEAR)
    return np.asarray(pil_img, dtype=np.uint8)


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
