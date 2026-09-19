"""Synthetic Thai-font glyph renderer and degradation pipeline.

Renders 72 TIS-620 classes from OFL fonts with realistic degradations matching the
real corpus (aspect-ratio preserved scaling, stroke weight changes, affine jitter,
binarisation thresholding, and speckle/erosion noise).
"""

from __future__ import annotations

import glob
import logging
import math
import os
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from thaichar.classes import CLASS_CODES, category, code_to_char, code_to_index

logger = logging.getLogger(__name__)

COMBINING_MARK_CODES: set[int] = {
    c for c in CLASS_CODES if unicodedata.category(code_to_char(c)) == "Mn"
}

UPPER_MARK_CODES: set[int] = {
    209,  # ั MAI HAN-AKAT
    212,  # ิ SARA I
    213,  # ี SARA II
    214,  # ึ SARA UE
    215,  # ื SARA UEE
    231,  # ็ MAITAIKHU
    232,  # ่ MAI EK
    233,  # ้ MAI THO
    234,  # ๊ MAI TRI
    236,  # ์ THANTHAKHAT
}

LOWER_MARK_CODES: set[int] = {
    216,  # ุ SARA U
    217,  # ู SARA UU
}


@dataclass
class GlyphTemplate:
    """Pre-rendered high-resolution clean glyph mask."""

    code: int
    char: str
    font_name: str
    image_u8: np.ndarray  # tight-cropped uint8, bg=255, ink=0
    base_used: str | None  # for combining marks, which base consonant was used


class FontLibrary:
    """Loads fonts and pre-renders high-resolution clean glyph templates."""

    def __init__(self, font_dir: str = "assets/fonts", render_size: int = 128) -> None:
        self.font_dir = font_dir
        self.render_size = render_size
        self.font_paths = sorted(glob.glob(os.path.join(font_dir, "*.ttf")))
        if not self.font_paths:
            raise FileNotFoundError(f"No .ttf fonts found in {font_dir}")

        self.fonts: dict[str, ImageFont.FreeTypeFont] = {}
        self.skipped: dict[str, list[tuple[int, str]]] = {}
        self.templates: dict[int, list[GlyphTemplate]] = {c: [] for c in CLASS_CODES}

        self._load_and_pre_render()

    def _load_and_pre_render(self) -> None:
        for fpath in self.font_paths:
            fname = os.path.basename(fpath)
            font_stem = os.path.splitext(fname)[0]
            try:
                font = ImageFont.truetype(fpath, size=self.render_size)
                self.fonts[font_stem] = font
                self.skipped[font_stem] = []
            except Exception as e:
                logger.warning("Failed to load font %s: %s", fname, e)
                continue

            for code in CLASS_CODES:
                char = code_to_char(code)
                is_mark = code in COMBINING_MARK_CODES

                if not is_mark:
                    tmpl = self._render_standalone(font, font_stem, code, char)
                    if tmpl is not None:
                        self.templates[code].append(tmpl)
                    else:
                        self.skipped[font_stem].append((code, char))
                        logger.info("Font %s skipped standalone code %d (%s)", font_stem, code, char)
                else:
                    tmpls = self._render_combining(font, font_stem, code, char)
                    if tmpls:
                        self.templates[code].extend(tmpls)
                    else:
                        self.skipped[font_stem].append((code, char))
                        logger.info("Font %s skipped combining code %d (%s)", font_stem, code, char)

    def _render_standalone(
        self,
        font: ImageFont.FreeTypeFont,
        font_name: str,
        code: int,
        char: str,
    ) -> GlyphTemplate | None:
        # Check support
        bbox = font.getbbox(char)
        mask = font.getmask(char)
        if bbox is None or mask.size[0] == 0 or mask.size[1] == 0:
            return None

        # Render on large canvas
        canvas_size = max(200, self.render_size * 2)
        im = Image.new("L", (canvas_size, canvas_size), 255)
        draw = ImageDraw.Draw(im)
        draw.text((canvas_size // 4, canvas_size // 4), char, font=font, fill=0)

        arr = np.asarray(im, dtype=np.uint8)
        ink_rows, ink_cols = np.where(arr < 128)
        if len(ink_rows) == 0:
            return None

        cropped = arr[ink_rows.min() : ink_rows.max() + 1, ink_cols.min() : ink_cols.max() + 1].copy()
        return GlyphTemplate(code=code, char=char, font_name=font_name, image_u8=cropped, base_used=None)

    def _render_combining(
        self,
        font: ImageFont.FreeTypeFont,
        font_name: str,
        code: int,
        char: str,
    ) -> list[GlyphTemplate]:
        bases = ["อ", "ก", "ป"] if code in UPPER_MARK_CODES else ["อ", "ก"]
        results: list[GlyphTemplate] = []
        canvas_size = max(250, self.render_size * 2)
        origin = (canvas_size // 4, canvas_size // 3)

        for base in bases:
            im_bm = Image.new("L", (canvas_size, canvas_size), 255)
            ImageDraw.Draw(im_bm).text(origin, base + char, font=font, fill=0)

            im_b = Image.new("L", (canvas_size, canvas_size), 255)
            ImageDraw.Draw(im_b).text(origin, base, font=font, fill=0)

            arr_bm = np.asarray(im_bm, dtype=np.uint8)
            arr_b = np.asarray(im_b, dtype=np.uint8)

            # Inked in base+mark but not in base alone
            mark_ink = (arr_bm < 128) & (arr_b >= 128)
            base_ink = arr_b < 128

            if mark_ink.sum() == 0:
                continue

            # Reject if mark touches base (8-connectivity dilation of mark touches base)
            kernel = np.ones((3, 3), dtype=np.uint8)
            dilated = cv2.dilate(mark_ink.astype(np.uint8), kernel)
            if np.any((dilated > 0) & base_ink):
                continue

            rows, cols = np.where(mark_ink)
            if len(rows) == 0:
                continue

            r0, r1 = rows.min(), rows.max() + 1
            c0, c1 = cols.min(), cols.max() + 1

            # Build tight-cropped mark image: background 255, mark ink 0
            mark_crop = np.where(mark_ink[r0:r1, c0:c1], np.uint8(0), np.uint8(255))

            results.append(
                GlyphTemplate(
                    code=code,
                    char=char,
                    font_name=font_name,
                    image_u8=mark_crop,
                    base_used=base,
                )
            )

        return results


def degrade_glyph(
    template_u8: np.ndarray,
    target_h: int,
    rng: np.random.Generator,
    degraded: bool = True,
) -> np.ndarray:
    """Degrade a high-resolution clean glyph template into a realistic low-res sample.

    Parameters
    ----------
    template_u8 : ndarray[H, W] uint8
        Clean high-res glyph crop (255 bg, 0 ink).
    target_h : int
        Target height sampled around the real class median height.
    rng : np.random.Generator
        Random generator.
    degraded : bool
        If False, produce a clean undegraded downscaled variant.

    Returns
    -------
    ndarray[h, w] uint8
        Tight-cropped binary glyph {0, 255}.
    """
    h_orig, w_orig = template_u8.shape[:2]

    if not degraded:
        # Clean downscale: maintain aspect ratio
        target_h = max(4, target_h)
        target_w = max(2, int(round(w_orig * (target_h / max(1, h_orig)))))
        down = cv2.resize(template_u8, (target_w, target_h), interpolation=cv2.INTER_AREA)
        bin_img = np.where(down > 128, np.uint8(255), np.uint8(0))
        r, c = np.where(bin_img < 128)
        if len(r) > 0:
            return bin_img[r.min() : r.max() + 1, c.min() : c.max() + 1]
        return bin_img

    cur = template_u8.copy()

    # 1. Random stroke weight: dilate/erode 0-2 px at high res
    mask = (cur < 128).astype(np.uint8) * 255
    kw = int(rng.integers(0, 3))
    if kw > 0:
        kernel = np.ones((kw + 1, kw + 1), np.uint8)
        if rng.random() < 0.5:
            mask = cv2.dilate(mask, kernel)
        else:
            eroded = cv2.erode(mask, kernel)
            if eroded.sum() > 0:
                mask = eroded
    cur = np.where(mask > 127, np.uint8(0), np.uint8(255))

    # 2. Small rotation +/-3 deg, shear +/-5 deg
    angle = rng.uniform(-3.0, 3.0)
    shear = rng.uniform(-5.0, 5.0)
    h_c, w_c = cur.shape[:2]
    cx, cy = w_c / 2.0, h_c / 2.0
    rad = math.radians(angle)
    tan_s = math.tan(math.radians(shear))
    cos_a = math.cos(rad)
    sin_a = math.sin(rad)

    a00 = cos_a
    a01 = -sin_a + cos_a * tan_s
    a10 = sin_a
    a11 = cos_a + sin_a * tan_s
    m02 = cx - (a00 * cx + a01 * cy)
    m12 = cy - (a10 * cx + a11 * cy)
    m_aff = np.array([[a00, a01, m02], [a10, a11, m12]], dtype=np.float32)

    cur = cv2.warpAffine(
        cur,
        m_aff,
        (w_c, h_c),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,
    )

    # 3. Downscale to target height (aspect preserved)
    target_h = max(4, target_h)
    target_w = max(2, int(round(w_c * (target_h / max(1, h_c)))))
    down = cv2.resize(cur, (target_w, target_h), interpolation=cv2.INTER_AREA)

    # 4. Binarise at random threshold U(100, 160)
    thresh = rng.uniform(100.0, 160.0)
    bin_img = np.where(down > thresh, np.uint8(255), np.uint8(0))

    # 5. Speckle noise p=0.3
    if rng.random() < 0.3:
        frac = rng.uniform(0.01, 0.03)
        n_speckle = int(round(bin_img.size * frac))
        if n_speckle > 0:
            ys = rng.integers(0, bin_img.shape[0], size=n_speckle)
            xs = rng.integers(0, bin_img.shape[1], size=n_speckle)
            vals = rng.choice(np.array([0, 255], dtype=np.uint8), size=n_speckle)
            bin_img[ys, xs] = vals

    # 6. Optional 1-px erosion p=0.2
    if rng.random() < 0.2:
        eroded = cv2.erode(bin_img, np.ones((3, 3), np.uint8))
        if np.mean(eroded < 128) >= 0.01:
            bin_img = eroded

    # 7. Output must be tight-cropped uint8 {0, 255}
    rows, cols = np.where(bin_img < 128)
    if len(rows) == 0:
        # Fallback to clean crop if noise eliminated all ink
        return degrade_glyph(template_u8, target_h, rng, degraded=False)

    cropped = bin_img[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]
    return cropped.astype(np.uint8)


def build_synth_cache(
    images: list[np.ndarray],
    paths: list[str],
    out_path: str,
) -> None:
    """Save rendered images and paths into an .npz cache compatible with load_cache."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    n = len(images)
    offsets = np.zeros(n, dtype=np.int64)
    widths = np.zeros(n, dtype=np.int32)
    heights = np.zeros(n, dtype=np.int32)
    flat_parts: list[np.ndarray] = []

    offset = 0
    for i, img in enumerate(images):
        h, w = img.shape[:2]
        flat_parts.append(img.ravel())
        offsets[i] = offset
        widths[i] = w
        heights[i] = h
        offset += h * w

    data = np.concatenate(flat_parts) if flat_parts else np.empty(0, dtype=np.uint8)
    path_arr = np.array(paths, dtype=str)

    np.savez(
        out_path,
        data=data,
        offsets=offsets,
        widths=widths,
        heights=heights,
        paths=path_arr,
    )
    mb = os.path.getsize(out_path) / (1024 * 1024)
    logger.info("Saved synthetic cache to %s (%.1f MB, %d glyphs)", out_path, mb, n)
