"""Augmentation presets and operations for Thai glyphs.

All operations operate on square uint8 canvas images (white background 255, dark ink 0),
typically 20–60 px canvas size before final resize, so morphology acts at near-native
stroke scale.
"""

from __future__ import annotations

import math
from typing import Callable, Sequence

import cv2
import numpy as np

PRESETS: list[str] = ["none", "base", "morph", "full", "randaug", "trivial"]


class RandomAffine:
    """Random affine transformation: rotation, shear, scale, translation.

    No reflections or flips are performed (Thai script is chirality-sensitive).
    """

    def __init__(
        self,
        rot: float = 8.0,
        shear: float = 10.0,
        scale: tuple[float, float] = (0.85, 1.15),
        translate: float = 0.08,
        p: float = 1.0,
    ) -> None:
        self.rot = rot
        self.shear = shear
        self.scale = scale
        self.translate = translate
        self.p = p

    def __call__(self, img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        if rng.random() > self.p:
            return img
        h, w = img.shape[:2]
        angle = rng.uniform(-self.rot, self.rot)
        shear_deg = rng.uniform(-self.shear, self.shear)
        s = rng.uniform(self.scale[0], self.scale[1])
        tx = rng.uniform(-self.translate, self.translate) * w
        ty = rng.uniform(-self.translate, self.translate) * h

        rad = math.radians(angle)
        shear_rad = math.radians(shear_deg)
        cos_a = math.cos(rad) * s
        sin_a = math.sin(rad) * s
        tan_s = math.tan(shear_rad)

        cx, cy = w / 2.0, h / 2.0
        a00 = cos_a
        a01 = -sin_a + cos_a * tan_s
        a10 = sin_a
        a11 = cos_a + sin_a * tan_s

        m02 = cx + tx - (a00 * cx + a01 * cy)
        m12 = cy + ty - (a10 * cx + a11 * cy)

        m = np.array([[a00, a01, m02], [a10, a11, m12]], dtype=np.float32)
        return cv2.warpAffine(
            img,
            m,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=255,
        )


class MarginJitter:
    """Randomly re-pad / crop the canvas edges by +/- frac of canvas size."""

    def __init__(self, frac: float = 0.15, p: float = 1.0) -> None:
        self.frac = frac
        self.p = p

    def __call__(self, img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        if rng.random() > self.p:
            return img
        h, w = img.shape[:2]
        delta_scale = rng.uniform(-self.frac, self.frac)
        new_side = max(4, int(round(h * (1.0 + delta_scale))))
        if new_side == h:
            return img

        if new_side > h:
            pad_total = new_side - h
            pad_top = int(rng.integers(0, pad_total + 1))
            pad_bottom = pad_total - pad_top
            pad_left = int(rng.integers(0, pad_total + 1))
            pad_right = pad_total - pad_left
            canvas = np.pad(
                img,
                ((pad_top, pad_bottom), (pad_left, pad_right)),
                mode="constant",
                constant_values=255,
            )
        else:
            crop_total = h - new_side
            crop_top = int(rng.integers(0, crop_total + 1))
            crop_left = int(rng.integers(0, crop_total + 1))
            canvas = img[crop_top : crop_top + new_side, crop_left : crop_left + new_side]

        return cv2.resize(canvas, (w, h), interpolation=cv2.INTER_NEAREST)


class StrokeWidth:
    """Dilate or erode ink with 3x3 kernel (1 iteration)."""

    def __init__(self, p: float = 0.4, kernel_size: int = 3) -> None:
        self.p = p
        self.kernel_size = kernel_size
        self.kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)

    def __call__(self, img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        if rng.random() > self.p:
            return img
        # ink is 0, bg is 255. Inverted mask has ink=255, bg=0.
        ink_mask = (img < 128).astype(np.uint8) * 255
        if rng.random() < 0.5:
            # Thicken ink: dilate ink mask
            res_mask = cv2.dilate(ink_mask, self.kernel, iterations=1)
        else:
            # Thin ink: erode ink mask
            res_mask = cv2.erode(ink_mask, self.kernel, iterations=1)
        return np.where(res_mask > 127, np.uint8(0), np.uint8(255))


class ResolutionJitter:
    """Downscale then upscale back (bilinear/nearest random), re-binarise with p=0.5."""

    def __init__(self, p: float = 0.4, scale: tuple[float, float] = (0.5, 1.0)) -> None:
        self.p = p
        self.scale = scale

    def __call__(self, img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        if rng.random() > self.p:
            return img
        h, w = img.shape[:2]
        s = rng.uniform(self.scale[0], self.scale[1])
        low_h = max(2, int(round(h * s)))
        low_w = max(2, int(round(w * s)))

        down_interp = cv2.INTER_NEAREST if rng.random() < 0.5 else cv2.INTER_LINEAR
        up_interp = cv2.INTER_NEAREST if rng.random() < 0.5 else cv2.INTER_LINEAR

        down = cv2.resize(img, (low_w, low_h), interpolation=down_interp)
        up = cv2.resize(down, (w, h), interpolation=up_interp)

        if rng.random() < 0.5:
            up = np.where(up > 127, np.uint8(255), np.uint8(0))
        return up.astype(np.uint8)


class Elastic:
    """Grid warp deformation using cv2.remap."""

    def __init__(
        self,
        p: float = 0.3,
        alpha_frac: float = 0.15,
        sigma_frac: float = 0.08,
    ) -> None:
        self.p = p
        self.alpha_frac = alpha_frac
        self.sigma_frac = sigma_frac

    def __call__(self, img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        if rng.random() > self.p:
            return img
        h, w = img.shape[:2]
        alpha = h * self.alpha_frac
        sigma = max(1.0, h * self.sigma_frac)

        rand_x = rng.uniform(-1.0, 1.0, (h, w)).astype(np.float32)
        rand_y = rng.uniform(-1.0, 1.0, (h, w)).astype(np.float32)

        ksize = int(math.ceil(sigma * 3)) * 2 + 1
        dx = cv2.GaussianBlur(rand_x, (ksize, ksize), sigma) * alpha
        dy = cv2.GaussianBlur(rand_y, (ksize, ksize), sigma) * alpha

        grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        map_x = grid_x + dx
        map_y = grid_y + dy

        warped = cv2.remap(
            img,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=255,
        )
        return warped.astype(np.uint8)


class Speckle:
    """Salt and pepper noise."""

    def __init__(self, p: float = 0.3, frac_range: tuple[float, float] = (0.01, 0.03)) -> None:
        self.p = p
        self.frac_range = frac_range

    def __call__(self, img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        if rng.random() > self.p:
            return img
        h, w = img.shape[:2]
        frac = rng.uniform(self.frac_range[0], self.frac_range[1])
        n_noisy = int(round(h * w * frac))
        if n_noisy == 0:
            return img
        out = img.copy()
        ys = rng.integers(0, h, size=n_noisy)
        xs = rng.integers(0, w, size=n_noisy)
        vals = rng.choice(np.array([0, 255], dtype=np.uint8), size=n_noisy)
        out[ys, xs] = vals
        return out


class GaussianBlur:
    """Gaussian blur with small sigma."""

    def __init__(self, p: float = 0.2, sigma_max: float = 0.6) -> None:
        self.p = p
        self.sigma_max = sigma_max

    def __call__(self, img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        if rng.random() > self.p:
            return img
        sigma = rng.uniform(0.1, self.sigma_max)
        ksize = int(math.ceil(sigma * 3)) * 2 + 1
        blurred = cv2.GaussianBlur(img, (ksize, ksize), sigmaX=sigma, borderType=cv2.BORDER_CONSTANT)
        return blurred.astype(np.uint8)


class RandomErasing:
    """Randomly erase a small rectangle (fill white 255 or black 0)."""

    def __init__(self, p: float = 0.25, max_side_frac: float = 0.20) -> None:
        self.p = p
        self.max_side_frac = max_side_frac

    def __call__(self, img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        if rng.random() > self.p:
            return img
        h, w = img.shape[:2]
        max_h = max(2, int(round(h * self.max_side_frac)))
        max_w = max(2, int(round(w * self.max_side_frac)))
        eh = int(rng.integers(1, max_h + 1))
        ew = int(rng.integers(1, max_w + 1))
        y0 = int(rng.integers(0, max(1, h - eh + 1)))
        x0 = int(rng.integers(0, max(1, w - ew + 1)))
        val = np.uint8(255) if rng.random() < 0.7 else np.uint8(0)
        out = img.copy()
        out[y0 : y0 + eh, x0 : x0 + ew] = val
        return out


class Rebinarize:
    """Threshold at 128 to restore 1-bit crisp binary look."""

    def __init__(self, p: float = 0.5, threshold: int = 128) -> None:
        self.p = p
        self.threshold = threshold

    def __call__(self, img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        if rng.random() > self.p:
            return img
        return np.where(img > self.threshold, np.uint8(255), np.uint8(0))


def _build_scaled_op(op_name: str, magnitude: float) -> Callable[[np.ndarray, np.random.Generator], np.ndarray]:
    """Create an operation scaled by magnitude in [0, 1]."""
    mag = max(0.0, min(1.0, magnitude))
    if op_name == "affine":
        return RandomAffine(
            rot=8.0 * mag,
            shear=10.0 * mag,
            scale=(1.0 - 0.15 * mag, 1.0 + 0.15 * mag),
            translate=0.08 * mag,
            p=1.0,
        )
    elif op_name == "margin":
        return MarginJitter(frac=0.15 * mag, p=1.0)
    elif op_name == "stroke":
        return StrokeWidth(p=1.0, kernel_size=3)
    elif op_name == "res_jitter":
        return ResolutionJitter(p=1.0, scale=(1.0 - 0.5 * mag, 1.0))
    elif op_name == "elastic":
        return Elastic(p=1.0, alpha_frac=0.15 * mag, sigma_frac=0.08)
    elif op_name == "speckle":
        return Speckle(p=1.0, frac_range=(0.01 * mag, 0.03 * mag))
    elif op_name == "blur":
        return GaussianBlur(p=1.0, sigma_max=max(0.1, 0.6 * mag))
    elif op_name == "erasing":
        return RandomErasing(p=1.0, max_side_frac=0.20 * mag)
    elif op_name == "rebinarize":
        return Rebinarize(p=1.0)
    else:
        raise ValueError(f"Unknown op name: {op_name}")


CANDIDATE_OPS: list[str] = [
    "affine",
    "margin",
    "stroke",
    "res_jitter",
    "elastic",
    "speckle",
    "blur",
    "erasing",
    "rebinarize",
]


def get_transform(preset: str, seed: int | None = None) -> Callable[[np.ndarray], np.ndarray]:
    """Return an augmentation callable for the given preset name.

    Presets:
      - 'none': identity.
      - 'base': RandomAffine + MarginJitter.
      - 'morph': base + StrokeWidth + ResolutionJitter.
      - 'full': morph + Elastic + Speckle + GaussianBlur + RandomErasing + Rebinarize.
      - 'randaug': RandAugment N=2, M in [0, 10] uniform magnitude.
      - 'trivial': TrivialAugment: one op with uniform random magnitude.

    Every preset guarantees the output contains >= 1% ink pixels; if not, returns
    the input unchanged.
    """
    if preset not in PRESETS:
        raise ValueError(f"Unknown preset: {preset!r}. Expected one of {PRESETS}")

    _state = {"rng": np.random.default_rng(seed)}

    if preset == "none":
        return lambda img: img.copy()

    # Pre-construct fixed pipeline for fixed presets
    if preset == "base":
        ops = [RandomAffine(), MarginJitter()]
    elif preset == "morph":
        ops = [RandomAffine(), MarginJitter(), StrokeWidth(), ResolutionJitter()]
    elif preset == "full":
        ops = [
            RandomAffine(),
            MarginJitter(),
            StrokeWidth(),
            ResolutionJitter(),
            Elastic(),
            Speckle(),
            GaussianBlur(),
            RandomErasing(),
            Rebinarize(),
        ]
    else:
        ops = []

    def transform(img: np.ndarray) -> np.ndarray:
        if preset == "none":
            return img.copy()
        rng = _state["rng"]

        out = img.copy()

        if preset in ("base", "morph", "full"):
            for op in ops:
                out = op(out, rng)
        elif preset == "randaug":
            # N=2 ops, random magnitude M in [0, 10]
            chosen_ops = rng.choice(CANDIDATE_OPS, size=2, replace=True)
            mag = float(rng.uniform(0.0, 1.0))
            for op_name in chosen_ops:
                op = _build_scaled_op(op_name, mag)
                out = op(out, rng)
        elif preset == "trivial":
            # 1 op, uniform random magnitude
            chosen_op = str(rng.choice(CANDIDATE_OPS))
            mag = float(rng.uniform(0.0, 1.0))
            op = _build_scaled_op(chosen_op, mag)
            out = op(out, rng)

        # Guarantee >= 1% ink pixels; otherwise return input unchanged
        ink_frac = float(np.mean(out < 128))
        if ink_frac < 0.01:
            return img.copy()

        return out

    def reseed(new_seed: int | None) -> None:
        """Re-seed the internal RNG (call per DataLoader worker to avoid duplicated draws)."""
        _state["rng"] = np.random.default_rng(new_seed)

    transform.reseed = reseed  # type: ignore[attr-defined]
    transform.preset = preset  # type: ignore[attr-defined]
    return transform
