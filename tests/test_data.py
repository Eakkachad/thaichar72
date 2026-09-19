"""Tests for data preparation pipeline (TASK-02)."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

from thaichar.data import (
    GlyphCache,
    ThaiGlyphDataset,
    build_clean_index,
    load_cache,
    make_doc_split,
    make_stratified_split,
)
from thaichar.transforms import fit_to_square


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def clean_df() -> pd.DataFrame:
    return build_clean_index()


@pytest.fixture(scope="module")
def cache() -> GlyphCache:
    return load_cache("data/cache/glyphs.npz")


# ---------------------------------------------------------------------------
# Split tests
# ---------------------------------------------------------------------------

class TestSplits:
    """Split determinism and correctness."""

    def test_stratified_deterministic(self, clean_df: pd.DataFrame) -> None:
        """Same seed → identical assignment."""
        s1 = make_stratified_split(clean_df, seed=42)
        s2 = make_stratified_split(clean_df, seed=42)
        assert s1.equals(s2)

    def test_doc_deterministic(self, clean_df: pd.DataFrame) -> None:
        """Same seed → identical doc split."""
        s1 = make_doc_split(clean_df, seed=42)
        s2 = make_doc_split(clean_df, seed=42)
        assert s1.equals(s2)

    def test_stratified_no_overlap(self, clean_df: pd.DataFrame) -> None:
        """Train ∩ val = ∅ for stratified split."""
        split = make_stratified_split(clean_df, seed=42)
        train_idx = set(clean_df.index[split == "train"])
        val_idx = set(clean_df.index[split == "val"])
        assert len(train_idx & val_idx) == 0

    def test_doc_no_overlap(self, clean_df: pd.DataFrame) -> None:
        """Train ∩ val = ∅ for doc split."""
        split = make_doc_split(clean_df, seed=42)
        train_idx = set(clean_df.index[split == "train"])
        val_idx = set(clean_df.index[split == "val"])
        assert len(train_idx & val_idx) == 0

    def test_every_class_with_n_ge_2_has_val(self, clean_df: pd.DataFrame) -> None:
        """Every class with n ≥ 2 has at least 1 val sample."""
        split = make_stratified_split(clean_df, seed=42)
        for code in clean_df["code"].unique():
            mask = clean_df["code"] == code
            n = int(mask.sum())
            if n >= 2:
                n_val = int((split[mask] == "val").sum())
                assert n_val >= 1, f"Code {code} has n={n} but 0 val samples"


# ---------------------------------------------------------------------------
# Transform tests
# ---------------------------------------------------------------------------

class TestFitToSquare:
    """fit_to_square correctness."""

    def test_output_shape(self) -> None:
        """Output is exactly size×size."""
        img = np.zeros((10, 30), dtype=np.uint8)
        out = fit_to_square(img, 64)
        assert out.shape == (64, 64)

    def test_preserves_aspect_ratio(self) -> None:
        """A 10×30 glyph: after resize, width/height ratio within 10% of 1/3 inside the glyph region."""
        img = np.zeros((10, 30), dtype=np.uint8)  # ink = 0
        size = 64
        margin = 0.1
        out = fit_to_square(img, size, margin=margin)

        # The glyph should fit within size/(1+2*margin) of the square
        max_glyph_extent = size / (1 + 2 * margin)

        # Find bounding box of ink pixels
        ink = out < 128
        if ink.any():
            rows = np.where(ink.any(axis=1))[0]
            cols = np.where(ink.any(axis=0))[0]
            glyph_h = rows[-1] - rows[0] + 1
            glyph_w = cols[-1] - cols[0] + 1

            assert glyph_h <= max_glyph_extent + 1, \
                f"Glyph height {glyph_h} exceeds {max_glyph_extent}"
            # Aspect ratio of original: w/h = 30/10 = 3.0
            # So in the resized version, ratio should be close to 3.0
            actual_ratio = glyph_w / glyph_h
            expected_ratio = 30 / 10  # = 3.0
            assert abs(actual_ratio - expected_ratio) / expected_ratio < 0.10, \
                f"Aspect ratio {actual_ratio:.3f} too far from {expected_ratio:.3f}"

    def test_no_ink_on_border(self) -> None:
        """No ink touches the border of the output."""
        img = np.zeros((10, 30), dtype=np.uint8)
        out = fit_to_square(img, 64, margin=0.1)
        # Check all 4 borders
        assert out[0, :].min() > 127, "Ink on top border"
        assert out[-1, :].min() > 127, "Ink on bottom border"
        assert out[:, 0].min() > 127, "Ink on left border"
        assert out[:, -1].min() > 127, "Ink on right border"


# ---------------------------------------------------------------------------
# Dataset tests
# ---------------------------------------------------------------------------

class TestDataset:
    """ThaiGlyphDataset shape checks."""

    def test_shape_3ch_64(self, clean_df: pd.DataFrame, cache: GlyphCache) -> None:
        """Default: channels=3, size=64 → x [3,64,64]."""
        ds = ThaiGlyphDataset(clean_df.head(10), cache, size=64, channels=3)
        x, y, g = ds[0]
        assert x.shape == (3, 64, 64), f"Got {x.shape}"
        assert y.dtype == getattr(__import__("torch"), "int64")
        assert g.shape == (4,)

    def test_shape_1ch_32(self, clean_df: pd.DataFrame, cache: GlyphCache) -> None:
        """channels=1, size=32 → x [1,32,32]."""
        ds = ThaiGlyphDataset(clean_df.head(10), cache, size=32, channels=1)
        x, y, g = ds[0]
        assert x.shape == (1, 32, 32), f"Got {x.shape}"


# ---------------------------------------------------------------------------
# Cache performance
# ---------------------------------------------------------------------------

class TestCachePerformance:
    """Cache random-access speed."""

    def test_2000_random_samples_under_2s(self, cache: GlyphCache) -> None:
        """Loading 2000 random samples < 2 s."""
        rng = np.random.default_rng(0)
        indices = rng.integers(0, len(cache), size=2000)
        t0 = time.time()
        for i in indices:
            _ = cache[i]
        elapsed = time.time() - t0
        assert elapsed < 2.0, f"Took {elapsed:.2f} s"
