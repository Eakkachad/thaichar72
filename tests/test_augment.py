"""Tests for augmentation presets, channel encodings, and canvas transforms (TASK-03)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from thaichar.augment import PRESETS, get_transform
from thaichar.data import (
    GlyphCache,
    ThaiGlyphDataset,
    build_clean_index,
    load_cache,
    merge_sources,
)
from thaichar.transforms import (
    encode_channels,
    fit_to_square,
    pad_to_square_canvas,
    resize_square,
)


@pytest.fixture(scope="module")
def clean_df() -> pd.DataFrame:
    return build_clean_index()


@pytest.fixture(scope="module")
def cache() -> GlyphCache:
    return load_cache("data/cache/glyphs.npz")


class TestCanvasTransforms:
    """Test canvas padding, resizing, and equivalence to fit_to_square."""

    def test_pad_and_resize_bit_exact_with_fit_to_square(self, cache: GlyphCache) -> None:
        """pad_to_square_canvas + resize_square equals fit_to_square bit-exactly."""
        rng = np.random.default_rng(42)
        indices = rng.choice(len(cache), size=30, replace=False)

        for idx in indices:
            img = cache[int(idx)]
            for size in (32, 64, 96):
                for margin in (0.05, 0.1, 0.2):
                    expected = fit_to_square(img, size, margin=margin)
                    canvas = pad_to_square_canvas(img, margin=margin)
                    actual = resize_square(canvas, size)
                    assert np.array_equal(actual, expected), (
                        f"Mismatch for size={size}, margin={margin} at index {idx}"
                    )


class TestEncodeChannels:
    """Test multi-channel encodings: gray1, gray3, and onoff."""

    def test_encode_channels_shapes(self) -> None:
        """encode_channels produces correct shapes for all modes."""
        square = np.full((64, 64), 255, dtype=np.uint8)
        square[20:44, 20:44] = 0

        # gray1
        g1 = encode_channels(square, "gray1")
        assert g1.shape == (1, 64, 64)
        assert g1.dtype == np.float32
        assert 0.0 <= g1.min() and g1.max() <= 1.0

        # gray3
        g3 = encode_channels(square, "gray3")
        assert g3.shape == (3, 64, 64)
        assert g3.dtype == np.float32

        # onoff
        onoff = encode_channels(square, "onoff")
        assert onoff.shape == (3, 64, 64)
        assert onoff.dtype == np.float32
        # All standardised channels should be in [-1, 1]
        assert onoff.min() >= -1.0 - 1e-5
        assert onoff.max() <= 1.0 + 1e-5

    def test_dataset_onoff_shape(self, clean_df: pd.DataFrame, cache: GlyphCache) -> None:
        """ThaiGlyphDataset with channel_mode='onoff' returns [3, 64, 64]."""
        ds = ThaiGlyphDataset(clean_df.head(10), cache, size=64, channel_mode="onoff")
        x, y, g = ds[0]
        assert x.shape == (3, 64, 64)
        assert isinstance(x, torch.Tensor)
        assert x.dtype == torch.float32
        assert y.dtype == torch.int64
        assert g.shape == (4,)


class TestAugmentationPresets:
    """Test augmentation presets over real canvases."""

    def test_presets_over_300_real_canvases(self, cache: GlyphCache) -> None:
        """Every preset over 300 random real canvases -> same dtype, square, >=1% ink, values in {0..255}."""
        rng = np.random.default_rng(123)
        sample_indices = rng.choice(len(cache), size=300, replace=False)
        canvases = [pad_to_square_canvas(cache[int(i)], margin=0.1) for i in sample_indices]

        for preset in PRESETS:
            tf = get_transform(preset, seed=42)
            for i, canvas in enumerate(canvases):
                out = tf(canvas)

                assert out.dtype == np.uint8, f"Preset {preset} yielded dtype {out.dtype}"
                assert out.ndim == 2, f"Preset {preset} yielded ndim {out.ndim}"
                assert out.shape[0] == out.shape[1], (
                    f"Preset {preset} output not square: {out.shape}"
                )
                assert int(out.min()) >= 0 and int(out.max()) <= 255, (
                    f"Preset {preset} values out of range: [{out.min()}, {out.max()}]"
                )

                ink_frac = float(np.mean(out < 128))
                assert ink_frac >= 0.01, (
                    f"Preset {preset} sample {i} lost ink: {ink_frac:.4f} < 0.01"
                )

                if preset == "none":
                    assert np.array_equal(out, canvas), "Preset 'none' was not identity"


class TestMergeSources:
    """Test merging multiple data sources into a single dataset."""

    def test_merge_sources_and_dataset(self, clean_df: pd.DataFrame, cache: GlyphCache) -> None:
        """merge_sources creates valid unified index and MultiCache."""
        df1 = clean_df.iloc[:5].copy()
        df2 = clean_df.iloc[5:12].copy()

        df_merged, multi_cache = merge_sources([(df1, cache), (df2, cache)])
        assert len(df_merged) == 12
        assert len(multi_cache) == 2 * len(cache.paths)

        ds = ThaiGlyphDataset(df_merged, multi_cache, size=32, channel_mode="gray1")
        assert len(ds) == 12
        x, y, g = ds[11]
        assert x.shape == (1, 32, 32)
