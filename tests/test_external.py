"""Tests for external dataset mapping, binarization, and caching (TASK-07)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import pytest

from thaichar.classes import code_to_char, code_to_index
from thaichar.data import load_cache
from thaichar.external import (
    binarize_for_cache,
    build_external_cache,
    map_char_to_class,
)


# ---------------------------------------------------------------------------
# Character & transliteration mapping tests
# ---------------------------------------------------------------------------

class TestMapCharToClass:
    """Test map_char_to_class with Thai characters and transliterations."""

    def test_mapped_character(self) -> None:
        """ก (code 161) maps to class 0."""
        assert map_char_to_class("ก") == 0

    def test_unmapped_character(self) -> None:
        """ฅ (code 165) is not in our 72 classes and maps to None."""
        assert map_char_to_class("ฅ") is None

    def test_other_unmapped_characters(self) -> None:
        """Other excluded characters map to None."""
        unmapped_chars = ["ฆ", "ฌ", "ฎ", "ฦ", "ะ", "ำ", "๋", "ํ"]
        for ch in unmapped_chars:
            assert map_char_to_class(ch) is None, f"Expected {ch} to map to None"

    def test_transliterations(self) -> None:
        """Consonant names and transliterations map correctly."""
        assert map_char_to_class("ko kai") == 0
        assert map_char_to_class("kho khai") == 1
        assert map_char_to_class("kho khon") is None  # ฅ
        assert map_char_to_class("THAI CHARACTER KO KAI") == 0

    def test_digits(self) -> None:
        """Digit names and digits map to their class indices."""
        assert map_char_to_class("๐") == 62
        assert map_char_to_class("0") == 62
        assert map_char_to_class("thai digit zero") == 62
        assert map_char_to_class("๙") == 71
        assert map_char_to_class("9") == 71

    def test_invalid_input(self) -> None:
        """Empty string or unknown names return None."""
        assert map_char_to_class("") is None
        assert map_char_to_class("xyz_random_word") is None


# ---------------------------------------------------------------------------
# Binarize & crop tests
# ---------------------------------------------------------------------------

class TestBinarizeForCache:
    """Test binarize_for_cache: Otsu thresholding, tight crop, ink dark."""

    def _make_hollow_box(
        self,
        canvas_h: int = 50,
        canvas_w: int = 50,
        box_y: int = 10,
        box_x: int = 15,
        box_h: int = 25,
        box_w: int = 20,
        thickness: int = 3,
        bg_val: int = 255,
        ink_val: int = 20,
    ) -> np.ndarray:
        """Helper to create a hollow rectangle."""
        arr = np.full((canvas_h, canvas_w), bg_val, dtype=np.uint8)
        # Top and bottom horizontal borders
        arr[box_y : box_y + thickness, box_x : box_x + box_w] = ink_val
        arr[box_y + box_h - thickness : box_y + box_h, box_x : box_x + box_w] = ink_val
        # Left and right vertical borders
        arr[box_y : box_y + box_h, box_x : box_x + thickness] = ink_val
        arr[box_y : box_y + box_h, box_x + box_w - thickness : box_x + box_w] = ink_val
        return arr

    def test_returns_uint8_with_0_and_255(self) -> None:
        """Output contains only {0, 255} and is uint8."""
        img = self._make_hollow_box()
        cropped = binarize_for_cache(img)
        assert cropped.dtype == np.uint8
        unique = set(np.unique(cropped))
        assert unique.issubset({0, 255})
        assert 0 in unique and 255 in unique

    def test_ink_is_dark(self) -> None:
        """Ink pixels are 0, background pixels are 255."""
        img = self._make_hollow_box()
        cropped = binarize_for_cache(img)
        # Corners of the hollow box bounding box are ink (0)
        assert cropped[0, 0] == 0
        # Interior center is background (255)
        cy, cx = cropped.shape[0] // 2, cropped.shape[1] // 2
        assert cropped[cy, cx] == 255

    def test_tight_cropped_zero_margin(self) -> None:
        """Glyph is cropped to exact bounding box with 0 margin."""
        img = self._make_hollow_box(box_y=8, box_x=12, box_h=28, box_w=18)
        cropped = binarize_for_cache(img)
        assert cropped.shape == (28, 18)
        # All 4 borders must touch ink (0)
        assert cropped[0, :].min() == 0, "Top border must touch ink"
        assert cropped[-1, :].min() == 0, "Bottom border must touch ink"
        assert cropped[:, 0].min() == 0, "Left border must touch ink"
        assert cropped[:, -1].min() == 0, "Right border must touch ink"

    def test_inverts_dark_background(self) -> None:
        """Light ink on dark background is correctly inverted to dark ink on white."""
        img = self._make_hollow_box(bg_val=20, ink_val=240)
        cropped = binarize_for_cache(img)
        assert cropped[0, 0] == 0, "Border must be ink (0)"
        cy, cx = cropped.shape[0] // 2, cropped.shape[1] // 2
        assert cropped[cy, cx] == 255, "Inside must be background (255)"

    def test_rejects_empty_image(self) -> None:
        """All-white image with no ink raises ValueError."""
        img = np.full((40, 40), 255, dtype=np.uint8)
        with pytest.raises(ValueError):
            binarize_for_cache(img)

    def test_rejects_excessive_ink_fraction(self) -> None:
        """Solid ink block with ink fraction > 90% raises ValueError."""
        img = np.full((100, 100), 255, dtype=np.uint8)
        img[40:60, 40:60] = 0  # 20x20 ink block on 100x100 white background
        img[50, 50] = 255      # small white pixel inside so crop doesn't collapse
        with pytest.raises(ValueError, match="outside"):
            binarize_for_cache(img)


# ---------------------------------------------------------------------------
# Cache round-trip tests (using 20 synthetic images, no network)
# ---------------------------------------------------------------------------

class TestBuildExternalCacheRoundTrip:
    """Test build_external_cache round-trips through load_cache."""

    @pytest.fixture
    def synthetic_dataset(self, tmp_path: Path) -> tuple[pd.DataFrame, Path, Path]:
        """Generate 20 synthetic grey images and metadata DataFrame."""
        img_dir = tmp_path / "images"
        img_dir.mkdir()

        # Generate 20 synthetic images: 16 mapped, 4 unmapped
        # Mapped codes: 161 (ก), 162 (ข), 164 (ค), 240 (๐)
        # Unmapped codes: 165 (ฅ), 166 (ฆ)
        sample_codes = [
            161, 161, 161, 161,
            162, 162, 162, 162,
            164, 164, 164, 164,
            240, 240, 240, 240,
            165, 165,  # unmapped (ฅ)
            166, 166,  # unmapped (ฆ)
        ]

        records: list[dict] = []
        for i, code in enumerate(sample_codes):
            # Create a 40x40 canvas with a hollow box of size (15+i) x (12+i%5)
            h = 15 + (i % 10)
            w = 12 + (i % 8)
            canvas = np.full((50, 50), 255, dtype=np.uint8)
            # Hollow box
            canvas[5 : 5 + 2, 5 : 5 + w] = 10
            canvas[5 + h - 2 : 5 + h, 5 : 5 + w] = 10
            canvas[5 : 5 + h, 5 : 5 + 2] = 10
            canvas[5 : 5 + h, 5 + w - 2 : 5 + w] = 10

            img_path = img_dir / f"syn_{i:02d}_code_{code}.png"
            Image.fromarray(canvas).save(img_path)

            lbl = map_char_to_class(code_to_char(code))
            records.append({
                "src_path": str(img_path),
                "source": "syn_test",
                "orig_label": f"label_{code}",
                "char": code_to_char(code),
                "code": code,
                "label": lbl if lbl is not None else -1,
                "writer_id": f"W{(i % 4) + 1:02d}",
            })

        df = pd.DataFrame(records)
        out_npz = tmp_path / "glyphs_external.npz"
        out_csv = tmp_path / "index.csv"
        return df, out_npz, out_csv

    def test_build_and_load_round_trip(
        self, synthetic_dataset: tuple[pd.DataFrame, Path, Path]
    ) -> None:
        """build_external_cache writes valid .npz that load_cache correctly loads."""
        df, out_npz, out_csv = synthetic_dataset

        n_saved, n_skipped = build_external_cache(
            df, out_npz=out_npz, index_csv=out_csv, n_workers=1
        )

        # 16 mapped samples should be kept, 4 unmapped rows filtered
        assert n_saved == 16
        assert n_skipped == 0
        assert out_npz.exists()
        assert out_csv.exists()

        # Check index.csv
        index_df = pd.read_csv(out_csv)
        assert len(index_df) == 16
        assert list(index_df.columns) == [
            "path", "code", "label", "char", "category",
            "width", "height", "ink_frac", "group"
        ]
        # Only mapped labels
        assert (index_df["label"] >= 0).all()
        assert not index_df["code"].isin([165, 166]).any()

        # Load with load_cache
        cache = load_cache(str(out_npz))
        assert len(cache) == 16

        # Check every glyph in cache
        for i in range(len(cache)):
            glyph = cache[i]
            row = index_df.iloc[i]
            assert glyph.shape == (int(row["height"]), int(row["width"]))
            assert glyph.dtype == np.uint8
            assert set(np.unique(glyph)).issubset({0, 255})
            # Check ink fraction matches
            calc_ink = float(np.mean(glyph == 0))
            assert abs(calc_ink - float(row["ink_frac"])) < 1e-5
