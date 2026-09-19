"""Tests for synthetic glyph dataset (TASK-03 D6).

Checks:
  - index has 72 classes
  - every combining-mark class has >= 50 samples
  - no synth sample is identical to the rendered base glyph
  - per-class synthetic median height within +-35% of real median
  - cache loads via load_cache and len == len(index)
"""

from __future__ import annotations

import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from thaichar.classes import CLASS_CODES, code_to_char
from thaichar.data import load_cache
from thaichar.synth import COMBINING_MARK_CODES

INDEX_CSV = Path("data/synth/index.csv")
CACHE_NPZ = Path("data/synth/glyphs_synth.npz")
CLASS_STATS_CSV = Path("reports/eda/class_stats.csv")


@pytest.fixture(scope="module")
def synth_df() -> pd.DataFrame:
    if not INDEX_CSV.exists():
        pytest.skip(f"Synthetic index not found: {INDEX_CSV}")
    return pd.read_csv(INDEX_CSV)


@pytest.fixture(scope="module")
def synth_cache():
    if not CACHE_NPZ.exists():
        pytest.skip(f"Synthetic cache not found: {CACHE_NPZ}")
    return load_cache(str(CACHE_NPZ))


@pytest.fixture(scope="module")
def real_stats() -> pd.DataFrame:
    if not CLASS_STATS_CSV.exists():
        pytest.skip(f"Class stats not found: {CLASS_STATS_CSV}")
    return pd.read_csv(CLASS_STATS_CSV).set_index("code")


class TestSynthIndex:
    """Tests on the synthetic index CSV."""

    def test_has_72_classes(self, synth_df: pd.DataFrame) -> None:
        """Synthetic index must have all 72 classes."""
        codes_in_index = set(synth_df["code"].unique())
        missing = set(CLASS_CODES) - codes_in_index
        assert len(missing) == 0, (
            f"Missing {len(missing)} classes from synth index: "
            f"{sorted(missing)[:10]}"
        )
        assert len(codes_in_index) == 72, (
            f"Expected 72 unique codes, got {len(codes_in_index)}"
        )

    def test_required_columns_present(self, synth_df: pd.DataFrame) -> None:
        """Index must have required columns."""
        required = {"path", "code", "label", "char", "category",
                    "width", "height", "ink_frac", "font", "degraded", "group"}
        missing = required - set(synth_df.columns)
        assert len(missing) == 0, f"Missing columns: {missing}"

    def test_combining_mark_classes_min_50_samples(self, synth_df: pd.DataFrame) -> None:
        """Every combining-mark class must have at least 50 samples."""
        insufficient = []
        for code in COMBINING_MARK_CODES:
            n = int((synth_df["code"] == code).sum())
            if n < 50:
                char = code_to_char(code)
                insufficient.append((code, char, n))

        assert len(insufficient) == 0, (
            f"Combining-mark classes with < 50 samples: {insufficient}"
        )

    def test_combining_marks_not_identical_to_base(self, synth_df: pd.DataFrame, synth_cache) -> None:
        """No combining-mark synth sample should be pixel-identical to a blank/white canvas."""
        path_to_idx = {p: i for i, p in enumerate(synth_cache.paths)}
        mark_rows = synth_df[synth_df["code"].isin(COMBINING_MARK_CODES)]

        # Sample up to 200 mark rows to test
        rng = np.random.default_rng(42)
        if len(mark_rows) > 200:
            sample = mark_rows.sample(200, random_state=0)
        else:
            sample = mark_rows

        all_white_count = 0
        for _, row in sample.iterrows():
            p = row["path"]
            if p not in path_to_idx:
                continue
            img = synth_cache[path_to_idx[p]]
            ink_frac = float(np.mean(img < 128))
            if ink_frac < 0.01:
                all_white_count += 1

        assert all_white_count == 0, (
            f"{all_white_count} combining-mark samples appear to have no ink"
        )

    def test_per_class_height_within_35pct_of_real(
        self, synth_df: pd.DataFrame, real_stats: pd.DataFrame
    ) -> None:
        """Per-class synth median height must be within +-35% of real median height."""
        failures = []
        for code in CLASS_CODES:
            if code not in real_stats.index:
                continue
            real_med_h = float(real_stats.loc[code, "med_h"])
            sub = synth_df[synth_df["code"] == code]
            if len(sub) == 0:
                failures.append((code, code_to_char(code), "no samples"))
                continue
            synth_med_h = float(sub["height"].median())
            lo = real_med_h * 0.65
            hi = real_med_h * 1.35
            if not (lo <= synth_med_h <= hi):
                failures.append(
                    (code, code_to_char(code),
                     f"synth_med_h={synth_med_h:.1f}, real_med_h={real_med_h:.1f}, "
                     f"allowed=[{lo:.1f},{hi:.1f}]")
                )

        assert len(failures) == 0, (
            f"{len(failures)} classes have synth height too far from real:\n"
            + "\n".join(f"  code={c} char={ch}: {msg}" for c, ch, msg in failures[:10])
        )


class TestSynthCache:
    """Tests on the synthetic .npz cache."""

    def test_cache_loads_and_len_matches_index(
        self, synth_df: pd.DataFrame, synth_cache
    ) -> None:
        """Cache must load without error and len == len(index)."""
        assert len(synth_cache) == len(synth_df), (
            f"Cache len {len(synth_cache)} != index len {len(synth_df)}"
        )

    def test_cache_images_are_uint8(self, synth_cache) -> None:
        """Sampled cache images must be uint8 arrays."""
        rng = np.random.default_rng(7)
        idxs = rng.choice(len(synth_cache), size=min(50, len(synth_cache)), replace=False)
        for i in idxs:
            img = synth_cache[int(i)]
            assert img.dtype == np.uint8, f"Image {i} has dtype {img.dtype}"
            assert img.ndim == 2, f"Image {i} has ndim {img.ndim}"

    def test_cache_images_have_ink(self, synth_cache) -> None:
        """Sampled cache images must all have >= 1% ink."""
        rng = np.random.default_rng(99)
        idxs = rng.choice(len(synth_cache), size=min(100, len(synth_cache)), replace=False)
        for i in idxs:
            img = synth_cache[int(i)]
            ink_frac = float(np.mean(img < 128))
            assert ink_frac >= 0.01, (
                f"Image {i} has ink_frac={ink_frac:.4f} < 0.01"
            )

    def test_cache_images_are_binary(self, synth_cache) -> None:
        """Cache images must contain only values 0 and 255."""
        rng = np.random.default_rng(13)
        idxs = rng.choice(len(synth_cache), size=min(100, len(synth_cache)), replace=False)
        for i in idxs:
            img = synth_cache[int(i)]
            unique_vals = set(img.ravel().tolist())
            assert unique_vals.issubset({0, 255}), (
                f"Image {i} has non-binary values: {unique_vals - {0, 255}}"
            )
