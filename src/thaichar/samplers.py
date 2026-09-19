"""Sampling strategies for imbalanced Thai glyph data."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler


def build_sampler(
    labels: np.ndarray | list[int],
    kind: str = "none",
    cap: float = 10.0,
) -> WeightedRandomSampler | None:
    """Build a weighted random sampler for imbalanced classes.

    Parameters
    ----------
    labels : array-like
        Integer class labels for every sample.
    kind : str
        'none' → None (uniform),
        'sqrt_inv' → weight ∝ n_c^-0.5,
        'inv' → weight ∝ n_c^-1.0.
    cap : float
        Maximum per-sample weight relative to uniform (1/N).

    Returns
    -------
    WeightedRandomSampler | None
    """
    if kind == "none":
        return None

    labels_arr = np.asarray(labels, dtype=np.int64)
    n = len(labels_arr)
    classes, counts = np.unique(labels_arr, return_counts=True)
    count_map = dict(zip(classes.tolist(), counts.tolist()))

    if kind == "sqrt_inv":
        power = 0.5
    elif kind == "inv":
        power = 1.0
    else:
        raise ValueError(f"Unknown sampler kind: {kind!r}")

    # Per-class weight ∝ n_c^-power
    class_weight = {c: max(count_map[c], 1) ** (-power) for c in count_map}

    # Per-sample weights
    sample_weights = np.array([class_weight[int(l)] for l in labels_arr], dtype=np.float64)

    # Normalise so mean = 1.0 (uniform baseline)
    sample_weights /= sample_weights.mean()

    # Cap
    uniform_w = 1.0  # after normalisation, uniform is 1.0
    max_w = cap * uniform_w
    sample_weights = np.minimum(sample_weights, max_w)

    return WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).double(),
        num_samples=n,
        replacement=True,
    )
