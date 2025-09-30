"""Utility functions for Extreme Value (Generalized Pareto) analysis."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.stats import genpareto


@dataclass(frozen=True)
class ExtremeValueResult:
    """Container for Generalized Pareto extreme value analysis results."""

    return_periods: np.ndarray
    return_levels: np.ndarray
    lower_bounds: np.ndarray
    upper_bounds: np.ndarray
    shape: float
    scale: float
    exceedances: np.ndarray
    threshold: float
    exceedance_rate: float


def cluster_exceedances(x: np.ndarray, threshold: float, tail: str) -> np.ndarray:
    """Return the cluster peaks that exceed *threshold*.

    The GUI performs a crude declustering by splitting the signal on mean level
    crossings and picking the most extreme value in each segment. Re-use the
    same logic here so that the behaviour can be tested without the GUI.
    """

    if tail not in {"upper", "lower"}:
        raise ValueError("tail must be 'upper' or 'lower'")

    mean_val = np.mean(x)
    cross_type = np.greater if tail == "upper" else np.less
    cross_indices = np.where(np.diff(cross_type(x, mean_val)))[0]
    if cross_indices.size == 0 or cross_indices[-1] != len(x) - 1:
        cross_indices = np.append(cross_indices, len(x) - 1)

    clustered_peaks: list[float] = []
    for i in range(len(cross_indices) - 1):
        segment = x[cross_indices[i] : cross_indices[i + 1]]
        peak = np.max(segment) if tail == "upper" else np.min(segment)
        if (tail == "upper" and peak > threshold) or (
            tail == "lower" and peak < threshold
        ):
            clustered_peaks.append(peak)

    return np.asarray(clustered_peaks, dtype=float)


def calculate_extreme_value_statistics(
    t: np.ndarray,
    x: np.ndarray,
    threshold: float,
    *,
    tail: str = "upper",
    return_periods_hours: Sequence[float] = (0.1, 0.5, 1, 3, 5),
    confidence_level: float = 95.0,
    n_bootstrap: int = 500,
    rng: np.random.Generator | None = None,
    clustered_peaks: np.ndarray | None = None,
) -> ExtremeValueResult:
    """Estimate return levels using the Generalized Pareto distribution.

    Parameters
    ----------
    t, x:
        Time stamps and samples of the signal.
    threshold:
        Level above which exceedances are analysed.
    tail:
        "upper" for high extremes, "lower" for low extremes.
    return_periods_hours:
        Iterable of return periods (in hours) for which to compute levels.
    confidence_level:
        Percent confidence level for the bootstrap interval.
    n_bootstrap:
        Number of bootstrap iterations.
    rng:
        Optional :class:`numpy.random.Generator` for deterministic bootstrapping.
    """

    if tail not in {"upper", "lower"}:
        raise ValueError("tail must be 'upper' or 'lower'")

    if clustered_peaks is None:
        clustered_peaks = cluster_exceedances(x, threshold, tail)
    else:
        clustered_peaks = np.asarray(clustered_peaks, dtype=float)
    if clustered_peaks.size == 0:
        raise ValueError("No exceedances found above the provided threshold")

    excesses = clustered_peaks - threshold
    c, loc, scale = genpareto.fit(excesses, floc=0)

    exceed_prob = clustered_peaks.size / (t[-1] - t[0])
    return_periods = np.asarray(tuple(return_periods_hours), dtype=float)
    return_secs = return_periods * 3600
    return_levels = threshold + (scale / c) * ((exceed_prob * return_secs) ** c - 1)

    rng = np.random.default_rng() if rng is None else rng
    boot_levels: list[np.ndarray] = []
    for _ in range(n_bootstrap):
        sample = rng.choice(excesses, size=excesses.size, replace=True)
        try:
            bc, _, bscale = genpareto.fit(sample, floc=0)
        except Exception:
            continue
        boot_level = threshold + (bscale / bc) * ((exceed_prob * return_secs) ** bc - 1)
        if np.isnan(boot_level).any():
            continue
        if not ((boot_level > -1e6).all() and (boot_level < 1e6).all()):
            continue
        boot_levels.append(boot_level)

    if boot_levels:
        boot_arr = np.vstack(boot_levels)
        ci_alpha = 100 - confidence_level
        lower_bounds = np.percentile(boot_arr, ci_alpha / 2, axis=0)
        upper_bounds = np.percentile(boot_arr, 100 - ci_alpha / 2, axis=0)
    else:
        lower_bounds = upper_bounds = np.full(return_levels.shape, np.nan)

    return ExtremeValueResult(
        return_periods=return_periods,
        return_levels=return_levels,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        shape=float(c),
        scale=float(scale),
        exceedances=clustered_peaks,
        threshold=float(threshold),
        exceedance_rate=float(exceed_prob),
    )


__all__ = [
    "ExtremeValueResult",
    "calculate_extreme_value_statistics",
    "cluster_exceedances",
]

