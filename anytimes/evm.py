"""Utility functions for Extreme Value (Generalized Pareto) analysis."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.stats import genpareto


def _gpd_negative_log_likelihood(
    params: np.ndarray, excesses: np.ndarray
) -> float:
    """Return the negative log-likelihood of a GPD sample."""

    shape, scale = params
    if scale <= 0 or np.any(1.0 + shape * excesses / scale <= 0.0):
        return float("inf")

    log_term = np.log1p(shape * excesses / scale)
    n = excesses.size
    return n * np.log(scale) + (1.0 / shape + 1.0) * np.sum(log_term)


def _gpd_parameter_covariance(
    *, shape: float, scale: float, excesses: np.ndarray
) -> np.ndarray | None:
    """Return an observed-information covariance matrix for ``shape`` and ``scale``."""

    params = np.asarray([shape, scale], dtype=float)
    if np.any(~np.isfinite(params)):
        return None

    # Scale the perturbation relative to the magnitude of each parameter so that
    # the finite-difference Hessian remains well conditioned for both large and
    # small values.
    step = np.maximum(np.abs(params), 1.0) * 1e-5
    hessian = np.empty((2, 2), dtype=float)

    for i in range(2):
        for j in range(2):
            offset_i = np.zeros(2, dtype=float)
            offset_j = np.zeros(2, dtype=float)
            offset_i[i] = step[i]
            offset_j[j] = step[j]

            f_pp = _gpd_negative_log_likelihood(
                params + offset_i + offset_j, excesses
            )
            f_pm = _gpd_negative_log_likelihood(
                params + offset_i - offset_j, excesses
            )
            f_mp = _gpd_negative_log_likelihood(
                params - offset_i + offset_j, excesses
            )
            f_mm = _gpd_negative_log_likelihood(
                params - offset_i - offset_j, excesses
            )

            hessian[i, j] = (f_pp - f_pm - f_mp + f_mm) / (4.0 * step[i] * step[j])

    if not np.all(np.isfinite(hessian)):
        return None

    try:
        covariance = np.linalg.inv(hessian)
    except np.linalg.LinAlgError:
        return None

    if np.any(~np.isfinite(covariance)):
        return None

    return covariance


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



def declustering_boundaries(signal: np.ndarray, tail: str) -> np.ndarray:
    r"""Return indices that split *signal* at mean crossings.

    The GUI declustering routine separates the record whenever it crosses the
    mean level and then keeps the most extreme value from each segment.  This
    helper mirrors that behaviour so that both the GUI and the reusable module
    use identical clustering logic.
    """

    if signal.size == 0:
        return np.array([0], dtype=int)

    mean_val = float(np.mean(signal))
    cross_type = np.greater if tail == "upper" else np.less
    # ``np.diff`` operates on the boolean mask and highlights sign changes.
    crossings = np.where(np.diff(cross_type(signal, mean_val)))[0] + 1

    # Always include the start and end of the record so that every sample is
    # covered even if no crossings are detected.
    boundaries = np.concatenate(([0], crossings, [signal.size]))
    _, unique_indices = np.unique(boundaries, return_index=True)
    return boundaries[np.sort(unique_indices)]



def cluster_exceedances(x: np.ndarray, threshold: float, tail: str) -> np.ndarray:
    """Return the cluster peaks that exceed *threshold*.

    The GUI performs a crude declustering by splitting the signal on mean level
    crossings and picking the most extreme value in each segment. Re-use the
    same logic here so that the behaviour can be tested without the GUI.
    """

    if tail not in {"upper", "lower"}:
        raise ValueError("tail must be 'upper' or 'lower'")


    boundaries = declustering_boundaries(x, tail)

    clustered_peaks: list[float] = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        if end <= start:
            continue
        segment = x[start:end]
        peak = float(np.max(segment) if tail == "upper" else np.min(segment))

        if (tail == "upper" and peak > threshold) or (
            tail == "lower" and peak < threshold
        ):
            clustered_peaks.append(peak)

    return np.asarray(clustered_peaks, dtype=float)



def _prepare_tail_arrays(
    t: np.ndarray, x: np.ndarray, tail: str
) -> tuple[np.ndarray, np.ndarray]:
    """Return sorted copies of ``t`` and ``x`` suitable for analysis."""

    if t.shape != x.shape:
        raise ValueError("t and x must have matching shapes")

    if t.size < 2:
        raise ValueError("At least two samples are required for extreme value analysis")

    order = np.argsort(t)
    t_sorted = np.asarray(t, dtype=float)[order]
    x_sorted = np.asarray(x, dtype=float)[order]

    if not np.all(np.isfinite(t_sorted)) or not np.all(np.isfinite(x_sorted)):
        raise ValueError("t and x must contain only finite values")

    if t_sorted[-1] == t_sorted[0]:
        raise ValueError("Time array must span a non-zero duration")

    return t_sorted, x_sorted


def _return_levels(
    *,
    threshold: float,
    scale: float,
    shape: float,
    exceedance_rate: float,
    return_durations: np.ndarray,
    tail: str,

) -> np.ndarray:
    r"""Compute return levels using the OrcaFlex convention.

    The OrcaFlex documentation defines the return level for a storm of
    duration ``T`` hours as ::

        z_T = u + \frac{\sigma}{\xi} \left( (\lambda T)^{\xi} - 1 \right)

    where ``u`` is the threshold, ``\sigma`` is the scale, ``\xi`` is the
    shape and ``\lambda`` is the mean cluster rate per hour.  The lower-tail
    expression mirrors the upper-tail result but subtracts the positive GPD
    excursion instead of adding it.  The limit as ``\xi`` tends to zero is
    handled analytically.
    """

    scaled_rate = exceedance_rate * return_durations
    if np.any(scaled_rate <= 0):
        raise ValueError("Return durations must be positive")

    with np.errstate(divide="ignore", invalid="ignore"):
        if abs(shape) < 1e-9:
            excursion = scale * np.log(scaled_rate)
        else:
            excursion = (scale / shape) * (np.power(scaled_rate, shape) - 1.0)

    if tail == "upper":
        return threshold + excursion
    return threshold - excursion



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
        Number of multivariate normal samples used to approximate the
        confidence interval.  Set to zero to skip the interval calculation.
    rng:
        Optional :class:`numpy.random.Generator` for deterministic bootstrapping.
    """

    if tail not in {"upper", "lower"}:
        raise ValueError("tail must be 'upper' or 'lower'")


    t, x = _prepare_tail_arrays(np.asarray(t, dtype=float), np.asarray(x, dtype=float), tail)


    if clustered_peaks is None:
        clustered_peaks = cluster_exceedances(x, threshold, tail)
    else:
        clustered_peaks = np.asarray(clustered_peaks, dtype=float)
    if clustered_peaks.size == 0:
        raise ValueError("No exceedances found above the provided threshold")


    if tail == "upper":
        excesses = clustered_peaks - threshold
    else:
        excesses = threshold - clustered_peaks
    if np.any(excesses <= 0):
        raise ValueError("Threshold must be exceeded by all clustered peaks")

    shape, loc, scale = genpareto.fit(excesses, floc=0)

    duration_seconds = float(t[-1] - t[0])
    exceed_rate = clustered_peaks.size / (duration_seconds / 3600.0)
    return_periods = np.asarray(tuple(return_periods_hours), dtype=float)
    if np.any(return_periods <= 0):
        raise ValueError("Return periods must be positive")
    return_secs = return_periods * 3600
    return_levels = _return_levels(
        threshold=threshold,
        scale=scale,
        shape=shape,
        exceedance_rate=exceed_rate,
        return_durations=return_secs / 3600.0,
        tail=tail,
    )

    covariance = _gpd_parameter_covariance(
        shape=shape, scale=scale, excesses=excesses
    )

    ci_alpha = 100.0 - confidence_level
    lower_bounds = np.full(return_levels.shape, np.nan)
    upper_bounds = np.full(return_levels.shape, np.nan)

    if covariance is not None and n_bootstrap > 0:
        rng = np.random.default_rng() if rng is None else rng
        try:
            samples = rng.multivariate_normal(
                mean=np.array([shape, scale], dtype=float),
                cov=covariance,
                size=int(n_bootstrap),
            )
        except ValueError:
            samples = np.empty((0, 2))

        if samples.size:
            valid_levels: list[np.ndarray] = []
            for sh, sc in samples:
                if sc <= 0:
                    continue
                if np.any(1.0 + sh * excesses / sc <= 0.0):
                    continue

                boot_level = _return_levels(
                    threshold=threshold,
                    scale=sc,
                    shape=sh,
                    exceedance_rate=exceed_rate,
                    return_durations=return_secs / 3600.0,
                    tail=tail,
                )

                if np.isnan(boot_level).any():
                    continue
                valid_levels.append(boot_level)

            if valid_levels:
                boot_arr = np.vstack(valid_levels)
                lower_bounds = np.percentile(boot_arr, ci_alpha / 2, axis=0)
                upper_bounds = np.percentile(boot_arr, 100.0 - ci_alpha / 2, axis=0)

    return ExtremeValueResult(
        return_periods=return_periods,
        return_levels=return_levels,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        shape=float(shape),
        scale=float(scale),
        exceedances=clustered_peaks,
        threshold=float(threshold),

        exceedance_rate=float(exceed_rate),

    )


__all__ = [
    "ExtremeValueResult",
    "calculate_extreme_value_statistics",
    "cluster_exceedances",

    "declustering_boundaries",

]

