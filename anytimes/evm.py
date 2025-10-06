"""Utility functions for Extreme Value analysis.

This module contains both the original built-in Generalized Pareto
implementation as well as an optional interface to the ``pyextremes``
package.  The GUI can now choose between the two engines at run time.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Mapping, Sequence

import matplotlib.ticker as mticker
import numpy as np
from scipy.stats import genpareto


_PYEXTREMES_PLOTTING_POSITIONS = {
    "ecdf",
    "hazen",
    "weibull",
    "tukey",
    "blom",
    "median",
    "cunnane",
    "gringorten",
    "beard",
}


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


def _normalise_tail(tail: str) -> str:
    """Return the canonical tail label (``"upper"`` or ``"lower"``)."""

    tail_key = tail.lower()
    if tail_key in {"upper", "high"}:
        return "upper"
    if tail_key in {"lower", "low"}:
        return "lower"
    raise ValueError("tail must be 'upper'/'high' or 'lower'/'low'")


@dataclass(frozen=True)
class ExtremeValueResult:
    """Container for extreme value analysis results."""

    return_periods: np.ndarray
    return_levels: np.ndarray
    lower_bounds: np.ndarray
    upper_bounds: np.ndarray
    shape: float
    scale: float
    exceedances: np.ndarray
    threshold: float
    exceedance_rate: float
    engine: str = "builtin"
    metadata: Mapping[str, object] | None = None
    diagnostic_figure: object | None = None


@contextmanager
def _temporary_numpy_seed(seed: int | None):
    """Context manager that temporarily sets the global NumPy RNG seed."""

    if seed is None:
        yield
        return

    state = np.random.get_state()
    np.random.seed(int(seed) & 0xFFFF_FFFF)
    try:
        yield
    finally:
        np.random.set_state(state)



def declustering_boundaries(signal: np.ndarray, tail: str) -> np.ndarray:
    r"""Return indices that split *signal* at mean crossings.

    The GUI declustering routine separates the record whenever it crosses the
    mean level and then keeps the most extreme value from each segment.  This
    helper mirrors that behaviour so that both the GUI and the reusable module
    use identical clustering logic.  ``tail`` may be supplied as ``"upper"`` /
    ``"high"`` for maxima or ``"lower"`` / ``"low"`` for minima.
    """

    tail = _normalise_tail(tail)

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
    same logic here so that the behaviour can be tested without the GUI.  The
    ``tail`` argument accepts either ``"upper"`` / ``"high"`` or ``"lower"`` /
    ``"low"`` aliases.
    """

    tail = _normalise_tail(tail)

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

    tail = _normalise_tail(tail)

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



_DEFAULT_RETURN_PERIODS_HOURS = (0.1, 0.5, 1, 3, 5)


def calculate_extreme_value_statistics(
    t: np.ndarray,
    x: np.ndarray,
    threshold: float,
    *,
    tail: str = "upper",
    return_periods_hours: Sequence[float] | None = _DEFAULT_RETURN_PERIODS_HOURS,
    confidence_level: float = 95.0,
    n_bootstrap: int = 500,
    rng: np.random.Generator | None = None,
    clustered_peaks: np.ndarray | None = None,
    sample_exceedance_rate: bool = False,
    engine: str = "builtin",
    pyextremes_options: Mapping[str, object] | None = None,
) -> ExtremeValueResult:
    """Estimate return levels for the requested extreme value ``engine``.

    The default ``engine='builtin'`` reproduces the historical behaviour using
    SciPy's Generalized Pareto fitting.  Selecting ``engine='pyextremes'``
    dispatches the computation to :mod:`pyextremes` while keeping the return
    signature identical.  ``tail`` accepts both ``"upper"`` / ``"high"`` and
    ``"lower"`` / ``"low"`` labels.
    """

    tail = _normalise_tail(tail)

    engine_key = (engine or "builtin").lower()
    if engine_key in {"builtin", "gpd", "scipy"}:
        builtin_return_periods = (
            _DEFAULT_RETURN_PERIODS_HOURS
            if return_periods_hours is None
            else return_periods_hours
        )
        return _calculate_extreme_value_statistics_builtin(
            t,
            x,
            threshold,
            tail=tail,
            return_periods_hours=builtin_return_periods,
            confidence_level=confidence_level,
            n_bootstrap=n_bootstrap,
            rng=rng,
            clustered_peaks=clustered_peaks,
            sample_exceedance_rate=sample_exceedance_rate,
        )

    if engine_key == "pyextremes":
        return _calculate_extreme_value_statistics_pyextremes(
            t,
            x,
            threshold,
            tail=tail,
            return_periods_hours=return_periods_hours,
            confidence_level=confidence_level,
            rng=rng,
            options=pyextremes_options or {},
        )

    raise ValueError(f"Unknown extreme value engine '{engine}'")


def _calculate_extreme_value_statistics_builtin(
    t: np.ndarray,
    x: np.ndarray,
    threshold: float,
    *,
    tail: str,
    return_periods_hours: Sequence[float] | None,
    confidence_level: float,
    n_bootstrap: int,
    rng: np.random.Generator | None,
    clustered_peaks: np.ndarray | None,
    sample_exceedance_rate: bool,
) -> ExtremeValueResult:
    """Original Generalized Pareto implementation used by the GUI."""

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

    shape, _loc, scale = genpareto.fit(excesses, floc=0)

    duration_seconds = float(t[-1] - t[0])
    duration_hours = duration_seconds / 3600.0
    exceed_rate = clustered_peaks.size / duration_hours
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
        rng = np.random.default_rng(0) if rng is None else rng
        try:
            samples = rng.multivariate_normal(
                mean=np.array([shape, scale], dtype=float),
                cov=covariance,
                size=int(n_bootstrap),
            )
        except ValueError:
            samples = np.empty((0, 2))

        if samples.size:
            shapes = samples[:, 0]
            scales = samples[:, 1]

            finite_mask = np.isfinite(shapes) & np.isfinite(scales)
            if not finite_mask.all():
                shapes = shapes[finite_mask]
                scales = scales[finite_mask]

            if shapes.size:
                max_excess = float(np.max(excesses))

                valid_mask = scales > 0
                if max_excess > 0.0:
                    support_limit = -shapes * max_excess
                    valid_mask &= (shapes >= 0.0) | (scales > support_limit)

                if not valid_mask.all():
                    shapes = shapes[valid_mask]
                    scales = scales[valid_mask]

                if shapes.size:
                    durations = return_secs / 3600.0
                    if sample_exceedance_rate and clustered_peaks.size > 0:
                        rate_samples = rng.gamma(
                            shape=clustered_peaks.size,
                            scale=1.0 / duration_hours,
                            size=shapes.size,
                        )
                    else:
                        rate_samples = np.full(shapes.size, exceed_rate)

                    rate = rate_samples[:, np.newaxis] * (durations[np.newaxis, :])
                    shape_mat = shapes[:, np.newaxis]
                    scale_mat = scales[:, np.newaxis]

                    excursion = np.where(
                        np.isclose(shape_mat, 0.0, atol=1e-9),
                        scale_mat * np.log(rate),
                        (scale_mat / shape_mat) * (np.power(rate, shape_mat) - 1.0),
                    )

                    if tail == "upper":
                        boot_levels = threshold + excursion
                    else:
                        boot_levels = threshold - excursion

                    lower_bounds = np.percentile(boot_levels, ci_alpha / 2, axis=0)
                    upper_bounds = np.percentile(
                        boot_levels, 100.0 - ci_alpha / 2, axis=0
                    )

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
        engine="builtin",
        metadata={"method": "scipy_genpareto"},
        diagnostic_figure=None,
    )


def _calculate_extreme_value_statistics_pyextremes(
    t: np.ndarray,
    x: np.ndarray,
    threshold: float,
    *,
    tail: str,
    return_periods_hours: Sequence[float],
    confidence_level: float,
    rng: np.random.Generator | None,
    options: Mapping[str, object],
) -> ExtremeValueResult:
    """Estimate return levels using :mod:`pyextremes`."""

    try:
        import pandas as pd
        from pyextremes import EVA
    except ImportError as exc:  # pragma: no cover - defensive guard
        raise ImportError(
            "pyextremes must be installed to use engine='pyextremes'"
        ) from exc

    t_arr, x_arr = _prepare_tail_arrays(
        np.asarray(t, dtype=float), np.asarray(x, dtype=float), tail
    )

    if t_arr.size < 2:
        raise ValueError("At least two samples are required for extreme value analysis")

    start_time = pd.Timestamp("1970-01-01")
    offsets = pd.to_timedelta(t_arr - t_arr[0], unit="s")
    index = start_time + offsets
    series = pd.Series(x_arr, index=index, name="signal")

    method = str(options.get("method", "POT")).upper()
    extremes_type = "high" if tail == "upper" else "low"

    eva = EVA(series)

    def _coerce_timedelta(value, *, default_unit="s", argument="value"):
        if value is None:
            return None
        if isinstance(value, pd.Timedelta):
            return value
        if isinstance(value, str):
            return pd.to_timedelta(value)
        if isinstance(value, (int, float)):
            return pd.to_timedelta(value, unit=default_unit)
        raise TypeError(f"Invalid type for {argument}: {type(value)!r}")

    metadata: dict[str, object] = {
        "method": method,
        "extremes_type": extremes_type,
    }

    plotting_position_opt = options.get("plotting_position", "weibull")
    if plotting_position_opt is None:
        plotting_position = "weibull"
    else:
        plotting_position = str(plotting_position_opt).lower()
    if plotting_position not in _PYEXTREMES_PLOTTING_POSITIONS:
        raise ValueError(
            f"Unsupported pyextremes plotting_position '{plotting_position_opt}'"
        )
    metadata["plotting_position"] = plotting_position

    if method == "POT":
        r_value = options.get("r")
        if r_value is None:
            diffs = np.diff(t_arr)
            positive_diffs = diffs[diffs > 0]
            if positive_diffs.size:
                median_step = float(np.median(positive_diffs))
            else:
                median_step = 0.0
            r_td = _coerce_timedelta(median_step, argument="r")
        else:
            r_td = _coerce_timedelta(r_value, argument="r")

        eva.get_extremes(
            method="POT",
            extremes_type=extremes_type,
            threshold=float(threshold),
            r=r_td,
        )
        metadata["declustering_window"] = r_td
    elif method == "BM":
        block_size = options.get("block_size", "24h")
        block_td = _coerce_timedelta(block_size, argument="block_size")
        eva.get_extremes(
            method="BM",
            extremes_type=extremes_type,
            block_size=block_td,
        )
        metadata["block_size"] = block_td
    else:
        raise ValueError(f"Unsupported pyextremes method '{method}'")

    distribution = options.get("distribution")
    if distribution is None:
        distribution = "genpareto" if method == "POT" else "genextreme"

    fit_kwargs = dict(options.get("fit_kwargs", {}))
    eva.fit_model(distribution=distribution, **fit_kwargs)
    metadata["distribution"] = distribution

    exceedances = eva.extremes.values
    if exceedances.size == 0:
        raise ValueError("PyExtremes did not identify any exceedances")

    params = dict(eva.model.distribution.mle_parameters)
    shape = params.get("c")
    if shape is None:
        shape = params.get("shape")
    if shape is None:
        shape = params.get("xi")
    scale = params.get("scale")

    shape = float(shape) if shape is not None else float("nan")
    scale = float(scale) if scale is not None else float("nan")

    duration_seconds = float(t_arr[-1] - t_arr[0])
    duration_hours = duration_seconds / 3600.0
    if duration_hours <= 0:
        raise ValueError("Time array must span a non-zero duration")

    exceed_rate = exceedances.size / duration_hours

    return_period_size = _coerce_timedelta(
        options.get("return_period_size", "1h"),
        argument="return_period_size",
    )
    metadata["return_period_size"] = return_period_size

    base_hours = return_period_size / np.timedelta64(1, "h")
    if base_hours <= 0:
        raise ValueError("return_period_size must be positive")

    diagnostic_periods: np.ndarray | None = None

    if return_periods_hours is None:
        from pyextremes.extremes import return_periods as _pyext_return_periods

        observed_return_values = _pyext_return_periods.get_return_periods(
            ts=series,
            extremes=eva.extremes,
            extremes_method=method,
            extremes_type=extremes_type,
            block_size=options.get("block_size") if method == "BM" else None,
            return_period_size=return_period_size,
            plotting_position=plotting_position,
        )

        observed_periods = observed_return_values.loc[:, "return period"].astype(float)
        if observed_periods.empty:
            raise ValueError("PyExtremes did not identify any return periods")

        min_period = float(np.nanmin(observed_periods))
        max_period = float(np.nanmax(observed_periods))

        if not np.isfinite(min_period) or not np.isfinite(max_period):
            raise ValueError("PyExtremes produced invalid return periods")

        if max_period <= min_period:
            max_period = min_period * 1.1 if min_period > 0 else 1.0

        diagnostic_periods = np.linspace(min_period, max_period, 100, dtype=float)
        return_periods = np.asarray(_DEFAULT_RETURN_PERIODS_HOURS, dtype=float)
    else:
        return_periods = np.asarray(tuple(return_periods_hours), dtype=float)
        if np.any(return_periods <= 0):
            raise ValueError("Return periods must be positive")

    metadata["return_periods_hours"] = tuple(return_periods)

    pyext_return_periods = return_periods / float(base_hours)


    if "diagnostic_return_periods" in options:
        diagnostic_return_periods_opt = options.get("diagnostic_return_periods")
    else:
        diagnostic_return_periods_opt = None

    if diagnostic_return_periods_opt is None:
        if diagnostic_periods is not None:
            diagnostic_return_periods = diagnostic_periods
        else:
            diagnostic_return_periods = pyext_return_periods
    else:
        diagnostic_return_periods = np.asarray(
            tuple(diagnostic_return_periods_opt), dtype=float
        )
        if np.any(diagnostic_return_periods <= 0):
            raise ValueError("diagnostic_return_periods must be positive")
        diagnostic_return_periods = diagnostic_return_periods / float(base_hours)

    metadata["diagnostic_return_periods"] = (
        None
        if diagnostic_return_periods_opt is None
        else tuple(np.asarray(diagnostic_return_periods_opt, dtype=float))
    )

    alpha = float(confidence_level) / 100.0
    alpha = min(max(alpha, 1e-6), 0.999999)

    n_samples = int(options.get("n_samples", 400))
    metadata["n_samples"] = n_samples

    seed = None
    if rng is not None:
        seed = int(rng.integers(0, 2**31 - 1))

    with _temporary_numpy_seed(seed):
        return_values = eva.get_return_value(
            return_period=pyext_return_periods,
            return_period_size=return_period_size,
            alpha=alpha,
            n_samples=n_samples,
        )

    return_levels, ci_lower, ci_upper = return_values

    diagnostic_figure = None
    try:
        diagnostic_figure, _ = eva.plot_diagnostic(
            return_period=diagnostic_return_periods,
            return_period_size=return_period_size,
            alpha=alpha,
            plotting_position=plotting_position,
        )
        if diagnostic_figure is not None:
            for ax in diagnostic_figure.axes:
                ax.grid(True, linestyle="--", alpha=0.5)
                ax.set_axisbelow(True)

                title = ax.get_title().strip().lower()
                if title == "return values plot":
                    x_min, x_max = ax.get_xlim()
                    dense_ticks: tuple[int, ...] = ()

                    if (
                        np.isfinite(x_min)
                        and np.isfinite(x_max)
                        and x_max >= 1.0
                    ):
                        lower = max(1, int(np.ceil(x_min)))
                        upper = int(np.floor(x_max))

                        if upper >= lower:
                            tick_count = upper - lower + 1
                            if tick_count <= 15:
                                dense_ticks = tuple(range(lower, upper + 1))

                    if dense_ticks:

                        if ax.get_xscale() != "linear":
                            ax.set_xscale("linear")


                        def _format_return_period_tick(value: float, _pos: int) -> str:
                            if not np.isfinite(value) or value < 1.0:
                                return ""
                            return f"{int(round(value))}"

                        ax.xaxis.set_major_locator(
                            mticker.FixedLocator(list(dense_ticks))
                        )
                        ax.xaxis.set_major_formatter(
                            mticker.FuncFormatter(_format_return_period_tick)
                        )
                        ax.xaxis.set_minor_locator(mticker.NullLocator())
                    else:

                        if ax.get_xscale() != "log":
                            ax.set_xscale("log")

                        locator = mticker.LogLocator(
                            base=10.0, subs=tuple(range(1, 10))
                        )
                        ax.xaxis.set_major_locator(locator)
                        ax.xaxis.set_major_formatter(
                            mticker.LogFormatter(labelOnlyBase=False)
                        )
                        ax.xaxis.set_minor_locator(mticker.NullLocator())

    except Exception:  # pragma: no cover - plotting should not fail analysis
        diagnostic_figure = None

    return_levels = np.asarray(return_levels, dtype=float)
    if return_levels.ndim == 0:
        return_levels = return_levels[np.newaxis]

    def _as_array(value) -> np.ndarray:
        if value is None:
            return np.full(return_levels.shape, np.nan)
        arr = np.asarray(value, dtype=float)
        if arr.ndim == 0:
            arr = arr[np.newaxis]
        return arr

    lower_bounds = _as_array(ci_lower)
    upper_bounds = _as_array(ci_upper)

    return ExtremeValueResult(
        return_periods=return_periods,
        return_levels=return_levels,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        shape=shape,
        scale=scale,
        exceedances=np.asarray(exceedances, dtype=float),
        threshold=float(threshold),
        exceedance_rate=float(exceed_rate),
        engine="pyextremes",
        metadata=metadata,
        diagnostic_figure=diagnostic_figure,
    )


__all__ = [
    "ExtremeValueResult",
    "calculate_extreme_value_statistics",
    "cluster_exceedances",

    "declustering_boundaries",

]

