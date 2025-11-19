"""Utilities for fatigue damage calculations independent of the GUI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from anyqats.fatigue.rainflow import count_cycles
from anyqats.fatigue.sn import SNCurve, minersum


@dataclass(slots=True)
class FatigueSeries:
    """Container for prepared series data used in the fatigue tool."""

    label: str
    values: np.ndarray
    duration: float
    source_file: str = ""
    variable_name: str = ""


@dataclass(slots=True)
class FatigueResult:
    """Summary of the fatigue damage computed for a single series."""

    label: str
    total_cycles: float
    damage: float
    max_range: float
    exposure_hours: float


@dataclass(slots=True)
class FatigueSummary:
    """Aggregated fatigue metrics across multiple series."""

    total_damage: float
    total_exposure_hours: float | None
    estimated_life_years: float | None


class FatigueComputationError(ValueError):
    """Raised when the fatigue calculator cannot process the input data."""


_DEF_LOG_MSG = "{label}: {message}"


def compute_fatigue_damage(
    series_list: Sequence[FatigueSeries],
    exposure_hours: Sequence[float],
    curve: SNCurve,
    *,
    scf: float = 1.0,
    thickness: float | None = None,
    load_basis: str = "stress",
    curve_type: str = "sn",
    unit_factor: float = 1.0,
    reference_strength: float | None = None,
) -> tuple[list[FatigueResult], list[str]]:
    """Calculate fatigue damage for each prepared series.

    Parameters
    ----------
    series_list:
        Iterable with pre-filtered series ready for fatigue calculations.
    exposure_hours:
        Exposure duration in hours for each series (same order as ``series_list``).
    curve:
        The fatigue curve definition used for the calculation.
    scf:
        Stress concentration factor applied to all ranges.
    thickness:
        Optional thickness used for thickness correction in ``minersum``.
    load_basis:
        Either ``"stress"`` for S-N curves or ``"tension"`` for T-N curves.
    curve_type:
        Type of fatigue curve (``"sn"`` or ``"tn"``). Used to validate ``load_basis``.
    unit_factor:
        Conversion applied to the time series samples (e.g. kN -> MN).
    reference_strength:
        Optional breaking strength used to normalise T-N curves when values are
        provided in absolute units. The value should use the same base units as
        the converted time series (after ``unit_factor`` is applied).
    """

    if len(series_list) != len(exposure_hours):
        raise FatigueComputationError("Exposure list must match the number of series.")

    if curve_type == "sn" and load_basis != "stress":
        raise FatigueComputationError("S-N curves require stress-based interpretation.")
    if curve_type == "tn" and load_basis != "tension":
        raise FatigueComputationError("T-N curves require tension-based interpretation.")

    if reference_strength is not None and reference_strength <= 0:
        raise FatigueComputationError("Reference breaking strength must be positive.")

    results: list[FatigueResult] = []
    logs: list[str] = []

    for series, hours in zip(series_list, exposure_hours):
        label = series.label
        valid = np.isfinite(series.values)
        if not np.any(valid):
            logs.append(_DEF_LOG_MSG.format(label=label, message="skipped (no valid samples)"))
            continue

        filtered = series.values[valid]
        if filtered.size < 2:
            logs.append(_DEF_LOG_MSG.format(label=label, message="skipped (not enough samples)"))
            continue

        exposure = float(hours)
        if exposure <= 0:
            logs.append(
                _DEF_LOG_MSG.format(label=label, message="skipped (non-positive exposure time)")
            )
            continue

        signal_duration = float(series.duration)
        if signal_duration <= 0:
            logs.append(
                _DEF_LOG_MSG.format(label=label, message="skipped (invalid series duration)")
            )
            continue

        converted = filtered * float(unit_factor)
        if reference_strength:
            converted = converted / float(reference_strength)

        cycles = count_cycles(converted)
        if cycles.size == 0:
            logs.append(_DEF_LOG_MSG.format(label=label, message="skipped (no cycles detected)"))
            continue

        cycle_rate = cycles[:, 2] / signal_duration
        duration_seconds = exposure * 3600.0

        damage = minersum(cycles[:, 0], cycle_rate, curve, td=duration_seconds, scf=scf, th=thickness)
        total_cycles = float(np.sum(cycles[:, 2]))
        max_range = float(np.max(cycles[:, 0]))

        results.append(
            FatigueResult(
                label=label,
                total_cycles=total_cycles,
                damage=float(damage),
                max_range=max_range,
                exposure_hours=exposure,
            )
        )
        logs.append(
            _DEF_LOG_MSG.format(
                label=label,
                message=(
                    f"damage={damage:.6g} for {exposure:.3f} h exposure over "
                    f"{total_cycles:.2f} rainflow cycles (max range {max_range:.3f})."
                ),
            )
        )

    return results, logs


def summarize_damage(
    results: Sequence[FatigueResult], exposure_hours: Sequence[float] | None = None
) -> FatigueSummary:
    """Summarize fatigue results over all series.

    Parameters
    ----------
    results:
        Collection of :class:`FatigueResult` entries to aggregate.
    exposure_hours:
        Optional exposure hours used in the calculation. When provided, the
        total exposure and an estimated fatigue life (based on Miner’s rule)
        are returned.
    """

    total_damage = float(np.sum([res.damage for res in results])) if results else 0.0

    total_exposure_hours: float | None = None
    estimated_life_years: float | None = None

    if exposure_hours is not None:
        total_exposure_hours = float(np.sum(exposure_hours))
        if total_damage > 0 and total_exposure_hours > 0:
            total_exposure_years = total_exposure_hours / 8760.0
            estimated_life_years = total_exposure_years / total_damage

    return FatigueSummary(
        total_damage=total_damage,
        total_exposure_hours=total_exposure_hours,
        estimated_life_years=estimated_life_years,
    )


__all__ = [
    "FatigueSeries",
    "FatigueResult",
    "FatigueSummary",
    "FatigueComputationError",
    "compute_fatigue_damage",
    "summarize_damage",
]
