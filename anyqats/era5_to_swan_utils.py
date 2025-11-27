"""Utilities for validating ERA5-derived boundary inputs for SWAN.

The functions here normalise meteorological/wave directions and verify that
applied wind directions stay reasonably aligned with the propagated wave
fields when they reach a SWAN boundary.
"""
from __future__ import annotations


from pathlib import Path

from typing import Mapping, Sequence

import numpy as np


class MisalignedBoundaryError(ValueError):
    """Raised when boundary wind and wave directions differ too much."""


def _normalise_angles(angles: Sequence[float]) -> np.ndarray:
    """Normalise angles to the range [0, 360).

    Parameters
    ----------
    angles : Sequence[float]
        A sequence of angles in degrees.

    Returns
    -------
    numpy.ndarray
        Angles wrapped to the 0–360 degree interval.
    """

    return np.mod(np.asarray(angles, dtype=float), 360.0)


def calculate_misalignment(wind_dirs: Sequence[float], wave_dirs: Sequence[float]) -> np.ndarray:
    """Return the absolute angular difference between wind and wave directions.

    Parameters
    ----------
    wind_dirs : Sequence[float]
        Wind directions in degrees.
    wave_dirs : Sequence[float]
        Wave directions in degrees.

    Returns
    -------
    numpy.ndarray
        Absolute misalignment between wind and wave directions in degrees.
    """

    wind = _normalise_angles(wind_dirs)
    wave = _normalise_angles(wave_dirs)
    if wind.shape != wave.shape:
        raise ValueError("Wind and wave direction arrays must share the same shape.")

    # Wrap the difference to [-180, 180] before taking the magnitude. This gives the
    # smallest angular distance regardless of the direction wrap-around.
    return np.abs((wind - wave + 180.0) % 360.0 - 180.0)



def _write_report(report_path: Path, lines: Sequence[str]) -> None:
    """Persist a human-readable report, creating parent folders if needed."""

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines))


def validate_boundary_alignment(
    boundary_wind_dirs: Mapping[str, Sequence[float]],
    boundary_wave_dirs: Mapping[str, Sequence[float]],
    *,
    tolerance: float = 45.0,

    report_path: str | Path | None = None,

) -> None:
    """Ensure wind directions applied at each boundary are aligned with wave directions.

    The check is agnostic to how the directions were produced; it simply enforces that
    the two inputs share boundary keys and that their differences stay within the
    supplied tolerance. A :class:`MisalignedBoundaryError` is raised when the tolerance
    is exceeded so callers can decide how to handle the inconsistency (log a warning,
    adjust the forcing or abort the run).

    Parameters
    ----------
    boundary_wind_dirs : Mapping[str, Sequence[float]]
        Wind directions keyed by boundary name.
    boundary_wave_dirs : Mapping[str, Sequence[float]]
        Wave directions keyed by boundary name.
    tolerance : float, optional
        Maximum allowed angular difference in degrees (default 45).

    report_path : str or pathlib.Path, optional
        File path where a human-readable report is written. Parent directories are
        created automatically. When provided, a short summary is written for both
        passing and failing validations. If not provided, no report is saved.

    Raises
    ------
    MisalignedBoundaryError
        If wind and wave directions differ by more than ``tolerance`` for any entry.
    ValueError
        If the provided boundary keys differ between wind and wave dictionaries or the
        direction arrays cannot be compared.
    """

    missing = set(boundary_wind_dirs).symmetric_difference(boundary_wave_dirs)
    if missing:
        raise ValueError(
            "Wind and wave boundary keys must match before validation; mismatches: "
            f"{sorted(missing)}"
        )

    offending = []

    boundaries = sorted(boundary_wind_dirs)

    for boundary in boundary_wind_dirs:
        misalignment = calculate_misalignment(
            boundary_wind_dirs[boundary], boundary_wave_dirs[boundary]
        )
        exceed = np.argwhere(misalignment > tolerance).ravel()
        for idx in exceed:
            offending.append(
                (
                    boundary,
                    int(idx),
                    float(boundary_wind_dirs[boundary][idx]),
                    float(boundary_wave_dirs[boundary][idx]),
                    float(misalignment[idx]),
                )
            )

    if offending:
        summary = "; ".join(
            f"{b}[{i}]: wind={w:.1f}°, wave={v:.1f}° (Δ={d:.1f}°)"
            for b, i, w, v, d in offending
        )


        if report_path is not None:
            lines = [
                "Boundary alignment check failed.",
                f"Tolerance: {tolerance:.1f} degrees.",
                "Offending entries:",
                *(f"- {entry}" for entry in summary.split("; ")),
            ]
            _write_report(Path(report_path), lines)


        raise MisalignedBoundaryError(
            "Boundary wind/wave directions differ more than allowed tolerance. "
            f"Tolerance={tolerance:.1f}°. Offending entries: {summary}"
        )


    if report_path is not None:
        passed_lines = [
            "Boundary alignment check passed.",
            f"Tolerance: {tolerance:.1f} degrees.",
            f"Checked boundaries: {', '.join(boundaries) if boundaries else 'none'}.",
        ]
        _write_report(Path(report_path), passed_lines)

