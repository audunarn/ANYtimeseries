"""Multiprocessing helpers for ERA5 → SWAN conversions.

Before dispatching tasks, use :func:`validate_boundary_alignment` to ensure wind
forcings remain compatible with incoming wave directions at each boundary.
"""
from __future__ import annotations

from typing import Mapping, Sequence

from .era5_to_swan_utils import MisalignedBoundaryError, validate_boundary_alignment


def prepare_mp_inputs(
    *,
    boundary_wave_dirs: Mapping[str, Sequence[float]],
    boundary_wind_dirs: Mapping[str, Sequence[float]],
    tolerance: float = 45.0,
    report_path: str | None = None,
) -> None:
    """Validate boundary wind/wave alignment before parallel processing.

    Parameters
    ----------
    boundary_wave_dirs : Mapping[str, Sequence[float]]
        Wave directions keyed by boundary.
    boundary_wind_dirs : Mapping[str, Sequence[float]]
        Wind directions keyed by boundary.
    tolerance : float, optional
        Maximum allowed angular difference between wind and wave directions.
    report_path : str, optional
        When provided, a text report is written to this path describing the
        alignment results.

    Raises
    ------
    MisalignedBoundaryError
        If the angular difference exceeds ``tolerance`` for any boundary entry.
    ValueError
        If boundary keys differ or direction arrays cannot be compared.
    """

    validate_boundary_alignment(
        boundary_wind_dirs,
        boundary_wave_dirs,
        tolerance=tolerance,
        report_path=report_path,
    )
