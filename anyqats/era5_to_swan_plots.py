"""Plot-specific helpers for ERA5 → SWAN conversions.

The plotting pipeline can leverage :func:`validate_boundary_alignment` to make sure
wind forcing stays consistent with the wave directions at each boundary. Violations
are surfaced early so the user can correct the input data before producing plots or
running SWAN.
"""
from __future__ import annotations

from typing import Mapping, Sequence

from .era5_to_swan_utils import MisalignedBoundaryError, validate_boundary_alignment


def prepare_plot_inputs(
    *,
    boundary_wave_dirs: Mapping[str, Sequence[float]],
    boundary_wind_dirs: Mapping[str, Sequence[float]],
    tolerance: float = 45.0,
) -> None:
    """Validate that wind directions align with wave directions before plotting.

    Parameters
    ----------
    boundary_wave_dirs : Mapping[str, Sequence[float]]
        Wave directions keyed by boundary.
    boundary_wind_dirs : Mapping[str, Sequence[float]]
        Wind directions keyed by boundary.
    tolerance : float, optional
        Maximum allowed angular difference between wind and wave directions.

    Raises
    ------
    MisalignedBoundaryError
        If the angular difference exceeds ``tolerance`` for any boundary entry.
    ValueError
        If boundary keys differ or direction arrays cannot be compared.
    """

    validate_boundary_alignment(boundary_wind_dirs, boundary_wave_dirs, tolerance=tolerance)
