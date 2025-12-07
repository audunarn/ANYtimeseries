"""Helper utilities for SWAN pre-processing tasks.

This module provides shared helpers for generating overview plots of the
target domain and handling scatter-point bookkeeping used by
``era5_to_swan_mp.py``.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import pandas as pd


# Default temporal resolution for non-stationary SWAN simulations.
#
# Some callers import this module purely for shared constants used when building
# SWAN configuration UIs. In those scenarios we want a well-defined fallback
# instead of triggering a ``NameError`` when the constant is missing from the
# provided configuration dictionary. A one-hour step aligns with the ERA5
# forcing files handled by the rest of this repository.
NONSTAT_DT: int = 3600


@dataclass(frozen=True)
class Domain:
    """Geographic domain bounds.

    The bounds are stored as decimal degrees and mapped to a rectangular
    patch on the overview plot.
    """

    lon_min: float
    lon_max: float
    lat_min: float
    lat_max: float

    @property
    def corners(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        return (self.lon_min, self.lat_min), (self.lon_max, self.lat_max)


@dataclass(frozen=True)
class ScatterPoint:
    """Single scatter output location."""

    lon: float
    lat: float
    label: str | None = None


def plot_domain_overview(
    domain: Domain,
    output_file: Path,
    scatter_points: Sequence[ScatterPoint] | None = None,
) -> None:
    """Save a quick-look map of the requested domain.

    Parameters
    ----------
    domain:
        Geographic bounds used for the ERA5 extraction.
    output_file:
        File path where the plot should be stored.
    scatter_points:
        Optional list of scatter output locations to highlight on the
        map.
    """

    fig, ax = plt.subplots(figsize=(7, 5))
    rect_lon = domain.lon_max - domain.lon_min
    rect_lat = domain.lat_max - domain.lat_min

    # Draw domain outline
    ax.add_patch(
        plt.Rectangle(
            (domain.lon_min, domain.lat_min),
            rect_lon,
            rect_lat,
            fill=False,
            lw=2,
            color="tab:blue",
            label="Domain extent",
        )
    )

    # Scatter locations
    for idx, point in enumerate(scatter_points or []):
        label = point.label or f"Scatter {idx + 1}"
        ax.scatter(point.lon, point.lat, s=40, color="tab:orange", label=label)
        ax.text(point.lon, point.lat, f" {label}", va="center", ha="left")

    ax.set_xlabel("Longitude [deg]")
    ax.set_ylabel("Latitude [deg]")
    ax.set_title("Domain overview")
    ax.grid(True, linestyle=":", linewidth=0.6)

    # Avoid duplicate legend entries for multiple scatter points
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    if unique:
        ax.legend(unique.values(), unique.keys(), loc="best")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_file, dpi=200)
    plt.close(fig)


def _parse_scatter_points(values: Iterable[str]) -> List[ScatterPoint]:
    points: List[ScatterPoint] = []
    for value in values:
        try:
            lon_str, lat_str, *label_parts = value.split(",")
            lon = float(lon_str)
            lat = float(lat_str)
            label = ",".join(label_parts).strip() or None
        except ValueError as exc:  # pragma: no cover - defensive guard
            raise argparse.ArgumentTypeError(
                "Scatter points must be provided as 'lon,lat[,label]'"
            ) from exc

        points.append(ScatterPoint(lon=lon, lat=lat, label=label))
    return points


def _validate_required_columns(df: pd.DataFrame, required: Sequence[str]) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            "Scatter data is missing required columns: " + ", ".join(missing)
        )


def _aligned_limits(x: pd.Series, y: pd.Series) -> Tuple[float, float]:
    finite_x = x.dropna()
    finite_y = y.dropna()
    if finite_x.empty or finite_y.empty:
        return 0.0, 1.0

    minimum = float(min(finite_x.min(), finite_y.min()))
    maximum = float(max(finite_x.max(), finite_y.max()))
    if minimum == maximum:
        maximum = minimum + 1.0
    return minimum, maximum


def plot_offshore_nearshore_scatter(df: pd.DataFrame, output_file: Path) -> None:
    """Create offshore/nearshore Hs and Tp scatter plots with aligned axes."""

    _validate_required_columns(
        df,
        ["Hs_offshore", "Hs_nearshore", "Tp_offshore", "Tp_nearshore"],
    )

    fig, (ax_hs, ax_tp) = plt.subplots(1, 2, figsize=(10, 5))

    # Significant wave height scatter
    hs_limits = _aligned_limits(df["Hs_offshore"], df["Hs_nearshore"])
    ax_hs.scatter(df["Hs_offshore"], df["Hs_nearshore"], alpha=0.7)
    ax_hs.plot(hs_limits, hs_limits, "k--", linewidth=1, label="1:1 line")
    ax_hs.set_xlim(hs_limits)
    ax_hs.set_ylim(hs_limits)
    ax_hs.set_xlabel("Offshore Hs [m]")
    ax_hs.set_ylabel("Nearshore Hs [m]")
    ax_hs.set_title("Significant wave height")
    ax_hs.grid(True, linestyle=":", linewidth=0.6)
    ax_hs.legend()

    # Peak period scatter
    tp_limits = _aligned_limits(df["Tp_offshore"], df["Tp_nearshore"])
    ax_tp.scatter(df["Tp_offshore"], df["Tp_nearshore"], alpha=0.7)
    ax_tp.plot(tp_limits, tp_limits, "k--", linewidth=1, label="1:1 line")
    ax_tp.set_xlim(tp_limits)
    ax_tp.set_ylim(tp_limits)
    ax_tp.set_xlabel("Offshore Tp [s]")
    ax_tp.set_ylabel("Nearshore Tp [s]")
    ax_tp.set_title("Peak period")
    ax_tp.grid(True, linestyle=":", linewidth=0.6)
    ax_tp.legend()

    fig.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=200)
    plt.close(fig)


def run_scatter_mode(
    domain: Domain,
    scatter_points: Sequence[ScatterPoint],
    out_dir: Path,
    *,
    scatter_data: Path | None = None,
) -> None:
    """Execute scatter extraction and export a domain map.

    The actual data extraction is left to downstream tooling; this
    function focuses on the bookkeeping needed to keep scatter output
    and the domain map together.
    """

    out_dir.mkdir(parents=True, exist_ok=True)

    # Placeholder for potential scatter data export could go here.
    overview_path = out_dir / "domain_overview.png"
    plot_domain_overview(domain, overview_path, scatter_points)

    if scatter_data:
        df = pd.read_csv(scatter_data)
        scatter_plot = out_dir / "scatter_offshore_nearshore.png"
        plot_offshore_nearshore_scatter(df, scatter_plot)


def parse_scatter_args(values: Iterable[str]) -> List[ScatterPoint]:
    """CLI-friendly wrapper that validates scatter coordinates."""

    return _parse_scatter_points(values)

