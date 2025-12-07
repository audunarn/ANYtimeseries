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


def run_scatter_mode(domain: Domain, scatter_points: Sequence[ScatterPoint], out_dir: Path) -> None:
    """Execute scatter extraction and export a domain map.

    The actual data extraction is left to downstream tooling; this
    function focuses on the bookkeeping needed to keep scatter output
    and the domain map together.
    """

    out_dir.mkdir(parents=True, exist_ok=True)

    # Placeholder for potential scatter data export could go here.
    overview_path = out_dir / "domain_overview.png"
    plot_domain_overview(domain, overview_path, scatter_points)


def parse_scatter_args(values: Iterable[str]) -> List[ScatterPoint]:
    """CLI-friendly wrapper that validates scatter coordinates."""

    return _parse_scatter_points(values)

