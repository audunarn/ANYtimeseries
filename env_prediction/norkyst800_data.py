"""Utility plotting helpers for Norkyst800 fields.

This module focuses on making fjord-scale plots visually clear by combining:
- a perceptually smooth colour map,
- terrain-like bathymetry shading when available,
- and high-detail coastline overlays (10m Natural Earth).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


def _first_existing_name(candidates: Iterable[str], available: Iterable[str]) -> str | None:
    available_set = set(available)
    for name in candidates:
        if name in available_set:
            return name
    return None


def _pick_coords(ds: xr.Dataset, var_name: str) -> tuple[np.ndarray, np.ndarray]:
    da = ds[var_name]
    coord_candidates_lon = ("lon", "longitude", "xlon", "nav_lon")
    coord_candidates_lat = ("lat", "latitude", "xlat", "nav_lat")

    lon_name = _first_existing_name(coord_candidates_lon, da.coords)
    lat_name = _first_existing_name(coord_candidates_lat, da.coords)

    if lon_name is None or lat_name is None:
        lon_name = lon_name or _first_existing_name(coord_candidates_lon, ds.variables)
        lat_name = lat_name or _first_existing_name(coord_candidates_lat, ds.variables)

    if lon_name is None or lat_name is None:
        raise ValueError(
            "Could not infer longitude/latitude variable names. "
            "Expected one of: lon/longitude/nav_lon and lat/latitude/nav_lat."
        )

    lon = ds[lon_name].values
    lat = ds[lat_name].values
    return lon, lat


def _select_time_slice(da: xr.DataArray, time_index: int) -> xr.DataArray:
    for dim in ("time", "ocean_time", "t"):
        if dim in da.dims:
            return da.isel({dim: time_index})
    return da


def plot_fjord_map(
    netcdf_path: str | Path,
    variable: str,
    time_index: int = 0,
    extent: tuple[float, float, float, float] | None = None,
    output_path: str | Path = "fjord_plot.png",
) -> Path:
    """Create a fjord-focused map plot with high coastline detail.

    Parameters
    ----------
    netcdf_path:
        NetCDF file containing Norkyst800-like data.
    variable:
        Name of field to color-plot.
    time_index:
        Time index used when `variable` has a time-like dimension.
    extent:
        Optional map extent as `(min_lon, max_lon, min_lat, max_lat)`.
    output_path:
        Output image path.
    """
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
    except ImportError as exc:  # pragma: no cover - runtime guidance
        raise ImportError(
            "cartopy is required for detailed map plotting. "
            "Install with: pip install cartopy"
        ) from exc

    ds = xr.open_dataset(netcdf_path)
    if variable not in ds:
        raise KeyError(f"Variable '{variable}' not found in dataset. Available: {list(ds.data_vars)}")

    field = _select_time_slice(ds[variable], time_index)
    lon, lat = _pick_coords(ds, variable)

    fig = plt.figure(figsize=(11, 8), dpi=180)
    ax = plt.axes(projection=ccrs.PlateCarree())

    if extent is None:
        ax.set_extent([float(np.nanmin(lon)), float(np.nanmax(lon)), float(np.nanmin(lat)), float(np.nanmax(lat))])
    else:
        ax.set_extent(extent)

    # Optional soft terrain shading where bathymetry/topography exists.
    bathy_name = _first_existing_name(("h", "depth", "bathymetry", "topo"), ds.variables)
    if bathy_name is not None:
        bathy = ds[bathy_name].values
        ax.pcolormesh(
            lon,
            lat,
            bathy,
            transform=ccrs.PlateCarree(),
            cmap="Greys",
            alpha=0.22,
            shading="auto",
            zorder=1,
        )

    # Better-looking colour plot for the requested variable.
    mesh = ax.pcolormesh(
        lon,
        lat,
        field.values,
        transform=ccrs.PlateCarree(),
        cmap="turbo",
        shading="auto",
        zorder=2,
    )

    # High-detail coastline and land layers suitable for fjord-scale views.
    land_10m = cfeature.NaturalEarthFeature("physical", "land", "10m", facecolor="#efe9df")
    coast_10m = cfeature.NaturalEarthFeature("physical", "coastline", "10m", facecolor="none")
    lakes_10m = cfeature.NaturalEarthFeature("physical", "lakes", "10m", facecolor="#dfeff9")
    rivers_10m = cfeature.NaturalEarthFeature("physical", "rivers_lake_centerlines", "10m", facecolor="none")

    ax.add_feature(land_10m, edgecolor="none", zorder=3)
    ax.add_feature(lakes_10m, edgecolor="none", zorder=4)
    ax.add_feature(coast_10m, edgecolor="#1a1a1a", linewidth=0.65, zorder=5)
    ax.add_feature(rivers_10m, edgecolor="#6f9fcf", linewidth=0.35, zorder=5)

    gl = ax.gridlines(draw_labels=True, linewidth=0.35, color="#7c8a99", alpha=0.5, linestyle="--")
    gl.top_labels = False
    gl.right_labels = False

    cbar = fig.colorbar(mesh, ax=ax, shrink=0.86, pad=0.03)
    cbar.set_label(f"{variable} ({field.attrs.get('units', 'no units')})")

    ax.set_title(f"Norkyst800: {variable} (time index={time_index})", fontsize=12)
    fig.tight_layout()

    output_path = Path(output_path)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    ds.close()
    return output_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create high-detail fjord maps from Norkyst800 data.")
    parser.add_argument("netcdf", type=Path, help="Path to NetCDF file")
    parser.add_argument("variable", help="Variable to plot")
    parser.add_argument("--time-index", type=int, default=0, help="Time index for 3D/4D variables")
    parser.add_argument(
        "--extent",
        type=float,
        nargs=4,
        metavar=("MIN_LON", "MAX_LON", "MIN_LAT", "MAX_LAT"),
        help="Optional map extent for fjord zoom",
    )
    parser.add_argument("--output", type=Path, default=Path("fjord_plot.png"), help="Output image path")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    out = plot_fjord_map(
        netcdf_path=args.netcdf,
        variable=args.variable,
        time_index=args.time_index,
        extent=tuple(args.extent) if args.extent else None,
        output_path=args.output,
    )
    print(f"Saved detailed map plot to {out}")


if __name__ == "__main__":
    main()
