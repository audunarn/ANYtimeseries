from __future__ import annotations
import argparse
import json
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable
from xml.sax.saxutils import escape

import numpy as np
import plotly.graph_objects as go
import xarray as xr
from plotly.colors import sample_colorscale
from plotly.subplots import make_subplots

SPLIT_REPORT_FILES = True
AUTO_OPEN_SPLIT_FILES = True
SHORELINE_DYNAMIC_BUFFER_CELLS = 2

POINT_AARSET = (65.010180, 11.746670)
POINT_AARSET2 = (65.008869, 11.745154)
POINT_EITER = (64.971570, 11.720669)
POINT_EITER2 = (64.970490, 11.727023)


POINT_COORD = (POINT_AARSET,POINT_AARSET2, POINT_EITER,POINT_EITER2)
DEFAULT_ARROW_RESOLUTION = 100

# PyCharm-friendly configuration block:
# Set USE_CODE_CONFIG=True and edit these values when running directly from an IDE.
USE_CODE_CONFIG = True
INPUT_DIRECTORY: str | tuple[str, ...] = \
        (r"\\wsl.localhost\Ubuntu-24.04\home\audunarn\run_wave_env\r50L75Z_SWAN",
         r"\\wsl.localhost\Ubuntu-24.04\home\audunarn\run_wave_env\r125L75Z_SWAN",
        r"\\wsl.localhost\Ubuntu-24.04\home\audunarn\run_wave_env\r250L75Z_SWAN",
        r"\\wsl.localhost\Ubuntu-24.04\home\audunarn\run_wave_env\r500L75Z_SWAN",
     )
NC_FILE_NAME: str | None = None  # Example: "Ex1_Sula500_20200120.nc"
BOT_FILE_NAME: str | None = None  # Example: "Ex1_Sula500_SWAN.bot"
WIND_ARROW_RESOLUTION = DEFAULT_ARROW_RESOLUTION
EXPORT_HS_FORMAT: str | None = "kmz"  # Set to "geojson", "kml", or "kmz" for PyCharm export mode.
EXPORT_TIME_INDEX = "MAX"
SPEC_FILE_NAME: str | None = None  # Example: "Ex1_Sula500_20200120_spec.nc"
PLOT_SPEC_DIRECTIONAL = False
SPEC_DIR_THETA_STEP_DEG = 5.0
SPEC_DIR_SPREADING_S = 5.0
PLOT_ENGINE = "plotly"


@dataclass(frozen=True)
class Inputs:
    directories: tuple[Path, ...]
    nc_files: tuple[Path | None, ...]
    bot_files: tuple[Path | None, ...]
    point_coord: tuple[float, float] | tuple[tuple[float, float], ...]
    wind_arrow_resolution: int | None
    export_hs_format: str | None
    export_time_index: int | str
    spec_file: Path | None
    plot_spec_directional: bool
    spec_dir_theta_step_deg: float
    spec_dir_spreading_s: float
    plot_engine: str


@dataclass(frozen=True)
class RunData:
    label: str
    grid: GridInfo
    hs: np.ndarray
    tp: np.ndarray
    depth: np.ndarray
    wind_speed: np.ndarray | None
    wind_from: np.ndarray | None
    wave_from: np.ndarray | None
    ssh: np.ndarray | None
    current_speed: np.ndarray | None
    current_from: np.ndarray | None
    water_mask: np.ndarray
    mask_keep_static: np.ndarray
    mask_keep_time: np.ndarray


@dataclass(frozen=True)
class GridInfo:
    lon: np.ndarray
    lat: np.ndarray
    lon2d: np.ndarray
    lat2d: np.ndarray
    times: np.ndarray


@dataclass(frozen=True)
class PoiIndex:
    iy: int
    ix: int
    lon: float
    lat: float


@dataclass(frozen=True)
class CompositeFields:
    grid: GridInfo
    mask: np.ndarray
    depth: np.ndarray
    hs: np.ndarray | None
    tp: np.ndarray | None
    tps: np.ndarray | None
    ssh: np.ndarray | None
    current_speed: np.ndarray | None
    current_from: np.ndarray | None
    wind_speed: np.ndarray | None
    wind_from: np.ndarray | None


@dataclass(frozen=True)
class OutputsPayload:
    runs: list[RunData]
    times: np.ndarray
    pois: tuple[PoiIndex, ...]
    run_tps_list: list[np.ndarray]
    finest_tps: np.ndarray
    hs_min: float
    hs_max: float
    tp_min: float
    tp_max: float
    tps_min: float
    tps_max: float
    ts_y_min: float
    ts_y_max: float


@dataclass(frozen=True)
class InputsPayload:
    runs: list[RunData]
    times: np.ndarray
    pois: tuple[PoiIndex, ...]
    composite: CompositeFields
    has_wind: bool
    ws_min: float
    ws_max: float
    cur_min: float
    cur_max: float
    ssh_min: float
    ssh_max: float

def _build_merged_depth_point_cloud(
    runs: list[RunData],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build one merged bathymetry point cloud from all nested runs, using the
    already-computed finest-precedence masks.

    Returns
    -------
    x : (n_points,)
        Longitude of merged kept points.
    y : (n_points,)
        Latitude of merged kept points.
    depth_cube : (nt, n_points)
        Positive depth values [m] for each merged point and time.
    times : (nt,)
        Time axis from the finest run.
    """
    if not runs:
        raise ValueError("No runs supplied.")

    nt = runs[-1].depth.shape[0]
    times = runs[-1].grid.times

    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    depth_parts: list[np.ndarray] = []

    for run in runs:
        depth = _align_time_count(run.depth, nt)

        # Keep only points that survive finest-precedence masking
        keep_static = np.any(run.mask_keep_time, axis=0)
        if not np.any(keep_static):
            continue

        x_parts.append(np.asarray(run.grid.lon2d[keep_static], dtype=float).ravel())
        y_parts.append(np.asarray(run.grid.lat2d[keep_static], dtype=float).ravel())

        # Shape: (nt, n_kept_points)
        depth_kept = np.asarray(depth[:, keep_static], dtype=float)
        wet_kept = np.asarray(run.mask_keep_time[:, keep_static], dtype=bool)

        # Mask dry values
        depth_kept = np.where(wet_kept, depth_kept, np.nan)

        depth_parts.append(depth_kept)

    if not x_parts or not depth_parts:
        raise ValueError("No merged depth points available for 3D plotting.")

    x = np.concatenate(x_parts)
    y = np.concatenate(y_parts)
    depth_cube = np.concatenate(depth_parts, axis=1)

    # Drop points that are NaN for all time steps
    valid_cols = np.any(np.isfinite(depth_cube), axis=0)
    x = x[valid_cols]
    y = y[valid_cols]
    depth_cube = depth_cube[:, valid_cols]

    if depth_cube.shape[1] == 0:
        raise ValueError("Merged depth point cloud is empty after filtering.")

    return x, y, depth_cube, times

def _prepare_tps(
    ds: xr.Dataset,
    time_dim: str = "time",
    y_dim: str = "latitude",
    x_dim: str = "longitude",
) -> np.ndarray:
    """
    Return smoothed peak period on shape (time, latitude, longitude).

    Tries common variable names first, and falls back to 'tp' if no dedicated
    TPS variable exists.
    """
    tps_name = _first_present(ds, ["tps", "tpsmoo", "tp_smooth", "smoothed_peak_period"])
    if tps_name is None:
        tps_name = "tp"

    if tps_name not in ds:
        raise KeyError(
            f"Could not find TPS variable. Tried tps/tpsmoo/tp_smooth/smoothed_peak_period/tp in dataset variables: {list(ds.variables)}"
        )

    da = ds[tps_name]
    if not all(d in da.dims for d in [time_dim, y_dim, x_dim]):
        raise ValueError(
            f"Variable '{tps_name}' must include dimensions ({time_dim}, {y_dim}, {x_dim}). "
            f"Found dims={da.dims}"
        )

    return np.asarray(da.transpose(time_dim, y_dim, x_dim).values)

def _build_mask_boundary_lines(
    grid: GridInfo,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build line segments along the boundary of a boolean cell mask.

    The returned x/y arrays are suitable for a Plotly line trace with NaN
    separators between segments.

    Parameters
    ----------
    grid : GridInfo
        Grid with 1D lon/lat coordinates.
    mask : (ny, nx) bool
        True for cells to keep / outline.

    Returns
    -------
    x_line, y_line : 1D arrays
        Line coordinates with NaN separators.
    """
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError("mask must be a 2D boolean array.")

    lon_e = _cell_edges(grid.lon)
    lat_e = _cell_edges(grid.lat)

    ny, nx = mask.shape
    xs: list[float] = []
    ys: list[float] = []

    def _add_seg(x0: float, y0: float, x1: float, y1: float) -> None:
        xs.extend([x0, x1, np.nan])
        ys.extend([y0, y1, np.nan])

    for iy in range(ny):
        for ix in range(nx):
            if not mask[iy, ix]:
                continue

            x0 = float(lon_e[ix])
            x1 = float(lon_e[ix + 1])
            y0 = float(lat_e[iy])
            y1 = float(lat_e[iy + 1])

            # Left edge
            if ix == 0 or not mask[iy, ix - 1]:
                _add_seg(x0, y0, x0, y1)

            # Right edge
            if ix == nx - 1 or not mask[iy, ix + 1]:
                _add_seg(x1, y0, x1, y1)

            # Bottom edge
            if iy == 0 or not mask[iy - 1, ix]:
                _add_seg(x0, y0, x1, y0)

            # Top edge
            if iy == ny - 1 or not mask[iy + 1, ix]:
                _add_seg(x0, y1, x1, y1)

    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)

def _build_merged_boundary_lines(
    runs: list[RunData],
    dynamic_buffer_cells: int = 2,
) -> list[tuple[np.ndarray, np.ndarray, float | None]]:
    """
    Build shoreline lines directly from each run's own grid and already-computed
    finest-precedence masks, while avoiding artificial boundaries between nested
    regions.

    Physical shoreline comes from run.water_mask.
    Visible ownership of the composite comes from run.mask_keep_time.
    """
    all_lines: list[tuple[np.ndarray, np.ndarray, float | None]] = []

    for run in runs:
        nt = run.mask_keep_time.shape[0]
        if nt == 0:
            continue

        ref_visible = np.asarray(run.mask_keep_time[0], dtype=bool)
        changing_cells = np.any(run.mask_keep_time != ref_visible[None, :, :], axis=0)

        dynamic_region = _dilate_boolean_mask(changing_cells, iterations=dynamic_buffer_cells)
        stable_region = ~dynamic_region

        dynamic_segments_raw: list[tuple[str, float, float, float, float | None]] = []
        for t in range(nt):
            ssh_t = run.ssh[t] if run.ssh is not None else None
            dynamic_segments_raw.extend(
                _extract_boundary_axis_segments(
                    water_mask=run.water_mask[t],
                    lon=run.grid.lon,
                    lat=run.grid.lat,
                    region_mask=dynamic_region,
                    ssh_field=ssh_t,
                    visible_mask=run.mask_keep_time[t],
                    treat_grid_edge_as_boundary=False,
                )
            )
        dynamic_lines = _deduplicate_axis_segments(dynamic_segments_raw)

        if run.ssh is not None:
            ssh_stable = np.nanmean(np.where(run.mask_keep_time, run.ssh, np.nan), axis=0)
        else:
            ssh_stable = None

        stable_segments_raw = _extract_boundary_axis_segments(
            water_mask=run.water_mask[0],
            lon=run.grid.lon,
            lat=run.grid.lat,
            region_mask=stable_region,
            ssh_field=ssh_stable,
            visible_mask=ref_visible,
            treat_grid_edge_as_boundary=False,
        )
        stable_lines = _merge_axis_segments(stable_segments_raw)

        all_lines.extend(stable_lines)
        all_lines.extend(dynamic_lines)

    return all_lines


def _nearest_index_map(src_vals: np.ndarray, dst_vals: np.ndarray) -> np.ndarray:
    src_vals = np.asarray(src_vals, dtype=float)
    dst_vals = np.asarray(dst_vals, dtype=float)
    out = np.empty(dst_vals.size, dtype=int)
    for i, v in enumerate(dst_vals):
        out[i] = int(np.argmin(np.abs(src_vals - v)))
    return out


def _merge_field_to_outer_grid(
    runs: list[RunData],
    field_getter: Callable[[RunData], np.ndarray | None],
) -> tuple[GridInfo, np.ndarray | None, np.ndarray]:
    base = runs[0]
    grid = base.grid
    nt = base.depth.shape[0]
    ny = grid.lat.size
    nx = grid.lon.size

    merged = np.full((nt, ny, nx), np.nan, dtype=float)
    merged_mask = np.zeros((nt, ny, nx), dtype=bool)

    has_any = False
    for run in runs:
        src_vals_full = field_getter(run)
        if src_vals_full is None:
            continue
        has_any = True

        src_lon = np.asarray(run.grid.lon, dtype=float)
        src_lat = np.asarray(run.grid.lat, dtype=float)

        lon_in = (grid.lon >= float(np.nanmin(src_lon))) & (grid.lon <= float(np.nanmax(src_lon)))
        lat_in = (grid.lat >= float(np.nanmin(src_lat))) & (grid.lat <= float(np.nanmax(src_lat)))

        ix_dst = np.flatnonzero(lon_in)
        iy_dst = np.flatnonzero(lat_in)
        if ix_dst.size == 0 or iy_dst.size == 0:
            continue

        ix_src = _nearest_index_map(src_lon, grid.lon[ix_dst])
        iy_src = _nearest_index_map(src_lat, grid.lat[iy_dst])

        src_vals_full = _align_time_count(np.asarray(src_vals_full, dtype=float), nt)

        for t in range(nt):
            src_keep = np.asarray(run.mask_keep_time[t], dtype=bool)

            sampled_keep = np.zeros((ny, nx), dtype=bool)
            sampled_keep[np.ix_(iy_dst, ix_dst)] = src_keep[np.ix_(iy_src, ix_src)]
            if not np.any(sampled_keep):
                continue

            sampled_field = np.full((ny, nx), np.nan, dtype=float)
            sampled_field[np.ix_(iy_dst, ix_dst)] = src_vals_full[t][np.ix_(iy_src, ix_src)]
            merged[t][sampled_keep] = sampled_field[sampled_keep]
            merged_mask[t][sampled_keep] = True

    if not has_any:
        return grid, None, merged_mask
    return grid, merged, merged_mask


def build_composite_fields(
    runs: list[RunData],
    datasets: list[tuple[str, xr.Dataset]],
) -> CompositeFields:
    dataset_by_label = {label: ds for label, ds in datasets}

    grid, depth, mask = _merge_field_to_outer_grid(runs, lambda run: run.depth)
    _, hs, _ = _merge_field_to_outer_grid(runs, lambda run: run.hs)
    _, tp, _ = _merge_field_to_outer_grid(runs, lambda run: run.tp)
    _, ssh, _ = _merge_field_to_outer_grid(runs, lambda run: run.ssh)
    _, current_speed, _ = _merge_field_to_outer_grid(runs, lambda run: run.current_speed)
    _, current_from, _ = _merge_field_to_outer_grid(runs, lambda run: run.current_from)
    _, wind_speed, _ = _merge_field_to_outer_grid(runs, lambda run: run.wind_speed)
    _, wind_from, _ = _merge_field_to_outer_grid(runs, lambda run: run.wind_from)

    run_tps_by_label: dict[str, np.ndarray] = {}
    for run in runs:
        ds = dataset_by_label[run.label]
        run_tps_by_label[run.label] = _align_time_count(_prepare_tps(ds), run.depth.shape[0])
    _, tps, _ = _merge_field_to_outer_grid(runs, lambda run: run_tps_by_label[run.label])

    if depth is None:
        raise ValueError("Composite depth merge failed unexpectedly.")

    return CompositeFields(
        grid=grid,
        mask=mask,
        depth=depth,
        hs=hs,
        tp=tp,
        tps=tps,
        ssh=ssh,
        current_speed=current_speed,
        current_from=current_from,
        wind_speed=wind_speed,
        wind_from=wind_from,
    )

def _merge_optional_field_to_outer_grid(
    runs: list[RunData],
    field_name: str,
) -> tuple[GridInfo, np.ndarray | None]:
    """
    Merge an optional time-varying field from all runs onto the coarsest /
    outermost grid using finest-precedence masking, but only inside each run's
    true spatial coverage.
    """
    grid, merged, _ = _merge_field_to_outer_grid(runs, lambda run: getattr(run, field_name))
    return grid, merged

def _merge_runs_to_finest_grid(
    runs: list[RunData],
) -> tuple[
    GridInfo,
    np.ndarray,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray,
]:
    """
    Merge all runs onto the coarsest / outermost grid so the full nested domain
    is visible, while finer runs overwrite coarser runs only inside their true
    spatial coverage.

    Returns:
        merged_grid
        merged_depth
        merged_hs
        merged_wave_from
        merged_ssh
        merged_mask
    """
    base = runs[0]   # outermost / coarsest grid
    grid = base.grid
    nt = base.depth.shape[0]
    ny = grid.lat.size
    nx = grid.lon.size

    merged_depth = np.full((nt, ny, nx), np.nan, dtype=float)
    merged_hs = np.full((nt, ny, nx), np.nan, dtype=float) if any(r.hs is not None for r in runs) else None
    merged_wave_from = np.full((nt, ny, nx), np.nan, dtype=float) if any(r.wave_from is not None for r in runs) else None
    merged_ssh = np.full((nt, ny, nx), np.nan, dtype=float) if any(r.ssh is not None for r in runs) else None
    merged_mask = np.zeros((nt, ny, nx), dtype=bool)

    def nearest_index_map(src_vals: np.ndarray, dst_vals: np.ndarray) -> np.ndarray:
        src_vals = np.asarray(src_vals, dtype=float)
        dst_vals = np.asarray(dst_vals, dtype=float)
        out = np.empty(dst_vals.size, dtype=int)
        for i, v in enumerate(dst_vals):
            out[i] = int(np.argmin(np.abs(src_vals - v)))
        return out

    for run in runs:
        src_lon = np.asarray(run.grid.lon, dtype=float)
        src_lat = np.asarray(run.grid.lat, dtype=float)

        # Only map outer-grid cells that are actually inside this run bounds
        lon_in = (grid.lon >= float(np.nanmin(src_lon))) & (grid.lon <= float(np.nanmax(src_lon)))
        lat_in = (grid.lat >= float(np.nanmin(src_lat))) & (grid.lat <= float(np.nanmax(src_lat)))

        ix_dst = np.flatnonzero(lon_in)
        iy_dst = np.flatnonzero(lat_in)

        if ix_dst.size == 0 or iy_dst.size == 0:
            continue

        ix_src = nearest_index_map(src_lon, grid.lon[ix_dst])
        iy_src = nearest_index_map(src_lat, grid.lat[iy_dst])

        for t in range(nt):
            src_keep = np.asarray(run.mask_keep_time[t], dtype=bool)

            sampled_keep = np.zeros((ny, nx), dtype=bool)
            sampled_keep[np.ix_(iy_dst, ix_dst)] = src_keep[np.ix_(iy_src, ix_src)]

            if not np.any(sampled_keep):
                continue

            sampled_depth = np.full((ny, nx), np.nan, dtype=float)
            sampled_depth[np.ix_(iy_dst, ix_dst)] = run.depth[t][np.ix_(iy_src, ix_src)]
            merged_depth[t][sampled_keep] = sampled_depth[sampled_keep]

            if merged_hs is not None and run.hs is not None:
                sampled_hs = np.full((ny, nx), np.nan, dtype=float)
                sampled_hs[np.ix_(iy_dst, ix_dst)] = run.hs[t][np.ix_(iy_src, ix_src)]
                merged_hs[t][sampled_keep] = sampled_hs[sampled_keep]

            if merged_wave_from is not None and run.wave_from is not None:
                sampled_wave = np.full((ny, nx), np.nan, dtype=float)
                sampled_wave[np.ix_(iy_dst, ix_dst)] = run.wave_from[t][np.ix_(iy_src, ix_src)]
                merged_wave_from[t][sampled_keep] = sampled_wave[sampled_keep]

            if merged_ssh is not None and run.ssh is not None:
                sampled_ssh = np.full((ny, nx), np.nan, dtype=float)
                sampled_ssh[np.ix_(iy_dst, ix_dst)] = run.ssh[t][np.ix_(iy_src, ix_src)]
                merged_ssh[t][sampled_keep] = sampled_ssh[sampled_keep]

            merged_mask[t][sampled_keep] = True

    return grid, merged_depth, merged_hs, merged_wave_from, merged_ssh, merged_mask

def _build_depth_contour_traces(
    depth_2d: np.ndarray,
    lon: np.ndarray,
    lat: np.ndarray,
    contour_levels: list[float] | None = None,
    label_levels: list[float] | None = None,
) -> list[go.Contour]:
    """
    Build lightweight bathymetry contour traces using only the exact
    requested contour levels.

    Parameters
    ----------
    depth_2d : np.ndarray
        2D bathymetry array, positive downward [m].
    lon : np.ndarray
        1D longitude array.
    lat : np.ndarray
        1D latitude array.
    contour_levels : list[float] | None
        Exact contour levels to draw.
    label_levels : list[float] | None
        Subset of contour levels that should be labeled.
        If None, all drawn levels are labeled.
    """
    z = np.asarray(depth_2d, dtype=float)

    if contour_levels is None:
        contour_levels = [10, 20, 50, 100, 200]

    if label_levels is None:
        label_levels = contour_levels

    finite = z[np.isfinite(z)]
    if finite.size == 0:
        return []

    zmin = float(np.nanmin(finite))
    zmax = float(np.nanmax(finite))

    traces: list[go.Contour] = []

    for lev in contour_levels:
        if not (zmin <= lev <= zmax):
            continue

        # Slightly emphasize larger/deeper contours
        if lev >= 100:
            width = 1.6
        elif lev >= 50:
            width = 1.3
        else:
            width = 1.0

        showlabels = lev in label_levels

        traces.append(
            go.Contour(
                z=z,
                x=lon,
                y=lat,
                autocontour=False,
                contours=dict(
                    start=lev,
                    end=lev,
                    size=1,              # irrelevant here since start=end
                    coloring="none",
                    showlabels=showlabels,
                    labelfont=dict(
                        size=10,
                        color="black",
                    ),
                ),
                line=dict(
                    color="black",
                    width=width,
                ),
                showscale=False,
                hovertemplate=(
                    "lon=%{x:.6f}<br>"
                    "lat=%{y:.6f}<br>"
                    "depth=%{z:.2f} m<extra></extra>"
                ),
                name=f"{lev} m contour",
            )
        )

    return traces

def _smart_text_position(
    x: float,
    y: float,
    x_values: np.ndarray,
    y_values: np.ndarray,
) -> str:
    """
    Choose a Plotly textposition so the label stays inside the plot as much as possible.
    """
    x_values = np.asarray(x_values, dtype=float)
    y_values = np.asarray(y_values, dtype=float)

    x_min = float(np.nanmin(x_values))
    x_max = float(np.nanmax(x_values))
    y_min = float(np.nanmin(y_values))
    y_max = float(np.nanmax(y_values))

    x_span = max(x_max - x_min, 1e-12)
    y_span = max(y_max - y_min, 1e-12)

    x_rel = (float(x) - x_min) / x_span
    y_rel = (float(y) - y_min) / y_span

    # Horizontal placement
    if x_rel > 0.80:
        hpos = "left"
    elif x_rel < 0.20:
        hpos = "right"
    else:
        hpos = "right"

    # Vertical placement
    if y_rel > 0.80:
        vpos = "bottom"
    elif y_rel < 0.20:
        vpos = "top"
    else:
        vpos = "top"

    return f"{vpos} {hpos}"

def _format_dir_listing(directory: Path, max_entries: int = 20) -> str:
    files = sorted(p.name for p in directory.iterdir() if p.is_file())
    preview = files[:max_entries]
    suffix = "\n  ..." if len(files) > max_entries else ""
    return "\n".join(f"  - {name}" for name in preview) + suffix


def _autodetect_file(
    directory: Path,
    suffix: str,
    preferred_stem_fragment: str | None = None,
    avoid_stem_fragment: str | None = None,
) -> Path:
    candidates = sorted(directory.glob(f"*{suffix}"))
    if not candidates:
        raise FileNotFoundError(f"No '{suffix}' files found in {directory}")

    if avoid_stem_fragment:
        filtered = [p for p in candidates if avoid_stem_fragment.lower() not in p.stem.lower()]
        if filtered:
            candidates = filtered

    if preferred_stem_fragment:
        preferred = [p for p in candidates if preferred_stem_fragment.lower() in p.stem.lower()]
        if preferred:
            return preferred[0]

    return candidates[0]


def _resolve_optional_file(
    directory: Path,
    user_file: str | None,
    suffix: str,
    preferred_stem_fragment: str | None = None,
    avoid_stem_fragment: str | None = None,
) -> Path:
    if user_file is None:
        detected = _autodetect_file(directory, suffix, preferred_stem_fragment, avoid_stem_fragment)
        print(f"Auto-detected {suffix} file: {detected.name}")
        return detected

    path = Path(user_file)
    if not path.is_absolute():
        path = directory / path
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return path


def _build_inputs_from_code_config() -> Inputs:
    raw_dirs = INPUT_DIRECTORY if isinstance(INPUT_DIRECTORY, (tuple, list)) else (INPUT_DIRECTORY,)
    directories = tuple(Path(d).expanduser().resolve() for d in raw_dirs)
    if not directories:
        raise ValueError("At least one input directory is required.")

    for directory in directories:
        if not directory.is_dir():
            raise NotADirectoryError(
                "INPUT_DIRECTORY is invalid. Update INPUT_DIRECTORY in the script before running from PyCharm. "
                f"Current value: {directory}"
            )
        print(f"Using directory: {directory}")
        print("Detected files in directory:")
        print(_format_dir_listing(directory) or "  (no files)")

    def _resolve_run_nc(directory: Path) -> Path | None:
        if NC_FILE_NAME is None:
            try:
                nc = _autodetect_file(directory, ".nc", avoid_stem_fragment="spec")
                print(f"[{directory.name}] Auto-detected .nc file: {nc.name}")
                return nc
            except FileNotFoundError:
                if PLOT_SPEC_DIRECTIONAL:
                    print(f"[{directory.name}] No non-spec .nc file found (allowed in spectral mode).")
                    return None
                raise
        return _resolve_optional_file(directory, NC_FILE_NAME, ".nc", avoid_stem_fragment="spec")

    nc_files = tuple(_resolve_run_nc(directory) for directory in directories)

    if SPEC_FILE_NAME is not None:
        spec_file = _resolve_optional_file(directories[0], SPEC_FILE_NAME, "_spec.nc")
    else:
        try:
            spec_file = _autodetect_file(directories[0], "_spec.nc")
            print(f"Auto-detected _spec.nc file: {spec_file.name}")
        except FileNotFoundError:
            spec_file = None

    def _resolve_run_bot(directory: Path) -> Path | None:
        if BOT_FILE_NAME is None:
            bot_candidates = sorted(list(directory.glob("*.bot")) + list(directory.glob("*.BOT")))
            bot = bot_candidates[0] if bot_candidates else None
            if bot is not None:
                print(f"[{directory.name}] Auto-detected .bot file: {bot.name}")
            else:
                print(f"[{directory.name}] No .bot/.BOT file found. Bathymetry contours will be skipped.")
            return bot
        return _resolve_optional_file(directory, BOT_FILE_NAME, ".bot")

    bot_files = tuple(_resolve_run_bot(directory) for directory in directories)

    war = None if WIND_ARROW_RESOLUTION == 0 else WIND_ARROW_RESOLUTION
    return Inputs(
        directories=directories,
        nc_files=nc_files,
        bot_files=bot_files,
        point_coord=_normalize_point_coords(POINT_COORD),
        wind_arrow_resolution=war,
        export_hs_format=EXPORT_HS_FORMAT,
        export_time_index=EXPORT_TIME_INDEX,
        spec_file=spec_file,
        plot_spec_directional=PLOT_SPEC_DIRECTIONAL,
        spec_dir_theta_step_deg=SPEC_DIR_THETA_STEP_DEG,
        spec_dir_spreading_s=SPEC_DIR_SPREADING_S,
        plot_engine=PLOT_ENGINE,
    )


def parse_args() -> Inputs:
    parser = argparse.ArgumentParser(
        description="Post-process SWAN/DNORA output with animated HS/TP/Wind visualizations.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "directory",
        nargs="+",
        help=(
            "One or more folders containing SWAN output files (.nc, .BOT/.bot).\n"
            "Typical folder contents:\n"
            "  - Ex1_Sula500_20200120.nc\n"
            "  - Ex1_Sula500_20200120_spec.nc\n"
            "  - Ex1_Sula500_SWAN.bot"
        ),
    )
    parser.add_argument("--nc-file", default=None, help="NetCDF filename (default: auto-detect *.nc).")
    parser.add_argument("--bot-file", default=None, help="Bathymetry filename (default: auto-detect *.bot/*.BOT).")
    parser.add_argument(
        "--point-lat",
        action="append",
        default=None,
        help=(
            "Point-of-interest latitude. Repeat flag for multiple points, or pass comma-separated values "
            '(e.g. --point-lat 65.01 --point-lat 64.97 or --point-lat "65.01,64.97").'
        ),
    )
    parser.add_argument(
        "--point-lon",
        action="append",
        default=None,
        help=(
            "Point-of-interest longitude. Repeat flag for multiple points, or pass comma-separated values "
            '(e.g. --point-lon 11.74 --point-lon 11.72 or --point-lon "11.74,11.72").'
        ),
    )
    parser.add_argument(
        "--wind-arrow-resolution",
        type=int,
        default=DEFAULT_ARROW_RESOLUTION,
        help="Use every Nth grid point for wind arrows. Set 0 to disable.",
    )
    parser.add_argument(
        "--export-hs-format",
        choices=["geojson", "kml", "kmz"],
        default=None,
        help="Export HS colormap for Google Earth import (GeoJSON/KML/KMZ) and exit.",
    )
    parser.add_argument(
        "--export-time-index",
        default="0",
        help='Time index to export for HS colormap. Use integer (e.g. "0") or "MAX".',
    )
    parser.add_argument(
        "--spec-file",
        default=None,
        help="Spectral NetCDF filename (default for spectral plot mode: auto-detect *_spec.nc).",
    )
    parser.add_argument(
        "--plot-spec-directional",
        action="store_true",
        help="Use oceanwaves to plot directional spectra from the spec file at requested point.",
    )
    parser.add_argument(
        "--spec-dir-theta-step-deg",
        type=float,
        default=5.0,
        help="Directional step size in degrees used for oceanwaves.as_directional (default: 5.0).",
    )
    parser.add_argument(
        "--spec-dir-spreading-s",
        type=float,
        default=5.0,
        help="Spreading parameter s for oceanwaves.as_directional (default: 5.0).",
    )
    parser.add_argument(
        "--plot-engine",
        choices=["plotly"],
        default="plotly",
        help="Plot engine for SWAN plots.",
    )
    parser.add_argument(
        "--split-report-files",
        dest="split_report_files",
        action="store_true",
        default=None,
        help="Write split HTML reports (default: current script setting).",
    )
    parser.add_argument(
        "--single-report-file",
        dest="split_report_files",
        action="store_false",
        help="Write a single combined HTML report.",
    )
    parser.add_argument(
        "--auto-open-split-files",
        dest="auto_open_split_files",
        action="store_true",
        default=None,
        help="Auto-open produced HTML reports in browser (default: current script setting).",
    )
    parser.add_argument(
        "--no-auto-open-split-files",
        dest="auto_open_split_files",
        action="store_false",
        help="Do not auto-open produced HTML reports.",
    )

    args = parser.parse_args()
    directories = tuple(Path(d).expanduser().resolve() for d in args.directory)
    for directory in directories:
        if not directory.is_dir():
            raise NotADirectoryError(f"Not a directory: {directory}")
        print(f"Using directory: {directory}")
        print("Detected files in directory:")
        print(_format_dir_listing(directory) or "  (no files)")

    def _resolve_run_nc(directory: Path) -> Path | None:
        if args.nc_file is None:
            try:
                nc = _autodetect_file(directory, ".nc", avoid_stem_fragment="spec")
                print(f"[{directory.name}] Auto-detected .nc file: {nc.name}")
                return nc
            except FileNotFoundError:
                if args.plot_spec_directional:
                    print(f"[{directory.name}] No non-spec .nc file found (allowed in spectral mode).")
                    return None
                raise
        return _resolve_optional_file(directory, args.nc_file, ".nc", avoid_stem_fragment="spec")

    nc_files = tuple(_resolve_run_nc(directory) for directory in directories)

    def _resolve_run_bot(directory: Path) -> Path | None:
        if args.bot_file is None:
            bot_candidates = sorted(list(directory.glob("*.bot")) + list(directory.glob("*.BOT")))
            if bot_candidates:
                print(f"[{directory.name}] Auto-detected .bot file: {bot_candidates[0].name}")
                return bot_candidates[0]
            print(f"[{directory.name}] No .bot/.BOT file found. Bathymetry contours will be skipped.")
            return None
        return _resolve_optional_file(directory, args.bot_file, ".bot")

    bot_files = tuple(_resolve_run_bot(directory) for directory in directories)

    wind_arrow_resolution = None if args.wind_arrow_resolution == 0 else args.wind_arrow_resolution
    if args.spec_file is not None:
        spec_file = _resolve_optional_file(directories[0], args.spec_file, "_spec.nc")
    else:
        try:
            spec_file = _autodetect_file(directories[0], "_spec.nc")
            print(f"Auto-detected _spec.nc file: {spec_file.name}")
        except FileNotFoundError:
            spec_file = None

    point_coords = _resolve_point_coordinates(args.point_lat, args.point_lon)

    if args.split_report_files is not None:
        globals()["SPLIT_REPORT_FILES"] = bool(args.split_report_files)
    if args.auto_open_split_files is not None:
        globals()["AUTO_OPEN_SPLIT_FILES"] = bool(args.auto_open_split_files)

    return Inputs(
        directories=directories,
        nc_files=nc_files,
        bot_files=bot_files,
        point_coord=point_coords,
        wind_arrow_resolution=wind_arrow_resolution,
        export_hs_format=args.export_hs_format,
        export_time_index=args.export_time_index,
        spec_file=spec_file,
        plot_spec_directional=bool(args.plot_spec_directional),
        spec_dir_theta_step_deg=float(args.spec_dir_theta_step_deg),
        spec_dir_spreading_s=float(args.spec_dir_spreading_s),
        plot_engine=str(args.plot_engine),
    )


def _normalize_point_coords(
    point_coord: tuple[float, float] | tuple[tuple[float, float], ...],
) -> tuple[tuple[float, float], ...]:
    if len(point_coord) == 2 and all(np.isscalar(v) for v in point_coord):
        lat, lon = point_coord
        return ((float(lat), float(lon)),)
    return tuple((float(lat), float(lon)) for lat, lon in point_coord)


def _resolve_point_coordinates(
    point_lats: list[str] | None,
    point_lons: list[str] | None,
) -> tuple[tuple[float, float], ...]:
    """
    Resolve requested POIs from CLI arguments.

    Behavior:
    - If neither --point-lat nor --point-lon is given, use POINT_COORD exactly as configured.
      This may be a single (lat, lon) tuple or a tuple of (lat, lon) tuples.
    - Otherwise parse the provided lat/lon values and return a tuple of point tuples.
    """
    if not point_lats and not point_lons:
        return _normalize_point_coords(POINT_COORD)

    def _parse_many(values: list[str] | None) -> list[float]:
        out: list[float] = []
        if not values:
            return out
        for raw in values:
            for token in str(raw).split(","):
                t = token.strip()
                if t:
                    out.append(float(t))
        return out

    lats = _parse_many(point_lats)
    lons = _parse_many(point_lons)

    if not lats or not lons:
        raise ValueError(
            "Both --point-lat and --point-lon must be provided when overriding default point coordinates."
        )

    if len(lats) != len(lons):
        raise ValueError(
            f"Number of point latitudes ({len(lats)}) must match number of point longitudes ({len(lons)})."
        )

    return tuple((float(lat), float(lon)) for lat, lon in zip(lats, lons))

def _first_present(ds: xr.Dataset, candidates: Iterable[str]) -> str | None:
    for name in candidates:
        if name in ds:
            return name
    return None


def _cell_edges(coords: np.ndarray) -> np.ndarray:
    c = np.asarray(coords, dtype=float)
    if c.ndim != 1 or c.size < 2:
        raise ValueError("Expected 1D coordinate array with at least 2 entries.")
    edges = np.empty(c.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (c[:-1] + c[1:])
    edges[0] = c[0] - 0.5 * (c[1] - c[0])
    edges[-1] = c[-1] + 0.5 * (c[-1] - c[-2])
    return edges


def _rgba_for_value(value: float, vmin: float, vmax: float, alpha: int = 180) -> tuple[int, int, int, int]:
    if not np.isfinite(value):
        return 0, 0, 0, 0
    if vmax <= vmin:
        norm = 0.5
    else:
        norm = max(0.0, min(1.0, (value - vmin) / (vmax - vmin)))
    color = sample_colorscale("Jet", [norm])[0]
    rgb = color[color.find("(") + 1 : color.find(")")].split(",")
    r, g, b = (int(float(rgb[0])), int(float(rgb[1])), int(float(rgb[2])))
    return r, g, b, alpha


def _kml_abgr(a: int, r: int, g: int, b: int) -> str:
    return f"{a:02x}{b:02x}{g:02x}{r:02x}"


def _export_hs_geojson_or_kml(ds: xr.Dataset, out_path: Path, export_format: str, time_index: int) -> Path:
    if "hs" not in ds:
        raise KeyError("Dataset has no 'hs' variable to export.")
    hs = np.asarray(ds["hs"].transpose("time", "latitude", "longitude").values)
    lon = np.asarray(ds["longitude"].values, dtype=float)
    lat = np.asarray(ds["latitude"].values, dtype=float)
    times = np.asarray(ds["time"].values)

    nt = hs.shape[0]
    if time_index < 0 or time_index >= nt:
        raise IndexError(f"time_index={time_index} is out of range [0, {nt - 1}].")

    hs_t = hs[time_index]
    vmin = float(np.nanmin(hs_t))
    vmax = float(np.nanmax(hs_t))
    lon_e = _cell_edges(lon)
    lat_e = _cell_edges(lat)
    time_text = str(times[time_index]) if times.size else str(time_index)

    if export_format == "geojson":
        features = []
        for iy in range(lat.size):
            for ix in range(lon.size):
                val = float(hs_t[iy, ix])
                if not np.isfinite(val):
                    continue
                r, g, b, a = _rgba_for_value(val, vmin, vmax)
                ring = [
                    [float(lon_e[ix]), float(lat_e[iy])],
                    [float(lon_e[ix + 1]), float(lat_e[iy])],
                    [float(lon_e[ix + 1]), float(lat_e[iy + 1])],
                    [float(lon_e[ix]), float(lat_e[iy + 1])],
                    [float(lon_e[ix]), float(lat_e[iy])],
                ]
                features.append(
                    {
                        "type": "Feature",
                        "properties": {
                            "hs_m": val,
                            "time": time_text,
                            "fill": f"#{r:02x}{g:02x}{b:02x}",
                            "fill-opacity": round(a / 255.0, 3),
                            "stroke": "#000000",
                            "stroke-opacity": 0.08,
                            "stroke-width": 0.3,
                        },
                        "geometry": {"type": "Polygon", "coordinates": [ring]},
                    }
                )
        payload = {
            "type": "FeatureCollection",
            "name": f"HS_colormap_t{time_index}",
            "properties": {"time": time_text, "hs_min": vmin, "hs_max": vmax, "units": "m", "colorscale": "Jet"},
            "features": features,
        }
        out_path.write_text(json.dumps(payload), encoding="utf-8")
        return out_path

    style_map: dict[str, str] = {}
    placemarks: list[str] = []
    for iy in range(lat.size):
        for ix in range(lon.size):
            val = float(hs_t[iy, ix])
            if not np.isfinite(val):
                continue
            r, g, b, a = _rgba_for_value(val, vmin, vmax)
            style_id = f"s_{r:02x}{g:02x}{b:02x}{a:02x}"
            if style_id not in style_map:
                style_map[style_id] = (
                    f'<Style id="{style_id}"><LineStyle><color>22000000</color><width>0.4</width></LineStyle>'
                    f"<PolyStyle><color>{_kml_abgr(a, r, g, b)}</color></PolyStyle></Style>"
                )
            c0 = f"{lon_e[ix]},{lat_e[iy]},0"
            c1 = f"{lon_e[ix + 1]},{lat_e[iy]},0"
            c2 = f"{lon_e[ix + 1]},{lat_e[iy + 1]},0"
            c3 = f"{lon_e[ix]},{lat_e[iy + 1]},0"
            placemarks.append(
                "<Placemark>"
                f"<name>{val:.2f} m</name>"
                f"<description>{escape(f'HS={val:.3f} m, time={time_text}')}</description>"
                f"<styleUrl>#{style_id}</styleUrl>"
                "<Polygon><outerBoundaryIs><LinearRing><coordinates>"
                f"{c0} {c1} {c2} {c3} {c0}"
                "</coordinates></LinearRing></outerBoundaryIs></Polygon>"
                "</Placemark>"
            )

    kml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<kml xmlns="http://www.opengis.net/kml/2.2"><Document>'
        f"<name>{escape(out_path.stem)}</name>"
        f"<description>{escape(f'HS colormap (Jet), time={time_text}, range={vmin:.3f}-{vmax:.3f} m')}</description>"
        f"{''.join(style_map.values())}"
        f"{''.join(placemarks)}"
        "</Document></kml>"
    )

    if export_format == "kml":
        out_path.write_text(kml, encoding="utf-8")
        return out_path

    if export_format == "kmz":
        with zipfile.ZipFile(out_path, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("doc.kml", kml)
        return out_path

    raise ValueError(f"Unsupported export format: {export_format}")


def _export_hs_geojson_or_kml_nested(runs: list[RunData], out_path: Path, export_format: str, time_index: int) -> Path:
    hs_values = [np.where(r.mask_keep_time[time_index], r.hs[time_index], np.nan) for r in runs]
    vmin = float(np.nanmin([np.nanmin(v) for v in hs_values]))
    vmax = float(np.nanmax([np.nanmax(v) for v in hs_values]))
    time_text = str(runs[-1].grid.times[time_index]) if runs[-1].grid.times.size else str(time_index)

    if export_format == "geojson":
        features = []
        for run, hs_t in zip(runs, hs_values):
            lon_e = _cell_edges(run.grid.lon)
            lat_e = _cell_edges(run.grid.lat)
            for iy in range(run.grid.lat.size):
                for ix in range(run.grid.lon.size):
                    val = float(hs_t[iy, ix])
                    if not np.isfinite(val):
                        continue
                    r, g, b, a = _rgba_for_value(val, vmin, vmax)
                    ring = [
                        [float(lon_e[ix]), float(lat_e[iy])],
                        [float(lon_e[ix + 1]), float(lat_e[iy])],
                        [float(lon_e[ix + 1]), float(lat_e[iy + 1])],
                        [float(lon_e[ix]), float(lat_e[iy + 1])],
                        [float(lon_e[ix]), float(lat_e[iy])],
                    ]
                    features.append({"type": "Feature", "properties": {"hs_m": val, "time": time_text, "run": run.label, "fill": f"#{r:02x}{g:02x}{b:02x}", "fill-opacity": round(a / 255.0, 3), "stroke": "#000000", "stroke-opacity": 0.08, "stroke-width": 0.3}, "geometry": {"type": "Polygon", "coordinates": [ring]}})
        payload = {"type": "FeatureCollection", "name": f"HS_colormap_t{time_index}", "properties": {"time": time_text, "hs_min": vmin, "hs_max": vmax, "units": "m", "colorscale": "Jet", "composite": "nested_finest_precedence"}, "features": features}
        out_path.write_text(json.dumps(payload), encoding="utf-8")
        return out_path

    style_map: dict[str, str] = {}
    placemarks: list[str] = []
    for run, hs_t in zip(runs, hs_values):
        lon_e = _cell_edges(run.grid.lon)
        lat_e = _cell_edges(run.grid.lat)
        for iy in range(run.grid.lat.size):
            for ix in range(run.grid.lon.size):
                val = float(hs_t[iy, ix])
                if not np.isfinite(val):
                    continue
                r, g, b, a = _rgba_for_value(val, vmin, vmax)
                style_id = f"s_{r:02x}{g:02x}{b:02x}{a:02x}"
                if style_id not in style_map:
                    style_map[style_id] = (f'<Style id="{style_id}"><LineStyle><color>22000000</color><width>0.4</width></LineStyle>' f"<PolyStyle><color>{_kml_abgr(a, r, g, b)}</color></PolyStyle></Style>")
                c0 = f"{lon_e[ix]},{lat_e[iy]},0"
                c1 = f"{lon_e[ix + 1]},{lat_e[iy]},0"
                c2 = f"{lon_e[ix + 1]},{lat_e[iy + 1]},0"
                c3 = f"{lon_e[ix]},{lat_e[iy + 1]},0"
                placemarks.append("<Placemark>" f"<name>{val:.2f} m</name>" f"<description>{escape(f'HS={val:.3f} m, time={time_text}, run={run.label}')}</description>" f"<styleUrl>#{style_id}</styleUrl>" "<Polygon><outerBoundaryIs><LinearRing><coordinates>" f"{c0} {c1} {c2} {c3} {c0}" "</coordinates></LinearRing></outerBoundaryIs></Polygon>" "</Placemark>")

    kml = ('<?xml version="1.0" encoding="UTF-8"?>' '<kml xmlns="http://www.opengis.net/kml/2.2"><Document>' f"<name>{escape(out_path.stem)}</name>" f"<description>{escape(f'HS nested colormap (Jet), time={time_text}, range={vmin:.3f}-{vmax:.3f} m')}</description>" f"{''.join(style_map.values())}" f"{''.join(placemarks)}" "</Document></kml>")
    if export_format == "kml":
        out_path.write_text(kml, encoding="utf-8")
        return out_path
    if export_format == "kmz":
        with zipfile.ZipFile(out_path, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("doc.kml", kml)
        return out_path
    raise ValueError(f"Unsupported export format: {export_format}")


def _resolve_export_time_index(ds: xr.Dataset, point_coord: tuple[float, float], export_time_index: int | str) -> int:
    if isinstance(export_time_index, int):
        return export_time_index

    token = str(export_time_index).strip()
    if token.upper() == "MAX":
        if "hs" not in ds:
            raise KeyError("Dataset has no 'hs' variable, cannot use export_time_index='MAX'.")
        grid = _build_grid_info(ds)
        poi = _find_nearest_grid_point(grid, point_coord)
        hs = np.asarray(ds["hs"].transpose("time", "latitude", "longitude").values)
        series = hs[:, poi.iy, poi.ix]
        if not np.isfinite(series).any():
            raise ValueError("HS values are all NaN at selected point; cannot resolve export_time_index='MAX'.")
        idx = int(np.nanargmax(series))
        print(
            f"Resolved export_time_index='MAX' to {idx} "
            f"(POI lat/lon={poi.lat:.6f},{poi.lon:.6f}, hs_max={float(series[idx]):.3f} m)."
        )
        return idx

    try:
        return int(token)
    except ValueError as exc:
        raise ValueError(f"Invalid export_time_index: {export_time_index!r}. Use integer or 'MAX'.") from exc


def _prepare_wind_direction_from(ds: xr.Dataset, time_dim: str = "time", y_dim: str = "latitude", x_dim: str = "longitude") -> np.ndarray | None:
    if "xwnd" not in ds or "ywnd" not in ds:
        return None
    u_da = ds["xwnd"]
    v_da = ds["ywnd"]
    if not all(d in u_da.dims for d in [time_dim, y_dim, x_dim]) or not all(d in v_da.dims for d in [time_dim, y_dim, x_dim]):
        raise ValueError("Variables 'xwnd' and 'ywnd' must include dimensions (time, latitude, longitude).")
    u = np.asarray(u_da.transpose(time_dim, y_dim, x_dim).values)
    v = np.asarray(v_da.transpose(time_dim, y_dim, x_dim).values)
    heading_to = (np.degrees(np.arctan2(u, v)) + 360.0) % 360.0
    return (heading_to + 180.0) % 360.0


def _prepare_wave_direction_from(ds: xr.Dataset, time_dim: str = "time", y_dim: str = "latitude", x_dim: str = "longitude") -> np.ndarray | None:
    """
    Return wave direction in nautical "from" convention (deg clockwise from North),
    typically available in SWAN as theta0.
    """
    wave_dir_name = _first_present(ds, ["theta0", "mdir", "wave_direction", "dir"])
    if not wave_dir_name:
        return None
    d_da = ds[wave_dir_name]
    if all(d in d_da.dims for d in [time_dim, y_dim, x_dim]):
        return np.asarray(d_da.transpose(time_dim, y_dim, x_dim).values)
    return None


def _prepare_wind_speed(ds: xr.Dataset, time_dim: str = "time", y_dim: str = "latitude", x_dim: str = "longitude") -> tuple[np.ndarray, str] | None:
    if "xwnd" not in ds or "ywnd" not in ds:
        return None
    u_da = ds["xwnd"]
    v_da = ds["ywnd"]
    if not all(d in u_da.dims for d in [time_dim, y_dim, x_dim]) or not all(d in v_da.dims for d in [time_dim, y_dim, x_dim]):
        raise ValueError("Variables 'xwnd' and 'ywnd' must include dimensions (time, latitude, longitude).")
    u = np.asarray(u_da.transpose(time_dim, y_dim, x_dim).values)
    v = np.asarray(v_da.transpose(time_dim, y_dim, x_dim).values)
    return np.sqrt(u**2 + v**2), "sqrt(xwnd^2+ywnd^2)"


def _prepare_current(
    ds: xr.Dataset, time_dim: str = "time", y_dim: str = "latitude", x_dim: str = "longitude"
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if "xcur" not in ds or "ycur" not in ds:
        return None, None
    u_da = ds["xcur"]
    v_da = ds["ycur"]
    if not all(d in u_da.dims for d in [time_dim, y_dim, x_dim]) or not all(d in v_da.dims for d in [time_dim, y_dim, x_dim]):
        raise ValueError("Variables 'xcur' and 'ycur' must include dimensions (time, latitude, longitude).")
    u = np.asarray(u_da.transpose(time_dim, y_dim, x_dim).values)
    v = np.asarray(v_da.transpose(time_dim, y_dim, x_dim).values)
    speed = np.sqrt(u**2 + v**2)
    heading_to = (np.degrees(np.arctan2(u, v)) + 360.0) % 360.0
    heading_from = (heading_to + 180.0) % 360.0
    return speed, heading_from


def _prepare_ssh(ds: xr.Dataset, time_dim: str = "time", y_dim: str = "latitude", x_dim: str = "longitude") -> np.ndarray | None:
    ssh_name = _first_present(ds, ["ssh", "sea_surface_height", "zeta"])
    if not ssh_name:
        return None
    ssh_da = ds[ssh_name]
    if not all(d in ssh_da.dims for d in [time_dim, y_dim, x_dim]):
        return None
    return np.asarray(ssh_da.transpose(time_dim, y_dim, x_dim).values)

def _prepare_water_mask(ds: xr.Dataset, time_dim: str = "time", y_dim: str = "latitude", x_dim: str = "longitude") -> np.ndarray:
    if "depth" not in ds:
        raise KeyError("Dataset is missing required 'depth' variable for land mask.")
    depth_da = ds["depth"]
    if not all(d in depth_da.dims for d in [time_dim, y_dim, x_dim]):
        raise ValueError("Variable 'depth' must include dimensions (time, latitude, longitude).")
    depth = np.asarray(depth_da.transpose(time_dim, y_dim, x_dim).values)
    return np.isfinite(depth) & (depth > 0.0)


def _build_wind_arrow_frame_data(heading_from_2d: np.ndarray, lon2d: np.ndarray, lat2d: np.ndarray, arrow_resolution: int | None) -> tuple[np.ndarray, np.ndarray]:
    if arrow_resolution is None:
        return np.array([]), np.array([])
    if int(arrow_resolution) <= 0:
        raise ValueError("wind_arrow_resolution must be a positive integer or None.")

    x_all = lon2d.ravel()
    y_all = lat2d.ravel()
    h_all = np.asarray(heading_from_2d).ravel()
    valid = np.isfinite(x_all) & np.isfinite(y_all) & np.isfinite(h_all)
    idx = np.flatnonzero(valid)
    if idx.size == 0:
        return np.array([]), np.array([])

    sampled = idx[:: int(arrow_resolution)]
    x0 = x_all[sampled]
    y0 = y_all[sampled]
    heading_to = (h_all[sampled] + 180.0) % 360.0

    x_span = float(np.nanmax(x_all[valid]) - np.nanmin(x_all[valid]))
    y_span = float(np.nanmax(y_all[valid]) - np.nanmin(y_all[valid]))
    arrow_length = 0.06 * min(x_span, y_span)

    to_rad = np.deg2rad(heading_to)
    dx = arrow_length * np.sin(to_rad)
    dy = arrow_length * np.cos(to_rad)
    x1 = x0 + dx
    y1 = y0 + dy

    head_len = 0.35 * arrow_length
    head_ang = np.deg2rad(25.0)
    rev_rad = to_rad + np.pi
    wing1 = rev_rad + head_ang
    wing2 = rev_rad - head_ang
    xw1 = x1 + head_len * np.sin(wing1)
    yw1 = y1 + head_len * np.cos(wing1)
    xw2 = x1 + head_len * np.sin(wing2)
    yw2 = y1 + head_len * np.cos(wing2)

    line_x = np.column_stack([x0, x1, np.full_like(x0, np.nan), x1, xw1, np.full_like(x0, np.nan), x1, xw2, np.full_like(x0, np.nan)]).ravel()
    line_y = np.column_stack([y0, y1, np.full_like(y0, np.nan), y1, yw1, np.full_like(y0, np.nan), y1, yw2, np.full_like(y0, np.nan)]).ravel()
    return line_x, line_y


def read_bot(bot_path: Path, nx: int, ny: int) -> np.ndarray:
    with bot_path.open("r", encoding="utf-8", errors="ignore") as f:
        tokens = f.read().split()
    vals = []
    for t in tokens:
        try:
            vals.append(float(t))
        except ValueError:
            pass
    if len(vals) < nx * ny:
        raise ValueError(f"Not enough numeric values in BOT file: found {len(vals)}, expected at least {nx * ny}")
    return np.array(vals[-nx * ny :]).reshape((ny, nx))


def _build_grid_info(ds: xr.Dataset) -> GridInfo:
    lon = np.asarray(ds["longitude"].values)
    lat = np.asarray(ds["latitude"].values)
    lon2d, lat2d = np.meshgrid(lon, lat)
    times = np.asarray(ds["time"].values)
    return GridInfo(lon=lon, lat=lat, lon2d=lon2d, lat2d=lat2d, times=times)


def _grid_resolution_score(grid: GridInfo) -> float:
    dlon = np.diff(np.unique(grid.lon))
    dlat = np.diff(np.unique(grid.lat))
    sx = float(np.nanmedian(np.abs(dlon))) if dlon.size else np.inf
    sy = float(np.nanmedian(np.abs(dlat))) if dlat.size else np.inf
    return sx * sy


def _grid_bounds(grid: GridInfo) -> tuple[float, float, float, float]:
    return (
        float(np.nanmin(grid.lon)),
        float(np.nanmax(grid.lon)),
        float(np.nanmin(grid.lat)),
        float(np.nanmax(grid.lat)),
    )


def _prepare_runs_data(datasets: list[tuple[str, xr.Dataset]]) -> list[RunData]:
    raw: list[
        tuple[str, GridInfo, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray]
    ] = []
    for label, ds in datasets:
        grid = _build_grid_info(ds)
        hs = np.asarray(ds["hs"].transpose("time", "latitude", "longitude").values)
        tp = np.asarray(ds["tp"].transpose("time", "latitude", "longitude").values)
        depth = np.asarray(ds["depth"].transpose("time", "latitude", "longitude").values)
        wind = _prepare_wind_speed(ds)
        wind_speed = wind[0] if wind is not None else None
        wind_from = _prepare_wind_direction_from(ds)
        wave_from = _prepare_wave_direction_from(ds)
        ssh = _prepare_ssh(ds)
        current_speed, current_from = _prepare_current(ds)
        water_mask = _prepare_water_mask(ds)
        raw.append((label, grid, hs, tp, depth, wind_speed, wind_from, wave_from, ssh, current_speed, current_from, water_mask))

    raw_sorted = sorted(raw, key=lambda item: _grid_resolution_score(item[1]), reverse=True)
    finer_bounds: list[tuple[float, float, float, float]] = []
    runs_rev: list[RunData] = []
    for label, grid, hs, tp, depth, wind_speed, wind_from, wave_from, ssh, current_speed, current_from, water_mask in reversed(raw_sorted):
        mask_keep_static = np.ones_like(water_mask[0], dtype=bool)
        for lon_min, lon_max, lat_min, lat_max in finer_bounds:
            overlap = (
                (grid.lon2d >= lon_min)
                & (grid.lon2d <= lon_max)
                & (grid.lat2d >= lat_min)
                & (grid.lat2d <= lat_max)
            )
            mask_keep_static &= ~overlap
        mask_keep_time = water_mask & mask_keep_static[None, :, :]
        runs_rev.append(
            RunData(
                label=label,
                grid=grid,
                hs=hs,
                tp=tp,
                depth=depth,
                wind_speed=wind_speed,
                wind_from=wind_from,
                wave_from=wave_from,
                ssh=ssh,
                current_speed=current_speed,
                current_from=current_from,
                water_mask=water_mask,
                mask_keep_static=mask_keep_static,
                mask_keep_time=mask_keep_time,
            )
        )
        finer_bounds.append(_grid_bounds(grid))
    return list(reversed(runs_rev))
def _dilate_boolean_mask(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    """
    Simple 8-neighbour dilation without scipy.
    """
    out = np.asarray(mask, dtype=bool).copy()
    for _ in range(max(0, int(iterations))):
        p = np.pad(out, 1, mode="constant", constant_values=False)
        out = (
            p[1:-1, 1:-1]
            | p[:-2, 1:-1] | p[2:, 1:-1]
            | p[1:-1, :-2] | p[1:-1, 2:]
            | p[:-2, :-2] | p[:-2, 2:]
            | p[2:, :-2] | p[2:, 2:]
        )
    return out


def _merge_optional_field_to_finest_grid(
    runs: list[RunData],
    field_name: str,
) -> np.ndarray | None:
    """
    Merge an optional time-varying field from all runs onto the finest grid
    using the same finest-precedence logic as the bathymetry merge.

    Returns:
        merged_field: shape (time, ny, nx) or None if no run has that field
    """
    finest = runs[-1]
    grid = finest.grid
    nt = finest.depth.shape[0]
    ny = grid.lat.size
    nx = grid.lon.size

    has_any = any(getattr(run, field_name) is not None for run in runs)
    if not has_any:
        return None

    merged = np.full((nt, ny, nx), np.nan, dtype=float)
    merged_filled = np.zeros((nt, ny, nx), dtype=bool)

    def nearest_index_map(src_vals: np.ndarray, dst_vals: np.ndarray) -> np.ndarray:
        src_vals = np.asarray(src_vals, dtype=float)
        dst_vals = np.asarray(dst_vals, dtype=float)
        out = np.empty(dst_vals.size, dtype=int)
        for i, v in enumerate(dst_vals):
            out[i] = int(np.argmin(np.abs(src_vals - v)))
        return out

    for run in runs:
        src_field = getattr(run, field_name)
        if src_field is None:
            continue

        iy_map = nearest_index_map(run.grid.lat, grid.lat)
        ix_map = nearest_index_map(run.grid.lon, grid.lon)

        for t in range(nt):
            src_keep = run.mask_keep_time[t]
            src_vals = src_field[t]

            for iy_f, iy_s in enumerate(iy_map):
                for ix_f, ix_s in enumerate(ix_map):
                    if not src_keep[iy_s, ix_s]:
                        continue
                    if merged_filled[t, iy_f, ix_f]:
                        continue

                    val = src_vals[iy_s, ix_s]
                    if np.isfinite(val):
                        merged[t, iy_f, ix_f] = float(val)
                    merged_filled[t, iy_f, ix_f] = True

    return merged


def _extract_boundary_axis_segments(
    water_mask: np.ndarray,
    lon: np.ndarray,
    lat: np.ndarray,
    region_mask: np.ndarray | None = None,
    ssh_field: np.ndarray | None = None,
    visible_mask: np.ndarray | None = None,
    treat_grid_edge_as_boundary: bool = False,
) -> list[tuple[str, float, float, float, float | None]]:
    """
    Extract physical shoreline segments from a water mask.

    Parameters
    ----------
    water_mask : (ny, nx) bool
        Physical wet/dry mask. True means wet cell.
    lon, lat : 1D arrays
        Cell-center coordinates.
    region_mask : (ny, nx) bool | None
        Optional mask restricting where shoreline extraction is allowed.
    ssh_field : (ny, nx) float | None
        Optional SSH field used for shoreline coloring.
    visible_mask : (ny, nx) bool | None
        Optional composite visibility mask. True means this cell belongs to the
        visible nested composite.
    treat_grid_edge_as_boundary : bool
        If True, the outer edge of the array is treated as a boundary.
        For nested domains this should normally be False.
    """
    water_mask = np.asarray(water_mask, dtype=bool)
    ny, nx = water_mask.shape

    if visible_mask is None:
        visible_mask = np.ones_like(water_mask, dtype=bool)
    else:
        visible_mask = np.asarray(visible_mask, dtype=bool)

    lon_e = _cell_edges(lon)
    lat_e = _cell_edges(lat)

    segments: list[tuple[str, float, float, float, float | None]] = []

    for iy in range(ny):
        for ix in range(nx):
            if not water_mask[iy, ix]:
                continue
            if not visible_mask[iy, ix]:
                continue
            if region_mask is not None and not region_mask[iy, ix]:
                continue

            ssh_val: float | None = None
            if ssh_field is not None:
                v = float(ssh_field[iy, ix]) if np.isfinite(ssh_field[iy, ix]) else np.nan
                if np.isfinite(v):
                    ssh_val = v

            x0 = float(lon_e[ix])
            x1 = float(lon_e[ix + 1])
            y0 = float(lat_e[iy])
            y1 = float(lat_e[iy + 1])

            # IMPORTANT:
            # Only mark shoreline against a physically dry neighbor
            # that exists INSIDE the same grid.
            south_is_boundary = False
            north_is_boundary = False
            west_is_boundary = False
            east_is_boundary = False

            if iy > 0:
                south_is_boundary = not water_mask[iy - 1, ix]
            elif treat_grid_edge_as_boundary:
                south_is_boundary = True

            if iy < ny - 1:
                north_is_boundary = not water_mask[iy + 1, ix]
            elif treat_grid_edge_as_boundary:
                north_is_boundary = True

            if ix > 0:
                west_is_boundary = not water_mask[iy, ix - 1]
            elif treat_grid_edge_as_boundary:
                west_is_boundary = True

            if ix < nx - 1:
                east_is_boundary = not water_mask[iy, ix + 1]
            elif treat_grid_edge_as_boundary:
                east_is_boundary = True

            if south_is_boundary:
                segments.append(("h", y0, x0, x1, ssh_val))
            if north_is_boundary:
                segments.append(("h", y1, x0, x1, ssh_val))
            if west_is_boundary:
                segments.append(("v", x0, y0, y1, ssh_val))
            if east_is_boundary:
                segments.append(("v", x1, y0, y1, ssh_val))

    return segments


def _deduplicate_axis_segments(
    segments: list[tuple[str, float, float, float, float | None]],
    decimals: int = 12,
) -> list[tuple[np.ndarray, np.ndarray, float | None]]:
    """
    Deduplicate identical axis-aligned segments. Used for dynamic shoreline areas,
    where we want to preserve detail but avoid repeated segments from many time steps.
    """
    bucket: dict[tuple[str, float, float, float], list[float]] = {}

    for orient, const_coord, start, end, ssh_val in segments:
        s0 = min(float(start), float(end))
        s1 = max(float(start), float(end))
        key = (
            orient,
            round(float(const_coord), decimals),
            round(s0, decimals),
            round(s1, decimals),
        )
        if key not in bucket:
            bucket[key] = []
        if ssh_val is not None and np.isfinite(ssh_val):
            bucket[key].append(float(ssh_val))

    out: list[tuple[np.ndarray, np.ndarray, float | None]] = []
    for (orient, const_coord, s0, s1), vals in bucket.items():
        mean_ssh = float(np.mean(vals)) if vals else None
        if orient == "h":
            out.append(
                (
                    np.array([s0, s1], dtype=float),
                    np.array([const_coord, const_coord], dtype=float),
                    mean_ssh,
                )
            )
        else:
            out.append(
                (
                    np.array([const_coord, const_coord], dtype=float),
                    np.array([s0, s1], dtype=float),
                    mean_ssh,
                )
            )

    return out


def _merge_axis_segments(
    segments: list[tuple[str, float, float, float, float | None]],
    decimals: int = 12,
    tol: float = 1e-12,
) -> list[tuple[np.ndarray, np.ndarray, float | None]]:
    """
    Merge collinear contiguous axis-aligned segments. Used for stable shoreline areas.
    """
    groups: dict[tuple[str, float], list[tuple[float, float, float | None]]] = {}

    for orient, const_coord, start, end, ssh_val in segments:
        s0 = min(float(start), float(end))
        s1 = max(float(start), float(end))
        key = (orient, round(float(const_coord), decimals))
        groups.setdefault(key, []).append((s0, s1, ssh_val))

    out: list[tuple[np.ndarray, np.ndarray, float | None]] = []

    for (orient, const_coord), segs in groups.items():
        segs.sort(key=lambda item: (item[0], item[1]))

        cur_start, cur_end, cur_vals = segs[0][0], segs[0][1], []
        if segs[0][2] is not None and np.isfinite(segs[0][2]):
            cur_vals.append(float(segs[0][2]))

        for s0, s1, ssh_val in segs[1:]:
            if s0 <= cur_end + tol:
                cur_end = max(cur_end, s1)
                if ssh_val is not None and np.isfinite(ssh_val):
                    cur_vals.append(float(ssh_val))
            else:
                mean_ssh = float(np.mean(cur_vals)) if cur_vals else None
                if orient == "h":
                    out.append(
                        (
                            np.array([cur_start, cur_end], dtype=float),
                            np.array([const_coord, const_coord], dtype=float),
                            mean_ssh,
                        )
                    )
                else:
                    out.append(
                        (
                            np.array([const_coord, const_coord], dtype=float),
                            np.array([cur_start, cur_end], dtype=float),
                            mean_ssh,
                        )
                    )

                cur_start, cur_end = s0, s1
                cur_vals = []
                if ssh_val is not None and np.isfinite(ssh_val):
                    cur_vals.append(float(ssh_val))

        mean_ssh = float(np.mean(cur_vals)) if cur_vals else None
        if orient == "h":
            out.append(
                (
                    np.array([cur_start, cur_end], dtype=float),
                    np.array([const_coord, const_coord], dtype=float),
                    mean_ssh,
                )
            )
        else:
            out.append(
                (
                    np.array([const_coord, const_coord], dtype=float),
                    np.array([cur_start, cur_end], dtype=float),
                    mean_ssh,
                )
            )

    return out

def _plot_depth_hs_plane_3d(
    runs: list[RunData],
    point_coord: tuple[float, float] | tuple[tuple[float, float], ...],
    output_dir: Path,
    max_surface_points: int = 250,
    arrow_resolution: int | None = DEFAULT_ARROW_RESOLUTION,
    include_variable_plane: bool = False,
) -> str:
    """
    Build a static 3D bathymetry plot using Plotly Mesh3d, based on the merged
    point cloud after finest-precedence masking.

    Static / lightweight version:
    - one merged bathymetry mesh
    - shoreline geometry plotted for the whole analysis period at once
    - optional shoreline display modes:
        * SSH-colored (Jet), if SSH exists
        * all black
    - no time slider / no animation

    Plot interpretation:
    - The 3D mesh shows the merged bathymetry/depth field on the finest merged grid.
    - The shoreline / land-mask lines show the shoreline positions encountered over
      the full selected time period.
    - Areas with many overlapping shoreline segments indicate places where the
      shoreline changes position over time.
    - Areas with only a single shoreline trace indicate a stable shoreline.
    - Red diamonds are the selected POIs.
    """

    point_coords = _normalize_point_coords(point_coord)

    x, y, depth_cube, times = _build_merged_depth_point_cloud(runs)
    nt = depth_cube.shape[0]
    if nt == 0:
        raise ValueError("Depth point cloud has zero time steps.")

    depth_ref = np.nanmean(depth_cube, axis=0)
    if not np.isfinite(depth_ref).any():
        raise ValueError("No finite merged depth values available for 3D plotting.")

    finite_ref = np.isfinite(depth_ref)
    if not np.all(finite_ref):
        replacement = float(np.nanmean(depth_ref[finite_ref])) if np.any(finite_ref) else 0.0
        depth_ref = np.where(np.isfinite(depth_ref), depth_ref, replacement)

    # ------------------------------------------------------------------
    # Shoreline geometry
    # ------------------------------------------------------------------
    boundary_segments = _build_merged_boundary_lines(
        runs,
        dynamic_buffer_cells=SHORELINE_DYNAMIC_BUFFER_CELLS,
    )

    # ------------------------------------------------------------------
    # SSH handling for shoreline coloring
    # ------------------------------------------------------------------
    ssh_vals_all: list[float] = []
    ssh_segment_means: list[float | None] = []

    for _, _, ssh_val in boundary_segments:
        if ssh_val is None:
            ssh_segment_means.append(None)
            continue

        ssh_arr = np.asarray(ssh_val, dtype=float)
        if np.isfinite(ssh_arr).any():
            mean_val = float(np.nanmean(ssh_arr))
            ssh_segment_means.append(mean_val)
            ssh_vals_all.append(mean_val)
        else:
            ssh_segment_means.append(None)

    has_ssh_boundary = len(ssh_vals_all) > 0
    ssh_min = ssh_max = 0.0
    if has_ssh_boundary:
        ssh_min = float(np.nanmin(ssh_vals_all))
        ssh_max = float(np.nanmax(ssh_vals_all))
        ssh_min, ssh_max = _normalize_colorscale_limits(ssh_min, ssh_max)

    def _ssh_color(mean_val: float | None) -> str:
        if mean_val is None:
            return "black"
        if ssh_max <= ssh_min:
            norm = 0.5
        else:
            norm = max(0.0, min(1.0, (mean_val - ssh_min) / (ssh_max - ssh_min)))
        return sample_colorscale("Jet", [norm])[0]

    # ------------------------------------------------------------------
    # POIs
    # ------------------------------------------------------------------
    poi_lons: list[float] = []
    poi_lats: list[float] = []
    poi_labels: list[str] = []
    poi_hover: list[str] = []

    for idx, (lat0, lon0) in enumerate(point_coords):
        dist2 = (y - lat0) ** 2 + (x - lon0) ** 2
        ip = int(np.nanargmin(dist2))
        poi_lons.append(float(x[ip]))
        poi_lats.append(float(y[ip]))
        poi_labels.append(f"POI {idx + 1}")
        poi_hover.append(
            f"POI {idx + 1}<br>"
            f"requested lat={lat0:.6f}, lon={lon0:.6f}<br>"
            f"selected lat={y[ip]:.6f}, lon={x[ip]:.6f}<extra></extra>"
        )

    # ------------------------------------------------------------------
    # Mesh limits
    # ------------------------------------------------------------------
    depth_min = float(np.nanmin(depth_ref))
    depth_max = float(np.nanmax(depth_ref))
    depth_min, depth_max = _normalize_colorscale_limits(depth_min, depth_max)

    fig = go.Figure()

    # ------------------------------------------------------------------
    # Main mesh
    # ------------------------------------------------------------------
    fig.add_trace(
        go.Mesh3d(
            x=x,
            y=y,
            z=-depth_ref,
            intensity=depth_ref,
            colorscale="Earth",
            cmin=depth_min,
            cmax=depth_max,
            showscale=True,
            colorbar=dict(
                title="Depth [m]",
                x=-0.10,
                y=0.52,
                len=0.74,
                thickness=18,
            ),
            name="Merged bathymetry",
            hovertemplate=(
                "lon=%{x:.6f}<br>"
                "lat=%{y:.6f}<br>"
                "depth=%{intensity:.2f} m<extra></extra>"
            ),
            delaunayaxis="z",
            flatshading=False,
            opacity=1.0,
            lighting=dict(
                ambient=0.55,
                diffuse=0.80,
                specular=0.08,
                roughness=0.95,
                fresnel=0.02,
            ),
            lightposition=dict(x=100, y=200, z=300),
        )
    )

    # ------------------------------------------------------------------
    # Shoreline traces: only SSH-colored version
    # ------------------------------------------------------------------
    ssh_line_trace_indices: list[int] = []
    ssh_colorbar_trace_idx: int | None = None

    if boundary_segments:
        if has_ssh_boundary:
            for (x_line, y_line, _), mean_val in zip(boundary_segments, ssh_segment_means):
                fig.add_trace(
                    go.Scatter3d(
                        x=x_line,
                        y=y_line,
                        z=np.zeros(len(x_line), dtype=float),
                        mode="lines",
                        line=dict(color=_ssh_color(mean_val), width=4),
                        hoverinfo="skip",
                        showlegend=False,
                        visible=True,
                        name="SSH shoreline",
                    )
                )
                ssh_line_trace_indices.append(len(fig.data) - 1)

            fig.add_trace(
                go.Scatter3d(
                    x=[float(np.nanmin(x)), float(np.nanmin(x))],
                    y=[float(np.nanmin(y)), float(np.nanmin(y))],
                    z=[0.0, 0.0],
                    mode="markers",
                    marker=dict(
                        size=0.1,
                        color=[ssh_min, ssh_max],
                        colorscale="Jet",
                        cmin=ssh_min,
                        cmax=ssh_max,
                        showscale=True,
                        colorbar=dict(
                            title="SSH [m]",
                            x=1.05,
                            y=0.52,
                            len=0.74,
                            thickness=18,
                        ),
                    ),
                    opacity=0.0,
                    hoverinfo="skip",
                    showlegend=False,
                    visible=True,
                )
            )
            ssh_colorbar_trace_idx = len(fig.data) - 1

    # ------------------------------------------------------------------
    # POI markers
    # ------------------------------------------------------------------
    fig.add_trace(
        go.Scatter3d(
            x=poi_lons,
            y=poi_lats,
            z=[0.0] * len(poi_lons),
            mode="markers+text",
            marker=dict(color="red", size=6, symbol="diamond"),
            text=poi_labels,
            textposition="top center",
            hovertemplate=poi_hover,
            name="Points of interest",
            showlegend=True,
        )
    )

    # ------------------------------------------------------------------
    # Layout geometry
    # ------------------------------------------------------------------
    lon_span_deg = max(float(np.nanmax(x) - np.nanmin(x)), 1e-9)
    lat_span_deg = max(float(np.nanmax(y) - np.nanmin(y)), 1e-9)
    mean_lat_rad = np.deg2rad(float(np.nanmean(y)))

    x_span_m = max(lon_span_deg * 111_320.0 * max(np.cos(mean_lat_rad), 1e-6), 1.0)
    y_span_m = max(lat_span_deg * 111_320.0, 1.0)
    z_span_m = max(float(np.nanmax(depth_ref) - np.nanmin(depth_ref)), 1.0)

    xy_ref = max(x_span_m, y_span_m)
    z_ratio = max(0.03, min(1.0, z_span_m / xy_ref))

    # ------------------------------------------------------------------
    # Dropdown for shoreline display mode
    # ------------------------------------------------------------------
    updatemenus = []

    if has_ssh_boundary and ssh_line_trace_indices and ssh_colorbar_trace_idx is not None:
        n_mesh = 1
        n_ssh = len(ssh_line_trace_indices)
        n_ssh_cb = 1
        n_poi = 1

        visible_ssh_mode = (
                [True] * n_mesh +
                [True] * n_ssh +
                [True] * n_ssh_cb +
                [True] * n_poi
        )

        visible_off_mode = (
                [True] * n_mesh +
                [False] * n_ssh +
                [False] * n_ssh_cb +
                [True] * n_poi
        )

        updatemenus.append(
            dict(
                type="dropdown",
                direction="down",
                x=0.02,
                y=1.02,
                xanchor="left",
                yanchor="top",
                showactive=True,
                active=0,
                buttons=[
                    dict(
                        label="Shoreline: SSH colored",
                        method="update",
                        args=[{"visible": visible_ssh_mode}],
                    ),
                    dict(
                        label="Shoreline: Off",
                        method="update",
                        args=[{"visible": visible_off_mode}],
                    ),
                ],
            )
        )

    # ------------------------------------------------------------------
    # Explanation annotation
    # ------------------------------------------------------------------
    explanation_text = (
        "Explanation:<br>"
        "• Mesh = merged bathymetry/depth surface.<br>"
        "• Shoreline lines = shoreline positions observed over the full selected period.<br>"
        "• Dense/overlapping shoreline traces indicate locations where the shoreline moved over time.<br>"
        "• Single shoreline traces indicate stable coastline/land-mask position.<br>"
        "• Red diamonds = selected POIs."
    )

    fig.update_layout(
        title=(
            "3D Bathymetry and Full-Period Shoreline Footprint"
        ),
        scene=dict(
            xaxis_title="Longitude [deg]",
            yaxis_title="Latitude [deg]",
            zaxis_title="Depth [m] (visual only: depth * -1)",
            camera=dict(eye=dict(x=1.45, y=-1.65, z=0.95)),
            aspectmode="manual",
            aspectratio=dict(
                x=x_span_m / xy_ref,
                y=y_span_m / xy_ref,
                z=z_ratio,
            ),
        ),
        margin=dict(l=90, r=90, t=120, b=20),
        legend=dict(
            x=0.01,
            y=0.99,
            xanchor="left",
            yanchor="top",
            bgcolor="rgba(255,255,255,0.7)",
        ),
        updatemenus=updatemenus,
        annotations=[
            dict(
                text=explanation_text,
                x=0.5,
                y=1.08,
                xref="paper",
                yref="paper",
                showarrow=False,
                align="left",
                bgcolor="rgba(255,255,255,0.85)",
                bordercolor="rgba(0,0,0,0.2)",
                borderwidth=1,
                font=dict(size=12),
            )
        ],
    )

    return fig.to_html(full_html=False, config={"responsive": True}, auto_play=False)

def _build_2d_figure_html(fig: go.Figure) -> str:
    return fig.to_html(full_html=False, config={"responsive": True})


def _select_spec_point_dataset(spec_ds: xr.Dataset, point_coord: tuple[float, float]) -> xr.Dataset:
    """
    Backward-compatible single-point selector.
    Kept so existing callers do not break, but now uses the shared index resolver.
    """
    point_coords = _normalize_point_coords(point_coord)
    point_indices = _find_nearest_spec_point_indices(spec_ds, point_coords)
    return _select_spec_point_dataset_by_index(spec_ds, point_indices[0])
def _iter_selected_spec_point_datasets(
    spec_ds: xr.Dataset,
    point_coords: tuple[tuple[float, float], ...],
):
    """
    Yield unique spectral point datasets matched from the requested POIs.

    Yields:
        request_idx: index of the first requested POI mapping to this spectral point
        spec_point_idx: selected spectral point index in the dataset
        selected_ds: dataset sliced to that spectral point
        matched_point_coord: requested (lat, lon)
    """
    point_coords = _normalize_point_coords(point_coords)
    point_indices = _find_nearest_spec_point_indices(spec_ds, point_coords)

    seen: set[int] = set()
    for request_idx, (matched_point_coord, spec_point_idx) in enumerate(zip(point_coords, point_indices)):
        if spec_point_idx in seen:
            continue
        seen.add(spec_point_idx)
        yield request_idx, spec_point_idx, _select_spec_point_dataset_by_index(spec_ds, spec_point_idx), matched_point_coord


def _iter_all_spec_point_datasets(spec_ds: xr.Dataset):
    """
    Yield all spectral points directly from the spec dataset (no POI matching).

    Yields:
        display_idx: zero-based index in the iteration order
        spec_point_idx: spectral point index in the dataset
        selected_ds: dataset sliced to one spectral point
        spec_point_coord: spectral (lat, lon) if available, else None
    """
    if "points" in spec_ds.sizes and int(spec_ds.sizes["points"]) > 1:
        npoints = int(spec_ds.sizes["points"])
    else:
        npoints = 1

    lats = None
    lons = None
    if "latitude" in spec_ds and "longitude" in spec_ds:
        lats = np.asarray(spec_ds["latitude"].values, dtype=float).reshape(-1)
        lons = np.asarray(spec_ds["longitude"].values, dtype=float).reshape(-1)

    for display_idx in range(npoints):
        spec_point_idx = int(display_idx if npoints > 1 else 0)
        coord = None
        if lats is not None and lons is not None and display_idx < lats.size and display_idx < lons.size:
            coord = (float(lats[display_idx]), float(lons[display_idx]))
        yield display_idx, spec_point_idx, _select_spec_point_dataset_by_index(spec_ds, spec_point_idx), coord
def _select_spec_point_dataset_by_index(spec_ds: xr.Dataset, point_idx: int) -> xr.Dataset:
    """
    Select a single spectral point by index.
    """
    if "points" not in spec_ds.sizes or int(spec_ds.sizes["points"]) <= 1:
        return spec_ds
    return spec_ds.isel(points=int(point_idx))
def _find_nearest_spec_point_indices(
    spec_ds: xr.Dataset,
    point_coords: tuple[tuple[float, float], ...],
) -> tuple[int, ...]:
    """
    For each requested (lat, lon), find the nearest spectral point index.
    Returns one index per requested point.
    """
    point_coords = _normalize_point_coords(point_coords)

    if "points" not in spec_ds.sizes or int(spec_ds.sizes["points"]) <= 1:
        return tuple(0 for _ in point_coords)

    if "latitude" not in spec_ds or "longitude" not in spec_ds:
        raise KeyError("Spec dataset has multiple points but missing latitude/longitude coordinates.")

    lats = np.asarray(spec_ds["latitude"].values, dtype=float).reshape(-1)
    lons = np.asarray(spec_ds["longitude"].values, dtype=float).reshape(-1)

    if lats.size != lons.size:
        raise ValueError("Spec dataset latitude/longitude point coordinates have inconsistent sizes.")

    indices: list[int] = []
    for lat0, lon0 in point_coords:
        dist2 = (lats - lat0) ** 2 + (lons - lon0) ** 2
        point_idx = int(np.nanargmin(dist2))
        print(
            "Spec dataset point selection: "
            f"input(lat,lon)=({lat0:.6f},{lon0:.6f}), "
            f"selected idx={point_idx} at ({lats[point_idx]:.6f},{lons[point_idx]:.6f})"
        )
        indices.append(point_idx)

    return tuple(indices)
def _find_first_dim_name(ds: xr.Dataset, candidates: tuple[str, ...]) -> str | None:
    names = {name.lower(): name for name in ds.dims}
    for candidate in candidates:
        if candidate in names:
            return names[candidate]
    return None


def _find_first_coord_name(ds: xr.Dataset, candidates: tuple[str, ...]) -> str | None:
    all_names = set(ds.coords) | set(ds.variables)
    lowered = {name.lower(): name for name in all_names}
    for candidate in candidates:
        if candidate in lowered:
            return lowered[candidate]
    return None


def _detect_spec_axes_and_var(ds: xr.Dataset) -> tuple[str, str, str, xr.DataArray]:
    time_name = _find_first_dim_name(ds, ("time", "datetime"))
    freq_name = _find_first_coord_name(ds, ("frequency", "freq", "f"))
    dir_name = _find_first_coord_name(ds, ("direction", "dir", "theta"))
    if time_name is None or freq_name is None or dir_name is None:
        raise KeyError(
            "Could not auto-detect time/frequency/direction axes. "
            f"dims={list(ds.dims)}, vars={list(ds.variables)}"
        )

    freq_coord = np.asarray(ds[freq_name].values, dtype=float).reshape(-1)
    dir_coord = np.asarray(ds[dir_name].values, dtype=float).reshape(-1)
    if freq_coord.size < 2 or dir_coord.size < 2:
        raise ValueError("Frequency and direction coordinates must each contain at least 2 values.")

    best_name: str | None = None
    best_score = -1
    for name, da in ds.data_vars.items():
        dims = set(da.dims)
        score = int(time_name in dims) + int(freq_name in dims) + int(dir_name in dims)
        if score > best_score and score >= 3:
            best_name = name
            best_score = score
    if best_name is None:
        raise KeyError("Could not auto-detect a spectral density variable containing time/frequency/direction.")

    spec_da = ds[best_name]
    extra_dims = [d for d in spec_da.dims if d not in (time_name, freq_name, dir_name)]
    for extra_dim in extra_dims:
        spec_da = spec_da.isel({extra_dim: 0})
    spec_da = spec_da.transpose(time_name, freq_name, dir_name)
    return time_name, freq_name, dir_name, spec_da

def build_poi_overview_figure(
    datasets: list[tuple[str, xr.Dataset]],
    point_coord: tuple[float, float] | tuple[tuple[float, float], ...],
) -> go.Figure:
    runs = _prepare_runs_data(datasets)
    finest = runs[-1]

    point_coords = _normalize_point_coords(point_coord)
    pois = tuple(_find_nearest_grid_point(finest.grid, pc) for pc in point_coords)

    fig = go.Figure()

    depth_ref = np.where(finest.mask_keep_time[0], finest.depth[0], np.nan)

    lon_1d = finest.grid.lon if np.ndim(finest.grid.lon) == 1 else finest.grid.lon[0, :]
    lat_1d = finest.grid.lat if np.ndim(finest.grid.lat) == 1 else finest.grid.lat[:, 0]

    contour_traces = _build_depth_contour_traces(
        depth_2d=depth_ref,
        lon=lon_1d,
        lat=lat_1d,
        contour_levels=[20, 50, 100, 200],
        label_levels=[],
    )
    for tr in contour_traces:
        fig.add_trace(tr)

    poi_lons = [poi.lon for poi in pois]
    poi_lats = [poi.lat for poi in pois]
    poi_labels = [str(i + 1) for i in range(len(pois))]
    poi_hover = [
        f"POI {i + 1}<br>lon={poi.lon:.6f}<br>lat={poi.lat:.6f}<extra></extra>"
        for i, poi in enumerate(pois)
    ]

    fig.add_trace(
        go.Scatter(
            x=poi_lons,
            y=poi_lats,
            mode="markers+text",
            text=poi_labels,
            textposition="middle center",
            marker=dict(color="red", size=12, symbol="circle"),
            textfont=dict(color="white", size=10),
            hovertemplate=poi_hover,
            showlegend=False,
            name="POIs",
        )
    )

    lon_min = min(float(np.nanmin(r.grid.lon)) for r in runs)
    lon_max = max(float(np.nanmax(r.grid.lon)) for r in runs)
    lat_min = min(float(np.nanmin(r.grid.lat)) for r in runs)
    lat_max = max(float(np.nanmax(r.grid.lat)) for r in runs)

    fig.update_layout(
        title="POI overview and bathymetry",
        margin=dict(l=40, r=20, t=60, b=40),
    )
    fig.update_xaxes(title_text="Longitude", range=[lon_min, lon_max], fixedrange=True)
    fig.update_yaxes(title_text="Latitude", range=[lat_min, lat_max], fixedrange=True)

    return fig


def prepare_outputs_payload(
    datasets: list[tuple[str, xr.Dataset]],
    point_coord: tuple[float, float] | tuple[tuple[float, float], ...],
) -> OutputsPayload:
    runs = _prepare_runs_data(datasets)
    finest = runs[-1]
    point_coords = _normalize_point_coords(point_coord)
    pois = tuple(_find_nearest_grid_point(finest.grid, pc) for pc in point_coords)

    dataset_by_label = {label: ds for label, ds in datasets}
    run_tps_list: list[np.ndarray] = []
    for run in runs:
        tps = _prepare_tps(dataset_by_label[run.label])
        run_tps_list.append(_align_time_count(tps, run.hs.shape[0]))
    finest_tps = run_tps_list[-1]

    hs_min = float(min(np.nanmin(np.where(r.mask_keep_time, r.hs, np.nan)) for r in runs))
    hs_max = float(max(np.nanmax(np.where(r.mask_keep_time, r.hs, np.nan)) for r in runs))
    tp_min = float(min(np.nanmin(np.where(r.mask_keep_time, r.tp, np.nan)) for r in runs))
    tp_max = float(max(np.nanmax(np.where(r.mask_keep_time, r.tp, np.nan)) for r in runs))
    tps_min = float(min(np.nanmin(np.where(r.mask_keep_time, tps, np.nan)) for r, tps in zip(runs, run_tps_list)))
    tps_max = float(max(np.nanmax(np.where(r.mask_keep_time, tps, np.nan)) for r, tps in zip(runs, run_tps_list)))

    hs_min, hs_max = _normalize_colorscale_limits(hs_min, hs_max)
    tp_min, tp_max = _normalize_colorscale_limits(tp_min, tp_max)
    tps_min, tps_max = _normalize_colorscale_limits(tps_min, tps_max)

    poi_hs_series_list = [finest.hs[:, poi.iy, poi.ix] for poi in pois]
    poi_hs_concat = np.concatenate([np.asarray(v) for v in poi_hs_series_list]) if poi_hs_series_list else np.array([0.0])
    ts_y_min = float(np.nanmin(poi_hs_concat))
    ts_y_max = float(np.nanmax(poi_hs_concat))
    if not np.isfinite(ts_y_min) or not np.isfinite(ts_y_max) or np.isclose(ts_y_min, ts_y_max):
        ts_y_min, ts_y_max = _normalize_colorscale_limits(
            ts_y_min if np.isfinite(ts_y_min) else 0.0,
            ts_y_max if np.isfinite(ts_y_max) else 1.0,
        )

    return OutputsPayload(
        runs=runs,
        times=finest.grid.times,
        pois=pois,
        run_tps_list=run_tps_list,
        finest_tps=finest_tps,
        hs_min=hs_min,
        hs_max=hs_max,
        tp_min=tp_min,
        tp_max=tp_max,
        tps_min=tps_min,
        tps_max=tps_max,
        ts_y_min=ts_y_min,
        ts_y_max=ts_y_max,
    )


def prepare_inputs_payload(
    datasets: list[tuple[str, xr.Dataset]],
    point_coord: tuple[float, float] | tuple[tuple[float, float], ...],
) -> InputsPayload:
    runs = _prepare_runs_data(datasets)
    finest = runs[-1]
    point_coords = _normalize_point_coords(point_coord)
    pois = tuple(_find_nearest_grid_point(finest.grid, pc) for pc in point_coords)

    composite = build_composite_fields(runs, datasets)
    has_wind = all(r.wind_speed is not None for r in runs)

    ws_min = ws_max = 0.0
    if has_wind:
        ws_min = float(min(np.nanmin(np.where(r.mask_keep_time, r.wind_speed, np.nan)) for r in runs if r.wind_speed is not None))
        ws_max = float(max(np.nanmax(np.where(r.mask_keep_time, r.wind_speed, np.nan)) for r in runs if r.wind_speed is not None))
        ws_min, ws_max = _normalize_colorscale_limits(ws_min, ws_max)

    cur_min = cur_max = 0.0
    if composite.current_speed is not None:
        cur_min = float(np.nanmin(composite.current_speed))
        cur_max = float(np.nanmax(composite.current_speed))
        cur_min, cur_max = _normalize_colorscale_limits(cur_min, cur_max)

    ssh_min = ssh_max = 0.0
    if composite.ssh is not None:
        ssh_min = float(np.nanmin(composite.ssh))
        ssh_max = float(np.nanmax(composite.ssh))
        ssh_min, ssh_max = _normalize_colorscale_limits(ssh_min, ssh_max)

    return InputsPayload(
        runs=runs,
        times=finest.grid.times,
        pois=pois,
        composite=composite,
        has_wind=has_wind,
        ws_min=ws_min,
        ws_max=ws_max,
        cur_min=cur_min,
        cur_max=cur_max,
        ssh_min=ssh_min,
        ssh_max=ssh_max,
    )

def build_outputs_figure(
    datasets: list[tuple[str, xr.Dataset]],
    point_coord: tuple[float, float] | tuple[tuple[float, float], ...],
    wind_arrow_resolution: int | None,
) -> go.Figure:
    payload = prepare_outputs_payload(datasets=datasets, point_coord=point_coord)
    runs = payload.runs
    finest = runs[-1]
    nt = finest.hs.shape[0]
    times = payload.times
    pois = payload.pois
    run_tps_list = payload.run_tps_list
    finest_tps = payload.finest_tps
    hs_min, hs_max = payload.hs_min, payload.hs_max
    tp_min, tp_max = payload.tp_min, payload.tp_max
    tps_min, tps_max = payload.tps_min, payload.tps_max

    default_colorscale = "Jet"

    # ------------------------------------------------------------------
    # POI time series
    # ------------------------------------------------------------------
    poi_hs_series_list = [finest.hs[:, poi.iy, poi.ix] for poi in pois]
    poi_tp_series_list = [finest.tp[:, poi.iy, poi.ix] for poi in pois]
    poi_tps_series_list = [finest_tps[:, poi.iy, poi.ix] for poi in pois]

    ts_y_min, ts_y_max = payload.ts_y_min, payload.ts_y_max

    # ------------------------------------------------------------------
    # Figure layout
    # ------------------------------------------------------------------
    subplot_specs: list[list[dict[str, object]]] = [
        [{"type": "xy", "secondary_y": True}, {"type": "xy"}],
        [{"type": "xy"}, {"type": "xy"}],
    ]
    fig = make_subplots(
        rows=2,
        cols=2,
        specs=subplot_specs,
        subplot_titles=(
            "Point of interest",
            "TP",
            "HS",
            "TPS (Smoothed peak period)",
        ),
        horizontal_spacing=0.16,
        vertical_spacing=0.10,
    )

    # ------------------------------------------------------------------
    # Heatmaps
    # ------------------------------------------------------------------
    hs_trace_idxs: list[int] = []
    tp_trace_idxs: list[int] = []
    tps_trace_idxs: list[int] = []

    for idx, (run, tps) in enumerate(zip(runs, run_tps_list)):
        hs0 = np.where(run.mask_keep_time[0], run.hs[0], np.nan)
        tp0 = np.where(run.mask_keep_time[0], run.tp[0], np.nan)
        tps0 = np.where(run.mask_keep_time[0], tps[0], np.nan)

        fig.add_trace(
            go.Heatmap(
                z=tp0,
                x=run.grid.lon,
                y=run.grid.lat,
                zmin=tp_min,
                zmax=tp_max,
                colorscale=default_colorscale,
                colorbar=dict(title="TP [s]", len=0.35, x=1.04, y=0.79),
                showscale=(idx == 0),
            ),
            row=1,
            col=2,
        )
        tp_trace_idxs.append(len(fig.data) - 1)

        fig.add_trace(
            go.Heatmap(
                z=hs0,
                x=run.grid.lon,
                y=run.grid.lat,
                zmin=hs_min,
                zmax=hs_max,
                colorscale=default_colorscale,
                colorbar=dict(title="HS [m]", len=0.35, x=0.41, y=0.21),
                showscale=(idx == 0),
            ),
            row=2,
            col=1,
        )
        hs_trace_idxs.append(len(fig.data) - 1)

        fig.add_trace(
            go.Heatmap(
                z=tps0,
                x=run.grid.lon,
                y=run.grid.lat,
                zmin=tps_min,
                zmax=tps_max,
                colorscale=default_colorscale,
                colorbar=dict(title="TPS [s]", len=0.35, x=1.04, y=0.21),
                showscale=(idx == 0),
            ),
            row=2,
            col=2,
        )
        tps_trace_idxs.append(len(fig.data) - 1)

    # ------------------------------------------------------------------
    # Direction arrows on TP / HS / TPS
    # ------------------------------------------------------------------
    hs_arrow_trace_idxs: list[int] = []
    tp_arrow_trace_idxs: list[int] = []
    tps_arrow_trace_idxs: list[int] = []

    for run in runs:
        if run.wave_from is not None and wind_arrow_resolution is not None:
            hs_ax0, hs_ay0 = _build_wind_arrow_frame_data(
                np.where(run.mask_keep_time[0], run.wave_from[0], np.nan),
                run.grid.lon2d,
                run.grid.lat2d,
                wind_arrow_resolution,
            )
            tp_ax0, tp_ay0 = _build_wind_arrow_frame_data(
                np.where(run.mask_keep_time[0], run.wave_from[0], np.nan),
                run.grid.lon2d,
                run.grid.lat2d,
                wind_arrow_resolution,
            )
            tps_ax0, tps_ay0 = _build_wind_arrow_frame_data(
                np.where(run.mask_keep_time[0], run.wave_from[0], np.nan),
                run.grid.lon2d,
                run.grid.lat2d,
                wind_arrow_resolution,
            )
        else:
            hs_ax0, hs_ay0 = np.array([]), np.array([])
            tp_ax0, tp_ay0 = np.array([]), np.array([])
            tps_ax0, tps_ay0 = np.array([]), np.array([])

        fig.add_trace(
            go.Scatter(
                x=hs_ax0,
                y=hs_ay0,
                mode="lines",
                line=dict(color="black", width=1.1),
                hoverinfo="skip",
                showlegend=False,
                name=f"HS direction ({run.label})",
            ),
            row=2,
            col=1,
        )
        hs_arrow_trace_idxs.append(len(fig.data) - 1)

        fig.add_trace(
            go.Scatter(
                x=tp_ax0,
                y=tp_ay0,
                mode="lines",
                line=dict(color="black", width=1.1),
                hoverinfo="skip",
                showlegend=False,
                name=f"TP direction ({run.label})",
            ),
            row=1,
            col=2,
        )
        tp_arrow_trace_idxs.append(len(fig.data) - 1)

        fig.add_trace(
            go.Scatter(
                x=tps_ax0,
                y=tps_ay0,
                mode="lines",
                line=dict(color="black", width=1.1),
                hoverinfo="skip",
                showlegend=False,
                name=f"TPS direction ({run.label})",
            ),
            row=2,
            col=2,
        )
        tps_arrow_trace_idxs.append(len(fig.data) - 1)

    # ------------------------------------------------------------------
    # POI markers on maps
    # ------------------------------------------------------------------
    poi_marker_style = dict(
        mode="markers+text",
        marker=dict(color="red", size=12, symbol="circle"),
        textposition="middle center",
        textfont=dict(color="white", size=10),
        name="Point of interest",
        showlegend=False,
    )
    poi_lons = [poi.lon for poi in pois]
    poi_lats = [poi.lat for poi in pois]
    poi_labels = [str(i + 1) for i in range(len(pois))]
    poi_hover = [
        f"POI {i + 1}<br>lon={poi.lon:.6f}<br>lat={poi.lat:.6f}<extra></extra>"
        for i, poi in enumerate(pois)
    ]

    fig.add_trace(
        go.Scatter(x=poi_lons, y=poi_lats, text=poi_labels, hovertemplate=poi_hover, **poi_marker_style),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(x=poi_lons, y=poi_lats, text=poi_labels, hovertemplate=poi_hover, **poi_marker_style),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=poi_lons, y=poi_lats, text=poi_labels, hovertemplate=poi_hover, **poi_marker_style),
        row=2,
        col=2,
    )

    # ------------------------------------------------------------------
    # Time series panel (top-left) with table split
    # ------------------------------------------------------------------
    ts_x_domain = [float(v) for v in fig.layout.xaxis.domain]
    ts_y_domain = [float(v) for v in fig.layout.yaxis.domain]
    split_gap = 0.02 * (ts_y_domain[1] - ts_y_domain[0])
    split_y = ts_y_domain[0] + 0.53 * (ts_y_domain[1] - ts_y_domain[0])
    table_domain = {"x": ts_x_domain, "y": [split_y + split_gap, ts_y_domain[1]]}
    ts_plot_domain = [ts_y_domain[0], split_y - split_gap]

    fig.layout.yaxis.domain = ts_plot_domain
    fig.layout.xaxis.anchor = "y"

    table_trace_idx = len(fig.data)
    fig.add_trace(
        _build_table_trace(
            time_text=str(times[0]),
            pois=pois,
            rows=[
                ("HS [m]", tuple(float(finest.hs[0, poi.iy, poi.ix]) for poi in pois)),
                ("TP [s]", tuple(float(finest.tp[0, poi.iy, poi.ix]) for poi in pois)),
                ("TPS [s]", tuple(float(finest_tps[0, poi.iy, poi.ix]) for poi in pois)),
            ],
            domain=table_domain,
        ),
    )

    # HS lines
    for i, poi_hs_series in enumerate(poi_hs_series_list):
        fig.add_trace(
            go.Scatter(
                x=times,
                y=poi_hs_series,
                mode="lines",
                line=dict(color="royalblue", width=2),
                name=f"HS P{i + 1} [m] (left axis)",
                showlegend=True,
                hovertemplate=f"POI {i + 1}<br>Time=%{{x}}<br>HS=%{{y:.3f}} m<extra></extra>",
            ),
            row=1,
            col=1,
        )

    hs_timeseries_marker_trace_idx = len(fig.data)
    fig.add_trace(
        _build_hs_timeseries_marker_trace(times, poi_hs_series_list[0], time_idx=0),
        row=1,
        col=1,
    )

    # TP lines
    for i, poi_tp_series in enumerate(poi_tp_series_list):
        fig.add_trace(
            go.Scatter(
                x=times,
                y=poi_tp_series,
                mode="lines",
                line=dict(color="darkorange", width=2, dash="dot"),
                name=f"TP P{i + 1} [s] (right axis)",
                showlegend=True,
                hovertemplate=f"POI {i + 1}<br>Time=%{{x}}<br>TP=%{{y:.3f}} s<extra></extra>",
            ),
            row=1,
            col=1,
            secondary_y=True,
        )

    tp_timeseries_marker_trace_idx = len(fig.data)
    fig.add_trace(
        _build_tp_timeseries_marker_trace(times, poi_tp_series_list[0], time_idx=0),
        row=1,
        col=1,
        secondary_y=True,
    )

    # TPS lines on same right axis, dashed differently
    for i, poi_tps_series in enumerate(poi_tps_series_list):
        fig.add_trace(
            go.Scatter(
                x=times,
                y=poi_tps_series,
                mode="lines",
                line=dict(color="darkgreen", width=2, dash="dash"),
                name=f"TPS P{i + 1} [s] (right axis)",
                showlegend=True,
                hovertemplate=f"POI {i + 1}<br>Time=%{{x}}<br>TPS=%{{y:.3f}} s<extra></extra>",
            ),
            row=1,
            col=1,
            secondary_y=True,
        )

    tps_marker_trace_idx = len(fig.data)
    fig.add_trace(
        go.Scatter(
            x=[times[0]],
            y=[poi_tps_series_list[0][0]],
            mode="markers",
            marker=dict(color="darkgreen", size=10, symbol="square"),
            cliponaxis=False,
            name="Current TPS",
            showlegend=False,
            hovertemplate="Time=%{x}<br>TPS=%{y:.3f} s<extra></extra>",
        ),
        row=1,
        col=1,
        secondary_y=True,
    )

    time_cursor_trace_idx = len(fig.data)
    fig.add_trace(
        _build_time_cursor_trace(times, ts_y_min, ts_y_max, time_idx=0),
        row=1,
        col=1,
    )

    # ------------------------------------------------------------------
    # Dropdown controls
    # ------------------------------------------------------------------
    all_heatmap_trace_idxs = hs_trace_idxs + tp_trace_idxs + tps_trace_idxs
    all_heatmap_zmin: list[float] = (
        [hs_min] * len(hs_trace_idxs)
        + [tp_min] * len(tp_trace_idxs)
        + [tps_min] * len(tps_trace_idxs)
    )
    all_heatmap_zmax_full: list[float] = (
        [hs_max] * len(hs_trace_idxs)
        + [tp_max] * len(tp_trace_idxs)
        + [tps_max] * len(tps_trace_idxs)
    )

    percentage_options = [
        ("Max: 20%", 0.20),
        ("Max: 40%", 0.40),
        ("Max: 60%", 0.60),
        ("Max: 80%", 0.80),
        ("Max: 100%", 1.00),
    ]
    percentage_buttons = []
    for label, frac in percentage_options:
        zmax_values = [max(v * frac, 1e-12) for v in all_heatmap_zmax_full]
        percentage_buttons.append(
            dict(
                label=label,
                method="restyle",
                args=[
                    {"zmin": all_heatmap_zmin, "zmax": zmax_values},
                    all_heatmap_trace_idxs,
                ],
            )
        )

    colorscale_options = [
        ("Scale: Jet", "Jet"),
        ("Scale: Viridis", "Viridis"),
        ("Scale: Plasma", "Plasma"),
        ("Scale: Cividis", "Cividis"),
        ("Scale: Turbo", "Turbo"),
    ]
    colorscale_buttons = [
        dict(
            label=label,
            method="restyle",
            args=[{"colorscale": [scale] * len(all_heatmap_trace_idxs)}, all_heatmap_trace_idxs],
        )
        for label, scale in colorscale_options
    ]

    # ------------------------------------------------------------------
    # Animated trace indices
    # ------------------------------------------------------------------
    dynamic_trace_indices = (
        tp_trace_idxs
        + hs_trace_idxs
        + tps_trace_idxs
        + tp_arrow_trace_idxs
        + hs_arrow_trace_idxs
        + tps_arrow_trace_idxs
        + [table_trace_idx, hs_timeseries_marker_trace_idx, tp_timeseries_marker_trace_idx, tps_marker_trace_idx, time_cursor_trace_idx]
    )

    # ------------------------------------------------------------------
    # Frames
    # ------------------------------------------------------------------
    frames: list[go.Frame] = []
    for i in range(nt):
        frame_data: list[go.BaseTraceType] = []

        # TP heatmaps
        for run in runs:
            frame_data.append(
                go.Heatmap(
                    z=np.where(run.mask_keep_time[i], run.tp[i], np.nan),
                    x=run.grid.lon,
                    y=run.grid.lat,
                )
            )

        # HS heatmaps
        for run in runs:
            frame_data.append(
                go.Heatmap(
                    z=np.where(run.mask_keep_time[i], run.hs[i], np.nan),
                    x=run.grid.lon,
                    y=run.grid.lat,
                )
            )

        # TPS heatmaps
        for run, tps in zip(runs, run_tps_list):
            frame_data.append(
                go.Heatmap(
                    z=np.where(run.mask_keep_time[i], tps[i], np.nan),
                    x=run.grid.lon,
                    y=run.grid.lat,
                )
            )

        # TP arrows
        for run in runs:
            if run.wave_from is not None and wind_arrow_resolution is not None:
                tp_axi, tp_ayi = _build_wind_arrow_frame_data(
                    np.where(run.mask_keep_time[i], run.wave_from[i], np.nan),
                    run.grid.lon2d,
                    run.grid.lat2d,
                    wind_arrow_resolution,
                )
            else:
                tp_axi, tp_ayi = np.array([]), np.array([])
            frame_data.append(
                go.Scatter(
                    x=tp_axi,
                    y=tp_ayi,
                    mode="lines",
                    line=dict(color="black", width=1.1),
                    hoverinfo="skip",
                )
            )

        # HS arrows
        for run in runs:
            if run.wave_from is not None and wind_arrow_resolution is not None:
                hs_axi, hs_ayi = _build_wind_arrow_frame_data(
                    np.where(run.mask_keep_time[i], run.wave_from[i], np.nan),
                    run.grid.lon2d,
                    run.grid.lat2d,
                    wind_arrow_resolution,
                )
            else:
                hs_axi, hs_ayi = np.array([]), np.array([])
            frame_data.append(
                go.Scatter(
                    x=hs_axi,
                    y=hs_ayi,
                    mode="lines",
                    line=dict(color="black", width=1.1),
                    hoverinfo="skip",
                )
            )

        # TPS arrows
        for run in runs:
            if run.wave_from is not None and wind_arrow_resolution is not None:
                tps_axi, tps_ayi = _build_wind_arrow_frame_data(
                    np.where(run.mask_keep_time[i], run.wave_from[i], np.nan),
                    run.grid.lon2d,
                    run.grid.lat2d,
                    wind_arrow_resolution,
                )
            else:
                tps_axi, tps_ayi = np.array([]), np.array([])
            frame_data.append(
                go.Scatter(
                    x=tps_axi,
                    y=tps_ayi,
                    mode="lines",
                    line=dict(color="black", width=1.1),
                    hoverinfo="skip",
                )
            )

        frame_data.append(
            _build_table_trace(
                time_text=str(times[i]),
                pois=pois,
                rows=[
                    ("HS [m]", tuple(float(finest.hs[i, poi.iy, poi.ix]) for poi in pois)),
                    ("TP [s]", tuple(float(finest.tp[i, poi.iy, poi.ix]) for poi in pois)),
                    ("TPS [s]", tuple(float(finest_tps[i, poi.iy, poi.ix]) for poi in pois)),
                ],
                domain=table_domain,
            )
        )
        frame_data.append(_build_hs_timeseries_marker_trace(times, poi_hs_series_list[0], time_idx=i))
        frame_data.append(_build_tp_timeseries_marker_trace(times, poi_tp_series_list[0], time_idx=i))
        frame_data.append(
            go.Scatter(
                x=[times[i]],
                y=[poi_tps_series_list[0][i]],
                mode="markers",
                marker=dict(color="darkgreen", size=10, symbol="square"),
                cliponaxis=False,
                showlegend=False,
                hovertemplate="Time=%{x}<br>TPS=%{y:.3f} s<extra></extra>",
            )
        )
        frame_data.append(_build_time_cursor_trace(times, ts_y_min, ts_y_max, time_idx=i))

        frames.append(go.Frame(data=frame_data, traces=dynamic_trace_indices, name=str(i)))

    fig.frames = frames

    # ------------------------------------------------------------------
    # Slider
    # ------------------------------------------------------------------
    steps = [
        dict(
            method="animate",
            args=[[str(i)], dict(mode="immediate", frame=dict(duration=0, redraw=True), transition=dict(duration=0))],
            label=(str(times[i])[11:16] if "T" in str(times[i]) else str(times[i])),
        )
        for i in range(nt)
    ]

    fig.update_layout(
        title="SWAN outputs: Hs / Tp / Tps / POI time series",
        legend=dict(
            orientation="h",
            x=ts_x_domain[0] + 0.01,
            y=ts_plot_domain[0] - 0.035,
            xanchor="left",
            yanchor="top",
            bgcolor="rgba(255,255,255,0.7)",
            borderwidth=0,
        ),
        margin=dict(l=40, r=40, t=145, b=95),
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                x=0.02,
                y=-0.055,
                xanchor="left",
                yanchor="top",
                pad=dict(r=8, t=0),
                buttons=[
                    dict(
                        label="Play",
                        method="animate",
                        args=[None, dict(frame=dict(duration=250, redraw=True), fromcurrent=True, transition=dict(duration=0))]
                    ),
                    dict(
                        label="Pause",
                        method="animate",
                        args=[[None], dict(frame=dict(duration=0, redraw=True), mode="immediate", transition=dict(duration=0))]
                    ),
                ],
            ),
            dict(
                type="dropdown",
                direction="down",
                x=0.02,
                y=1.14,
                xanchor="left",
                yanchor="top",
                buttons=percentage_buttons,
                showactive=True,
                active=4,
            ),
            dict(
                type="dropdown",
                direction="down",
                x=0.19,
                y=1.14,
                xanchor="left",
                yanchor="top",
                buttons=colorscale_buttons,
                showactive=True,
                active=0,
            ),
        ],
        sliders=[
            dict(
                active=0,
                x=0.16,
                y=-0.055,
                len=0.78,
                xanchor="left",
                yanchor="top",
                pad=dict(t=0, b=0),
                steps=steps,
            )
        ],
    )

    # Axis titles
    fig.update_xaxes(title_text="Time", row=1, col=1)
    fig.update_yaxes(
        title_text="HS [m] (blue, left axis)",
        title_font=dict(color="royalblue"),
        tickfont=dict(color="royalblue"),
        row=1,
        col=1,
        secondary_y=False,
    )
    fig.update_yaxes(
        title_text="TP / TPS [s] (right axis)",
        title_font=dict(color="darkorange"),
        tickfont=dict(color="darkorange"),
        showgrid=False,
        zeroline=False,
        showline=True,
        ticks="outside",
        row=1,
        col=1,
        secondary_y=True,
    )

    fig.update_xaxes(title_text="Longitude", row=1, col=2)
    fig.update_yaxes(title_text="Latitude", row=1, col=2)
    fig.update_xaxes(title_text="Longitude", row=2, col=1)
    fig.update_yaxes(title_text="Latitude", row=2, col=1)
    fig.update_xaxes(title_text="Longitude", row=2, col=2)
    fig.update_yaxes(title_text="Latitude", row=2, col=2)

    lon_min = min(float(np.nanmin(r.grid.lon)) for r in runs)
    lon_max = max(float(np.nanmax(r.grid.lon)) for r in runs)
    lat_min = min(float(np.nanmin(r.grid.lat)) for r in runs)
    lat_max = max(float(np.nanmax(r.grid.lat)) for r in runs)

    fig.update_xaxes(range=[lon_min, lon_max], row=1, col=2)
    fig.update_xaxes(range=[lon_min, lon_max], row=2, col=1)
    fig.update_xaxes(range=[lon_min, lon_max], row=2, col=2)
    fig.update_yaxes(range=[lat_min, lat_max], row=1, col=2)
    fig.update_yaxes(range=[lat_min, lat_max], row=2, col=1)
    fig.update_yaxes(range=[lat_min, lat_max], row=2, col=2)

    return fig

def build_inputs_figure(
    datasets: list[tuple[str, xr.Dataset]],
    point_coord: tuple[float, float] | tuple[tuple[float, float], ...],
    wind_arrow_resolution: int | None,
) -> go.Figure:
    payload = prepare_inputs_payload(datasets=datasets, point_coord=point_coord)
    runs = payload.runs
    finest = runs[-1]
    composite = payload.composite
    nt = finest.hs.shape[0]
    times = payload.times
    pois = payload.pois
    has_wind = payload.has_wind
    ws_min, ws_max = payload.ws_min, payload.ws_max
    cur_min, cur_max = payload.cur_min, payload.cur_max
    ssh_min, ssh_max = payload.ssh_min, payload.ssh_max

    current_grid = composite.grid
    current_from_grid = composite.grid
    ssh_grid = composite.grid
    current_cube = composite.current_speed
    current_from_cube = composite.current_from
    ssh_cube = composite.ssh

    fig = make_subplots(
        rows=2,
        cols=2,
        specs=[
            [{"type": "domain"}, {"type": "xy"}],
            [{"type": "xy"}, {"type": "xy"}],
        ],
        subplot_titles=(
            "Input overview",
            "Wind speed",
            "Current speed",
            "Sea surface elevation",
        ),
        horizontal_spacing=0.10,
        vertical_spacing=0.12,
    )

    poi_lons = [poi.lon for poi in pois]
    poi_lats = [poi.lat for poi in pois]
    poi_labels = [str(i + 1) for i in range(len(pois))]
    poi_hover = [
        f"POI {i + 1}<br>lon={poi.lon:.6f}<br>lat={poi.lat:.6f}<extra></extra>"
        for i, poi in enumerate(pois)
    ]
    poi_marker_style = dict(
        mode="markers+text",
        marker=dict(color="red", size=12, symbol="circle"),
        textposition="middle center",
        textfont=dict(color="white", size=10),
        showlegend=False,
        hovertemplate=poi_hover,
        name="POIs",
    )

    time_start = str(times[0]) if times.size else "N/A"
    time_end = str(times[-1]) if times.size else "N/A"

    lon_min = min(float(np.nanmin(r.grid.lon)) for r in runs)
    lon_max = max(float(np.nanmax(r.grid.lon)) for r in runs)
    lat_min = min(float(np.nanmin(r.grid.lat)) for r in runs)
    lat_max = max(float(np.nanmax(r.grid.lat)) for r in runs)

    grid_lines = []
    for i, run in enumerate(runs, start=1):
        ny = run.grid.lat.size
        nx = run.grid.lon.size
        grid_lines.append(f"Run {i} ({run.label}): {ny} x {nx}")

    poi_lines = []
    for i, poi in enumerate(pois, start=1):
        poi_lines.append(f"POI {i}: lat={poi.lat:.6f}, lon={poi.lon:.6f}")

    overview_text = (
        "<b>Included inputs</b><br>"
        f"Time range: {time_start} → {time_end}<br>"
        f"Number of time steps: {nt}<br>"
        f"Domain: lon {lon_min:.4f} to {lon_max:.4f}, lat {lat_min:.4f} to {lat_max:.4f}<br>"
        f"Wind available: {'Yes' if has_wind else 'No'}<br>"
        f"Current available: {'Yes' if current_cube is not None else 'No'}<br>"
        f"SSH available: {'Yes' if ssh_cube is not None else 'No'}<br>"
        f"Arrow resolution: {wind_arrow_resolution if wind_arrow_resolution is not None else 'Off'}<br><br>"
        "<b>Nested runs</b><br>"
        + "<br>".join(grid_lines)
        + "<br><br><b>Points of interest</b><br>"
        + "<br>".join(poi_lines)
    )

    fig.add_trace(
        go.Table(
            header=dict(
                values=["Summary"],
                fill_color="lightgrey",
                align="left",
                font=dict(size=14),
            ),
            cells=dict(
                values=[[overview_text]],
                align="left",
                height=260,
                font=dict(size=13),
            ),
            domain=dict(x=[0.00, 0.44], y=[0.56, 1.00]),
        )
    )

    default_colorscale = "Jet"

    wind_heatmap_trace_idxs: list[int] = []
    wind_arrow_trace_idxs: list[int] = []
    current_trace_idx: int | None = None
    current_arrow_trace_idx: int | None = None
    ssh_trace_idx: int | None = None

    # Top-right: Wind
    if has_wind:
        for idx, run in enumerate(runs):
            ws0 = np.where(run.mask_keep_time[0], run.wind_speed[0], np.nan)
            fig.add_trace(
                go.Heatmap(
                    z=ws0,
                    x=run.grid.lon,
                    y=run.grid.lat,
                    zmin=ws_min,
                    zmax=ws_max,
                    colorscale=default_colorscale,
                    colorbar=dict(title="Wind [m/s]", len=0.30, x=1.03, y=0.79),
                    showscale=(idx == 0),
                ),
                row=1,
                col=2,
            )
            wind_heatmap_trace_idxs.append(len(fig.data) - 1)

            if run.wind_from is not None and wind_arrow_resolution is not None:
                wx0, wy0 = _build_wind_arrow_frame_data(
                    np.where(run.mask_keep_time[0], run.wind_from[0], np.nan),
                    run.grid.lon2d,
                    run.grid.lat2d,
                    wind_arrow_resolution,
                )
            else:
                wx0, wy0 = np.array([]), np.array([])

            fig.add_trace(
                go.Scatter(
                    x=wx0,
                    y=wy0,
                    mode="lines",
                    line=dict(color="black", width=1.3),
                    hoverinfo="skip",
                    showlegend=False,
                ),
                row=1,
                col=2,
            )
            wind_arrow_trace_idxs.append(len(fig.data) - 1)

        fig.add_trace(
            go.Scatter(x=poi_lons, y=poi_lats, text=poi_labels, **poi_marker_style),
            row=1,
            col=2,
        )
    else:
        fig.add_annotation(
            text="Wind not available in input dataset",
            xref="x domain",
            yref="y domain",
            x=0.78,
            y=0.78,
            showarrow=False,
            font=dict(color="gray", size=13),
        )

    # Bottom-left: Current
    if current_cube is not None:
        fig.add_trace(
            go.Heatmap(
                z=current_cube[0],
                x=current_grid.lon,
                y=current_grid.lat,
                zmin=cur_min,
                zmax=cur_max,
                colorscale=default_colorscale,
                colorbar=dict(title="Current [m/s]", len=0.30, x=0.47, y=0.21),
                showscale=True,
            ),
            row=2,
            col=1,
        )
        current_trace_idx = len(fig.data) - 1

        if current_from_cube is not None and wind_arrow_resolution is not None:
            cax0, cay0 = _build_wind_arrow_frame_data(
                current_from_cube[0],
                current_grid.lon2d,
                current_grid.lat2d,
                wind_arrow_resolution,
            )
        else:
            cax0, cay0 = np.array([]), np.array([])

        fig.add_trace(
            go.Scatter(
                x=cax0,
                y=cay0,
                mode="lines",
                line=dict(color="black", width=1.1),
                hoverinfo="skip",
                showlegend=False,
            ),
            row=2,
            col=1,
        )
        current_arrow_trace_idx = len(fig.data) - 1

        fig.add_trace(
            go.Scatter(x=poi_lons, y=poi_lats, text=poi_labels, **poi_marker_style),
            row=2,
            col=1,
        )
    else:
        fig.add_annotation(
            text="Current not available in input dataset",
            xref="x2 domain",
            yref="y2 domain",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(color="gray", size=13),
        )

    # Bottom-right: SSH
    if ssh_cube is not None:
        fig.add_trace(
            go.Heatmap(
                z=ssh_cube[0],
                x=ssh_grid.lon,
                y=ssh_grid.lat,
                zmin=ssh_min,
                zmax=ssh_max,
                colorscale=default_colorscale,
                colorbar=dict(title="SSH [m]", len=0.30, x=1.03, y=0.21),
                showscale=True,
            ),
            row=2,
            col=2,
        )
        ssh_trace_idx = len(fig.data) - 1

        fig.add_trace(
            go.Scatter(x=poi_lons, y=poi_lats, text=poi_labels, **poi_marker_style),
            row=2,
            col=2,
        )
    else:
        fig.add_annotation(
            text="SSH not available in input dataset",
            xref="x3 domain",
            yref="y3 domain",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(color="gray", size=13),
        )

    # animated trace indices
    dynamic_trace_indices: list[int] = []
    dynamic_trace_indices.extend(wind_heatmap_trace_idxs)
    dynamic_trace_indices.extend(wind_arrow_trace_idxs)
    if current_trace_idx is not None:
        dynamic_trace_indices.append(current_trace_idx)
    if current_arrow_trace_idx is not None:
        dynamic_trace_indices.append(current_arrow_trace_idx)
    if ssh_trace_idx is not None:
        dynamic_trace_indices.append(ssh_trace_idx)

    frames: list[go.Frame] = []
    for i in range(nt):
        frame_data: list[go.BaseTraceType] = []

        if has_wind:
            for run in runs:
                ws_i = np.where(run.mask_keep_time[i], run.wind_speed[i], np.nan)
                frame_data.append(
                    go.Heatmap(
                        z=ws_i,
                        x=run.grid.lon,
                        y=run.grid.lat,
                    )
                )

            for run in runs:
                if run.wind_from is not None and wind_arrow_resolution is not None:
                    wx, wy = _build_wind_arrow_frame_data(
                        np.where(run.mask_keep_time[i], run.wind_from[i], np.nan),
                        run.grid.lon2d,
                        run.grid.lat2d,
                        wind_arrow_resolution,
                    )
                else:
                    wx, wy = np.array([]), np.array([])
                frame_data.append(
                    go.Scatter(
                        x=wx,
                        y=wy,
                        mode="lines",
                        line=dict(color="black", width=1.3),
                        hoverinfo="skip",
                        showlegend=False,
                    )
                )

        if current_cube is not None:
            frame_data.append(
                go.Heatmap(
                    z=current_cube[i],
                    x=current_grid.lon,
                    y=current_grid.lat,
                )
            )

        if current_cube is not None:
            if current_from_cube is not None and wind_arrow_resolution is not None:
                cax, cay = _build_wind_arrow_frame_data(
                    current_from_cube[i],
                    current_grid.lon2d,
                    current_grid.lat2d,
                    wind_arrow_resolution,
                )
            else:
                cax, cay = np.array([]), np.array([])
            frame_data.append(
                go.Scatter(
                    x=cax,
                    y=cay,
                    mode="lines",
                    line=dict(color="black", width=1.1),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

        if ssh_cube is not None:
            frame_data.append(
                go.Heatmap(
                    z=ssh_cube[i],
                    x=ssh_grid.lon,
                    y=ssh_grid.lat,
                )
            )

        frames.append(
            go.Frame(
                data=frame_data,
                traces=dynamic_trace_indices,
                name=str(i),
            )
        )

    fig.frames = frames

    # controls for all input heatmaps
    all_input_heatmap_trace_idxs = wind_heatmap_trace_idxs.copy()
    all_input_heatmap_zmin = [ws_min] * len(wind_heatmap_trace_idxs)
    all_input_heatmap_zmax_full = [ws_max] * len(wind_heatmap_trace_idxs)

    if current_trace_idx is not None:
        all_input_heatmap_trace_idxs.append(current_trace_idx)
        all_input_heatmap_zmin.append(cur_min)
        all_input_heatmap_zmax_full.append(cur_max)

    if ssh_trace_idx is not None:
        all_input_heatmap_trace_idxs.append(ssh_trace_idx)
        all_input_heatmap_zmin.append(ssh_min)
        all_input_heatmap_zmax_full.append(ssh_max)

    percentage_options = [
        ("Max: 20%", 0.20),
        ("Max: 40%", 0.40),
        ("Max: 60%", 0.60),
        ("Max: 80%", 0.80),
        ("Max: 100%", 1.00),
    ]
    percentage_buttons = []
    for label, frac in percentage_options:
        zmax_values = [max(v * frac, 1e-12) for v in all_input_heatmap_zmax_full]
        percentage_buttons.append(
            dict(
                label=label,
                method="restyle",
                args=[
                    {"zmin": all_input_heatmap_zmin, "zmax": zmax_values},
                    all_input_heatmap_trace_idxs,
                ],
            )
        )

    colorscale_options = [
        ("Scale: Jet", "Jet"),
        ("Scale: Viridis", "Viridis"),
        ("Scale: Plasma", "Plasma"),
        ("Scale: Cividis", "Cividis"),
        ("Scale: Turbo", "Turbo"),
    ]
    colorscale_buttons = [
        dict(
            label=label,
            method="restyle",
            args=[{"colorscale": [scale] * len(all_input_heatmap_trace_idxs)}, all_input_heatmap_trace_idxs],
        )
        for label, scale in colorscale_options
    ]

    steps = [
        dict(
            method="animate",
            args=[[str(i)], dict(mode="immediate", frame=dict(duration=0, redraw=True), transition=dict(duration=0))],
            label=(str(times[i])[11:16] if "T" in str(times[i]) else str(times[i])),
        )
        for i in range(nt)
    ]

    fig.update_layout(
        title="SWAN post-processing: Inputs overview",
        margin=dict(l=40, r=55, t=125, b=95),
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                x=0.02,
                y=-0.055,
                xanchor="left",
                yanchor="top",
                buttons=[
                    dict(
                        label="Play",
                        method="animate",
                        args=[None, dict(frame=dict(duration=250, redraw=True), fromcurrent=True, transition=dict(duration=0))]
                    ),
                    dict(
                        label="Pause",
                        method="animate",
                        args=[[None], dict(frame=dict(duration=0, redraw=True), mode="immediate", transition=dict(duration=0))]
                    ),
                ],
            ),
            dict(
                type="dropdown",
                direction="down",
                x=0.02,
                y=1.14,
                xanchor="left",
                yanchor="top",
                buttons=percentage_buttons,
                showactive=True,
                active=4,
            ),
            dict(
                type="dropdown",
                direction="down",
                x=0.19,
                y=1.14,
                xanchor="left",
                yanchor="top",
                buttons=colorscale_buttons,
                showactive=True,
                active=0,
            ),
        ],
        sliders=[
            dict(
                active=0,
                x=0.16,
                y=-0.055,
                len=0.78,
                xanchor="left",
                yanchor="top",
                pad=dict(t=0, b=0),
                steps=steps,
            )
        ],
    )

    fig.update_xaxes(title_text="Longitude", row=1, col=2, fixedrange=True)
    fig.update_yaxes(title_text="Latitude", row=1, col=2, fixedrange=True)
    fig.update_xaxes(title_text="Longitude", row=2, col=1, fixedrange=True)
    fig.update_yaxes(title_text="Latitude", row=2, col=1, fixedrange=True)
    fig.update_xaxes(title_text="Longitude", row=2, col=2, fixedrange=True)
    fig.update_yaxes(title_text="Latitude", row=2, col=2, fixedrange=True)

    fig.update_xaxes(range=[lon_min, lon_max], fixedrange=True, row=1, col=2)
    fig.update_yaxes(range=[lat_min, lat_max], fixedrange=True, row=1, col=2)
    fig.update_xaxes(range=[lon_min, lon_max], fixedrange=True, row=2, col=1)
    fig.update_yaxes(range=[lat_min, lat_max], fixedrange=True, row=2, col=1)
    fig.update_xaxes(range=[lon_min, lon_max], fixedrange=True, row=2, col=2)
    fig.update_yaxes(range=[lat_min, lat_max], fixedrange=True, row=2, col=2)

    return fig



def _prepare_direction_axis(direction_raw: np.ndarray, spectrum: np.ndarray):
    import numpy as np

    direction_raw = np.asarray(direction_raw, dtype=float)
    spectrum = np.asarray(spectrum, dtype=float)

    is_rad = np.nanmax(np.abs(direction_raw)) <= (2*np.pi + 0.5)

    if is_rad:
        direction_deg = np.degrees(direction_raw)
        direction_rad = direction_raw.copy()
    else:
        direction_deg = direction_raw.copy()
        direction_rad = np.radians(direction_raw)

    direction_deg = np.mod(direction_deg, 360.0)
    direction_rad = np.mod(direction_rad, 2*np.pi)

    order = np.argsort(direction_deg)
    return direction_deg[order], direction_rad[order], spectrum[..., order]

def export_wave_statistics(spec_file: Path, point_coord: tuple[float, float] | tuple[tuple[float, float], ...] = POINT_COORD) -> None:
    import numpy as np
    import xarray as xr
    import pandas as pd

    ds = xr.open_dataset(spec_file)

    try:
        rows: list[dict[str, float | int]] = []

        for display_idx, spec_point_idx, selected_ds, spec_point_coord in _iter_all_spec_point_datasets(ds):
            _, freq_name, dir_name, da = _detect_spec_axes_and_var(selected_ds)

            freq = da[freq_name].values.astype(float)
            direction_raw = da[dir_name].values.astype(float)
            spectrum = da.values.astype(float)

            direction_deg, direction_rad, spectrum = _prepare_direction_axis(direction_raw, spectrum)

            hs = _compute_hs_series(freq, direction_rad, spectrum)

            mean_spec = np.nanmean(spectrum, axis=0)
            peak_idx = np.unravel_index(np.nanargmax(mean_spec), mean_spec.shape)
            fp = freq[peak_idx[0]]
            Tp = 1.0 / fp
            theta_p = direction_deg[peak_idx[1]]

            row: dict[str, float | int] = {
                "spectral_point_order": int(display_idx),
                "spec_point_index": int(spec_point_idx),
                "Hs_mean": float(np.nanmean(hs)),
                "Hs_max": float(np.nanmax(hs)),
                "Hs_p95": float(np.nanpercentile(hs, 95)),
                "Tp_peak": float(Tp),
                "Dir_peak": float(theta_p),
            }

            if spec_point_coord is not None:
                row["spec_lat"] = float(spec_point_coord[0])
                row["spec_lon"] = float(spec_point_coord[1])

            if "latitude" in selected_ds and "longitude" in selected_ds:
                sel_lat = np.asarray(selected_ds["latitude"].values, dtype=float).reshape(-1)
                sel_lon = np.asarray(selected_ds["longitude"].values, dtype=float).reshape(-1)
                if sel_lat.size:
                    row["spec_lat"] = float(sel_lat[0])
                if sel_lon.size:
                    row["spec_lon"] = float(sel_lon[0])

            rows.append(row)

        df = pd.DataFrame(rows)

        out_file = spec_file.parent / f"{spec_file.stem}_wave_stats.csv"
        df.to_csv(out_file, index=False)

        print(f"Saved: {out_file.resolve()}")

    finally:
        ds.close()

def _compute_hs_series(freq, direction_rad, spectrum):
    import numpy as np
    df = np.gradient(freq)
    dtheta = 2*np.pi / len(direction_rad)

    hs = []
    for i in range(spectrum.shape[0]):
        m0 = np.sum(spectrum[i] * df[:, None] * dtheta)
        hs.append(4*np.sqrt(max(m0, 0.0)))
    return np.array(hs)

def run_directional_spec_plot(inputs: Inputs) -> None:
    import numpy as np
    import xarray as xr
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    if inputs.spec_file is None:
        raise ValueError("A spec file is required.")

    spec_ds = xr.open_dataset(inputs.spec_file)

    try:
        any_written = False

        for display_idx, spec_point_idx, selected, spec_point_coord in _iter_all_spec_point_datasets(spec_ds):
            _, freq_name, dir_name, spec_da = _detect_spec_axes_and_var(selected)

            time_name = spec_da.dims[0]

            freq = np.asarray(spec_da[freq_name].values, dtype=float)
            direction_raw = np.asarray(spec_da[dir_name].values, dtype=float)
            times = np.asarray(spec_da[time_name].values)
            spectrum = np.asarray(spec_da.values, dtype=float)

            direction_deg, direction_rad, spectrum = _prepare_direction_axis(direction_raw, spectrum)

            df = np.gradient(freq)
            dtheta = 2 * np.pi / len(direction_rad)

            def compute_hs(spec2d):
                m0 = np.sum(spec2d * df[:, None] * dtheta)
                return 4 * np.sqrt(max(m0, 0.0))

            hs_series = np.array([compute_hs(spectrum[i]) for i in range(len(times))])

            valid = np.where(np.isfinite(hs_series) & (hs_series > 0))[0]
            if len(valid) == 0:
                print(
                    f"No valid spectra found for spectral point {display_idx + 1} "
                    f"(spec point index {spec_point_idx}). Skipping."
                )
                continue

            idx_max = valid[np.argmax(hs_series[valid])]
            idx_p95 = valid[np.argmin(np.abs(hs_series[valid] - np.percentile(hs_series[valid], 95)))]
            idx_med = valid[np.argmin(np.abs(hs_series[valid] - np.percentile(hs_series[valid], 50)))]
            idx_min = valid[np.argmin(hs_series[valid])]

            selection = [
                ("Max Hs", idx_max),
                ("P95 Hs", idx_p95),
                ("Median Hs", idx_med),
                ("Min Hs", idx_min),
            ]

            fig = make_subplots(
                rows=2,
                cols=2,
                subplot_titles=[s[0] for s in selection],
                horizontal_spacing=0.06,
                vertical_spacing=0.10,
            )

            period = 1.0 / freq

            for i, (label, idx) in enumerate(selection):
                spec2d = np.maximum(spectrum[idx], 1e-12)

                flat_idx = np.argmax(spec2d)
                fi, di = np.unravel_index(flat_idx, spec2d.shape)

                fp = freq[fi]
                Tp = 1.0 / fp
                theta_p = direction_deg[di]
                hs_val = hs_series[idx]

                log_spec = np.log10(spec2d + 1e-6)
                vmin = np.percentile(log_spec, 5)
                vmax = np.percentile(log_spec, 99)
                log_spec = np.clip(log_spec, vmin, vmax)

                row = i // 2 + 1
                col = i % 2 + 1

                fig.add_trace(
                    go.Heatmap(
                        x=direction_deg,
                        y=period,
                        z=log_spec,
                        colorscale="Blues",
                        showscale=(i == 0),
                        colorbar=dict(title="log10(S) [m²/Hz/deg]", len=0.4),
                    ),
                    row=row,
                    col=col,
                )

                fig.add_trace(
                    go.Scatter(
                        x=[theta_p],
                        y=[Tp],
                        mode="markers+text",
                        marker=dict(color="black", size=14, symbol="x"),
                        text=[f"Hs={hs_val:.2f} m<br>Tp={Tp:.2f} s<br>θ={theta_p:.1f}°"],
                        textposition="top right",
                        showlegend=False,
                    ),
                    row=row,
                    col=col,
                )

                fig.update_xaxes(
                    title_text="Wave direction [deg] (coming-from)",
                    row=row,
                    col=col,
                )

                fig.update_yaxes(
                    title_text="Wave period [s] (T = 1/f)",
                    row=row,
                    col=col,
                )

            spec_lat_text = ""
            spec_lon_text = ""
            if "latitude" in selected and "longitude" in selected:
                sel_lat = np.asarray(selected["latitude"].values, dtype=float).reshape(-1)
                sel_lon = np.asarray(selected["longitude"].values, dtype=float).reshape(-1)
                if sel_lat.size:
                    spec_lat_text = f", spec lat={sel_lat[0]:.6f}"
                if sel_lon.size:
                    spec_lon_text = f", spec lon={sel_lon[0]:.6f}"

            fig.update_layout(
                title=dict(
                    text=(
                        f"Directional wave spectra E(f,θ) — key sea states — {inputs.spec_file.name}"
                        f"<br>Spectral point {display_idx + 1} (index={spec_point_idx})"
                        f"{spec_lat_text}{spec_lon_text}"
                    ),
                    x=0.5,
                    xanchor="center",
                ),
                template="plotly_white",
                autosize=True,
                margin=dict(l=10, r=10, t=90, b=10),
                font=dict(size=13),
            )

            out_file = inputs.spec_file.parent / (
                f"{inputs.spec_file.stem}_dirspec_sp{display_idx + 1}_specpt{spec_point_idx}.html"
            )
            fig.write_html(
                out_file,
                include_plotlyjs="cdn",
                config={"responsive": True},
                full_html=True,
                default_height="100vh",
            )

            print(f"Saved: {out_file.resolve()}")
            fig.show()
            any_written = True

        if not any_written:
            raise ValueError("No valid spectra found for any selected spectral point.")

    finally:
        spec_ds.close()

def _find_nearest_grid_point(grid: GridInfo, point_coord: tuple[float, float]) -> PoiIndex:
    lat0, lon0 = point_coord
    dist2 = (grid.lat2d - lat0) ** 2 + (grid.lon2d - lon0) ** 2
    iy, ix = np.unravel_index(np.nanargmin(dist2), dist2.shape)
    return PoiIndex(iy=int(iy), ix=int(ix), lon=float(grid.lon2d[iy, ix]), lat=float(grid.lat2d[iy, ix]))

def _build_table_trace(
    time_text: str,
    pois: tuple[PoiIndex, ...],
    rows: list[tuple[str, tuple[float | None, ...]]],
    domain: dict[str, list[float]] | None = None,
) -> go.Table:
    point_headers = [str(i + 1) for i in range(len(pois))]
    row_labels = ["Time", "Point latitude", "Point longitude", *[label for label, _ in rows]]
    point_columns: list[list[str]] = []

    for idx, poi in enumerate(pois):
        time_cell = time_text if idx == 0 else ""
        col_vals = [time_cell, f"{poi.lat:.6f}", f"{poi.lon:.6f}"]
        for _, metric_values in rows:
            metric_val = metric_values[idx] if idx < len(metric_values) else None
            if metric_val is None or not np.isfinite(metric_val):
                col_vals.append("N/A")
            else:
                col_vals.append(f"{float(metric_val):.3f}")
        point_columns.append(col_vals)

    return go.Table(
        header=dict(values=["Field", *point_headers], fill_color="lightgrey", align="left"),
        cells=dict(
            values=[row_labels, *point_columns],
            align="left",
        ),
        domain=domain,
    )


def _build_hs_timeseries_marker_trace(times: np.ndarray, hs_series: np.ndarray, time_idx: int) -> go.Scatter:
    return go.Scatter(
        x=[times[time_idx]],
        y=[hs_series[time_idx]],
        mode="markers",
        marker=dict(color="crimson", size=10),
        cliponaxis=False,
        name="Current HS",
        showlegend=False,
        hovertemplate="Time=%{x}<br>HS=%{y:.3f} m<extra></extra>",
    )


def _build_tp_timeseries_marker_trace(times: np.ndarray, tp_series: np.ndarray, time_idx: int) -> go.Scatter:
    return go.Scatter(
        x=[times[time_idx]],
        y=[tp_series[time_idx]],
        mode="markers",
        marker=dict(color="darkorange", size=10, symbol="diamond"),
        cliponaxis=False,
        name="Current TP",
        showlegend=False,
        hovertemplate="Time=%{x}<br>TP=%{y:.3f} s<extra></extra>",
    )


def _normalize_colorscale_limits(vmin: float, vmax: float) -> tuple[float, float]:
    if np.isfinite(vmin) and np.isfinite(vmax) and np.isclose(vmin, vmax):
        delta = max(1e-6, abs(vmin) * 1e-6)
        return vmin - delta, vmax + delta
    return vmin, vmax


def _align_time_count(data: np.ndarray, nt: int) -> np.ndarray:
    if data.ndim == 2:
        return np.repeat(data[None, :, :], repeats=nt, axis=0)
    if data.ndim != 3:
        raise ValueError("Expected 2D or 3D array for time alignment.")
    if data.shape[0] == nt:
        return data
    if data.shape[0] > nt:
        return data[:nt]
    out = np.empty((nt, data.shape[1], data.shape[2]), dtype=data.dtype)
    out[: data.shape[0]] = data
    out[data.shape[0] :] = data[-1]
    return out


def _build_time_cursor_trace(times: np.ndarray, y_min: float, y_max: float, time_idx: int) -> go.Scatter:
    x = times[time_idx]
    y_pad = 0.02 * (y_max - y_min) if np.isfinite(y_min) and np.isfinite(y_max) else 0.0
    return go.Scatter(
        x=[x, x],
        y=[y_min - y_pad, y_max + y_pad],
        mode="lines",
        line=dict(color="black", width=2, dash="dot"),
        name="Current time",
        showlegend=False,
        hoverinfo="skip",
        cliponaxis=False,
    )


def build_figure(
    datasets: list[tuple[str, xr.Dataset]],
    point_coord: tuple[float, float] | tuple[tuple[float, float], ...],
    wind_arrow_resolution: int | None,
) -> go.Figure:
    runs = _prepare_runs_data(datasets)
    nt = runs[-1].hs.shape[0]
    finest = runs[-1]
    point_coords = _normalize_point_coords(point_coord)
    pois = tuple(_find_nearest_grid_point(finest.grid, pc) for pc in point_coords)

    hs_min = float(min(np.nanmin(np.where(r.mask_keep_time, r.hs, np.nan)) for r in runs))
    hs_max = float(max(np.nanmax(np.where(r.mask_keep_time, r.hs, np.nan)) for r in runs))
    tp_min = float(min(np.nanmin(np.where(r.mask_keep_time, r.tp, np.nan)) for r in runs))
    tp_max = float(max(np.nanmax(np.where(r.mask_keep_time, r.tp, np.nan)) for r in runs))

    has_wind = all(r.wind_speed is not None for r in runs)
    ws_min = ws_max = 0.0
    if has_wind:
        ws_min = float(
            min(np.nanmin(np.where(r.mask_keep_time, r.wind_speed, np.nan)) for r in runs if r.wind_speed is not None)
        )
        ws_max = float(
            max(np.nanmax(np.where(r.mask_keep_time, r.wind_speed, np.nan)) for r in runs if r.wind_speed is not None)
        )

    hs_min, hs_max = _normalize_colorscale_limits(hs_min, hs_max)
    tp_min, tp_max = _normalize_colorscale_limits(tp_min, tp_max)
    if has_wind:
        ws_min, ws_max = _normalize_colorscale_limits(ws_min, ws_max)

    times = finest.grid.times

    poi_hs_series_list = [finest.hs[:, poi.iy, poi.ix] for poi in pois]
    poi_tp_series_list = [finest.tp[:, poi.iy, poi.ix] for poi in pois]
    poi_hs_concat = np.concatenate([np.asarray(v) for v in poi_hs_series_list]) if poi_hs_series_list else np.array([0.0])
    ts_y_min = float(np.nanmin(poi_hs_concat))
    ts_y_max = float(np.nanmax(poi_hs_concat))
    if not np.isfinite(ts_y_min) or not np.isfinite(ts_y_max) or np.isclose(ts_y_min, ts_y_max):
        ts_y_min, ts_y_max = _normalize_colorscale_limits(
            ts_y_min if np.isfinite(ts_y_min) else 0.0,
            ts_y_max if np.isfinite(ts_y_max) else 1.0,
        )

    current_cube: np.ndarray | None = finest.current_speed
    current_from_cube: np.ndarray | None = finest.current_from
    has_current = current_cube is not None
    cur_min = cur_max = 0.0
    if current_cube is not None:
        current_cube = _align_time_count(current_cube, nt)
        current_cube = np.where(finest.mask_keep_time, current_cube, np.nan)
        if current_from_cube is not None:
            current_from_cube = _align_time_count(current_from_cube, nt)
            current_from_cube = np.where(finest.mask_keep_time, current_from_cube, np.nan)
        cur_min = float(np.nanmin(current_cube))
        cur_max = float(np.nanmax(current_cube))
        cur_min, cur_max = _normalize_colorscale_limits(cur_min, cur_max)

    subplot_specs: list[list[dict[str, object]]] = [
        [{"type": "xy", "secondary_y": True}, {"type": "xy"}],
        [{"type": "xy"}, {"type": "xy"}],
    ]
    fig = make_subplots(
        rows=len(subplot_specs),
        cols=len(subplot_specs[0]),
        specs=subplot_specs,
        subplot_titles=(
            "Point of interest",
            "TP",
            "HS",
            "Wind and current" if has_wind and has_current else "Wind speed",
        ),
        horizontal_spacing=0.16,
        vertical_spacing=0.10,
    )

    default_colorscale = "Jet"

    hs_trace_idxs: list[int] = []
    tp_trace_idxs: list[int] = []
    ws_trace_idxs: list[int] = []

    for idx, run in enumerate(runs):
        hs0 = np.where(run.mask_keep_time[0], run.hs[0], np.nan)
        tp0 = np.where(run.mask_keep_time[0], run.tp[0], np.nan)

        fig.add_trace(
            go.Heatmap(
                z=hs0,
                x=run.grid.lon,
                y=run.grid.lat,
                zmin=hs_min,
                zmax=hs_max,
                colorscale=default_colorscale,
                colorbar=dict(title="HS [m]", len=0.35, x=0.41, y=0.21),
                showscale=(idx == 0),
            ),
            row=2,
            col=1,
        )
        hs_trace_idxs.append(len(fig.data) - 1)

        fig.add_trace(
            go.Heatmap(
                z=tp0,
                x=run.grid.lon,
                y=run.grid.lat,
                zmin=tp_min,
                zmax=tp_max,
                colorscale=default_colorscale,
                colorbar=dict(title="TP [s]", len=0.35, x=1.04, y=0.79),
                showscale=(idx == 0),
            ),
            row=1,
            col=2,
        )
        tp_trace_idxs.append(len(fig.data) - 1)

        if has_wind and run.wind_speed is not None:
            ws0 = np.where(run.mask_keep_time[0], run.wind_speed[0], np.nan)
            fig.add_trace(
                go.Heatmap(
                    z=ws0,
                    x=run.grid.lon,
                    y=run.grid.lat,
                    zmin=ws_min,
                    zmax=ws_max,
                    colorscale=default_colorscale,
                    colorbar=dict(title="Wind [m/s]", len=0.35, x=1.07, y=0.21),
                    showscale=(idx == 0),
                ),
                row=2,
                col=2,
            )
            ws_trace_idxs.append(len(fig.data) - 1)

    hs_arrow_trace_idxs: list[int] = []
    tp_arrow_trace_idxs: list[int] = []
    wind_arrow_trace_idxs: list[int] = []
    for run in runs:
        if run.wave_from is not None and wind_arrow_resolution is not None:
            hs_ax0, hs_ay0 = _build_wind_arrow_frame_data(
                np.where(run.mask_keep_time[0], run.wave_from[0], np.nan),
                run.grid.lon2d,
                run.grid.lat2d,
                wind_arrow_resolution,
            )
            tp_ax0, tp_ay0 = _build_wind_arrow_frame_data(
                np.where(run.mask_keep_time[0], run.wave_from[0], np.nan),
                run.grid.lon2d,
                run.grid.lat2d,
                wind_arrow_resolution,
            )
        else:
            hs_ax0, hs_ay0 = np.array([]), np.array([])
            tp_ax0, tp_ay0 = np.array([]), np.array([])

        if has_wind and run.wind_from is not None and wind_arrow_resolution is not None:
            lx0, ly0 = _build_wind_arrow_frame_data(
                np.where(run.mask_keep_time[0], run.wind_from[0], np.nan),
                run.grid.lon2d,
                run.grid.lat2d,
                wind_arrow_resolution,
            )
        else:
            lx0, ly0 = np.array([]), np.array([])

        fig.add_trace(
            go.Scatter(
                x=hs_ax0,
                y=hs_ay0,
                mode="lines",
                line=dict(color="black", width=1.1),
                hoverinfo="skip",
                name=f"HS direction ({run.label})",
                showlegend=False,
            ),
            row=2,
            col=1,
        )
        hs_arrow_trace_idxs.append(len(fig.data) - 1)

        fig.add_trace(
            go.Scatter(
                x=tp_ax0,
                y=tp_ay0,
                mode="lines",
                line=dict(color="black", width=1.1),
                hoverinfo="skip",
                name=f"TP direction ({run.label})",
                showlegend=False,
            ),
            row=1,
            col=2,
        )
        tp_arrow_trace_idxs.append(len(fig.data) - 1)

        fig.add_trace(
            go.Scatter(
                x=lx0,
                y=ly0,
                mode="lines",
                line=dict(color="black", width=1.5),
                hoverinfo="skip",
                name=f"Wind direction ({run.label})",
                showlegend=False,
            ),
            row=2,
            col=2,
        )
        wind_arrow_trace_idxs.append(len(fig.data) - 1)

    poi_marker_style = dict(
        mode="markers+text",
        marker=dict(color="red", size=12, symbol="circle"),
        textposition="middle center",
        textfont=dict(color="white", size=10),
        name="Point of interest",
        showlegend=False,
    )
    poi_lons = [poi.lon for poi in pois]
    poi_lats = [poi.lat for poi in pois]
    poi_labels = [str(i + 1) for i in range(len(pois))]
    poi_hover = [f"POI {i + 1}<br>lon={poi.lon:.6f}<br>lat={poi.lat:.6f}<extra></extra>" for i, poi in enumerate(pois)]

    marker_trace = go.Scatter(x=poi_lons, y=poi_lats, text=poi_labels, hovertemplate=poi_hover, **poi_marker_style)
    fig.add_trace(marker_trace, row=1, col=2)
    fig.add_trace(go.Scatter(x=poi_lons, y=poi_lats, text=poi_labels, hovertemplate=poi_hover, **poi_marker_style), row=2, col=1)

    if has_wind:
        fig.add_trace(go.Scatter(x=poi_lons, y=poi_lats, text=poi_labels, hovertemplate=poi_hover, **poi_marker_style), row=2, col=2)
    else:
        fig.add_annotation(
            text="Wind not available in input dataset",
            xref="x4 domain",
            yref="y5 domain",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(color="gray", size=13),
        )

    ts_x_domain = [float(v) for v in fig.layout.xaxis.domain]
    ts_y_domain = [float(v) for v in fig.layout.yaxis.domain]
    split_gap = 0.02 * (ts_y_domain[1] - ts_y_domain[0])
    split_y = ts_y_domain[0] + 0.53 * (ts_y_domain[1] - ts_y_domain[0])
    table_domain = {"x": ts_x_domain, "y": [split_y + split_gap, ts_y_domain[1]]}
    ts_plot_domain = [ts_y_domain[0], split_y - split_gap]

    fig.layout.yaxis.domain = ts_plot_domain
    fig.layout.xaxis.anchor = "y"

    table_trace_idx = len(fig.data)
    fig.add_trace(
        _build_table_trace(
            time_text=str(times[0]),
            pois=pois,
            rows=[
                ("HS [m]", tuple(float(finest.hs[0, poi.iy, poi.ix]) for poi in pois)),
                ("TP [s]", tuple(float(finest.tp[0, poi.iy, poi.ix]) for poi in pois)),
                (
                    "Wind speed [m/s]",
                    tuple(float(finest.wind_speed[0, poi.iy, poi.ix]) if finest.wind_speed is not None else None for poi in pois),
                ),
            ],
            domain=table_domain,
        ),
    )

    for i, poi_hs_series in enumerate(poi_hs_series_list):
        fig.add_trace(
            go.Scatter(
                x=times,
                y=poi_hs_series,
                mode="lines",
                line=dict(color="royalblue", width=2),
                name=f"HS P{i + 1} [m] (left axis)",
                showlegend=True,
                hovertemplate=f"POI {i + 1}<br>Time=%{{x}}<br>HS=%{{y:.3f}} m<extra></extra>",
            ),
            row=1,
            col=1,
        )

    hs_timeseries_marker_trace_idx = len(fig.data)
    fig.add_trace(_build_hs_timeseries_marker_trace(times, poi_hs_series_list[0], time_idx=0), row=1, col=1)

    for i, poi_tp_series in enumerate(poi_tp_series_list):
        fig.add_trace(
            go.Scatter(
                x=times,
                y=poi_tp_series,
                mode="lines",
                line=dict(color="darkorange", width=2, dash="dot"),
                name=f"TP P{i + 1} [s] (right axis)",
                showlegend=True,
                hovertemplate=f"POI {i + 1}<br>Time=%{{x}}<br>TP=%{{y:.3f}} s<extra></extra>",
            ),
            row=1,
            col=1,
            secondary_y=True,
        )

    tp_timeseries_marker_trace_idx = len(fig.data)
    fig.add_trace(_build_tp_timeseries_marker_trace(times, poi_tp_series_list[0], time_idx=0), row=1, col=1, secondary_y=True)

    time_cursor_trace_idx = len(fig.data)
    fig.add_trace(_build_time_cursor_trace(times, ts_y_min, ts_y_max, time_idx=0), row=1, col=1)

    current_trace_idx: int | None = None
    current_arrow_trace_idx: int | None = None
    if current_cube is not None:
        wind_x_domain = [float(v) for v in fig.layout.xaxis4.domain]
        wind_y_domain = [float(v) for v in fig.layout.yaxis5.domain]
        half_gap = 0.02 * (wind_x_domain[1] - wind_x_domain[0])
        mid_x = 0.5 * (wind_x_domain[0] + wind_x_domain[1])
        left_domain = [wind_x_domain[0], mid_x - half_gap]
        right_domain = [mid_x + half_gap, wind_x_domain[1]]
        fig.layout.xaxis4.domain = left_domain
        fig.layout.xaxis5 = go.layout.XAxis(domain=right_domain, anchor="y6", title="Longitude")
        fig.layout.yaxis6 = go.layout.YAxis(domain=wind_y_domain, anchor="x5", title="Latitude")

        fig.add_trace(
            go.Heatmap(
                z=current_cube[0],
                x=finest.grid.lon,
                y=finest.grid.lat,
                zmin=cur_min,
                zmax=cur_max,
                colorscale=default_colorscale,
                colorbar=dict(title="Current [m/s]", len=0.35, x=1.14, y=0.21),
                xaxis="x5",
                yaxis="y6",
            )
        )
        current_trace_idx = len(fig.data) - 1

        fig.add_trace(
            go.Scatter(
                x=poi_lons,
                y=poi_lats,
                text=poi_labels,
                hovertemplate=poi_hover,
                **poi_marker_style,
                xaxis="x5",
                yaxis="y6",
            )
        )

        if current_from_cube is not None and wind_arrow_resolution is not None:
            cax0, cay0 = _build_wind_arrow_frame_data(
                current_from_cube[0],
                finest.grid.lon2d,
                finest.grid.lat2d,
                wind_arrow_resolution,
            )
        else:
            cax0, cay0 = np.array([]), np.array([])

        fig.add_trace(
            go.Scatter(
                x=cax0,
                y=cay0,
                mode="lines",
                line=dict(color="black", width=1.1),
                hoverinfo="skip",
                name="Current direction",
                showlegend=False,
                xaxis="x5",
                yaxis="y6",
            )
        )
        current_arrow_trace_idx = len(fig.data) - 1

    all_heatmap_trace_idxs = hs_trace_idxs + tp_trace_idxs + ws_trace_idxs
    all_heatmap_zmin: list[float] = [hs_min] * len(hs_trace_idxs) + [tp_min] * len(tp_trace_idxs) + [ws_min] * len(ws_trace_idxs)
    all_heatmap_zmax_full: list[float] = [hs_max] * len(hs_trace_idxs) + [tp_max] * len(tp_trace_idxs) + [ws_max] * len(ws_trace_idxs)

    if current_trace_idx is not None:
        all_heatmap_trace_idxs.append(current_trace_idx)
        all_heatmap_zmin.append(cur_min)
        all_heatmap_zmax_full.append(cur_max)

    percentage_options = [
        ("Max: 20%", 0.20),
        ("Max: 40%", 0.40),
        ("Max: 60%", 0.60),
        ("Max: 80%", 0.80),
        ("Max: 100%", 1.00),
    ]
    percentage_buttons = []
    for label, frac in percentage_options:
        zmax_values = [max(v * frac, 1e-12) for v in all_heatmap_zmax_full]
        percentage_buttons.append(
            dict(
                label=label,
                method="restyle",
                args=[
                    {"zmin": all_heatmap_zmin, "zmax": zmax_values},
                    all_heatmap_trace_idxs,
                ],
            )
        )

    colorscale_options = [
        ("Scale: Jet", "Jet"),
        ("Scale: Viridis", "Viridis"),
        ("Scale: Plasma", "Plasma"),
        ("Scale: Cividis", "Cividis"),
        ("Scale: Turbo", "Turbo"),
    ]
    colorscale_buttons = [
        dict(
            label=label,
            method="restyle",
            args=[{"colorscale": [scale] * len(all_heatmap_trace_idxs)}, all_heatmap_trace_idxs],
        )
        for label, scale in colorscale_options
    ]

    dynamic_trace_indices = (
        hs_trace_idxs
        + tp_trace_idxs
        + ws_trace_idxs
        + hs_arrow_trace_idxs
        + tp_arrow_trace_idxs
        + wind_arrow_trace_idxs
        + [table_trace_idx, hs_timeseries_marker_trace_idx, tp_timeseries_marker_trace_idx, time_cursor_trace_idx]
    )
    if current_trace_idx is not None:
        dynamic_trace_indices.append(current_trace_idx)
    if current_arrow_trace_idx is not None:
        dynamic_trace_indices.append(current_arrow_trace_idx)

    frames: list[go.Frame] = []
    for i in range(nt):
        frame_data: list[go.BaseTraceType] = []

        for run in runs:
            frame_data.append(go.Heatmap(z=np.where(run.mask_keep_time[i], run.hs[i], np.nan), x=run.grid.lon, y=run.grid.lat))
        for run in runs:
            frame_data.append(go.Heatmap(z=np.where(run.mask_keep_time[i], run.tp[i], np.nan), x=run.grid.lon, y=run.grid.lat))
        if has_wind:
            for run in runs:
                if run.wind_speed is None:
                    continue
                frame_data.append(go.Heatmap(z=np.where(run.mask_keep_time[i], run.wind_speed[i], np.nan), x=run.grid.lon, y=run.grid.lat))

        for run in runs:
            if run.wave_from is not None and wind_arrow_resolution is not None:
                hs_axi, hs_ayi = _build_wind_arrow_frame_data(
                    np.where(run.mask_keep_time[i], run.wave_from[i], np.nan),
                    run.grid.lon2d,
                    run.grid.lat2d,
                    wind_arrow_resolution,
                )
            else:
                hs_axi, hs_ayi = np.array([]), np.array([])
            frame_data.append(go.Scatter(x=hs_axi, y=hs_ayi, mode="lines", line=dict(color="black", width=1.1), hoverinfo="skip"))

        for run in runs:
            if run.wave_from is not None and wind_arrow_resolution is not None:
                tp_axi, tp_ayi = _build_wind_arrow_frame_data(
                    np.where(run.mask_keep_time[i], run.wave_from[i], np.nan),
                    run.grid.lon2d,
                    run.grid.lat2d,
                    wind_arrow_resolution,
                )
            else:
                tp_axi, tp_ayi = np.array([]), np.array([])
            frame_data.append(go.Scatter(x=tp_axi, y=tp_ayi, mode="lines", line=dict(color="black", width=1.1), hoverinfo="skip"))

        if has_wind:
            for run in runs:
                if run.wind_from is not None and wind_arrow_resolution is not None:
                    lxi, lyi = _build_wind_arrow_frame_data(
                        np.where(run.mask_keep_time[i], run.wind_from[i], np.nan),
                        run.grid.lon2d,
                        run.grid.lat2d,
                        wind_arrow_resolution,
                    )
                else:
                    lxi, lyi = np.array([]), np.array([])
                frame_data.append(go.Scatter(x=lxi, y=lyi, mode="lines", line=dict(color="black", width=1.5), hoverinfo="skip"))

        frame_data.append(
            _build_table_trace(
                time_text=str(times[i]),
                pois=pois,
                rows=[
                    ("HS [m]", tuple(float(finest.hs[i, poi.iy, poi.ix]) for poi in pois)),
                    ("TP [s]", tuple(float(finest.tp[i, poi.iy, poi.ix]) for poi in pois)),
                    (
                        "Wind speed [m/s]",
                        tuple(float(finest.wind_speed[i, poi.iy, poi.ix]) if finest.wind_speed is not None else None for poi in pois),
                    ),
                ],
                domain=table_domain,
            )
        )
        frame_data.append(_build_hs_timeseries_marker_trace(times, poi_hs_series_list[0], time_idx=i))
        frame_data.append(_build_tp_timeseries_marker_trace(times, poi_tp_series_list[0], time_idx=i))
        frame_data.append(_build_time_cursor_trace(times, ts_y_min, ts_y_max, time_idx=i))

        if current_cube is not None:
            frame_data.append(go.Heatmap(z=current_cube[i], x=finest.grid.lon, y=finest.grid.lat, xaxis="x5", yaxis="y6"))
            if current_from_cube is not None and wind_arrow_resolution is not None:
                caxi, cayi = _build_wind_arrow_frame_data(
                    current_from_cube[i],
                    finest.grid.lon2d,
                    finest.grid.lat2d,
                    wind_arrow_resolution,
                )
            else:
                caxi, cayi = np.array([]), np.array([])
            frame_data.append(
                go.Scatter(
                    x=caxi,
                    y=cayi,
                    mode="lines",
                    line=dict(color="black", width=1.1),
                    hoverinfo="skip",
                    xaxis="x5",
                    yaxis="y6",
                )
            )

        frames.append(go.Frame(data=frame_data, traces=dynamic_trace_indices, name=str(i)))

    fig.frames = frames

    steps = [
        dict(
            method="animate",
            args=[[str(i)], dict(mode="immediate", frame=dict(duration=0, redraw=False), transition=dict(duration=0))],
            label=(str(times[i])[11:16] if "T" in str(times[i]) else str(times[i])),
        )
        for i in range(nt)
    ]

    fig.update_layout(
        title="SWAN post-processing: HS / TP / Wind speed (nested runs merged by finest precedence)",
        legend=dict(
            orientation="h",
            x=ts_x_domain[0] + 0.01,
            y=ts_plot_domain[0] - 0.035,
            xanchor="left",
            yanchor="top",
            bgcolor="rgba(255,255,255,0.7)",
            borderwidth=0,
        ),
        margin=dict(l=40, r=40, t=145, b=95),
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                x=0.02,
                y=-0.055,
                xanchor="left",
                yanchor="top",
                pad=dict(r=8, t=0),
                buttons=[
                    dict(label="Play", method="animate", args=[None, dict(frame=dict(duration=250, redraw=True), fromcurrent=True, transition=dict(duration=0))]),
                    dict(label="Pause", method="animate", args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate", transition=dict(duration=0))]),
                ],
            ),
            dict(
                type="dropdown",
                direction="down",
                x=0.02,
                y=1.14,
                xanchor="left",
                yanchor="top",
                buttons=percentage_buttons,
                showactive=True,
                active=4,
            ),
            dict(
                type="dropdown",
                direction="down",
                x=0.19,
                y=1.14,
                xanchor="left",
                yanchor="top",
                buttons=colorscale_buttons,
                showactive=True,
                active=0,
            ),
        ],
        sliders=[
            dict(
                active=0,
                x=0.16,
                y=-0.055,
                len=0.78,
                xanchor="left",
                yanchor="top",
                pad=dict(t=0, b=0),
                steps=steps,
            )
        ],
    )

    fig.update_xaxes(title_text="Time", row=1, col=1)
    fig.update_yaxes(
        title_text="HS [m] (blue, left axis)",
        title_font=dict(color="royalblue"),
        tickfont=dict(color="royalblue"),
        row=1,
        col=1,
        secondary_y=False,
    )
    fig.update_yaxes(
        title_text="TP [s] (orange, right axis)",
        title_font=dict(color="darkorange"),
        tickfont=dict(color="darkorange"),
        showgrid=False,
        zeroline=False,
        showline=True,
        ticks="outside",
        row=1,
        col=1,
        secondary_y=True,
    )
    fig.update_xaxes(title_text="Longitude", row=1, col=2)
    fig.update_yaxes(title_text="Latitude", row=1, col=2)
    fig.update_xaxes(title_text="Longitude", row=2, col=1)
    fig.update_yaxes(title_text="Latitude", row=2, col=1)
    fig.update_xaxes(title_text="Longitude", row=2, col=2)
    fig.update_yaxes(title_text="Latitude", row=2, col=2)

    lon_min = min(float(np.nanmin(r.grid.lon)) for r in runs)
    lon_max = max(float(np.nanmax(r.grid.lon)) for r in runs)
    lat_min = min(float(np.nanmin(r.grid.lat)) for r in runs)
    lat_max = max(float(np.nanmax(r.grid.lat)) for r in runs)
    fig.update_xaxes(range=[lon_min, lon_max], row=1, col=2)
    fig.update_xaxes(range=[lon_min, lon_max], row=2, col=1)
    fig.update_xaxes(range=[lon_min, lon_max], row=2, col=2)
    fig.update_yaxes(range=[lat_min, lat_max], row=1, col=2)
    fig.update_yaxes(range=[lat_min, lat_max], row=2, col=1)
    fig.update_yaxes(range=[lat_min, lat_max], row=2, col=2)

    return fig

def _build_leaflet_overlay_html(
    geojson_path: Path,
    bathy_geojson_path: Path | None = None
) -> str:

    geojson_name = geojson_path.name
    bathy_name = bathy_geojson_path.name if bathy_geojson_path else None

    return f"""
    <div id="map" style="width:100%; height:100%;"></div>

    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>

    <script>
    setTimeout(function() {{

        var map = L.map('map', {{
            zoomControl: true
        }});

        window.map = map;

        var osm = L.tileLayer(
            'https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png',
            {{
                maxZoom: 18,
                attribution: '&copy; OpenStreetMap contributors'
            }}
        ).addTo(map);

        var hsLayer = null;
        var bathyLayer = null;

        function applyNiceZoom(layer) {{
            if (!layer || !layer.getBounds || !layer.getBounds().isValid()) return;

            var bounds = layer.getBounds();

            // Add a comfortable margin around the data
            bounds = bounds.pad(0.35);

            map.fitBounds(bounds, {{
                padding: [40, 40],
                maxZoom: 11
            }});

            // Extra guard against over-zooming
            if (map.getZoom() > 11) {{
                map.setZoom(11);
            }}
        }}

        fetch('{geojson_name}')
            .then(res => res.json())
            .then(data => {{

                hsLayer = L.geoJSON(data, {{
                    style: function(feature) {{
                        return {{
                            fillColor: feature.properties.fill,
                            fillOpacity: 0.55,
                            color: "transparent",
                            weight: 0
                        }};
                    }}
                }}).addTo(map);

                applyNiceZoom(hsLayer);

            }})
            .catch(err => console.error("HS layer error:", err));

        {"fetch('" + bathy_name + "').then(res => res.json()).then(data => {\
            bathyLayer = L.geoJSON(data, {\
                style: function(feature) {\
                    return {\
                        color: '#000000',\
                        weight: 1,\
                        opacity: 0.6\
                    };\
                }\
            }).addTo(map);\
        }).catch(err => console.error('Bathy error:', err));" if bathy_name else ""}

    }}, 300);
    </script>
    """

def _export_bathymetry_contours(ds: xr.Dataset, out_path: Path, levels=None) -> Path:
    import numpy as np
    import json
    import matplotlib.pyplot as plt

    if "depth" not in ds:
        raise KeyError("Dataset missing 'depth' for bathymetry contours")

    depth = np.asarray(ds["depth"].isel(time=0).values)
    lon = np.asarray(ds["longitude"].values)
    lat = np.asarray(ds["latitude"].values)

    lon2d, lat2d = np.meshgrid(lon, lat)

    if levels is None:
        levels = [5, 10, 20, 50, 100]

    # Create contour object
    cs = plt.contour(lon2d, lat2d, depth, levels=levels)

    features = []

    # ✅ VERSION-SAFE: use allsegs instead of collections
    for level, segments in zip(cs.levels, cs.allsegs):
        for seg in segments:
            if len(seg) < 2:
                continue

            coords = seg.tolist()

            features.append({
                "type": "Feature",
                "properties": {
                    "depth_m": float(level)
                },
                "geometry": {
                    "type": "LineString",
                    "coordinates": coords
                }
            })

    plt.close()  # important to avoid memory leak

    geojson = {
        "type": "FeatureCollection",
        "features": features
    }

    out_path.write_text(json.dumps(geojson), encoding="utf-8")

    return out_path
def generate_metocean_report(
    inputs,
    html_2d_inputs,
    html_2d_outputs,
    html_3d,
    html_overlay,
    geojson_path=None,
    bathy_geojson_path=None,
):
    import xarray as xr
    from pathlib import Path
    import shutil
    import subprocess
    import webbrowser
    import time
    import socket
    import numpy as np

    spec_file = inputs.spec_file

    head_block = """
    <head>
    <style>
    html, body {
        margin: 0;
        height: 100%;
        overflow: hidden;
        font-family: sans-serif;
    }

    .tab {
        display: flex;
        height: 50px;
        background: #eee;
        border-bottom: 1px solid #ccc;
    }

    .tab button {
        flex: 1;
        border: none;
        background: #ddd;
        cursor: pointer;
        font-size: 14px;
    }

    .tab button:hover {
        background: #ccc;
    }

    .tabcontent {
        display: none;
        width: 100%;
        height: calc(100vh - 50px);
        overflow: hidden;
    }

    #map2d_inputs > div,
    #map2d_outputs > div,
    #map3d > div,
    #overlay > div,
    #dirspec > div {
        width: 100% !important;
        height: 100% !important;
    }

    .dashwrap {
        width: 100%;
        height: 100%;
        display: flex;
        flex-direction: column;
        overflow: hidden;
    }

    .subtab {
        display: flex;
        flex: 0 0 42px;
        height: 42px;
        background: #f5f5f5;
        border-bottom: 1px solid #ccc;
        overflow-x: auto;
        overflow-y: hidden;
        white-space: nowrap;
    }

    .subtab button {
        border: none;
        background: #e7e7e7;
        cursor: pointer;
        font-size: 13px;
        padding: 0 16px;
        height: 42px;
        flex: 0 0 auto;
    }

    .subtab button:hover {
        background: #d7d7d7;
    }

    .subtabcontent {
        display: none;
        flex: 1 1 auto;
        width: 100%;
        min-height: 0;
        overflow: hidden;
    }

    .subtabcontent.active {
        display: block;
    }

    .subtabcontent > div {
        width: 100% !important;
        height: 100% !important;
    }
    </style>

    <script>
    function openTab(name) {
        const tabs = ["dash", "dirspec", "map2d_inputs", "map2d_outputs", "map3d", "overlay"];

        tabs.forEach(t => {
            const el = document.getElementById(t);
            if (el) el.style.display = "none";
        });

        const active = document.getElementById(name);
        if (active) active.style.display = "block";

        setTimeout(() => {
            window.dispatchEvent(new Event('resize'));
            if (window.map) {
                window.map.invalidateSize();
            }
        }, 250);
    }

    function openDashSubtab(name) {
        const nodes = document.getElementsByClassName("subtabcontent");
        for (let i = 0; i < nodes.length; i++) {
            nodes[i].style.display = "none";
            nodes[i].classList.remove("active");
        }

        const active = document.getElementById(name);
        if (active) {
            active.style.display = "block";
            active.classList.add("active");
        }

        setTimeout(() => {
            window.dispatchEvent(new Event('resize'));
            if (window.map) {
                window.map.invalidateSize();
            }
        }, 250);
    }

    document.addEventListener("DOMContentLoaded", function() {
        setTimeout(() => {
            const defaultTab = document.getElementById("dash") ? "dash" : "map2d_outputs";
            openTab(defaultTab);

            const firstDashSubtab = document.querySelector(".subtabcontent");
            if (firstDashSubtab) {
                openDashSubtab(firstDashSubtab.id);
            }
        }, 150);
    });
    </script>
    </head>
    """

    def _single_page_html(title: str, body_html: str) -> str:
        return f"""
        <html>
        <head>
        <meta charset="utf-8">
        <title>{title}</title>
        <style>
        html, body {{
            margin: 0;
            width: 100%;
            height: 100%;
            overflow: hidden;
            font-family: sans-serif;
        }}

        .page {{
            width: 100vw;
            height: 100vh;
            overflow: hidden;
        }}

        .page > div {{
            width: 100% !important;
            height: 100% !important;
        }}

        .dashwrap {{
            width: 100%;
            height: 100%;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }}

        .subtab {{
            display: flex;
            flex: 0 0 42px;
            height: 42px;
            background: #f5f5f5;
            border-bottom: 1px solid #ccc;
            overflow-x: auto;
            overflow-y: hidden;
            white-space: nowrap;
        }}

        .subtab button {{
            border: none;
            background: #e7e7e7;
            cursor: pointer;
            font-size: 13px;
            padding: 0 16px;
            height: 42px;
            flex: 0 0 auto;
        }}

        .subtab button:hover {{
            background: #d7d7d7;
        }}

        .subtabcontent,
        .dirspec-subtabcontent {{
            display: none;
            flex: 1 1 auto;
            width: 100%;
            min-height: 0;
            overflow: hidden;
        }}

        .subtabcontent.active,
        .dirspec-subtabcontent.active {{
            display: block;
        }}

        .subtabcontent > div,
        .dirspec-subtabcontent > div {{
            width: 100% !important;
            height: 100% !important;
        }}
        </style>

        <script>
        function openSubtabByClass(name, className) {{
            const nodes = document.getElementsByClassName(className);
            for (let i = 0; i < nodes.length; i++) {{
                nodes[i].style.display = "none";
                nodes[i].classList.remove("active");
            }}

            const active = document.getElementById(name);
            if (active) {{
                active.style.display = "block";
                active.classList.add("active");
            }}

            setTimeout(() => {{
                window.dispatchEvent(new Event('resize'));
                if (window.map) {{
                    window.map.invalidateSize();
                }}
            }}, 250);
        }}

        function openDashSubtab(name) {{
            openSubtabByClass(name, "subtabcontent");
        }}

        function openDirspecSubtab(name) {{
            openSubtabByClass(name, "dirspec-subtabcontent");
        }}

        document.addEventListener("DOMContentLoaded", function() {{
            setTimeout(() => {{
                const firstDashSubtab = document.querySelector(".subtabcontent");
                if (firstDashSubtab) {{
                    openDashSubtab(firstDashSubtab.id);
                }}

                const firstDirspecSubtab = document.querySelector(".dirspec-subtabcontent");
                if (firstDirspecSubtab) {{
                    openDirspecSubtab(firstDirspecSubtab.id);
                }}

                if (!firstDashSubtab && !firstDirspecSubtab) {{
                    window.dispatchEvent(new Event('resize'));
                    if (window.map) {{
                        window.map.invalidateSize();
                    }}
                }}
            }}, 250);
        }});
        </script>
        </head>
        <body>
        <div class="page">{body_html}</div>
        </body>
        </html>
        """

    def _write_html(path: Path, content: str) -> None:
        path.write_text(content, encoding="utf-8")
        print(f"Saved report: {path.resolve()}")

    def _copy_support_files(out_dir: Path) -> None:
        if geojson_path is not None:
            geojson_report_path = out_dir / geojson_path.name
            try:
                if geojson_path.resolve() != geojson_report_path.resolve():
                    shutil.copy(geojson_path, geojson_report_path)
                    print(f"Copied GeoJSON -> {geojson_report_path}")
            except Exception as e:
                print(f"GeoJSON copy failed: {e}")

        if bathy_geojson_path is not None:
            bathy_report_path = out_dir / bathy_geojson_path.name
            try:
                if bathy_geojson_path.resolve() != bathy_report_path.resolve():
                    shutil.copy(bathy_geojson_path, bathy_report_path)
                    print(f"Copied bathy GeoJSON -> {bathy_report_path}")
            except Exception as e:
                print(f"Bathy GeoJSON copy failed: {e}")

    dashboard_html = None
    dirspec_tab_html = None

    if spec_file is not None:
        ds = xr.open_dataset(spec_file)

        try:
            selected_entries = list(_iter_all_spec_point_datasets(ds))
            if not selected_entries:
                raise ValueError("No spectral points found in the spectral dataset.")

            dash_buttons_html: list[str] = []
            dash_contents_html: list[str] = []
            dirspec_buttons_html: list[str] = []
            dirspec_contents_html: list[str] = []

            for display_idx, spec_point_idx, selected_ds, spec_point_coord in selected_entries:
                _, freq_name, dir_name, da = _detect_spec_axes_and_var(selected_ds)

                freq = da[freq_name].values.astype(float)
                direction_raw = da[dir_name].values.astype(float)
                spectrum = da.values.astype(float)

                direction_deg, direction_rad, spectrum = _prepare_direction_axis(direction_raw, spectrum)

                period = 1.0 / freq
                df = np.gradient(freq)
                dtheta = 2 * np.pi / len(direction_rad)
                hs = _compute_hs_series(freq, direction_rad, spectrum)

                valid = np.where(np.isfinite(hs) & (hs > 0))[0]

                if valid.size == 0:
                    idx_max = 0
                    idx_med = 0
                    no_valid_sea_state = True
                else:
                    idx_max = valid[np.argmax(hs[valid])]
                    idx_med = valid[np.argmin(np.abs(hs[valid] - np.percentile(hs[valid], 50)))]
                    no_valid_sea_state = False

                def compute_all(spec):
                    if not np.isfinite(spec).any():
                        spec1d = np.full(spec.shape[0], np.nan, dtype=float)
                        energy_dir = np.full(spec.shape[1], np.nan, dtype=float)
                    else:
                        spec1d = np.nansum(spec, axis=1) * dtheta
                        energy_dir = np.nansum(spec * df[:, None], axis=0)
                    return spec1d, energy_dir

                if no_valid_sea_state:
                    nan_spec = np.full_like(spectrum[0], np.nan, dtype=float)
                    datasets = {
                        "Mean": nan_spec,
                        "Max": nan_spec,
                        "Median": nan_spec,
                    }
                else:
                    datasets = {
                        "Mean": np.nanmean(spectrum, axis=0),
                        "Max": spectrum[idx_max],
                        "Median": spectrum[idx_med],
                    }

                fig1 = make_subplots(
                    rows=2,
                    cols=2,
                    specs=[
                        [{"type": "surface"}, {"type": "polar"}],
                        [{"colspan": 2}, None],
                    ],
                )

                dir_grid, period_grid = np.meshgrid(direction_deg, period)
                dataset_names = list(datasets.keys())
                traces_per_dataset = 5

                log_datasets = {
                    name: np.log10(np.maximum(spec, 1e-12))
                    for name, spec in datasets.items()
                }

                surf_cmin = min(np.nanmin(v) for v in log_datasets.values())
                surf_cmax = max(np.nanmax(v) for v in log_datasets.values())

                for idx_name, (name, specX) in enumerate(datasets.items()):
                    spec1d, energy = compute_all(specX)
                    visible = idx_name == 0
                    zsurf = log_datasets[name]

                    fig1.add_trace(
                        go.Surface(
                            x=dir_grid,
                            y=period_grid,
                            z=zsurf,
                            colorscale="Jet",
                            cmin=surf_cmin,
                            cmax=surf_cmax,
                            showscale=True,
                            visible=visible,
                            name=name,
                            colorbar=dict(
                                title="log10(S) [m²/Hz/deg]",
                                x=0.46,
                                y=0.78,
                                len=0.38,
                                thickness=18,
                            ),
                        ),
                        row=1,
                        col=1,
                    )

                    fig1.add_trace(
                        go.Barpolar(
                            theta=(90.0 - direction_deg) % 360.0,
                            r=energy,
                            visible=visible,
                            name=name,
                        ),
                        row=1,
                        col=2,
                    )

                    peak_1d_idx = int(np.nanargmax(spec1d)) if np.isfinite(spec1d).any() else 0
                    Tp_1d_peak = float(period[peak_1d_idx]) if period.size else np.nan
                    S_1d_peak = float(spec1d[peak_1d_idx]) if spec1d.size else np.nan

                    SWELL_TP_MIN_S = 8.0
                    swell_mask = np.isfinite(period) & np.isfinite(spec1d) & (period >= SWELL_TP_MIN_S)

                    if np.any(swell_mask):
                        swell_local_idx = int(np.nanargmax(spec1d[swell_mask]))
                        swell_idx = np.flatnonzero(swell_mask)[swell_local_idx]
                        Tp_swell = float(period[swell_idx])
                        S_swell = float(spec1d[swell_idx])
                    else:
                        Tp_swell = np.nan
                        S_swell = np.nan

                    fig1.add_trace(
                        go.Scatter(
                            x=period,
                            y=spec1d,
                            visible=visible,
                            name=name,
                            mode="lines",
                        ),
                        row=2,
                        col=1,
                    )

                    fig1.add_trace(
                        go.Scatter(
                            x=[Tp_1d_peak],
                            y=[S_1d_peak],
                            visible=visible,
                            mode="markers+text",
                            marker=dict(color="red", size=11, symbol="diamond"),
                            text=[f"<b>Tp={Tp_1d_peak:.2f} s</b>"],
                            textposition=_smart_text_position(Tp_1d_peak, S_1d_peak, period, spec1d),
                            textfont=dict(color="red", size=14),
                            showlegend=False,
                            hovertemplate="Peak Tp=%{x:.2f} s<br>S=%{y:.4g}<extra></extra>",
                        ),
                        row=2,
                        col=1,
                    )

                    fig1.add_trace(
                        go.Scatter(
                            x=[Tp_swell],
                            y=[S_swell],
                            visible=visible,
                            mode="markers+text",
                            marker=dict(color="darkgreen", size=11, symbol="circle"),
                            text=[f"<b>Swell Tp={Tp_swell:.2f} s</b>"] if np.isfinite(Tp_swell) else [""],
                            textposition=_smart_text_position(
                                Tp_swell if np.isfinite(Tp_swell) else Tp_1d_peak,
                                S_swell if np.isfinite(S_swell) else S_1d_peak,
                                period,
                                spec1d,
                            ),
                            textfont=dict(color="darkgreen", size=14),
                            showlegend=False,
                            hovertemplate="Indicative swell Tp=%{x:.2f} s<br>S=%{y:.4g}<extra></extra>",
                        ),
                        row=2,
                        col=1,
                    )

                fig1.update_xaxes(title_text="Wave direction [deg] (coming-from)", row=1, col=1)
                fig1.update_yaxes(title_text="Wave period [s]", row=1, col=1)
                fig1.update_xaxes(title_text="Wave period [s]", row=2, col=1)
                fig1.update_yaxes(title_text="1D spectral energy [m²/Hz]", row=2, col=1)

                spec_lat_text = ""
                spec_lon_text = ""
                if "latitude" in selected_ds and "longitude" in selected_ds:
                    sel_lat = np.asarray(selected_ds["latitude"].values, dtype=float).reshape(-1)
                    sel_lon = np.asarray(selected_ds["longitude"].values, dtype=float).reshape(-1)
                    if sel_lat.size:
                        spec_lat_text = f", spec lat={sel_lat[0]:.6f}"
                    if sel_lon.size:
                        spec_lon_text = f", spec lon={sel_lon[0]:.6f}"

                buttons = []
                for idx_name, name in enumerate(dataset_names):
                    visible = [False] * (len(dataset_names) * traces_per_dataset)
                    start = idx_name * traces_per_dataset
                    for j in range(traces_per_dataset):
                        visible[start + j] = True

                    buttons.append(
                        dict(
                            label=name,
                            method="update",
                            args=[
                                {"visible": visible},
                                {
                                    "title.text": (
                                        f"{spec_file.name}"
                                        f"<br>Spectral point {display_idx + 1} (index={spec_point_idx}){spec_lat_text}{spec_lon_text}"
                                        f"<br>Showing: {name}"
                                    )
                                },
                            ],
                        )
                    )

                fig1.update_layout(
                    title=dict(
                        text=(
                            f"{spec_file.name}"
                            f"<br>Spectral point {display_idx + 1} (index={spec_point_idx}){spec_lat_text}{spec_lon_text}"
                            f"<br>Showing: {dataset_names[0]}"
                        ),
                        x=0.5,
                        xanchor="center",
                    ),
                    updatemenus=[
                        dict(
                            type="dropdown",
                            direction="down",
                            x=0.02,
                            y=1.12,
                            xanchor="left",
                            yanchor="top",
                            buttons=buttons,
                            showactive=True,
                        )
                    ],
                    polar=dict(
                        angularaxis=dict(
                            rotation=90,
                            direction="clockwise",
                        ),
                    ),
                    margin=dict(l=20, r=20, t=100, b=20),
                )

                fig1.add_annotation(
                    text="Nautical direction [deg]",
                    x=0.83,
                    y=0.98,
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                    font=dict(size=12),
                )

                fig1.update_scenes(
                    xaxis_title="Wave direction [deg] (coming-from)",
                    yaxis_title="Wave period [s]",
                    zaxis_title="log10(Spectral energy)",
                )

                html_dash = fig1.to_html(full_html=False, config={"responsive": True}, auto_play=False)
                subtab_id = f"dash_sp_{display_idx}_spec_{spec_point_idx}"

                dash_buttons_html.append(
                    f'<button onclick="openDashSubtab(\'{subtab_id}\')">Spectral point {display_idx + 1} (idx {spec_point_idx})</button>'
                )
                dash_contents_html.append(
                    f'<div id="{subtab_id}" class="subtabcontent">{html_dash}</div>'
                )

                idx_p95 = 0
                idx_min = 0
                if valid.size > 0:
                    idx_p95 = valid[np.argmin(np.abs(hs[valid] - np.percentile(hs[valid], 95)))]
                    idx_min = valid[np.argmin(hs[valid])]

                selection = [
                    ("Max Hs", idx_max),
                    ("P95 Hs", idx_p95),
                    ("Median Hs", idx_med),
                    ("Min Hs", idx_min),
                ]

                fig2 = make_subplots(
                    rows=2,
                    cols=2,
                    subplot_titles=[s[0] for s in selection],
                    horizontal_spacing=0.06,
                    vertical_spacing=0.10,
                )

                for i, (label, idx_sel) in enumerate(selection):
                    spec2d = np.maximum(spectrum[idx_sel], 1e-12)

                    flat_idx = int(np.argmax(spec2d))
                    fi, di = np.unravel_index(flat_idx, spec2d.shape)

                    fp = freq[fi]
                    Tp = 1.0 / fp
                    theta_p = direction_deg[di]
                    hs_val = hs[idx_sel] if idx_sel < len(hs) else np.nan

                    log_spec = np.log10(spec2d + 1e-6)
                    vmin = np.percentile(log_spec, 5)
                    vmax = np.percentile(log_spec, 99)
                    log_spec = np.clip(log_spec, vmin, vmax)

                    row = i // 2 + 1
                    col = i % 2 + 1

                    fig2.add_trace(
                        go.Heatmap(
                            x=direction_deg,
                            y=period,
                            z=log_spec,
                            colorscale="Blues",
                            showscale=(i == 0),
                            colorbar=dict(title="log10(S) [m²/Hz/deg]", len=0.4),
                        ),
                        row=row,
                        col=col,
                    )
                    label_pos = _smart_text_position(theta_p, Tp, direction_deg, period)
                    fig2.add_trace(
                        go.Scatter(
                            x=[theta_p],
                            y=[Tp],
                            mode="markers+text",
                            marker=dict(color="red", size=10, symbol="x"),
                            text=[f"Hs={hs_val:.2f} m<br>Tp={Tp:.2f} s<br>θ={theta_p:.1f}°"],
                            textposition=label_pos,
                            textfont=dict(size=16, color="red"),
                            showlegend=False,
                        ),
                        row=row,
                        col=col,
                    )

                    fig2.update_xaxes(title_text="Wave direction [deg] (coming-from)", row=row, col=col)
                    fig2.update_yaxes(title_text="Wave period [s] (T = 1/f)", row=row, col=col)

                fig2.update_layout(
                    title=dict(
                        text=(
                            f"Directional wave spectra E(f,θ) — key sea states — {spec_file.name}"
                            f"<br>Spectral point {display_idx + 1} (index={spec_point_idx}){spec_lat_text}{spec_lon_text}"
                        ),
                        x=0.5,
                        xanchor="center",
                    ),
                    template="plotly_white",
                    autosize=True,
                    margin=dict(l=10, r=10, t=90, b=10),
                    font=dict(size=13),
                )

                html_dirspec = fig2.to_html(full_html=False, config={"responsive": True}, auto_play=False)

                dirspec_subtab_id = f"dirspec_sp_{display_idx}_spec_{spec_point_idx}"
                dirspec_buttons_html.append(
                    f'<button onclick="openDirspecSubtab(\'{dirspec_subtab_id}\')">Spectral point {display_idx + 1} (idx {spec_point_idx})</button>'
                )
                dirspec_contents_html.append(
                    f'<div id="{dirspec_subtab_id}" class="dirspec-subtabcontent">'
                    "<div style='height:100%; overflow:auto; padding:10px; box-sizing:border-box;'>"
                    f"<div style='height:95vh; min-height:700px; margin-bottom:24px;'>{html_dirspec}</div>"
                    "</div>"
                    "</div>"
                )

            dashboard_html = (
                f'<div class="dashwrap">'
                f'<div class="subtab">{"".join(dash_buttons_html)}</div>'
                f'{"".join(dash_contents_html)}'
                f'</div>'
            )

            dirspec_tab_html = (
                f'<div class="dashwrap">'
                f'<div class="subtab">{"".join(dirspec_buttons_html)}</div>'
                f'{"".join(dirspec_contents_html)}'
                f'</div>'
            )

        finally:
            ds.close()

    split_reports = bool(globals().get("SPLIT_REPORT_FILES", False))
    auto_open_split = bool(globals().get("AUTO_OPEN_SPLIT_FILES", True))

    if spec_file is None:
        if split_reports:
            out_dir = inputs.directories[0]
            _copy_support_files(out_dir)

            files_to_write = {
                "map2d_inputs": out_dir / "metocean_report_map2d_inputs.html",
                "map2d_outputs": out_dir / "metocean_report_map2d_outputs.html",
                "map3d": out_dir / "metocean_report_map3d.html",
                "overlay": out_dir / "metocean_report_overlay.html",
            }

            _write_html(files_to_write["map2d_inputs"], _single_page_html("2D Inputs", html_2d_inputs))
            _write_html(files_to_write["map2d_outputs"], _single_page_html("2D Outputs", html_2d_outputs))
            _write_html(files_to_write["map3d"], _single_page_html("3D View", html_3d))
            _write_html(files_to_write["overlay"], _single_page_html("Map overlay", html_overlay))

            out_files = list(files_to_write.values())
        else:
            final_html = f"""
            <html>
            {head_block}
            <body>

            <div class="tab">
                <button onclick="openTab('map2d_outputs')">2D Outputs</button>
                <button onclick="openTab('map2d_inputs')">2D Inputs</button>
                <button onclick="openTab('map3d')">3D View</button>
                <button onclick="openTab('overlay')">Map overlay</button>
            </div>

            <div id="map2d_outputs" class="tabcontent">{html_2d_outputs}</div>
            <div id="map2d_inputs" class="tabcontent">{html_2d_inputs}</div>
            <div id="map3d" class="tabcontent">{html_3d}</div>
            <div id="overlay" class="tabcontent">{html_overlay}</div>

            </body>
            </html>
            """
            out_file = inputs.directories[0] / "metocean_report.html"
            _copy_support_files(out_file.parent)
            _write_html(out_file, final_html)
            out_files = [out_file]
    else:
        out_dir = spec_file.parent
        _copy_support_files(out_dir)

        if split_reports:
            files_to_write = {
                "dashboard": out_dir / f"{spec_file.stem}_dashboard.html",
                "dirspec": out_dir / f"{spec_file.stem}_dirspec.html",
                "map2d_inputs": out_dir / f"{spec_file.stem}_map2d_inputs.html",
                "map2d_outputs": out_dir / f"{spec_file.stem}_map2d_outputs.html",
                "map3d": out_dir / f"{spec_file.stem}_map3d.html",
                "overlay": out_dir / f"{spec_file.stem}_overlay.html",
            }

            _write_html(files_to_write["dashboard"], _single_page_html("Dashboard", dashboard_html))
            _write_html(files_to_write["dirspec"], _single_page_html("Directional spectra", dirspec_tab_html))
            _write_html(files_to_write["map2d_inputs"], _single_page_html("2D Inputs", html_2d_inputs))
            _write_html(files_to_write["map2d_outputs"], _single_page_html("2D Outputs", html_2d_outputs))
            _write_html(files_to_write["map3d"], _single_page_html("3D View", html_3d))
            _write_html(files_to_write["overlay"], _single_page_html("Map overlay", html_overlay))

            out_files = list(files_to_write.values())
        else:
            final_html = f"""
            <html>
            {head_block}
            <body>

            <div class="tab">
                <button onclick="openTab('dash')">Dashboard</button>
                <button onclick="openTab('dirspec')">Directional spectra</button>
                <button onclick="openTab('map2d_outputs')">2D Outputs</button>
                <button onclick="openTab('map2d_inputs')">2D Inputs</button>
                <button onclick="openTab('map3d')">3D View</button>
                <button onclick="openTab('overlay')">Map overlay</button>
            </div>

            <div id="dash" class="tabcontent">{dashboard_html}</div>
            <div id="dirspec" class="tabcontent">{dirspec_tab_html}</div>
            <div id="map2d_outputs" class="tabcontent">{html_2d_outputs}</div>
            <div id="map2d_inputs" class="tabcontent">{html_2d_inputs}</div>
            <div id="map3d" class="tabcontent">{html_3d}</div>
            <div id="overlay" class="tabcontent">{html_overlay}</div>

            </body>
            </html>
            """

            out_file = out_dir / f"{spec_file.stem}_full_report.html"
            _write_html(out_file, final_html)
            out_files = [out_file]

    serve_dir = out_files[0].parent

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]

    cmd = [
        sys.executable,
        "-m",
        "http.server",
        str(port),
        "--bind",
        "127.0.0.1",
        "--directory",
        str(serve_dir),
    ]

    print(f"Serving folder: {serve_dir}")
    print(f"Starting server on port {port}")

    try:
        subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        time.sleep(1.0)

        if split_reports:
            if auto_open_split:
                for path in out_files:
                    url = f"http://127.0.0.1:{port}/{path.name}"
                    print(f"Opening report: {url}")
                    webbrowser.open(url)
                    time.sleep(0.3)
            else:
                print("Split report files created:")
                for path in out_files:
                    print(f"  http://127.0.0.1:{port}/{path.name}")
        else:
            url = f"http://127.0.0.1:{port}/{out_files[0].name}"
            print(f"Opening report: {url}")
            webbrowser.open(url)

    except Exception as e:
        print(f"Could not auto-open browser: {e}")
        if split_reports:
            print("Open manually:")
            for path in out_files:
                print(f"  http://127.0.0.1:{port}/{path.name}")
        else:
            print(f"Open manually: http://127.0.0.1:{port}/{out_files[0].name}")


def _run_matplotlib_fallback(
    runs: list[RunData],
    point_coord: tuple[float, float] | tuple[tuple[float, float], ...],
    output_dir: Path,
) -> None:
    import matplotlib.pyplot as plt

    if not runs:
        print("No runs available for matplotlib fallback plotting.")
        return

    points = _normalize_point_coords(point_coord)
    finest_run = runs[-1]
    times = finest_run.grid.times

    for index, (plat, plon) in enumerate(points, start=1):
        poi = _nearest_grid_point(finest_run.grid, (plat, plon))
        hs_series = finest_run.hs[:, poi.iy, poi.ix]
        tp_series = finest_run.tp[:, poi.iy, poi.ix]
        ws_series = (
            finest_run.wind_speed[:, poi.iy, poi.ix]
            if finest_run.wind_speed is not None
            else np.full_like(hs_series, np.nan, dtype=float)
        )

        fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
        axes[0].plot(times, hs_series, color="tab:blue")
        axes[0].set_ylabel("Hs [m]")
        axes[0].grid(alpha=0.3)

        axes[1].plot(times, tp_series, color="tab:purple")
        axes[1].set_ylabel("Tp [s]")
        axes[1].grid(alpha=0.3)

        axes[2].plot(times, ws_series, color="tab:green")
        axes[2].set_ylabel("Wind [m/s]")
        axes[2].set_xlabel("Time")
        axes[2].grid(alpha=0.3)

        fig.suptitle(f"SWAN matplotlib fallback - POI {index}: ({plat:.6f}, {plon:.6f})")
        fig.tight_layout()
        out_path = output_dir / f"swan_mpl_poi_{index:02d}.png"
        fig.savefig(out_path, dpi=160)
        plt.close(fig)
        print(f"Matplotlib fallback plot written: {out_path.resolve()}")

def main() -> None:
    use_code_config = USE_CODE_CONFIG and len(sys.argv) == 1
    inputs = _build_inputs_from_code_config() if use_code_config else parse_args()

    point_coords = _normalize_point_coords(inputs.point_coord)
    first_point_coord = point_coords[0]

    html_2d_inputs = ""
    html_2d_outputs = ""
    html_3d = ""
    html_overlay = "<div style='padding:20px'>No overlay available</div>"

    nc_paths = [p for p in inputs.nc_files if p is not None]

    if nc_paths:
        datasets = [(path.parent.name, xr.open_dataset(path)) for path in nc_paths]
        try:
            runs = _prepare_runs_data(datasets)
            fig_2d_outputs = build_outputs_figure(
                datasets=datasets,
                point_coord=inputs.point_coord,
                wind_arrow_resolution=inputs.wind_arrow_resolution,
            )
            html_2d_outputs = fig_2d_outputs.to_html(
                full_html=False,
                config={"responsive": True},
                auto_play=False,
            )

            fig_2d_inputs = build_inputs_figure(
                datasets=datasets,
                point_coord=inputs.point_coord,
                wind_arrow_resolution=inputs.wind_arrow_resolution,
            )
            html_2d_inputs = fig_2d_inputs.to_html(
                full_html=False,
                config={"responsive": True},
                auto_play=False,
            )

            html_3d = _plot_depth_hs_plane_3d(
                runs=runs,
                point_coord=inputs.point_coord,
                output_dir=inputs.directories[0],
                arrow_resolution=inputs.wind_arrow_resolution,
            )

            if inputs.export_hs_format is not None:
                finest_ds = min(
                    datasets,
                    key=lambda item: _grid_resolution_score(_build_grid_info(item[1]))
                )[1]

                resolved_time_index = _resolve_export_time_index(
                    ds=finest_ds,
                    point_coord=first_point_coord,
                    export_time_index=inputs.export_time_index,
                )

                suffix = {"geojson": ".geojson", "kml": ".kml", "kmz": ".kmz"}[inputs.export_hs_format]

                output_path = inputs.directories[0] / (
                    f"{'_'.join(p.parent.name for p in nc_paths)}_hs_t{resolved_time_index}{suffix}"
                )

                print(f"Export directory: {inputs.directories[0].resolve()}")

                exported = _export_hs_geojson_or_kml_nested(
                    runs,
                    output_path,
                    inputs.export_hs_format,
                    resolved_time_index,
                )

                print(f"HS colormap export written: {exported.resolve()}")

                if inputs.export_hs_format == "kmz":
                    print("KMZ content: doc.kml (inside the .kmz archive)")

                geojson_path = output_path.with_suffix(".geojson")

                _export_hs_geojson_or_kml_nested(
                    runs,
                    geojson_path,
                    "geojson",
                    resolved_time_index,
                )

                bathy_path = geojson_path.with_name(geojson_path.stem + "_bathy.geojson")

                _export_bathymetry_contours(
                    finest_ds,
                    bathy_path,
                )

                if geojson_path.exists():
                    html_overlay = _build_leaflet_overlay_html(
                        geojson_path=geojson_path,
                        bathy_geojson_path=bathy_path if bathy_path.exists() else None,
                    )

        finally:
            for _, ds in datasets:
                ds.close()

    generate_metocean_report(
        inputs,
        html_2d_inputs=html_2d_inputs,
        html_2d_outputs=html_2d_outputs,
        html_3d=html_3d,
        html_overlay=html_overlay,
        geojson_path=geojson_path if "geojson_path" in locals() else None,
        bathy_geojson_path=bathy_path if "bathy_path" in locals() else None,
    )


if __name__ == "__main__":
    main()
