#!/usr/bin/env python3
"""Post-process SWAN/DNORA output with minimal user input.

Features:
- Ask for the directory once (or accept via CLI argument)
- Auto-detect `.BOT` file when no BOT filename is supplied
- Auto-detect `.nc` file when no netCDF filename is supplied
- Plot `Hs`, `Tp`, and wind speed including direction arrows
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


@dataclass(frozen=True)
class TimeseriesData:
    time: np.ndarray
    hs: np.ndarray
    tp: np.ndarray
    wind_speed: np.ndarray
    wind_dir_deg: np.ndarray


HS_CANDIDATES = ("hs", "swh", "significant_wave_height", "Hsig")
TP_CANDIDATES = ("tp", "peak_period", "peak_wave_period", "Tm01")
WIND_SPEED_CANDIDATES = ("wind_speed", "wspd", "ws", "windspd")
WIND_DIR_CANDIDATES = ("wind_dir", "wdir", "wd", "winddir")
WIND_U_CANDIDATES = ("wind_u", "u10", "uwnd", "x_wind")
WIND_V_CANDIDATES = ("wind_v", "v10", "vwnd", "y_wind")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Postprocess SWAN/DNORA netCDF output")
    parser.add_argument("directory", nargs="?", type=Path, help="Run directory containing .nc/.BOT")
    parser.add_argument("--nc", type=str, default=None, help="netCDF filename (optional, auto-detect when omitted)")
    parser.add_argument("--bot", type=str, default=None, help="BOT filename (optional, auto-detect when omitted)")
    parser.add_argument("--show-folder", action="store_true", help="Print a concise folder content overview")
    parser.add_argument("--point-index", type=int, default=None, help="Optional flattened grid-point index")
    return parser.parse_args()


def prompt_missing_inputs(args: argparse.Namespace) -> tuple[Path, str | None, str | None]:
    directory = args.directory
    if directory is None:
        directory = Path(input("Directory path: ").strip())

    nc_name = args.nc
    bot_name = args.bot

    if nc_name is None and args.directory is None:
        typed = input("NC filename (Enter for auto-detect): ").strip()
        nc_name = typed or None
    if bot_name is None and args.directory is None:
        typed = input("BOT filename (Enter for auto-detect): ").strip()
        bot_name = typed or None

    return directory.expanduser().resolve(), nc_name, bot_name


def list_folder(directory: Path) -> None:
    print("\nTypical folder content (detected):")
    for item in sorted(directory.iterdir(), key=lambda p: (p.suffix.lower(), p.name.lower())):
        kind = "DIR" if item.is_dir() else item.suffix.lower() or "FILE"
        size = "-" if item.is_dir() else f"{item.stat().st_size / 1024:.1f} KB"
        print(f"  {item.name:<40} {kind:>8} {size:>10}")


def _pick_best_file(files: Sequence[Path], suffix: str) -> Path:
    if not files:
        raise FileNotFoundError(f"No {suffix} file found.")

    if suffix.lower() == ".nc":
        filtered = [f for f in files if "spec" not in f.name.lower()]
        if len(filtered) == 1:
            return filtered[0]
        if filtered:
            files = filtered

    if len(files) == 1:
        return files[0]

    return sorted(files, key=lambda p: (-p.stat().st_size, p.name.lower()))[0]


def autodetect_file(directory: Path, suffix: str, explicit_name: str | None) -> Path:
    if explicit_name:
        path = directory / explicit_name
        if not path.exists():
            raise FileNotFoundError(f"Requested file does not exist: {path}")
        return path

    candidates = [p for p in directory.iterdir() if p.is_file() and p.suffix.lower() == suffix.lower()]
    picked = _pick_best_file(candidates, suffix)
    print(f"Auto-detected {suffix} file: {picked.name}")
    return picked


def _get_var(ds: xr.Dataset, names: Iterable[str]) -> xr.DataArray | None:
    for name in names:
        if name in ds:
            return ds[name]
    return None


def _to_series(da: xr.DataArray, point_index: int | None = None) -> xr.DataArray:
    if "time" not in da.dims:
        raise ValueError(f"Variable '{da.name}' has no 'time' dimension.")

    non_time_dims = [d for d in da.dims if d != "time"]
    if not non_time_dims:
        return da

    stacked = da.stack(point=non_time_dims)
    index = point_index if point_index is not None else int(stacked.sizes["point"] // 2)
    return stacked.isel(point=index)


def _wind_from_uv(ds: xr.Dataset, point_index: int | None) -> tuple[xr.DataArray | None, xr.DataArray | None]:
    u = _get_var(ds, WIND_U_CANDIDATES)
    v = _get_var(ds, WIND_V_CANDIDATES)
    if u is None or v is None:
        return None, None

    u_s = _to_series(u, point_index)
    v_s = _to_series(v, point_index)
    speed = np.hypot(u_s, v_s)
    direction = (np.degrees(np.arctan2(u_s, v_s)) + 360.0) % 360.0
    return speed, direction


def load_timeseries(nc_file: Path, point_index: int | None = None) -> TimeseriesData:
    ds = xr.open_dataset(nc_file)
    try:
        hs = _get_var(ds, HS_CANDIDATES)
        tp = _get_var(ds, TP_CANDIDATES)
        if hs is None or tp is None:
            raise KeyError(
                f"Could not find required Hs/Tp variables. Found variables: {list(ds.data_vars)}"
            )

        hs_s = _to_series(hs, point_index)
        tp_s = _to_series(tp, point_index)

        wind_speed = _get_var(ds, WIND_SPEED_CANDIDATES)
        wind_dir = _get_var(ds, WIND_DIR_CANDIDATES)

        if wind_speed is not None and wind_dir is not None:
            ws_s = _to_series(wind_speed, point_index)
            wd_s = _to_series(wind_dir, point_index)
        else:
            ws_s, wd_s = _wind_from_uv(ds, point_index)
            if ws_s is None or wd_s is None:
                raise KeyError(
                    "Could not find wind speed/direction variables or wind U/V components. "
                    f"Found variables: {list(ds.data_vars)}"
                )

        time = hs_s["time"].values
        return TimeseriesData(
            time=np.asarray(time),
            hs=np.asarray(hs_s.values, dtype=float),
            tp=np.asarray(tp_s.values, dtype=float),
            wind_speed=np.asarray(ws_s.values, dtype=float),
            wind_dir_deg=np.asarray(wd_s.values, dtype=float),
        )
    finally:
        ds.close()


def _arrow_components_from_met_direction(speed: np.ndarray, direction_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Meteorological direction = direction coming *from*.
    # Convert to vector pointing where wind goes to for plotting arrows.
    theta = np.radians((direction_deg + 180.0) % 360.0)
    u = speed * np.sin(theta)
    v = speed * np.cos(theta)
    return u, v


def plot_timeseries(data: TimeseriesData, title: str) -> None:
    t = np.asarray(data.time)
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(13, 9), sharex=True)

    axes[0].plot(t, data.hs, color="tab:blue", lw=1.8)
    axes[0].set_ylabel("Hs [m]")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, data.tp, color="tab:purple", lw=1.8)
    axes[1].set_ylabel("Tp [s]")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(t, data.wind_speed, color="tab:green", lw=1.8, label="Wind speed")
    u, v = _arrow_components_from_met_direction(data.wind_speed, data.wind_dir_deg)

    n = len(t)
    stride = max(1, n // 40)
    axes[2].quiver(
        t[::stride],
        data.wind_speed[::stride],
        u[::stride],
        v[::stride],
        angles="xy",
        scale_units="xy",
        scale=max(np.nanmax(data.wind_speed), 1.0) * 8,
        width=0.002,
        color="tab:orange",
        alpha=0.85,
    )
    axes[2].set_ylabel("Wind [m/s]")
    axes[2].set_xlabel("Time")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc="upper right")

    axes[0].set_title(title)
    locator = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)
    axes[2].xaxis.set_major_locator(locator)
    axes[2].xaxis.set_major_formatter(formatter)

    fig.tight_layout()
    plt.show()


def main() -> None:
    args = parse_args()
    directory, nc_name, bot_name = prompt_missing_inputs(args)

    if not directory.exists() or not directory.is_dir():
        raise NotADirectoryError(f"Directory not found: {directory}")

    if args.show_folder:
        list_folder(directory)

    bot_path = autodetect_file(directory, ".bot", bot_name)
    nc_path = autodetect_file(directory, ".nc", nc_name)

    print(f"Using directory : {directory}")
    print(f"Using BOT file  : {bot_path.name}")
    print(f"Using netCDF file: {nc_path.name}")

    ts_data = load_timeseries(nc_path, point_index=args.point_index)
    plot_timeseries(ts_data, title=f"SWAN postprocessing — {nc_path.name}")


if __name__ == "__main__":
    main()
