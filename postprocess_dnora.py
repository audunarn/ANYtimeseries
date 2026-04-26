#!/usr/bin/env python3
"""Post-process SWAN/DNORA output with minimal user input."""

from __future__ import annotations

import argparse
from pathlib import Path

from anytimes.swanpost import autodetect_file, load_timeseries, plot_timeseries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Postprocess SWAN/DNORA netCDF output")
    parser.add_argument("directory", nargs="?", type=Path, help="Run directory containing .nc/.BOT")
    parser.add_argument("--nc", type=str, default=None, help="netCDF filename (optional, auto-detect when omitted)")
    parser.add_argument("--bot", type=str, default=None, help="BOT filename (optional, auto-detect when omitted)")
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


def main() -> None:
    args = parse_args()
    directory, nc_name, bot_name = prompt_missing_inputs(args)

    if not directory.exists() or not directory.is_dir():
        raise NotADirectoryError(f"Directory not found: {directory}")

    bot_path = autodetect_file(directory, ".bot", bot_name)
    nc_path = autodetect_file(directory, ".nc", nc_name)

    print(f"Using directory : {directory}")
    print(f"Using BOT file  : {bot_path.name}")
    print(f"Using netCDF file: {nc_path.name}")

    ts_data = load_timeseries(nc_path, point_index=args.point_index)
    plot_timeseries(ts_data, title=f"SWAN postprocessing — {nc_path.name}")


if __name__ == "__main__":
    main()
