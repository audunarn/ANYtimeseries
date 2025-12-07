"""
Utility script for exporting ERA5 data to SWAN inputs.

This module includes a simple scatter mode that extracts values at
user-specified coordinates. When scatter mode runs we also generate a
quick domain overview map showing the spatial extent alongside any
scatter locations.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from swan_helpers import Domain, parse_scatter_args, run_scatter_mode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ERA5 to SWAN converter (multi-process)")
    parser.add_argument("--lon-min", type=float, required=True, help="Minimum longitude")
    parser.add_argument("--lon-max", type=float, required=True, help="Maximum longitude")
    parser.add_argument("--lat-min", type=float, required=True, help="Minimum latitude")
    parser.add_argument("--lat-max", type=float, required=True, help="Maximum latitude")
    parser.add_argument(
        "--scatter",
        nargs="*",
        metavar="LON,LAT[,LABEL]",
        help="Enable scatter mode with one or more lon/lat entries",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("output"),
        help="Directory for generated outputs",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    domain = Domain(args.lon_min, args.lon_max, args.lat_min, args.lat_max)

    if args.scatter:
        scatter_points = parse_scatter_args(args.scatter)
        run_scatter_mode(domain, scatter_points, Path(args.out_dir))
    else:  # pragma: no cover - only executed outside scatter mode
        raise SystemExit("Scatter mode requested but no scatter points provided.")


if __name__ == "__main__":
    main()
