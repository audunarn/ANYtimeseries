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

from swan_helpers import (
    Domain,
    ManualWaveConfig,
    parse_scatter_args,
    run_scatter_mode,
    write_manual_run_configuration,
)


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
    parser.add_argument(
        "--scatter-data",
        type=Path,
        default=None,
        help=(
            "Optional CSV file containing offshore/nearshore Hs and Tp columns "
            "for scatter plotting"
        ),
    )

    manual = parser.add_argument_group("Manual wave height mode")
    manual.add_argument(
        "--run-manual",
        action="store_true",
        help="Execute manual SWAN mode alongside optional scatter extraction.",
    )
    manual.add_argument("--wave-height", type=float, help="Manual significant wave height (m)")
    manual.add_argument("--peak-period", type=float, help="Manual peak period (s)")
    manual.add_argument(
        "--wave-direction", type=float, help="Wave direction used for manual forcing (deg)"
    )
    manual.add_argument(
        "--wind",
        type=float,
        default=None,
        help="Aligned wind speed in m/s. Use zero to disable (OFF QUAD)",
    )
    manual.add_argument(
        "--physics",
        choices=["GEN2", "GEN3", "WESTHUYSEN"],
        default="GEN3",
        help="Physics option to test in manual mode",
    )
    manual.add_argument(
        "--flat-bottom",
        type=float,
        default=None,
        help="Optional constant depth (m) for flat-bottom runs",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    domain = Domain(args.lon_min, args.lon_max, args.lat_min, args.lat_max)

    if args.run_manual:
        missing = [
            name
            for name, value in {
                "--wave-height": args.wave_height,
                "--peak-period": args.peak_period,
                "--wave-direction": args.wave_direction,
            }.items()
            if value is None
        ]
        if missing:
            raise SystemExit(
                "Manual mode requires the following arguments: " + ", ".join(missing)
            )
        if args.flat_bottom is not None and args.flat_bottom <= 0:
            raise SystemExit("Flat bottom depth must be a positive value.")

        manual_config = ManualWaveConfig(
            wave_height=args.wave_height,
            peak_period=args.peak_period,
            direction=args.wave_direction,
            wind_speed=args.wind,
            physics=args.physics,
            flat_bottom_depth=args.flat_bottom,
        )

        out_dir = Path(args.out_dir)
        config_path = out_dir / "manual_run.txt"
        write_manual_run_configuration(manual_config, config_path)

    if args.scatter:
        scatter_points = parse_scatter_args(args.scatter)
        run_scatter_mode(
            domain,
            scatter_points,
            Path(args.out_dir),
            scatter_data=args.scatter_data,
        )
    elif not args.run_manual:  # pragma: no cover - only executed outside scatter mode
        raise SystemExit("Scatter mode requested but no scatter points provided.")


if __name__ == "__main__":
    main()
