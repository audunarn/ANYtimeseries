#!/usr/bin/env python
"""Run the ANYtimeSeries desktop application from this checkout."""

from __future__ import annotations


def main() -> None:
    """Launch the GUI using the package's maintained entry point."""

    from anytimes.anytimes_gui import main as gui_main

    gui_main()


if __name__ == "__main__":
    main()
