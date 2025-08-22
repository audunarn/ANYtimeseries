"""Support for launching the AnyQATS GUI.

This module exposes a :func:`main` function that serves as an entry point
for the desktop application.  It delegates to :func:`anyqats.cli.launch_app`
with default arguments, allowing the GUI to be started simply by executing
``python -m anyqats.app``.
"""

from __future__ import annotations


def main() -> None:
    """Launch the graphical user interface.

    The heavy lifting of setting up and starting the application lives in
    :func:`anyqats.cli.launch_app`.  Importing inside the function keeps import
    time light and avoids circular dependencies during package initialisation.
    """

    from ..cli import launch_app

    # Mirror the default behaviour of ``anyqats app`` which uses the current
    # working directory unless the user specifies otherwise.
    launch_app(home=False)


__all__ = ["main"]

