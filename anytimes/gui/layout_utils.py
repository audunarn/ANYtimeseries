"""Shared helpers for window sizing and responsive layouts."""
from __future__ import annotations

from typing import Tuple

from PySide6.QtGui import QGuiApplication


def apply_initial_size(
    widget,
    desired_width: float,
    desired_height: float,
    *,
    min_width: int = 720,
    min_height: int = 520,
    width_ratio: float = 0.9,
    height_ratio: float = 0.9,
) -> Tuple[int, int]:
    """Resize *widget* using the available screen geometry.

    Parameters
    ----------
    widget:
        The Qt widget or window whose size is being configured.
    desired_width, desired_height:
        The ideal window dimensions when plenty of screen space exists.
    min_width, min_height:
        Lower bounds that keep the UI usable even on compact displays.
    width_ratio, height_ratio:
        Fractions of the available screen size that will be used as an
        upper bound. The ratios ensure that a small border remains around
        dialogs so they do not overflow on low resolution screens.

    Returns
    -------
    tuple[int, int]
        The width and height applied to ``widget``.
    """

    screen = QGuiApplication.primaryScreen()
    if screen is None:
        width = int(desired_width)
        height = int(desired_height)
    else:
        available = screen.availableGeometry()
        width_limit = available.width()
        height_limit = available.height()

        width_cap = int(width_limit * width_ratio)
        height_cap = int(height_limit * height_ratio)

        width = int(min(width_limit, max(min_width, min(desired_width, width_cap))))
        height = int(min(height_limit, max(min_height, min(desired_height, height_cap))))

    widget.resize(width, height)
    widget.setMinimumSize(min(min_width, width), min(min_height, height))
    return width, height


__all__ = ["apply_initial_size"]

