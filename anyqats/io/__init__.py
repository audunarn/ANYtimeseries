# -*- coding: utf-8 -*-
"""Sub-package with io for various file formats."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Dict

__all__ = [
    "acis_sat_converter",
    "csv",
    "direct_access",
    "other",
    "sima",
    "sima_h5",
    "sintef_mat",
    "matlab",
]

_LAZY_MODULES: Dict[str, str] = {
    "acis_sat_converter": ".acis_sat_converter",
    "csv": ".csv",
    "direct_access": ".direct_access",
    "other": ".other",
    "sima": ".sima",
    "sima_h5": ".sima_h5",
    "sintef_mat": ".sintef_mat",
    "matlab": ".sintef_mat",
}

if TYPE_CHECKING:  # pragma: no cover - imported for static analysis only
    from . import acis_sat_converter, csv, direct_access, other, sima, sima_h5, sintef_mat


def __getattr__(name: str):
    module_name = _LAZY_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(module_name, __name__)
    globals()[name] = module
    return module
