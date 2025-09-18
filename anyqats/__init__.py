# -*- coding: utf-8 -*-
"""
Library for efficient processing and visualization of time series.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .ts import TimeSeries

__all__ = ["TimeSeries", "TsDB", "__version__", "__version_tuple__", "__summary__"]

if TYPE_CHECKING:  # pragma: no cover - used for static type checking only
    from .tsdb import TsDB


def __getattr__(name: str):
    """Provide lazy access to heavy submodules.

    Importing :mod:`anyqats` previously pulled in :mod:`anyqats.tsdb` which
    requires optional third-party libraries such as :mod:`h5py`.  The SAT
    converter only needs lightweight utilities, therefore we resolve the
    ``TsDB`` attribute on demand and raise a helpful error message when its
    optional dependencies are unavailable.
    """

    if name == "TsDB":
        try:
            from .tsdb import TsDB as _TsDB
        except ModuleNotFoundError as exc:  # pragma: no cover - depends on runtime deps
            if exc.name == "h5py":
                raise ImportError(
                    "anyqats.TsDB requires the optional dependency 'h5py'. "
                    "Install h5py to enable SIMA HDF5 I/O support."
                ) from exc
            raise
        globals()["TsDB"] = _TsDB
        return _TsDB
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

# version
try:
    from ._version import __version__, __version_tuple__
except ImportError:
    __version__ = "0.0.0"
    __version_tuple__ = (0, 0, 0)

# summary
__summary__ = __doc__
