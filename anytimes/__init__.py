"""ANYtimes public package interface."""

__version__ = "1.1.0"

try:
    from . import anytimes_gui  # noqa: F401
    from .anytimes_gui import *  # noqa: F401,F403
except Exception:  # pragma: no cover - GUI dependencies are optional during tests
    anytimes_gui = None  # type: ignore[assignment]

from . import evm  # noqa: F401

__all__ = [
    "__version__",
    *[name for name in globals() if not name.startswith("_")],
]
