"""AnytimeSeries Qt GUI entry point and public exports."""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Use software rendering for QtWebEngine to avoid "context lost" errors on
# systems without proper GPU support. Respect existing environment variables
# if they are already defined by the user.
os.environ.setdefault("QTWEBENGINE_DISABLE_GPU", "1")
os.environ.setdefault("QTWEBENGINE_CHROMIUM_FLAGS", "--disable-gpu")
os.environ.setdefault("QT_QUICK_BACKEND", "software")

from PySide6.QtWidgets import QApplication
from PySide6.QtWebEngineQuick import QtWebEngineQuick

if __package__ in {None, ""}:
    # Allow running the module as a script (``python anytimes/anytimes_gui.py``)
    # by ensuring the package root is importable before falling back to an
    # absolute import. When the module is imported as part of the package,
    # ``__package__`` is already set and the relative import below is used.
    package_root = Path(__file__).resolve().parent
    sys.path.insert(0, str(package_root.parent))

    from anytimes.gui import (
        ORCAFLEX_VARIABLE_MAP,
        MATH_FUNCTIONS,
        FileLoader,
        EVMWindow,
        FatigueDialog,
        OrcaflexVariableSelector,
        SortableTableWidgetItem,
        StatsDialog,
        TimeSeriesEditorQt,
        VariableRowWidget,
        VariableTab,
        _find_xyz_triples,
        _looks_like_user_var,
        _matches_terms,
        _parse_search_terms,
        _safe,
        get_object_available_vars,
    )
else:
    from .gui import (
        ORCAFLEX_VARIABLE_MAP,
        MATH_FUNCTIONS,
        FileLoader,
        EVMWindow,
        FatigueDialog,
        OrcaflexVariableSelector,
        SortableTableWidgetItem,
        StatsDialog,
        TimeSeriesEditorQt,
        VariableRowWidget,
        VariableTab,
        _find_xyz_triples,
        _looks_like_user_var,
        _matches_terms,
        _parse_search_terms,
        _safe,
        get_object_available_vars,
    )

__all__ = [
    "ORCAFLEX_VARIABLE_MAP",
    "MATH_FUNCTIONS",
    "FileLoader",
    "EVMWindow",
    "FatigueDialog",
    "OrcaflexVariableSelector",
    "SortableTableWidgetItem",
    "StatsDialog",
    "TimeSeriesEditorQt",
    "VariableRowWidget",
    "VariableTab",
    "_find_xyz_triples",
    "_looks_like_user_var",
    "_matches_terms",
    "_parse_search_terms",
    "_safe",
    "get_object_available_vars",
    "main",
]


def main() -> None:
    """Launch the AnytimeSeries GUI."""
    app = QApplication(sys.argv)
    # Explicitly initialise QtWebEngine before creating any widgets. Without
    # this step, the bundled `QtWebEngineProcess.exe` may not start correctly
    # on Windows builds, leaving only an empty frame visible.
    QtWebEngineQuick.initialize()
    window = TimeSeriesEditorQt()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
