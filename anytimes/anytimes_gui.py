# AnytimeSeries PySide6 with universal FileLoader
import sys
import os
import warnings
import datetime

# Use software rendering for QtWebEngine to avoid "context lost" errors on
# systems without proper GPU support. Respect existing environment variables
# if they are already defined by the user.
os.environ.setdefault("QTWEBENGINE_DISABLE_GPU", "1")
os.environ.setdefault("QTWEBENGINE_CHROMIUM_FLAGS", "--disable-gpu")
os.environ.setdefault("QT_QUICK_BACKEND", "software")

import re
import numpy as np
import pandas as pd
import scipy.io
import json
import subprocess
import anyqats as qats
from anyqats import TimeSeries, TsDB
from collections.abc import Sequence
from array import array
from PySide6.QtWidgets import (

    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QGridLayout,
    QListWidget, QTabWidget, QLabel, QLineEdit, QCheckBox, QRadioButton,
    QFileDialog, QProgressBar, QTextEdit, QGroupBox, QSplitter, QComboBox,
    QSpinBox, QScrollArea, QDoubleSpinBox, QListWidgetItem, QAbstractItemView,
    QListWidget, QMessageBox, QDialog, QVBoxLayout, QPlainTextEdit, QPushButton,
    QInputDialog, QApplication, QTableWidget, QTableWidgetItem, QHeaderView,

    QStyleFactory, QSizePolicy, QSpacerItem,

)

from PySide6.QtCore import Qt, Slot, Signal, QEvent, QTimer, QUrl
from PySide6.QtGui import (
    QTextCursor, QKeySequence, QGuiApplication, QPalette, QColor, QKeyEvent
)
from PySide6.QtWebEngineWidgets import QWebEngineView
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure


"""
FileLoader for TimeSeriesEditorQt
Supports generic (csv, mat, h5, etc) and OrcaFlex .sim files with Qt-based variable selection dialog.
"""


# You should provide these in your app context:
# from anyqats import TsDB, TimeSeries
# import OrcFxAPI (for .sim files)
class SortableTableWidgetItem(QTableWidgetItem):
    def __lt__(self, other):
        my = self.data(Qt.ItemDataRole.UserRole)
        other_val = other.data(Qt.ItemDataRole.UserRole)
        if isinstance(my, (int, float)) and isinstance(other_val, (int, float)):
            return my < other_val
        return super().__lt__(other)

ORCAFLEX_VARIABLE_MAP = {'Vessel': ['X', 'Y', 'Z', 'Dynamic x', 'Dynamic y', 'Dynamic z', 'Rotation 1', 'Rotation 2', 'Rotation 3', 'Dynamic Rx', 'Dynamic Ry', 'Dynamic Rz', 'Sea surface Z', 'Sea surface clearance', 'Sea velocity', 'Sea X velocity', 'Sea Y velocity', 'Sea Z velocity', 'Sea acceleration', 'Sea X acceleration', 'Sea Y acceleration', 'Sea Z acceleration', 'Disturbed sea surface Z', 'Disturbed sea surface clearance', 'Disturbed sea velocity', 'Disturbed sea X velocity', 'Disturbed sea Y velocity', 'Disturbed sea Z velocity', 'Disturbed sea acceleration', 'Disturbed sea X acceleration', 'Disturbed sea Y acceleration', 'Disturbed sea Z acceleration', 'Air gap', 'Velocity', 'GX velocity', 'GY velocity', 'GZ velocity', 'x velocity', 'y velocity', 'z velocity', 'Angular velocity', 'x angular velocity', 'y angular velocity', 'z angular velocity', 'Acceleration', 'GX acceleration', 'GY acceleration', 'GZ acceleration', 'x acceleration', 'y acceleration', 'z acceleration', 'Acceleration rel. g', 'x acceleration rel. g', 'y acceleration rel. g', 'z acceleration rel. g', 'Angular acceleration', 'x angular acceleration', 'y angular acceleration', 'z angular acceleration', 'Primary X', 'Primary Y', 'Primary Z', 'Primary rotation 1', 'Primary rotation 2', 'Primary rotation 3', 'Primary velocity', 'Primary x velocity', 'Primary y velocity', 'Primary z velocity', 'Primary angular velocity', 'Primary x angular velocity', 'Primary y angular velocity', 'Primary z angular velocity', 'Primary acceleration', 'Primary x acceleration', 'Primary y acceleration', 'Primary z acceleration', 'Primary angular acceleration', 'Primary x angular acceleration', 'Primary y angular acceleration', 'Primary z angular acceleration', 'Primary LF X', 'Primary LF Y', 'Primary LF Z', 'Primary LF rotation 1', 'Primary LF rotation 2', 'Primary LF rotation 3', 'Primary WF surge', 'Primary WF sway', 'Primary WF heave', 'Primary WF roll', 'Primary WF pitch', 'Primary WF yaw', 'Total force', 'Total Lx force', 'Total Ly force', 'Total Lz force', 'Total moment', 'Total Lx moment', 'Total Ly moment', 'Total Lz moment', 'Connections force', 'Connections Lx force', 'Connections Ly force', 'Connections Lz force', 'Connections moment', 'Connections Lx moment', 'Connections Ly moment', 'Connections Lz moment', 'Connections GX force', 'Connections GY force', 'Connections GZ force', 'Connections GX moment', 'Connections GY moment', 'Connections GZ moment', 'Hydrostatic stiffness force', 'Hydrostatic stiffness Lx force', 'Hydrostatic stiffness Ly force', 'Hydrostatic stiffness Lz force', 'Hydrostatic stiffness moment', 'Hydrostatic stiffness Lx moment', 'Hydrostatic stiffness Ly moment', 'Hydrostatic stiffness Lz moment', 'Morison elements force', 'Morison elements Lx force', 'Morison elements Ly force', 'Morison elements Lz force', 'Morison elements moment', 'Morison elements Lx moment', 'Morison elements Ly moment', 'Morison elements Lz moment', 'Morison element drag force', 'Morison element Lx drag force', 'Morison element Ly drag force', 'Morison element Lz drag force', 'Morison element fluid inertia force', 'Morison element Lx fluid inertia force', 'Morison element Ly fluid inertia force', 'Morison element Lz fluid inertia force', 'Morison element segment proportion wet', 'Morison element segment relative velocity', 'Morison element segment normal relative velocity', 'Morison element segment x relative velocity', 'Morison element segment y relative velocity', 'Morison element segment z relative velocity', 'Morison element segment x drag coefficient', 'Morison element segment y drag coefficient', 'Morison element segment z drag coefficient', 'Morison element segment drag force', 'Morison element segment x drag force', 'Morison element segment y drag force', 'Morison element segment z drag force', 'Morison element segment fluid inertia force', 'Morison element segment x fluid inertia force', 'Morison element segment y fluid inertia force', 'Morison element segment z fluid inertia force', 'Applied force', 'Applied Lx force', 'Applied Ly force', 'Applied Lz force', 'Applied moment', 'Applied Lx moment', 'Applied Ly moment', 'Applied Lz moment', 'Wave (1st order) force', 'Wave (1st order) Lx force', 'Wave (1st order) Ly force', 'Wave (1st order) Lz force', 'Wave (1st order) moment', 'Wave (1st order) Lx moment', 'Wave (1st order) Ly moment', 'Wave (1st order) Lz moment', 'Wave drift (2nd order) force', 'Wave drift (2nd order) Lx force', 'Wave drift (2nd order) Ly force', 'Wave drift (2nd order) Lz force', 'Wave drift (2nd order) moment', 'Wave drift (2nd order) Lx moment', 'Wave drift (2nd order) Ly moment', 'Wave drift (2nd order) Lz moment', 'Sum frequency force', 'Sum frequency Lx force', 'Sum frequency Ly force', 'Sum frequency Lz force', 'Sum frequency moment', 'Sum frequency Lx moment', 'Sum frequency Ly moment', 'Sum frequency Lz moment', 'Added mass & damping force', 'Added mass & damping Lx force', 'Added mass & damping Ly force', 'Added mass & damping Lz force', 'Added mass & damping moment', 'Added mass & damping Lx moment', 'Added mass & damping Ly moment', 'Added mass & damping Lz moment'], 'Constraint': ['Displacement', 'x', 'y', 'z', 'Angular displacement', 'Rx', 'Ry', 'Rz', 'Velocity', 'x velocity', 'y velocity', 'z velocity', 'Angular velocity', 'x angular velocity', 'y angular velocity', 'z angular velocity', 'Acceleration', 'x acceleration', 'y acceleration', 'z acceleration', 'Angular acceleration', 'x angular acceleration', 'y angular acceleration', 'z angular acceleration', 'In-frame X', 'In-frame Y', 'In-frame Z', 'In-frame dynamic x', 'In-frame dynamic y', 'In-frame dynamic z', 'In-frame azimuth', 'In-frame declination', 'In-frame gamma', 'In-frame dynamic Rx', 'In-frame dynamic Ry', 'In-frame dynamic Rz', 'In-frame velocity', 'In-frame GX velocity', 'In-frame GY velocity', 'In-frame GZ velocity', 'In-frame x velocity', 'In-frame y velocity', 'In-frame z velocity', 'In-frame angular velocity', 'In-frame x angular velocity', 'In-frame y angular velocity', 'In-frame z angular velocity', 'In-frame acceleration', 'In-frame GX acceleration', 'In-frame GY acceleration', 'In-frame GZ acceleration', 'In-frame x acceleration', 'In-frame y acceleration', 'In-frame z acceleration', 'In-frame angular acceleration', 'In-frame x angular acceleration', 'In-frame y angular acceleration', 'In-frame z angular acceleration', 'In-frame connection force', 'In-frame connection GX force', 'In-frame connection GY force', 'In-frame connection GZ force', 'In-frame connection Lx force', 'In-frame connection Ly force', 'In-frame connection Lz force', 'In-frame connection moment', 'In-frame connection GX moment', 'In-frame connection GY moment', 'In-frame connection GZ moment', 'In-frame connection Lx moment', 'In-frame connection Ly moment', 'In-frame connection Lz moment', 'Out-frame X', 'Out-frame Y', 'Out-frame Z', 'Out-frame dynamic x', 'Out-frame dynamic y', 'Out-frame dynamic z', 'Out-frame azimuth', 'Out-frame declination', 'Out-frame gamma', 'Out-frame dynamic Rx', 'Out-frame dynamic Ry', 'Out-frame dynamic Rz', 'Out-frame velocity', 'Out-frame GX velocity', 'Out-frame GY velocity', 'Out-frame GZ velocity', 'Out-frame x velocity', 'Out-frame y velocity', 'Out-frame z velocity', 'Out-frame angular velocity', 'Out-frame x angular velocity', 'Out-frame y angular velocity', 'Out-frame z angular velocity', 'Out-frame acceleration', 'Out-frame GX acceleration', 'Out-frame GY acceleration', 'Out-frame GZ acceleration', 'Out-frame x acceleration', 'Out-frame y acceleration', 'Out-frame z acceleration', 'Out-frame angular acceleration', 'Out-frame x angular acceleration', 'Out-frame y angular acceleration', 'Out-frame z angular acceleration', 'Out-frame connection force', 'Out-frame connection GX force', 'Out-frame connection GY force', 'Out-frame connection GZ force', 'Out-frame connection Lx force', 'Out-frame connection Ly force', 'Out-frame connection Lz force', 'Out-frame connection moment', 'Out-frame connection GX moment', 'Out-frame connection GY moment', 'Out-frame connection GZ moment', 'Out-frame connection Lx moment', 'Out-frame connection Ly moment', 'Out-frame connection Lz moment'], '6Dbuoy': ['X', 'Y', 'Z', 'Dynamic x', 'Dynamic y', 'Dynamic z', 'Rotation 1', 'Rotation 2', 'Rotation 3', 'Azimuth', 'Declination', 'Dynamic Rx', 'Dynamic Ry', 'Dynamic Rz', 'Velocity', 'GX velocity', 'GY velocity', 'GZ velocity', 'x velocity', 'y velocity', 'z velocity', 'Angular velocity', 'x angular velocity', 'y angular velocity', 'z angular velocity', 'Acceleration', 'GX acceleration', 'GY acceleration', 'GZ acceleration', 'x acceleration', 'y acceleration', 'z acceleration', 'Acceleration rel. g', 'x acceleration rel. g', 'y acceleration rel. g', 'z acceleration rel. g', 'Angular acceleration', 'x angular acceleration', 'y angular acceleration', 'z angular acceleration', 'Dry length', 'Wetted volume', 'Sea surface Z', 'Sea surface clearance', 'Sea velocity', 'Sea X velocity', 'Sea Y velocity', 'Sea Z velocity', 'Sea acceleration', 'Sea X acceleration', 'Sea Y acceleration', 'Sea Z acceleration', 'Applied force', 'Applied Lx force', 'Applied Ly force', 'Applied Lz force', 'Applied moment', 'Applied Lx moment', 'Applied Ly moment', 'Applied Lz moment', 'dummy'], 'Line': ['Tension per 50 mm', 'End force', 'End moment', 'End force Ez angle', 'End force Exy angle', 'End force Ezx angle', 'End force Ezy angle', 'End force azimuth', 'End force declination', 'No moment azimuth', 'No moment declination', 'End Ex force', 'End Ey force', 'End Ez force', 'End Ex moment', 'End Ey moment', 'End Ez moment', 'End Lx force', 'End Ly force', 'End Lz force', 'End Lx moment', 'End Ly moment', 'End Lz moment', 'End GX force', 'End GY force', 'End GZ force', 'End GX moment', 'End GY moment', 'End GZ moment', 'X', 'Y', 'Z', 'Dynamic x', 'Dynamic y', 'Dynamic z', 'Azimuth', 'Declination', 'Gamma', 'Twist', 'Node azimuth', 'Node declination', 'Node gamma', 'Dynamic Rx', 'Dynamic Ry', 'Dynamic Rz', 'Layback', 'Velocity', 'GX velocity', 'GY velocity', 'GZ velocity', 'x velocity', 'y velocity', 'z velocity', 'Acceleration', 'GX acceleration', 'GY acceleration', 'GZ acceleration', 'x acceleration', 'y acceleration', 'z acceleration', 'Acceleration rel. g', 'x acceleration rel. g', 'y acceleration rel. g', 'z acceleration rel. g', 'Effective tension', 'Wall tension', 'Normalised tension', 'Sidewall pressure', 'Total mean axial strain', 'Direct tensile strain', 'Max bending strain', 'Max pipelay von Mises strain', 'Worst ZZ strain', 'ZZ strain', 'Contents density', 'Contents temperature', 'Contents pressure', 'Contents flow rate', 'Contents flow velocity', 'Fluid incidence angle', 'Bend moment', 'x bend moment', 'y bend moment', 'Bend moment component', 'In plane bend moment', 'Out of plane bend moment', 'Curvature', 'Normalised curvature', 'x curvature', 'y curvature', 'Curvature component', 'In plane curvature', 'Out of plane curvature', 'Bend radius', 'x bend radius', 'y bend radius', 'Bend radius component', 'In plane bend radius', 'Out of plane bend radius', 'Shear force', 'x shear force', 'y shear force', 'Shear force component', 'In plane shear force', 'Out of plane shear force', 'Max von Mises stress', 'Bending stress', 'Max bending stress', 'Pm', 'Pb', 'Worst ZZ stress', 'Direct tensile stress', 'Worst hoop stress', 'Max xy shear stress', 'Internal pressure', 'External pressure', 'Net internal pressure', 'von Mises stress', 'RR stress', 'CC stress', 'ZZ stress', 'RC stress', 'RZ stress', 'CZ stress', 'API RP 2RD stress', 'API RP 2RD utilisation', 'API STD 2RD method 1', 'API STD 2RD method 2', 'API RP 1111 LLD', 'API RP 1111 CLD', 'API RP 1111 BEP', 'API RP 1111 max combined', 'DNV OS F101 disp. controlled', 'DNV OS F101 load controlled', 'DNV ST F101 disp. controlled', 'DNV ST F101 load controlled', 'DNV ST F101 simplified strain', 'DNV ST F101 simplified stress', 'DNV ST F101 tension utilisation', 'DNV OS F201 LRFD', 'DNV OS F201 WSD', 'PD 8010 allowable stress check', 'PD 8010 axial compression check', 'PD 8010 bending check', 'PD 8010 torsion check', 'PD 8010 load combinations check', 'PD 8010 bending strain check', 'Line clearance', 'Line centreline clearance', 'Line horizontal centreline clearance', 'Line vertical centreline clearance', 'Whole line clearance', 'Whole line centreline clearance', 'Whole line horizontal centreline clearance', 'Whole line vertical centreline clearance', 'Seabed clearance', 'Vertical seabed clearance', 'Line clash force', 'Line clash impulse', 'Line clash energy', 'Solid contact force', 'Seabed normal penetration/D', 'Seabed normal resistance', 'Seabed normal resistance/D', 'Arc length', 'Expansion factor', 'Ez angle', 'Exy angle', 'Ezx angle', 'Ezy angle', 'Relative velocity', 'Normal relative velocity', 'x relative velocity', 'y relative velocity', 'z relative velocity', 'Strouhal frequency', 'Reynolds number', 'x drag coefficient', 'y drag coefficient', 'z drag coefficient', 'Lift coefficient', 'Drag force', 'Normal drag force', 'x drag force', 'y drag force', 'z drag force', 'Lift force', 'Fluid inertia force', 'x fluid inertia force', 'y fluid inertia force', 'z fluid inertia force', 'Morison force', 'Normal Morison force', 'x Morison force', 'y Morison force', 'z Morison force', 'Sea surface Z', 'Depth', 'Sea surface clearance', 'Proportion wet', 'Sea velocity', 'Sea X velocity', 'Sea Y velocity', 'Sea Z velocity', 'Sea acceleration', 'Sea X acceleration', 'Sea Y acceleration', 'Sea Z acceleration'], 'Environment': ['Elevation', 'Velocity', 'X velocity', 'Y velocity', 'Z velocity', 'Acceleration', 'X acceleration', 'Y acceleration', 'Z acceleration', 'Current speed', 'Current direction', 'Current X velocity', 'Current Y velocity', 'Current Z velocity', 'Current acceleration', 'Current X acceleration', 'Current Y acceleration', 'Current Z acceleration', 'Wind speed', 'Wind direction', 'Wind X velocity', 'Wind Y velocity', 'Wind Z velocity', 'Static pressure', 'Density', 'Seabed Z'], 'General': ['Time', 'Ramp', 'Implicit solver iteration count', 'Implicit solver time step']}
MATH_FUNCTIONS = [
    "sin()", "cos()", "tan()", "sqrt()", "exp()", "log()", "abs()",
    "min()", "max()", "radians()", "degrees()", "pow(x, y)"
]

def _find_xyz_triples(
    varnames: list[str], warn_if_fallback: bool = True
) -> list[tuple[str, str, str]]:
    """
    Return a list of (x, y, z) triplets found in *varnames*.

    Strategy
    --------
    1.  Look for “perfect” matches that differ **only** by the axis token.
        • Tokens recognised (case-insensitive):
              x, y, z
              xpos, ypos, zpos
              posx, posy, posz
              <anything>_x, _y, _z               (trailing axis)
        • The remaining stem (with the axis part removed) must be identical.

    2.  Any variables not matched in step 1 are grouped naïvely in the
        original order (batch of 3).  If this fallback is used and
        *warn_if_fallback* is True, a warning dialog is shown.

    Returns
    -------
    list of (x_name, y_name, z_name)  – may be empty on failure.
    """
    import re, tkinter.messagebox as mb, itertools, collections as _C

    # ── regex for axis detection ───────────────────────────────────
    X_PAT = re.compile(r"(?:\b|_)(x|xpos|posx)(?:\b|_)?", re.I)
    Y_PAT = re.compile(r"(?:\b|_)(y|ypos|posy)(?:\b|_)?", re.I)
    Z_PAT = re.compile(r"(?:\b|_)(z|zpos|posz)(?:\b|_)?", re.I)

    def _axis(nm: str) -> str | None:
        if X_PAT.search(nm):
            return "x"
        if Y_PAT.search(nm):
            return "y"
        if Z_PAT.search(nm):
            return "z"
        return None

    # ── step 1  • perfect matches  ─────────────────────────────────
    stems: _C.defaultdict[str, dict[str, str]] = _C.defaultdict(dict)

    for nm in varnames:
        ax = _axis(nm)
        if not ax:
            continue
        # strip *only* the first axis occurrence to obtain a stem
        stem = X_PAT.sub("", nm, count=1)
        stem = Y_PAT.sub("", stem, count=1)
        stem = Z_PAT.sub("", stem, count=1)
        stems[stem][ax] = nm

    perfect = [
        (d["x"], d["y"], d["z"]) for d in stems.values() if {"x", "y", "z"} <= d.keys()
    ]

    matched = set(itertools.chain.from_iterable(perfect))
    leftovers = [nm for nm in varnames if nm not in matched]

    # ── step 2  • positional fallback  ─────────────────────────────
    fallback = []
    if leftovers:
        for i in range(0, len(leftovers), 3):
            trio = leftovers[i : i + 3]
            if len(trio) == 3:
                fallback.append(tuple(trio))

        if fallback and warn_if_fallback:
            mb.showwarning(
                "Ambiguous XYZ pairing",
                "One or more triplets were built by simple ordering because "
                "their axis letters could not be identified uniquely.\n\n"
                "Check that the colours / legends in the animation make sense!",
            )

    return perfect + fallback
# Patterns used to detect user-defined variables when loading files
_USER_PATTERNS = (
    r"_shift0$",
    r"_shiftNZ$",
    r"_shiftCommon",
    r"_f\d+$",
    r"\bmean\(",
    r"\bsqrt_sum_of_squares\(",
    r"×1000$",
    r"÷1000$",
    r"_rad$",
    r"_deg$",
    r"×10$",
    r"÷10$",
    r"×2",
    r"÷2$",
    # ── NEW:  apply-values “p / m / x / d” suffixes ───────────────
    r"_p\d+(?:\.\d+)?(?:_f\d+)?$",
    r"_m\d+(?:\.\d+)?(?:_f\d+)?$",
    r"_x\d+(?:\.\d+)?(?:_f\d+)?$",
    r"_d\d+(?:\.\d+)?(?:_f\d+)?$",
)
_user_regex = re.compile("|".join(_USER_PATTERNS))
_SAFE_RE = re.compile(r"\W")  # "not [A-Za-z0-9_]"

def _safe(name: str) -> str:
    """Return ``name`` converted to a valid Python identifier."""
    s = _SAFE_RE.sub("_", name)
    if s and s[0].isdigit():
        s = "_" + s
    return s

def _looks_like_user_var(name: str) -> bool:
    return bool(_user_regex.search(name))

def _parse_search_terms(text: str) -> list[list[str]]:
    """Return a list of search term groups from ``text``.

    The input may contain comma separated terms. If ``",,"`` occurs in the
    text, a literal comma is also included as a search term.  A term enclosed
    in ``!!`` is interpreted as a group of comma separated alternatives which
    will match if *any* of the alternatives are found.

    Example
    -------
    ``"coords, !!x,y,z!!, hor"`` results in ``[['coords'], ['x', 'y', 'z'],
    ['hor']]``.
    """

    text = text.lower()
    include_comma = ",," in text
    placeholders: dict[str, list[str]] = {}

    def _replace(match: re.Match) -> str:
        idx = len(placeholders)
        placeholder = f"\x00{idx}\x00"
        tokens = [t.strip() for t in match.group(1).split(',') if t.strip()]
        placeholders[placeholder] = tokens or [""]
        return placeholder

    # Temporarily replace !!..!! groups with placeholders to avoid splitting
    # them on commas
    text_no_groups = re.sub(r"!!(.*?)!!", _replace, text)

    groups: list[list[str]] = []
    for tok in [t.strip() for t in text_no_groups.split(',') if t.strip()]:
        if tok in placeholders:
            groups.append(placeholders[tok])
        else:
            groups.append([tok])

    if include_comma:
        groups.append([','])

    return groups

def _matches_terms(name: str, terms: list[list[str]]) -> bool:
    """Return ``True`` if ``name`` matches all search ``terms``.

    Each element in ``terms`` is a list of alternatives; at least one
    alternative must be present in ``name`` for the term to match.
    """

    if not terms:
        return True

    name_l = name.lower()
    return all(any(t in name_l for t in group) for group in terms)
# --------- Main Window ---------
class VariableRowWidget(QWidget):
    def __init__(self, var_key, display_label, parent=None):
        super().__init__(parent)

        # Checkbox
        self.checkbox = QCheckBox()

        # Entry field for numeric input
        self.value_entry = QLineEdit()
        self.value_entry.setFixedWidth(70)

        # Label for the variable name
        self.label = QLabel(display_label)

        # Layout setup: checkbox → entry field → label
        layout = QHBoxLayout(self)
        layout.addWidget(self.checkbox)
        layout.addWidget(self.value_entry)
        layout.addWidget(self.label)
        layout.addStretch(1)

        layout.setContentsMargins(2, 2, 2, 2)
        self.setLayout(layout)

class TimeSeriesEditorQt(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AnytimeSeries - time series editor (Qt/PySide6)")



        # Palette and style for theme switching
        app = QApplication.instance()
        self.default_palette = app.palette()
        self.default_style = app.style().objectName()
        # Reuse a single style instance when toggling themes to avoid
        # crashes from Python garbage-collecting temporary QStyle objects
        self._fusion_style = QStyleFactory.create("Fusion")



        # =======================
        # DATA STRUCTURES
        # =======================
        self.tsdbs = []                # List of anyqats.TsDB instances (one per file)
        self.file_paths = []           # List of file paths (order matches tsdbs)
        self.user_variables = set()    # User-defined/calculated variables

        self.var_checkboxes = {}       # key: variable key → QCheckBox
        self.var_offsets = {}          # key: variable key → QLineEdit for numeric offset

        # These lists must be filled before refresh_variable_tabs()
        self.common_var_keys = []      # e.g. ["Heave", "Surge"]
        self.file_var_keys = {}        # dict: file name → [var1, var2, ...]
        self.user_var_keys = []        # e.g. ["result_var1", ...]
        self.var_labels = {}           # Optional: key → display label

        self.file_loader = FileLoader(
            orcaflex_varmap=ORCAFLEX_VARIABLE_MAP,
            parent_gui=self,
        )
        # Progress updates while loading files
        self.file_loader.progress_callback = self.update_progressbar

        # =======================
        # LAYOUT: MAIN SPLITTER
        # =======================
        main_splitter = QSplitter(Qt.Horizontal)

        # -----------------------
        # LEFT: Variable Tabs
        # -----------------------
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)

        # Quick navigation buttons
        btn_row = QHBoxLayout()
        self.goto_common_btn = QPushButton("Go to Common")
        self.goto_user_btn = QPushButton("Go to User Variables")
        self.unselect_all_btn = QPushButton("Unselect All")
        self.select_pos_btn = QPushButton("Select all by list pos.")
        btn_row.addWidget(self.goto_common_btn)
        btn_row.addWidget(self.goto_user_btn)
        btn_row.addWidget(self.unselect_all_btn)
        btn_row.addWidget(self.select_pos_btn)
        left_layout.addLayout(btn_row)

        # Tab widget for variables (common, per-file, user)
        self.tabs = QTabWidget()
        self.tabs.setMinimumWidth(380)  # Make the variable panel wider
        left_layout.addWidget(self.tabs)

        main_splitter.addWidget(left_widget)

        # -----------------------
        # RIGHT: Controls and Analysis
        # -----------------------
        right_widget = QWidget()
        # Use a vertical layout so an optional embedded plot can span

        # the full width below the control sections when embedded

        self.right_outer_layout = QVBoxLayout(right_widget)
        self.top_row_layout = QHBoxLayout()
        self.right_outer_layout.addLayout(self.top_row_layout)

        self.controls_widget = QWidget()
        self.controls_layout = QVBoxLayout(self.controls_widget)

        self.extra_widget = QWidget()
        self.extra_layout = QVBoxLayout(self.extra_widget)
        self.extra_stretch = QSpacerItem(0, 0, QSizePolicy.Minimum, QSizePolicy.Expanding)

        # ---- File controls ----
        self.file_ctrls_layout = QHBoxLayout()
        self.load_btn = QPushButton("Load time series file")
        self.save_btn = QPushButton("Save Files")
        self.clear_btn = QPushButton("Clear All")
        self.save_values_btn = QPushButton("Save Values…")
        self.load_values_btn = QPushButton("Load Values…")
        self.export_csv_btn = QPushButton("Export Selected to CSV")
        self.export_dt_input = QLineEdit("0")
        self.export_dt_input.setFixedWidth(50)
        self.export_dt_input.setToolTip("Resample dt (0 = no resample)")
        self.clear_orcaflex_btn = QPushButton("Clear OrcaFlex Selection")
        self.reselect_orcaflex_btn = QPushButton("Re-select OrcaFlex Variables")
        # Hidden until a .sim file is loaded
        self.clear_orcaflex_btn.hide()
        self.reselect_orcaflex_btn.hide()
        self.file_ctrls_layout.addWidget(self.load_btn)
        self.file_ctrls_layout.addWidget(self.save_btn)
        self.file_ctrls_layout.addWidget(self.clear_btn)
        self.file_ctrls_layout.addWidget(self.save_values_btn)
        self.file_ctrls_layout.addWidget(self.load_values_btn)
        self.file_ctrls_layout.addWidget(self.export_csv_btn)
        self.file_ctrls_layout.addWidget(self.export_dt_input)
        self.file_ctrls_layout.addWidget(self.clear_orcaflex_btn)
        self.file_ctrls_layout.addWidget(self.reselect_orcaflex_btn)
        self.file_ctrls_layout.addStretch(1)

        self.theme_embed_widget = QWidget()
        self.theme_embed_layout = QVBoxLayout(self.theme_embed_widget)
        self.theme_switch = QCheckBox("Dark Theme")
        self.embed_plot_cb = QCheckBox("Embed Plot")
        self.theme_embed_layout.addWidget(self.theme_switch)
        self.theme_embed_layout.addWidget(self.embed_plot_cb)
        self.file_ctrls_layout.addWidget(self.theme_embed_widget)
        self.controls_layout.addLayout(self.file_ctrls_layout)

        # Progress bar
        self.progress = QProgressBar()

        # --- Transformations ---
        self.transform_group = QGroupBox("Quick transformations")
        transform_layout = QVBoxLayout(self.transform_group)

        row1 = QHBoxLayout()
        self.mult_by_1000_btn = QPushButton("Multiply by 1000")
        self.div_by_1000_btn = QPushButton("Divide by 1000")
        self.mult_by_10_btn = QPushButton("Multiply by 10")
        self.div_by_10_btn = QPushButton("Divide by 10")
        self.mult_by_2_btn = QPushButton("Multiply by 2")
        self.div_by_2_btn = QPushButton("Divide by 2")
        self.mult_by_neg1_btn = QPushButton("Multiply by -1")
        row1.addWidget(self.mult_by_1000_btn)
        row1.addWidget(self.div_by_1000_btn)
        row1.addWidget(self.mult_by_10_btn)
        row1.addWidget(self.div_by_10_btn)
        row1.addWidget(self.mult_by_2_btn)
        row1.addWidget(self.div_by_2_btn)
        row1.addWidget(self.mult_by_neg1_btn)
        transform_layout.addLayout(row1)

        row2 = QHBoxLayout()
        self.radians_btn = QPushButton("Radians")
        self.degrees_btn = QPushButton("Degrees")
        row2.addWidget(self.radians_btn)
        row2.addWidget(self.degrees_btn)
        transform_layout.addLayout(row2)

        row3 = QHBoxLayout()
        self.shift_mean0_btn = QPushButton("Shift Mean → 0")
        self.shift_min0_btn = QPushButton("Shift Min to Zero")
        self.ignore_anomalies_cb = QCheckBox("Ignore anomalies (lowest 1%) for shifting.")
        row3.addWidget(self.shift_mean0_btn)
        row3.addWidget(self.shift_min0_btn)
        row3.addWidget(self.ignore_anomalies_cb)
        transform_layout.addLayout(row3)

        row4 = QHBoxLayout()
        self.sqrt_sum_btn = QPushButton("Sqrt(sum of squares)")
        self.mean_of_sel_btn = QPushButton("Mean")
        self.abs_btn = QPushButton("Absolute")
        self.rolling_avg_btn = QPushButton("Rolling Avg")
        row4.addWidget(self.sqrt_sum_btn)
        row4.addWidget(self.mean_of_sel_btn)
        row4.addWidget(self.abs_btn)
        row4.addWidget(self.rolling_avg_btn)
        transform_layout.addLayout(row4)

        row5 = QHBoxLayout()
        row5.addWidget(QLabel("Tol [%]:"))
        self.shift_tol_entry = QLineEdit("0.01")
        self.shift_tol_entry.setFixedWidth(60)
        row5.addWidget(self.shift_tol_entry)
        row5.addWidget(QLabel("Min count:"))
        self.shift_cnt_entry = QLineEdit("10")
        self.shift_cnt_entry.setFixedWidth(60)
        row5.addWidget(self.shift_cnt_entry)
        self.shift_min_nz_btn = QPushButton(
            "Shift Min -> 0 : if repeted minima as per input"
        )
        self.shift_common_max_btn = QPushButton(
            "Common Shift Min -> 0 : if repeted minima as per input"
        )
        row5.addWidget(self.shift_min_nz_btn)
        row5.addWidget(self.shift_common_max_btn)
        transform_layout.addLayout(row5)


        # Progress bar is shown by itself unless the plot is embedded
        self.controls_layout.addWidget(self.progress)
        # Row used when embedding the plot to move transformations next to the
        # progress bar
        self.progress_transform_row = QHBoxLayout()


        # ---- Offset Group ----
        offset_group = QGroupBox("Apply operation from variable input fields")
        offset_layout = QVBoxLayout(offset_group)
        offset_layout.addWidget(QLabel('Examples: add "+1 / 1" substract "-1" divide "/2" multiply "*2"'))
        self.apply_value_user_var_cb = QCheckBox("Create user variable instead of overwriting?")
        offset_layout.addWidget(self.apply_value_user_var_cb)
        self.apply_values_btn = QPushButton("Apply Values")
        offset_layout.addWidget(self.apply_values_btn)
        self.controls_layout.addWidget(offset_group)

        # ---- File list group ----
        file_group = QGroupBox("Loaded Files")
        file_list_layout = QVBoxLayout(file_group)
        self.file_list = QListWidget()
        self.file_list.setMinimumWidth(220)
        self.remove_file_btn = QPushButton("Remove File")
        file_list_layout.addWidget(self.file_list)
        file_list_layout.addWidget(self.remove_file_btn)
        self.controls_layout.addWidget(file_group)

        # ---- Time window controls ----
        time_group = QGroupBox("Time Window (for Plot/Stats/Transform)")
        time_layout = QHBoxLayout(time_group)
        time_layout.addWidget(QLabel("Start:"))
        self.time_start = QLineEdit()
        self.time_start.setFixedWidth(60)
        time_layout.addWidget(self.time_start)
        time_layout.addWidget(QLabel("End:"))
        self.time_end = QLineEdit()
        self.time_end.setFixedWidth(60)
        time_layout.addWidget(self.time_end)
        self.reset_time_window_btn = QPushButton("Reset")
        time_layout.addWidget(self.reset_time_window_btn)
        self.controls_layout.addWidget(time_group)

        # ---- Frequency filtering controls ----
        self.freq_group = QGroupBox("Apply frequency filter to transformations and calculations")
        freq_layout = QGridLayout(self.freq_group)
        self.filter_none_rb = QRadioButton("None")
        self.filter_lowpass_rb = QRadioButton("Low-pass")
        self.filter_highpass_rb = QRadioButton("High-pass")
        self.filter_bandpass_rb = QRadioButton("Band-pass")
        self.filter_bandblock_rb = QRadioButton("Band-block")
        self.filter_none_rb.setChecked(True)
        self.lowpass_cutoff = QLineEdit("0.01")
        self.highpass_cutoff = QLineEdit("0.1")
        self.bandpass_low = QLineEdit("0.0")
        self.bandpass_high = QLineEdit("0.0")
        self.bandblock_low = QLineEdit("0.0")
        self.bandblock_high = QLineEdit("0.0")

        row = 0
        freq_layout.addWidget(self.filter_none_rb, row, 0, 1, 2)
        row += 1
        freq_layout.addWidget(self.filter_lowpass_rb, row, 0)
        freq_layout.addWidget(QLabel("below"), row, 1)
        freq_layout.addWidget(self.lowpass_cutoff, row, 2)
        freq_layout.addWidget(QLabel("Hz"), row, 3)
        row += 1
        freq_layout.addWidget(self.filter_highpass_rb, row, 0)
        freq_layout.addWidget(QLabel("above"), row, 1)
        freq_layout.addWidget(self.highpass_cutoff, row, 2)
        freq_layout.addWidget(QLabel("Hz"), row, 3)
        row += 1
        freq_layout.addWidget(self.filter_bandpass_rb, row, 0)
        freq_layout.addWidget(QLabel("between"), row, 1)
        freq_layout.addWidget(self.bandpass_low, row, 2)
        freq_layout.addWidget(QLabel("Hz and"), row, 3)
        freq_layout.addWidget(self.bandpass_high, row, 4)
        freq_layout.addWidget(QLabel("Hz"), row, 5)
        row += 1
        freq_layout.addWidget(self.filter_bandblock_rb, row, 0)
        freq_layout.addWidget(QLabel("between"), row, 1)
        freq_layout.addWidget(self.bandblock_low, row, 2)
        freq_layout.addWidget(QLabel("Hz and"), row, 3)
        freq_layout.addWidget(self.bandblock_high, row, 4)
        freq_layout.addWidget(QLabel("Hz"), row, 5)

        self.controls_layout.addWidget(self.freq_group)

        # ---- Tools (EVA + QATS) ----
        self.tools_group = QGroupBox("Tools")
        tools_layout = QHBoxLayout(self.tools_group)
        self.launch_qats_btn = QPushButton("Open in AnyQATS")
        self.evm_tool_btn = QPushButton("Open Extreme Value Statistics Tool")
        tools_layout.addWidget(self.launch_qats_btn)
        tools_layout.addWidget(self.evm_tool_btn)
        self.controls_layout.addWidget(self.tools_group)


        # ---- Plot controls ----
        self.plot_group = QGroupBox("Plot Controls")

        plot_group = self.plot_group  # backward compatibility for older refs

        plot_layout = QVBoxLayout(self.plot_group)
        plot_btn_row = QHBoxLayout()
        self.plot_selected_btn = QPushButton("Plot Selected (one graph)")
        self.plot_side_by_side_btn = QPushButton("Plot Selected (side-by-side)")
        grid_col = QVBoxLayout()
        grid_col.addWidget(self.plot_side_by_side_btn)
        self.plot_same_axes_cb = QCheckBox("Same axes")
        grid_col.addWidget(self.plot_same_axes_cb)
        self.plot_mean_btn = QPushButton("Plot Mean")
        self.plot_rolling_btn = QPushButton("Rolling Mean")
        self.animate_xyz_btn = QPushButton("Animate XYZ scatter (all points)")

        selected_col = QVBoxLayout()
        selected_col.addWidget(self.plot_selected_btn)
        self.plot_extrema_cb = QCheckBox("Mark max/min")
        selected_col.addWidget(self.plot_extrema_cb)

        plot_btn_row.addLayout(selected_col)
        plot_btn_row.addLayout(grid_col)
        plot_btn_row.addWidget(self.plot_mean_btn)
        plot_btn_row.addWidget(self.plot_rolling_btn)
        plot_btn_row.addWidget(self.animate_xyz_btn)
        self.plot_selected_btn.clicked.connect(self.plot_selected)
        # Use an explicit slot for side-by-side plotting so that the optional
        # ``checked`` argument emitted by QPushButton.clicked() is ignored and
        # the ``grid`` flag is always forwarded correctly.
        self.plot_side_by_side_btn.clicked.connect(self.plot_selected_side_by_side)
        self.plot_mean_btn.clicked.connect(self.plot_mean)
        self.plot_rolling_btn.clicked.connect(lambda: self.plot_selected(mode="rolling"))
        self.animate_xyz_btn.clicked.connect(self.animate_xyz_scatter_many)
        self.plot_raw_cb = QCheckBox("Raw")
        self.plot_raw_cb.setChecked(True)
        self.plot_lowpass_cb = QCheckBox("Low-pass")
        self.plot_highpass_cb = QCheckBox("High-pass")
        plot_btn_row.addWidget(self.plot_raw_cb)
        plot_btn_row.addWidget(self.plot_lowpass_cb)
        plot_btn_row.addWidget(self.plot_highpass_cb)
        plot_btn_row.addWidget(QLabel("Engine:"))
        self.plot_engine_combo = QComboBox()
        self.plot_engine_combo.addItems(["plotly", "bokeh", "default"])
        plot_btn_row.addWidget(self.plot_engine_combo)
        self.include_raw_mean_cb = QCheckBox("Show components (used in mean)")
        plot_btn_row.addWidget(self.include_raw_mean_cb)
        plot_layout.addLayout(plot_btn_row)
        # Label trimming controls
        trim_row = QHBoxLayout()
        trim_row.addWidget(QLabel("Trim label to keep:"))
        trim_row.addWidget(QLabel("Left:"))
        self.label_trim_left = QSpinBox()
        self.label_trim_left.setMaximum(1000)
        self.label_trim_left.setValue(10)
        trim_row.addWidget(self.label_trim_left)
        trim_row.addWidget(QLabel("Right:"))
        self.label_trim_right = QSpinBox()
        self.label_trim_right.setMaximum(1000)
        self.label_trim_right.setValue(60)
        trim_row.addWidget(self.label_trim_right)
        plot_layout.addLayout(trim_row)
        # Y-axis label
        yaxis_row = QHBoxLayout()
        yaxis_row.addWidget(QLabel("Y-axis label (optional):"))
        self.yaxis_label = QLineEdit("Value")
        yaxis_row.addWidget(self.yaxis_label)
        plot_layout.addLayout(yaxis_row)

        # Rolling mean window
        rolling_row = QHBoxLayout()
        rolling_row.addWidget(QLabel("Rolling mean window:"))
        self.rolling_window = QSpinBox()
        self.rolling_window.setMinimum(1)
        self.rolling_window.setMaximum(1000000)

        self.rolling_window.setValue(1)
        rolling_row.addWidget(self.rolling_window)
        plot_layout.addLayout(rolling_row)

        self.controls_layout.addWidget(self.plot_group)
        self.controls_layout.addWidget(self.transform_group)


        # ---- Calculator ----
        self.calc_group = QGroupBox("Calculator")
        calc_layout = QVBoxLayout(self.calc_group)
        calc_layout.addWidget(QLabel(
            "Define a new variable (e.g., result_name = f1_var1 + f2_var2) where f1 and f2 refer to file IDs in the loaded list (c_ common var, u_ user var)."
        ))
        self.calc_entry = QTextEdit()
        calc_layout.addWidget(self.calc_entry)
        calc_btn_row = QHBoxLayout()
        self.calc_btn = QPushButton("Calculate")
        self.calc_help_btn = QPushButton("?")
        calc_btn_row.addWidget(self.calc_btn)
        calc_btn_row.addWidget(self.calc_help_btn)
        calc_layout.addLayout(calc_btn_row)
        self.controls_layout.addWidget(self.calc_group)

        # Autocomplete popup for the calculator
        self.autocomplete_popup = QListWidget(self)
        self.autocomplete_popup.setWindowFlags(Qt.Popup)

        self.autocomplete_popup.setFocusPolicy(Qt.NoFocus)
        self.autocomplete_popup.setFocusProxy(self.calc_entry)


        # Do not steal focus when shown so typing can continue
        self.autocomplete_popup.setAttribute(Qt.WA_ShowWithoutActivating)
        self.autocomplete_popup.hide()

        # Connect calculator signals
        self.calc_btn.clicked.connect(self.calculate_series)
        self.calc_help_btn.clicked.connect(self.show_calc_help)
        self.calc_entry.textChanged.connect(self._update_calc_suggestions)
        self.autocomplete_popup.itemClicked.connect(self._insert_calc_suggestion)
        self.calc_entry.installEventFilter(self)
        self.autocomplete_popup.installEventFilter(self)

        # ---- Analysis ----
        self.analysis_group = QGroupBox("Analysis")
        analysis_layout = QVBoxLayout(self.analysis_group)
        self.show_stats_btn = QPushButton("Show statistic for selected variables")
        self.show_stats_btn.clicked.connect(self.show_stats)
        analysis_layout.addWidget(self.show_stats_btn)
        analysis_btn_row = QHBoxLayout()
        self.psd_btn = QPushButton("PSD")
        self.cycle_range_btn = QPushButton("Cycle Range")
        self.cycle_mean_btn = QPushButton("Range-Mean")
        self.cycle_mean3d_btn = QPushButton("Range-Mean 3-D")
        analysis_btn_row.addWidget(self.psd_btn)
        analysis_btn_row.addWidget(self.cycle_range_btn)
        analysis_btn_row.addWidget(self.cycle_mean_btn)
        analysis_btn_row.addWidget(self.cycle_mean3d_btn)
        analysis_layout.addLayout(analysis_btn_row)
        self.controls_layout.addWidget(self.analysis_group)
        # Plot controls below analysis
        self.controls_layout.addWidget(plot_group)



        self.plot_view = QWebEngineView()
        self.plot_view.setMinimumHeight(300)
        self.plot_view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        # Match the dark theme when embedding Plotly by removing the default
        # light border around the web view. Background color is updated when
        # themes toggle via ``apply_dark_palette``/``apply_light_palette``.
        self.plot_view.setStyleSheet("border:0px;")
        self._temp_plot_file = None  # temporary HTML used for embedded plots
        # Placeholder for embedded Matplotlib canvas
        self._mpl_canvas = None
        # plot_view is shown when the "Embed Plot" option is enabled

        self.controls_layout.addStretch(1)
        self.extra_layout.addItem(self.extra_stretch)

        self.top_row_layout.addWidget(self.controls_widget)
        # extra_widget will be inserted when embed is enabled
        # Plot view occupies full width below the top row
        self.right_outer_layout.addWidget(self.plot_view)
        self.right_outer_layout.setStretch(0, 0)
        self.right_outer_layout.setStretch(1, 1)
        self.plot_view.hide()
        main_splitter.addWidget(right_widget)
        main_splitter.setStretchFactor(0, 1)
        main_splitter.setStretchFactor(1, 2)

        # ---- Set main container ----
        container = QWidget()
        container.setAutoFillBackground(True)
        container_layout = QHBoxLayout(container)
        container_layout.addWidget(main_splitter)
        self.setCentralWidget(container)
        self.setAutoFillBackground(True)

        # =======================
        # SIGNALS AND ACTIONS
        # =======================
        self.load_btn.clicked.connect(self.load_files)
        self.remove_file_btn.clicked.connect(self.remove_selected_file)
        self.clear_btn.clicked.connect(self.clear_all_files)
        self.goto_common_btn.clicked.connect(lambda: self.tabs.setCurrentIndex(0))
        self.goto_user_btn.clicked.connect(lambda: self.tabs.setCurrentIndex(self.tabs.count() - 1))
        self.unselect_all_btn.clicked.connect(self._unselect_all_variables)
        self.select_pos_btn.clicked.connect(self._select_all_by_list_pos)
        self.file_list.currentRowChanged.connect(self.highlight_file_tab)
        self.apply_values_btn.clicked.connect(self.apply_values)
        self.mult_by_1000_btn.clicked.connect(self.multiply_by_1000)
        self.div_by_1000_btn.clicked.connect(self.divide_by_1000)
        self.mult_by_10_btn.clicked.connect(self.multiply_by_10)
        self.div_by_10_btn.clicked.connect(self.divide_by_10)
        self.mult_by_2_btn.clicked.connect(self.multiply_by_2)
        self.div_by_2_btn.clicked.connect(self.divide_by_2)
        self.mult_by_neg1_btn.clicked.connect(self.multiply_by_neg1)
        self.mean_of_sel_btn.clicked.connect(self.mean_of_selected)
        self.sqrt_sum_btn.clicked.connect(self.sqrt_sum_of_squares)
        self.abs_btn.clicked.connect(self.abs_var)
        self.rolling_avg_btn.clicked.connect(self.rolling_average)
        self.radians_btn.clicked.connect(self.to_radians)
        self.degrees_btn.clicked.connect(self.to_degrees)
        self.shift_min0_btn.clicked.connect(self.shift_min_to_zero)
        self.shift_mean0_btn.clicked.connect(self.shift_mean_to_zero)
        self.save_btn.clicked.connect(self.save_files)
        self.save_values_btn.clicked.connect(self.save_entry_values)
        self.load_values_btn.clicked.connect(self.load_entry_values)
        self.export_csv_btn.clicked.connect(self.export_selected_to_csv)
        self.shift_min_nz_btn.clicked.connect(self.shift_repeated_neg_min)
        self.shift_common_max_btn.clicked.connect(self.shift_common_max)
        self.launch_qats_btn.clicked.connect(self.launch_qats)
        self.evm_tool_btn.clicked.connect(self.open_evm_tool)
        self.reselect_orcaflex_btn.clicked.connect(self.reselect_orcaflex_variables)
        self.psd_btn.clicked.connect(lambda: self.plot_selected(mode="psd"))
        self.cycle_range_btn.clicked.connect(lambda: self.plot_selected(mode="cycle"))
        self.cycle_mean_btn.clicked.connect(lambda: self.plot_selected(mode="cycle_rm"))
        self.cycle_mean3d_btn.clicked.connect(lambda: self.plot_selected(mode="cycle_rm3d"))
        self.plot_rolling_btn.clicked.connect(lambda: self.plot_selected(mode="rolling"))

        self.theme_switch.stateChanged.connect(self.toggle_dark_theme)
        self.embed_plot_cb.stateChanged.connect(self.toggle_embed_layout)
        self.plot_engine_combo.currentTextChanged.connect(self._on_engine_changed)


        # ==== Populate variable tabs on startup ====
        self.refresh_variable_tabs()
        # Apply the light palette by default
        self.apply_dark_palette()
        self.theme_switch.setChecked(True)
        self.toggle_embed_layout('')
        self.embed_plot_cb.setChecked(True)

    def eventFilter(self, obj, event):

        if obj is self.calc_entry and event.type() == QEvent.Type.KeyPress:
            if self.autocomplete_popup.isVisible():
                if event.key() in (Qt.Key_Up, Qt.Key_Down):
                    self._navigate_autocomplete(event)
                    return True
                if event.key() in (Qt.Key_Return, Qt.Key_Enter, Qt.Key_Tab):
                    self._insert_calc_suggestion()
                    return True
            if event.key() == Qt.Key_Escape:
                self.autocomplete_popup.hide()
                return True

        if obj is self.autocomplete_popup and event.type() == QEvent.Type.KeyPress:
            if event.key() in (Qt.Key_Up, Qt.Key_Down):
                self._navigate_autocomplete(event)
                return True
            if event.key() in (Qt.Key_Return, Qt.Key_Enter):
                self._insert_calc_suggestion()
                return True
            if event.key() == Qt.Key_Escape:
                self.autocomplete_popup.hide()
                return True

            # Forward other keystrokes to the calculator entry
            fwd = QKeyEvent(
                event.type(),
                event.key(),
                event.modifiers(),
                event.text(),
                event.isAutoRepeat(),
                event.count(),
            )
            QApplication.sendEvent(self.calc_entry, fwd)
            return True

        return super().eventFilter(obj, event)

    # ---- Calculator helpers -------------------------------------------------
    def _navigate_autocomplete(self, event):
        count = self.autocomplete_popup.count()
        if count == 0:
            return
        idx = self.autocomplete_popup.currentRow()
        if event.key() == Qt.Key_Down:
            idx = (idx + 1) % count
        elif event.key() == Qt.Key_Up:
            idx = (idx - 1) % count
        self.autocomplete_popup.setCurrentRow(idx)

    def _insert_calc_suggestion(self):
        import re

        item = self.autocomplete_popup.currentItem()
        if not item:
            return
        token = self._calc_match_lookup.get(item.text(), "")
        cursor = self.calc_entry.textCursor()
        text_before = self.calc_entry.toPlainText()[: cursor.position()]
        m = re.search(r"([A-Za-z0-9_]+)$", text_before)
        if m:
            cursor.movePosition(QTextCursor.Left, QTextCursor.KeepAnchor, len(m.group(1)))
        cursor.insertText(token)
        self.calc_entry.setTextCursor(cursor)
        self.autocomplete_popup.hide()
        self.calc_entry.setFocus()

    def _build_calc_variable_list(self):
        self.calc_variables = []
        self.calc_var_filemap = {}
        for i, tsdb in enumerate(self.tsdbs):
            tag = f"f{i + 1}"
            filename = os.path.basename(self.file_paths[i])
            for key in tsdb.getm().keys():
                safe = f"{tag}_{_safe(key)}"
                self.calc_variables.append(safe)
                self.calc_var_filemap[safe] = filename
        if self.tsdbs:
            common_set = set(self.tsdbs[0].getm().keys())
            for db in self.tsdbs[1:]:
                common_set &= set(db.getm().keys())
            for key in sorted(common_set):
                safe = f"c_{_safe(key)}"
                self.calc_variables.append(safe)
                self.calc_var_filemap[safe] = "common"
        for key in getattr(self, "user_variables", set()):
            safe = f"c_{_safe(key)}"
            if safe not in self.calc_variables:
                filename = next((os.path.basename(fp) for tsdb, fp in zip(self.tsdbs, self.file_paths) if key in tsdb.getm()), "")
                self.calc_variables.append(safe)
                self.calc_var_filemap[safe] = filename

    def _update_calc_suggestions(self):
        import re

        text = self.calc_entry.toPlainText()
        text_until_cursor = self.calc_entry.toPlainText()[: self.calc_entry.textCursor().position()]
        if not text:
            self.autocomplete_popup.hide()
            return
        m = re.search(r"([A-Za-z0-9_]+)$", text_until_cursor)
        if not m:
            self.autocomplete_popup.hide()
            return
        token = m.group(1).lower()
        all_items = self.calc_variables + MATH_FUNCTIONS
        matches = [v for v in all_items if v.lower().startswith(token)]
        if not matches:
            self.autocomplete_popup.hide()
            return
        matches.sort(key=lambda v: (v not in self.calc_variables, v.lower()))
        self.autocomplete_popup.clear()
        self._calc_match_lookup = {}
        for item in matches:
            label = item if item not in self.calc_variables else f"{item}   ({self.calc_var_filemap.get(item, '')})"
            self._calc_match_lookup[label] = item
            self.autocomplete_popup.addItem(label)
        self.autocomplete_popup.setCurrentRow(0)
        pos = self.calc_entry.mapToGlobal(self.calc_entry.cursorRect().bottomLeft())
        self.autocomplete_popup.move(pos)
        self.autocomplete_popup.setFixedWidth(self.calc_entry.width())
        self.autocomplete_popup.setFixedHeight(min(6, len(matches)) * 22)
        self.autocomplete_popup.show()
        # Keep typing focus in the calculator entry
        self.calc_entry.setFocus()

    def calculate_series(self):
        """Evaluate the Calculator expression and create new series."""
        import traceback

        expr = self.calc_entry.toPlainText().strip()
        if not expr:
            QMessageBox.warning(self, "No Formula", "Please enter a formula.")
            return

        m_out = re.match(r"\s*([A-Za-z_]\w*)\s*=", expr)
        if not m_out:
            QMessageBox.critical(self, "No Assignment", "Write the formula like   result = <expression>")
            return
        base_output = m_out.group(1)

        t_window = None
        for tsdb in self.tsdbs:
            for ts in tsdb.getm().values():
                mask = self.get_time_window(ts)
                if mask is not None and np.any(mask):
                    t_window = ts.t[mask]
                    break
            if t_window is not None:
                break
        if t_window is None:
            QMessageBox.critical(self, "No Time Window", "Could not infer a valid time window.")
            return

        common_tokens = {m.group(1) for m in re.finditer(r"\bc_([\w\- ]+)\b", expr)}
        user_tokens = {m.group(1) for m in re.finditer(r"\bu_([\w\- ]+)", expr)}
        file_tags_used = {int(m.group(1)) for m in re.finditer(r"\bf(\d+)_", expr)}
        if not file_tags_used:
            file_tags_used = set(range(1, len(self.tsdbs) + 1))

        u_global = {u for u in user_tokens if not re.search(r"_f\d+$", u)}
        u_perfile = {u for u in user_tokens if re.search(r"_f\d+$", u)}

        known_user = getattr(self, "user_variables", set())
        missing = u_global - known_user
        if missing:
            QMessageBox.critical(self, "Unknown user variable", ", ".join(sorted(missing)))
            return

        def align_all_files(name):
            vecs = []
            for i, tsdb in enumerate(self.tsdbs):
                ts = tsdb.getm().get(name)
                if ts is None:
                    return None, f"'{name}' not in {os.path.basename(self.file_paths[i])}"
                idx = (ts.t >= t_window[0]) & (ts.t <= t_window[-1])
                t_part, x_part = ts.t[idx], ts.x[idx]
                if len(t_part) == 0:
                    vecs.append(np.full_like(t_window, np.nan))
                    continue
                if not np.array_equal(t_part, t_window):
                    t_common = t_window[(t_window >= t_part[0]) & (t_window <= t_part[-1])]
                    x_part = qats.TimeSeries(name, t_part, x_part).resample(t=t_common)
                    full = np.full_like(t_window, np.nan)
                    full[np.isin(t_window, t_common)] = x_part
                    x_part = full
                vecs.append(x_part.astype(float))
            return vecs, None

        aligned_common, aligned_u_global = {}, {}
        for k in common_tokens:
            v, err = align_all_files(k)
            if err:
                QMessageBox.critical(self, "Common variable error", err)
                return
            aligned_common[k] = v
        for k in u_global:
            v, err = align_all_files(k)
            if err:
                QMessageBox.critical(self, "User variable error", err)
                return
            aligned_u_global[k] = v

        aligned_u_perfile = {}
        for tok in u_perfile:
            m = re.match(r"(.+)_f(\d+)$", tok)
            if not m:
                continue
            src_idx = int(m.group(2)) - 1
            if src_idx >= len(self.tsdbs):
                QMessageBox.critical(self, "User variable error", f"File #{m.group(2)} does not exist.")
                return
            ts = self.tsdbs[src_idx].getm().get(tok)
            if ts is None:
                QMessageBox.critical(self, "User variable error", f"Variable '{tok}' not found in {os.path.basename(self.file_paths[src_idx])}")
                return
            idx = (ts.t >= t_window[0]) & (ts.t <= t_window[-1])
            t_part, x_part = ts.t[idx], ts.x[idx]
            if len(t_part) == 0:
                vec = np.full_like(t_window, np.nan)
            elif np.array_equal(t_part, t_window):
                vec = x_part.astype(float)
            else:
                t_common = t_window[(t_window >= t_part[0]) & (t_window <= t_part[-1])]
                vec = qats.TimeSeries(tok, t_part, x_part).resample(t=t_common)
                full = np.full_like(t_window, np.nan)
                full[np.isin(t_window, t_common)] = vec
                vec = full
            aligned_u_perfile[tok] = vec.astype(float)

        results = []
        for file_idx, tsdb in enumerate(self.tsdbs):
            f_no = file_idx + 1
            ctx = {}
            for i, db in enumerate(self.tsdbs):
                tag = f"f{i + 1}"
                for key, ts in db.getm().items():
                    idx = (ts.t >= t_window[0]) & (ts.t <= t_window[-1])
                    if not np.any(idx):
                        continue
                    t_part = ts.t[idx]
                    x_part = self.apply_filters(ts)[idx]
                    if not np.array_equal(t_part, t_window):
                        t_common = t_window[(t_window >= t_part[0]) & (t_window <= t_part[-1])]
                        x_part = qats.TimeSeries(key, t_part, x_part).resample(t=t_common)
                        full = np.full_like(t_window, np.nan)
                        full[np.isin(t_window, t_common)] = x_part
                        x_part = full
                    ctx[f"{tag}_{_safe(key)}"] = x_part.astype(float)

            for k, vecs in aligned_common.items():
                ctx[f"c_{_safe(k)}"] = vecs[file_idx]
            for k, vecs in aligned_u_global.items():
                ctx[f"c_{_safe(k)}"] = vecs[file_idx]
            for tok, vec in aligned_u_perfile.items():
                ctx[f"u_{tok}"] = vec

            ctx["time"] = t_window
            ctx.update({
                "np": np,
                "sin": np.sin,
                "cos": np.cos,
                "tan": np.tan,
                "exp": np.exp,
                "sqrt": np.sqrt,
                "log": np.log,
                "abs": np.abs,
                "min": np.min,
                "max": np.max,
                "power": np.power,
                "radians": np.radians,
                "degrees": np.degrees,
            })

            try:
                exec(expr, ctx)
                y = np.asarray(ctx[base_output], dtype=float)
                if y.ndim == 0:
                    y = np.full_like(t_window, y, dtype=float)
                if len(y) != len(t_window):
                    raise ValueError("Result length mismatch with time vector")

                create_common_output = len(file_tags_used) >= 2
                must_write_here = (create_common_output and f_no == min(file_tags_used)) or (not create_common_output and f_no in file_tags_used)
                if not must_write_here:
                    continue

                filt_tag = self._filter_tag()
                suffix = "" if create_common_output else f"_f{f_no}"
                out_name = base_output
                if filt_tag:
                    out_name += f"_{filt_tag}"
                out_name += suffix
                ts_new = qats.TimeSeries(out_name, t_window, y)

                tsdb.add(ts_new)

                if create_common_output:
                    for other_db in self.tsdbs:
                        if out_name not in other_db.getm():
                            other_db.add(ts_new.copy())

                self.user_variables = getattr(self, "user_variables", set())
                self.user_variables.add(out_name)
                results.append((tsdb, ts_new))

            except Exception as e:
                QMessageBox.critical(self, "Calculation Error", f"{os.path.basename(self.file_paths[file_idx])}:\n{e}\n\n{traceback.format_exc()}")
                return

        self.refresh_variable_tabs()

        if len(file_tags_used) >= 2:
            msg = base_output
        else:
            msg = ", ".join(f"{base_output}_f{n}" for n in sorted(file_tags_used))
        QMessageBox.information(self, "Success", f"New variable(s): {msg}")

    def show_calc_help(self):
        """Display calculator usage help in a message box."""

        if not self.tsdbs:
            QMessageBox.information(
                self,
                "Calculator Help",
                "No files loaded – load files to see available variable references.",
            )
            return

        lines = [
            "👁‍🗨  Calculator Help",
            "",
            "📌  Prefix cheat-sheet",
            "     fN_<var>    variable from file N   (N = 1, 2, …)",
            "     c_<var>     common variable (present in every file)",
            "     u_<var>     user-created variable (all files)",
            "     u_<var>_fN  user variable that lives only in file N",
            "",
            "📝  Examples",
            "     result = f1_AccX + f2_AccY",
            "     diff   = c_WAVE1 - u_MyVar_f1",
            "",
            "The file number N corresponds to the indices shown in the",
            "'Loaded Files' list:",
            "",
        ]

        for idx, path in enumerate(self.file_paths, start=1):
            lines.append(f"     {idx}. {os.path.basename(path)}")

        lines.extend(
            [
                "",
                "🧬  Built-in math helpers",
                "     sin, cos, tan, sqrt, exp, log",
                "     abs, min, max, power, radians, degrees",
                "",
                "💡  Tips",
                "  •  Any valid Python / NumPy expression works (np.mean, np.std, …).",
                "  •  Give the left-hand side any name you like – it becomes a new",
                "     user variable (and appears under the 'User Variables' tab).",
                "  •  Autocomplete suggests prefixes and math functions as you type.",
            ]
        )

        QMessageBox.information(self, "Calculator Help", "\n".join(lines))

    def populate_var_list(self, var_list_widget, variables):
        var_list_widget.clear()
        self.var_widgets = {}
        for varname in variables:
            row_widget = VariableRowWidget(varname)
            item = QListWidgetItem(var_list_widget)
            item.setSizeHint(row_widget.sizeHint())
            var_list_widget.addItem(item)
            var_list_widget.setItemWidget(item, row_widget)
            self.var_widgets[varname] = row_widget
        var_list_widget.setMinimumWidth(350)  # You can make it even wider if needed

    def show_selected(self):
        out = []
        for varname, row in self.var_widgets.items():
            if row.checkbox.isChecked():
                try:
                    val = float(row.input.text() or 0)
                except ValueError:
                    val = "Invalid"
                out.append(f"{varname}: checked, value = {val}")
            else:
                out.append(f"{varname}: not checked")
        from PySide6.QtWidgets import QMessageBox
        QMessageBox.information(self, "Selections", "\n".join(out))

    def make_variable_row(self, var_key, var_label, checked=False, initial_value=None):
        """Return a widget with checkbox, input field and variable label."""
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(2, 2, 2, 2)

        chk = QCheckBox()
        chk.setChecked(checked)
        offset_edit = QLineEdit()
        offset_edit.setFixedWidth(60)
        if initial_value is not None:
            offset_edit.setText(str(initial_value))
        label = QLabel(var_label)

        layout.addWidget(chk)
        layout.addWidget(offset_edit)
        layout.addWidget(label)
        layout.addStretch(1)
        row.setLayout(layout)

        # Register in dictionaries for later access
        self.var_checkboxes[var_key] = chk
        self.var_offsets[var_key] = offset_edit
        return row

    def populate_variable_tab(self, tab_widget, var_keys, var_labels=None):
        layout = QVBoxLayout(tab_widget)
        for key in var_keys:
            label = var_labels[key] if var_labels and key in var_labels else key
            row = self.make_variable_row(key, label)
            layout.addWidget(row)
        layout.addStretch(1)
        tab_widget.setLayout(layout)

    def apply_values(self):
        """Apply numeric edits entered for each selected variable."""
        import os

        def _parse(txt: str):
            txt = txt.strip()
            if not txt:
                return None
            if txt[0] in "+-*/":
                op, num = txt[0], txt[1:].strip()
            else:
                op, num = "+", txt
            if not num:
                return None
            try:
                val = float(num)
            except ValueError:
                return None
            if op == "/" and abs(val) < 1e-12:
                return None
            return op, val

        common_ops, per_file_ops = {}, {}
        for ukey, entry in self.var_offsets.items():
            parsed = _parse(entry.text())
            if parsed is None:
                continue
            if "::" in ukey:
                f, v = ukey.split("::", 1)
                per_file_ops[(f, v)] = parsed
            elif ":" in ukey:
                f, v = ukey.split(":", 1)
                per_file_ops[(f, v)] = parsed
            else:
                common_ops[ukey] = parsed

        if not (common_ops or per_file_ops):
            QMessageBox.information(self, "Apply Values", "No valid edits were entered.")
            return

        make_new = self.apply_value_user_var_cb.isChecked()
        applied = 0
        conflicts = []
        self.user_variables = getattr(self, "user_variables", set())

        def _fmt_val(v: float) -> str:
            txt = f"{v:g}"
            return txt.replace(".", "p")

        for file_idx, (tsdb, fp) in enumerate(zip(self.tsdbs, self.file_paths), start=1):
            fname = os.path.basename(fp)
            local_per = {v: op for (f, v), op in per_file_ops.items() if f == fname}
            for var, ts in list(tsdb.getm().items()):
                has_c = var in common_ops
                has_p = var in local_per
                if not (has_c or has_p):
                    continue

                if has_c and has_p:
                    (opC, valC), (opP, valP) = common_ops[var], local_per[var]
                    if opC == opP and abs(valC - valP) < 1e-12:
                        op_use, val_use = opC, valC
                    elif all(op in "+-" for op in (opC, opP)):
                        zeroC, zeroP = abs(valC) < 1e-12, abs(valP) < 1e-12
                        if zeroC and not zeroP:
                            op_use, val_use = opP, valP
                        elif zeroP and not zeroC:
                            op_use, val_use = opC, valC
                        else:
                            conflicts.append(f"{fname}:{var}  (+{valC} vs +{valP})")
                            continue
                    else:
                        conflicts.append(f"{fname}:{var}  ({opC}{valC} vs {opP}{valP})")
                        continue
                else:
                    op_use, val_use = common_ops[var] if has_c else local_per[var]

                if make_new:
                    op_code = {"+": "p", "-": "m", "*": "x", "/": "d"}[op_use]
                    filt_tag = self._filter_tag()
                    base = f"{var}_{op_code}{_fmt_val(val_use)}"
                    if filt_tag:
                        base += f"_{filt_tag}"
                    base += f"_f{file_idx}"
                    name = base
                    n = 1
                    while name in tsdb.getm():
                        name = f"{base}_{n}"
                        n += 1
                    if op_use == "+":
                        data = ts.x + val_use
                    elif op_use == "-":
                        data = ts.x - val_use
                    elif op_use == "*":
                        data = ts.x * val_use
                    elif op_use == "/":
                        data = ts.x / val_use
                    new_ts = TimeSeries(name, ts.t.copy(), data)
                    tsdb.add(new_ts)
                    self.user_variables.add(name)
                else:
                    if op_use == "+":
                        ts.x = ts.x + val_use
                    elif op_use == "-":
                        ts.x = ts.x - val_use
                    elif op_use == "*":
                        ts.x = ts.x * val_use
                    elif op_use == "/":
                        ts.x = ts.x / val_use
                applied += 1

        self._populate_variables(None)
        summary = [f"{'Created' if make_new else 'Edited'} {applied} series."]
        if conflicts:
            summary.append("\nConflicts (skipped):")
            summary.extend(f"  • {c}" for c in conflicts)
        QMessageBox.information(self, "Apply Values", "\n".join(summary))

    def get_selected_keys(self):
        """Return all checked variables from all VariableTabs except User Variables."""
        keys = []
        for i in range(self.tabs.count()):
            tab = self.tabs.widget(i)
            # Only check variable tabs, not user variables
            # You can skip last tab if it's user vars, or check label if you want.
            if hasattr(tab, "selected_variables"):
                keys.extend(tab.selected_variables())
        return list(set(keys))

    def _apply_transformation(self, func, suffix, announce=True):
        """
        Apply *func* to every selected time-series and push the result back
        into the corresponding TsDB.

          new-name = <orig_name>_<suffix>_fN[_k]
                     └───────────────┘  └┘ └┘
                          copy        N  clash-counter
        """
        import os
        from PySide6.QtCore import QTimer
        from anyqats import TimeSeries

        self.rebuild_var_lookup()
        made = []
        fnames = [os.path.basename(p) for p in self.file_paths]

        def _has_file_prefix(key: str) -> bool:
            """Return True if *key* is prefixed with any loaded file name."""
            for name in fnames:
                if key.startswith(f"{name}::") or key.startswith(f"{name}:"):
                    return True
            return False

        for f_idx, (tsdb, path) in enumerate(zip(self.tsdbs, self.file_paths), start=1):
            fname = os.path.basename(path)

            for u_key, chk in self.var_checkboxes.items():
                if not chk.isChecked():
                    continue

                # ── resolve unique-key to var name inside *this* file ─────────
                if u_key.startswith(f"{fname}::"):
                    varname = u_key.split("::", 1)[1]
                elif u_key.startswith(f"{fname}:"):
                    varname = u_key.split(":", 1)[1]
                elif not _has_file_prefix(u_key):
                    varname = u_key
                else:
                    continue

                ts = tsdb.getm().get(varname)
                if ts is None:
                    continue

                mask = self.get_time_window(ts)

                # 📌── accept slice OR ndarray ────────────────────────────────
                if isinstance(mask, slice):  # full window
                    t_win = ts.t[mask]
                    y_src = self.apply_filters(ts)[mask]
                else:  # boolean ndarray
                    if not mask.any():  # completely empty
                        continue
                    t_win = ts.t[mask]
                    y_src = self.apply_filters(ts)[mask]
                # ----------------------------------------------------------------

                y_new = func(y_src)

                # ── unique name inside this file ─────────────────────────────
                filt_tag = self._filter_tag()
                base = f"{ts.name}_{suffix}"
                if filt_tag:
                    base += f"_{filt_tag}"
                base += f"_f{f_idx}"
                new_name = base
                k = 1
                while new_name in tsdb.getm():
                    new_name = f"{base}_{k}"
                    k += 1

                tsdb.add(TimeSeries(new_name, t_win, y_new))
                made.append(new_name)

                # mark global user-var
                self.user_variables = getattr(self, "user_variables", set())
                self.user_variables.add(new_name)

        # ── GUI refresh & popup ──────────────────────────────────────────────
        if made:
            QTimer.singleShot(0, lambda: self._populate_variables(None))
            if announce:

                def _ok():
                    show = 10
                    if len(made) <= show:
                        msg = "\n".join(sorted(made))
                    else:
                        msg = (
                            "\n".join(sorted(made)[:show])
                            + f"\n… and {len(made) - show} more"
                        )
                    QMessageBox.information(self, "Transformation complete", msg)

                QTimer.singleShot(0, _ok)
        elif announce:
            QTimer.singleShot(
                0,
                lambda: QMessageBox.warning(
                    self,
                    "Nothing new",
                    "All requested series already exist – no new series created.",
                ),
            )

    def abs_var(self):
        import numpy as np
        from PySide6.QtWidgets import QMessageBox

        self._apply_transformation(lambda y: np.abs(y), "abs", True)

    def rolling_average(self):
        """Apply rolling mean to all selected series."""
        import pandas as pd

        window = 1
        if hasattr(self, "rolling_window"):
            try:
                window = max(1, int(self.rolling_window.value()))
            except Exception:
                window = 1

        func = lambda y, w=window: pd.Series(y).rolling(window=w, min_periods=1).mean().to_numpy()
        self._apply_transformation(func, "rollMean", True)

    def sqrt_sum_of_squares(self):
        """
        √(Σ xi²) on the currently-selected variables.

        • If you pick only *Common-tab* variables, every file gets its own
          result, named  sqrt_sum_of_squares(varA+varB)_fN

        • If you select explicit per-file keys (filename::var), each file
          gets exactly one result (the filename part is already unique).
        """
        import numpy as np, os, re
        from anyqats import TimeSeries
        from PySide6.QtCore import QTimer
        from PySide6.QtWidgets import QMessageBox


        self.rebuild_var_lookup()

        sel_keys = [k for k, ck in self.var_checkboxes.items() if ck.isChecked()]
        if not sel_keys:
            QMessageBox.warning(
                self, "No selection", "Select variables to apply the transformation."
            )
            return

        # ── helper: strip one trailing “_f<number>” (if any) ──────────────────
        _re_f = re.compile(r"_f\d+$")

        def _strip_f_suffix(name: str) -> str:
            return _re_f.sub("", name)

        multi_file = len(self.tsdbs) > 1
        common_pick = all("::" not in k for k in sel_keys)
        created = []

        self.user_variables = getattr(self, "user_variables", set())

        # ───────────────────────── COMMON-TAB BRANCH ──────────────────────────
        if common_pick:
            for f_idx, (tsdb, fp) in enumerate(zip(self.tsdbs, self.file_paths), 1):
                values, t_ref = [], None
                for k in sel_keys:
                    ts = tsdb.getm().get(k)
                    if ts is None:
                        continue
                    if t_ref is None:
                        t_ref = ts.t
                    elif not np.allclose(ts.t, t_ref):
                        QMessageBox.critical(
                            self,
                            "Time mismatch",
                            f"Time mismatch in {os.path.basename(fp)} for '{k}'",
                        )
                        return
                    values.append(ts.x)
                if not values:
                    continue

                y = np.sqrt(np.sum(np.vstack(values) ** 2, axis=0))

                # build *clean* base name (no duplicate _fN tails inside)
                clean_keys = [_strip_f_suffix(k) for k in sel_keys]
                base = f"sqrt_sum_of_squares({'+'.join(clean_keys)})"
                suffix = f"_f{f_idx}" if multi_file else ""
                name = f"{base}{suffix}"

                n = 1
                while name in tsdb.getm():
                    name = f"{base}{suffix}_{n}"
                    n += 1

                tsdb.add(TimeSeries(name, t_ref, y))
                self.user_variables.add(name)
                created.append(name)

        # ──────────────────────── PER-FILE-KEY BRANCH ─────────────────────────
        else:
            per_file = {}
            for k in sel_keys:
                if "::" not in k:
                    QMessageBox.critical(
                        self,
                        "Mixed selection",
                        "Choose either only common-tab or only per-file keys.",
                    )
                    return
                fname, var = k.split("::", 1)
                per_file.setdefault(fname, []).append(var)

            for tsdb, fp in zip(self.tsdbs, self.file_paths):
                fname = os.path.basename(fp)
                if fname not in per_file:
                    continue

                values, t_ref = [], None
                for v in per_file[fname]:
                    ts = tsdb.getm().get(v)
                    if ts is None:
                        continue
                    if t_ref is None:
                        t_ref = ts.t
                    elif not np.allclose(ts.t, t_ref):
                        QMessageBox.critical(
                            self,
                            "Time mismatch",
                            f"Time mismatch in {fname} for '{v}'",
                        )
                        return
                    values.append(ts.x)
                if not values:
                    continue

                y = np.sqrt(np.sum(np.vstack(values) ** 2, axis=0))

                clean = [_strip_f_suffix(v) for v in per_file[fname]]
                base = f"sqrt_sum_of_squares({'+'.join(clean)})"
                suffix = f"_f{self.file_paths.index(fp) + 1}"
                name = f"{base}{suffix}"

                n = 1
                while name in tsdb.getm():
                    name = f"{base}{suffix}_{n}"
                    n += 1

                tsdb.add(TimeSeries(name, t_ref, y))
                self.user_variables.add(name)
                created.append(name)

        # ─────────────────────────── GUI refresh ───────────────────────────────
        if created:
            QTimer.singleShot(0, self._populate_variables)
            print("✅ Added:", created)
        else:
            QTimer.singleShot(
                0,
                lambda: QMessageBox.warning(
                    self,
                    "No new series",
                    "All requested series already exist — no new series created.",
                ),
            )

    def mean_of_selected(self):
        """
        Compute the arithmetic mean of every *checked* variable.

        ─ Selection rules ─────────────────────────────────────────────
        • If you chose only Common-tab keys → one mean per file:
            mean(varA+varB)_fN

        • If you picked any per-file key   → one mean per file using the
          keys that belong to that very file.  (The filename already
          distinguishes them, so no extra suffix is added.)
        """
        import numpy as np, os, re
        from anyqats import TimeSeries
        from PySide6.QtWidgets import QMessageBox

        sel_keys = [k for k, ck in self.var_checkboxes.items() if ck.isChecked()]
        if not sel_keys:
            QMessageBox.warning(
                self, "No selection", "Select variables to apply the transformation."
            )
            return

        # ── regex: strip exactly one trailing “_f<number>” (if any) ──────────
        _re_f = re.compile(r"_f\d+$")
        _clean = lambda s: _re_f.sub("", s)

        common_pick = all("::" not in k for k in sel_keys)
        multi_file = len(self.tsdbs) > 1
        created = []

        self.user_variables = getattr(self, "user_variables", set())

        # ───────────────────────── helper ─────────────────────────────
        def _store(tsdb, name_base, t_ref, vals):
            """Add a new TimeSeries, ensuring uniqueness inside *tsdb*."""
            y = np.mean(np.vstack(vals), axis=0)
            new = name_base
            n = 1
            while new in tsdb.getm():
                new = f"{name_base}_{n}"
                n += 1
            tsdb.add(TimeSeries(new, t_ref, y))
            self.user_variables.add(new)
            created.append(new)

        # ─────────────────── COMMON-TAB BRANCH ────────────────────────
        if common_pick:
            clean_keys = [_clean(k) for k in sel_keys]

            for f_idx, (tsdb, fp) in enumerate(zip(self.tsdbs, self.file_paths), 1):
                vals, t_ref = [], None
                for k in sel_keys:
                    ts = tsdb.getm().get(k)
                    if ts is None:
                        continue
                    if t_ref is None:
                        t_ref = ts.t
                    elif not np.allclose(ts.t, t_ref):
                        QMessageBox.critical(
                            self,
                            "Time mismatch",
                            f"Time mismatch in {os.path.basename(fp)} for '{k}'",
                        )
                        return
                    vals.append(self.apply_filters(ts)[self.get_time_window(ts)])

                if not vals:
                    continue

                suffix = f"_f{f_idx}" if multi_file else ""
                namebase = f"mean({'+'.join(clean_keys)}){suffix}"
                _store(tsdb, namebase, t_ref, vals)

        # ────────────────── PER-FILE-KEY BRANCH ───────────────────────
        else:
            per_file = {}
            for k in sel_keys:
                if "::" not in k:
                    QMessageBox.critical(
                        self,
                        "Mixed selection",
                        "Pick either only common-tab or only per-file keys.",
                    )
                    return
                fname, var = k.split("::", 1)
                per_file.setdefault(fname, []).append(var)

            for tsdb, fp in zip(self.tsdbs, self.file_paths):
                fname = os.path.basename(fp)
                vars_here = per_file.get(fname)
                if not vars_here:
                    continue

                vals, t_ref = [], None
                for v in vars_here:
                    ts = tsdb.getm().get(v)
                    if ts is None:
                        continue
                    if t_ref is None:
                        t_ref = ts.t
                    elif not np.allclose(ts.t, t_ref):
                        QMessageBox.critical(
                            self,
                            "Time mismatch",
                            f"Time mismatch in {fname} for '{v}'",
                        )
                        return
                    vals.append(self.apply_filters(ts)[self.get_time_window(ts)])

                if not vals:
                    continue

                clean = [_clean(v) for v in vars_here]
                namebase = f"mean({'+'.join(clean)})"  # ← no _fN here
                _store(tsdb, namebase, t_ref, vals)

        # ───────────────────── GUI refresh ────────────────────────────
        if created:
            QTimer.singleShot(0, self._populate_variables)
            print("✅ Added mean series:", created)
        else:
            QTimer.singleShot(
                0,
                lambda: QMessageBox.warning(
                    self,
                    "No new series",
                    "All requested series already exist — no new series created.",
                ),
            )

    def multiply_by_1000(self):
        self._apply_transformation(lambda y: y * 1000, "×1000", True)

    def divide_by_1000(self):
        self._apply_transformation(lambda y: y / 1000, "÷1000", True)

    def multiply_by_10(self):
        self._apply_transformation(lambda y: y * 10, "×10", True)

    def divide_by_10(self):
        self._apply_transformation(lambda y: y / 10, "÷10", True)

    def multiply_by_2(self):
        self._apply_transformation(lambda y: y * 2, "×2", True)

    def divide_by_2(self):
        self._apply_transformation(lambda y: y / 2, "÷2", True)

    def multiply_by_neg1(self):
        self._apply_transformation(lambda y: y * -1, "×-1", True)

    def to_radians(self):
        import numpy as np

        self._apply_transformation(lambda y: np.radians(y), "rad", True)

    def to_degrees(self):
        import numpy as np

        self._apply_transformation(lambda y: np.degrees(y), "deg", True)

    def shift_min_to_zero(self):
        """Shift series so its minimum becomes zero **only** when that minimum is negative."""
        import numpy as np

        def shift(y: np.ndarray) -> np.ndarray:
            # (1) Find the reference minimum – optionally ignoring the lowest 1 %
            if self.ignore_anomalies_cb.isChecked():
                lower = np.sort(y)[int(len(y) * 0.01)]  # 1 % quantile
            else:
                lower = np.min(y)

            # (2) Do nothing if the series is already non-negative
            if lower >= 0:
                return y

            # (3) Otherwise shift the whole series up
            return y - lower

        # Create a new series with suffix “…_shift0”
        self._apply_transformation(shift, "shift0", True)

    def shift_repeated_neg_min(self):
        """
        Shift a series upward so that a *repeated* negative minimum becomes 0.

        The user supplies two numbers in the toolbar:

            Tol [%]   →  self.shift_tol_entry   (e.g. 0.001 = 0.001 %)
            Min count →  self.shift_cnt_entry   (integer ≥ 1)

        A shift is applied **only if**
          • the minimum value is negative, **and**
          • at least *Min count* samples lie within ±Tol % of that minimum.

        The new series are named  “<oldname>_shiftNZ”  (NZ = non-zero).
        """

        import numpy as np
        from PySide6.QtWidgets import QMessageBox

        # ── read parameters from the two entry boxes ──────────────────────
        try:
            tol_pct = float(self.shift_tol_entry.text()) / 100.0  # % → fraction
        except ValueError:
            QMessageBox.critical(
                self, "Invalid tolerance", "Enter a number in the Tol [%] box."
            )
            return

        try:
            min_count = int(self.shift_cnt_entry.text())
            if min_count < 1:
                raise ValueError
        except ValueError:
            QMessageBox.critical(
                self, "Invalid count", "Enter a positive integer in the Min count box."
            )
            return

        self.rebuild_var_lookup()

        # ── helper that is executed on every selected y-vector ────────────
        def _shift_if_plateau(y):
            y = np.asarray(y, dtype=float)
            if y.size == 0:
                return y

            ymin, ymax = y.min(), y.max()
            if ymin >= 0:
                return y  # already non-negative

            tol_abs = abs(ymin) * tol_pct  # absolute tolerance

            plate_cnt = np.count_nonzero(np.abs(y - ymin) <= tol_abs)
            print(plate_cnt, min_count, tol_pct, tol_abs)
            if plate_cnt >= min_count:
                return y - ymin  # shift so ymin → 0
            return y  # leave unchanged

        # reuse the generic helper (takes care of naming, user_variables, refresh)
        self._apply_transformation(_shift_if_plateau, "shiftNZ", True)

    def shift_common_max(self):
        """
        For each selected *common* variable (one that exists in ALL files),
        compute the negative‐minimum plateau‐based shift (if any) in each file,
        then take the LARGEST of those shifts and apply it to every selected common
        variable in every file.  New series are named "<oldname>_shiftCommon_fN".
        """

        import numpy as np

        # 1) Read tolerance [%] and minimum count
        try:
            tol_pct = float(self.shift_tol_entry.text()) / 100.0
        except ValueError:
            QMessageBox.critical(
                self, "Invalid tolerance", "Enter a number in the Tol [%] box."
            )
            return

        try:
            min_count = int(self.shift_cnt_entry.text())
            if min_count < 1:
                raise ValueError
        except ValueError:
            QMessageBox.critical(
                self, "Invalid count", "Enter a positive integer in the Min count box."
            )
            return

        # 2) Gather all currently selected common keys
        selected_common = [
            key
            for key, var in self.var_checkboxes.items()
            if var.isChecked() and "::" not in key and ":" not in key
        ]
        if not selected_common:
            QMessageBox.warning(
                self,
                "No Common Variables",
                "Select one or more common variables (in the Common tab) to shift.",
            )
            return

        # 3) Compute each file's candidate shift for each key
        all_shifts = []
        for key in selected_common:
            for tsdb in self.tsdbs:
                ts = tsdb.getm().get(key)
                if ts is None:
                    continue  # shouldn’t happen for a “common” key
                mask = self.get_time_window(ts)
                if mask is None or not np.any(mask):
                    continue
                y = self.apply_filters(ts)[mask]
                if y.size == 0:
                    continue

                ymin = np.min(y)
                if ymin >= 0:
                    continue

                tol_abs = abs(ymin) * tol_pct
                count = np.count_nonzero(np.abs(y - ymin) <= tol_abs)
                if count >= min_count:
                    all_shifts.append(-ymin)

        # 4) Find the largest shift
        if not all_shifts:
            QMessageBox.information(
                self,
                "No Shift Needed",
                "No common variable met the plateau criteria.",
            )
            return

        max_shift = max(all_shifts)
        if max_shift <= 0:
            QMessageBox.information(
                self,
                "No Shift Needed",
                "All selected series are already ≥ 0 or don't meet the count.",
            )
            return

        # 5) Temporarily turn OFF any per‐file checkboxes; leave only common‐keys ON:
        saved_state = {k: var.isChecked() for k, var in self.var_checkboxes.items()}
        try:
            # Turn OFF any per‐file or user‐variable checkboxes
            for unique_key in list(self.var_checkboxes.keys()):
                if "::" in unique_key or ":" in unique_key:
                    self.var_checkboxes[unique_key].setChecked(False)

            # Ensure each common key remains selected
            for key in selected_common:
                self.var_checkboxes[key].setChecked(True)

            # Call _apply_transformation (this will add one “_shiftCommon_fN” per file)
            self._apply_transformation(
                lambda y: y + max_shift, "shiftCommon", print_it=False
            )
        finally:
            # Restore the original check states
            for k, v in saved_state.items():
                self.var_checkboxes[k].setChecked(v)

        num_files = len(self.tsdbs)
        QMessageBox.information(
            self,
            "Success",
            f"Shifted {len(selected_common)} common variable(s) by {max_shift:.4g} across {num_files} files.",
        )

    def shift_mean_to_zero(self):
        """
        Shift each selected time-series vertically so that its *mean* becomes 0.

        ‣ If *Ignore anomalies* (self.ignore_anomalies_cb) is ticked,
          the mean is computed on the central 98 % (1-99 % percentiles) to
          reduce the influence of outliers — consistent with your other tools.

        Saved as:  <origName>_shiftMean0   (or _shiftMean0_1, _2, … if needed)
        """
        import numpy as np

        def _demean(y: np.ndarray) -> np.ndarray:
            if self.ignore_anomalies_cb.isChecked():
                # robust mean: trim 1 % at both ends
                p01, p99 = np.percentile(y, [1, 99])
                mask = (y >= p01) & (y <= p99)
                m = np.mean(y[mask]) if np.any(mask) else np.mean(y)
            else:
                m = np.mean(y)
            return y - m

        # suffix “shiftMean0” keeps the style of “shift0”, “shiftNZ”, …
        self._apply_transformation(_demean, "shiftMean0", True)

    @Slot()
    def load_files(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Open time series files", "", "All Files (*)")
        if not files:
            return
        self.update_progressbar(0, len(files))
        self.file_loader.progress_callback = self.update_progressbar
        sim_files = [fp for fp in files if fp.lower().endswith(".sim")]
        if sim_files:
            self.file_loader.preload_sim_models(sim_files)
        tsdbs, errors = self.file_loader.load_files(files)

        def _true_index(fp: str) -> int:
            if fp in self.file_paths:
                return self.file_paths.index(fp) + 1
            return len(self.file_paths) + 1

        for path, tsdb in zip(files, tsdbs):
            idx = _true_index(path)
            rename_map = {}
            for key in list(tsdb.getm().keys()):
                if not _looks_like_user_var(key):
                    continue
                m = re.search(r"_f(\d+)$", key)
                if m and int(m.group(1)) == idx:
                    continue
                base = re.sub(r"_f\d+$", "", key)
                new_key = f"{base}_f{idx}"
                dup = 1
                while new_key in tsdb.getm() or new_key in rename_map.values():
                    new_key = f"{base}_f{idx}_{dup}"
                    dup += 1
                rename_map[key] = new_key

            for old, new in rename_map.items():
                ts = tsdb.getm().pop(old)
                ts.name = new
                tsdb.getm()[new] = ts

            for k in tsdb.getm():
                if _looks_like_user_var(k):
                    self.user_variables.add(k)

            self.tsdbs.append(tsdb)
            self.file_paths.append(path)
            self.file_list.addItem(os.path.basename(path))
            #print(f"Loaded {path}: variables = {list(tsdb.getm().keys())}")
        if errors:
            QMessageBox.warning(self, "Errors occurred", "\n".join([f"{f}: {e}" for f, e in errors]))
        self.refresh_variable_tabs()

    def remove_selected_file(self):
        idx = self.file_list.currentRow()
        if idx < 0:
            return
        del self.tsdbs[idx]
        del self.file_paths[idx]
        self.file_list.takeItem(idx)
        self.refresh_variable_tabs()

    def clear_all_files(self):
        self.tsdbs.clear()
        self.file_paths.clear()
        self.file_list.clear()
        self.refresh_variable_tabs()

    def reselect_orcaflex_variables(self):
        """Re-open the OrcaFlex picker for currently loaded .sim files."""
        self.file_loader.reuse_orcaflex_selection = False
        sim_paths = [p for p in self.file_paths if p.lower().endswith(".sim")]
        if not sim_paths:
            return

        tsdb_map = self.file_loader.open_orcaflex_picker(sim_paths)
        if not tsdb_map:
            return

        for path in sim_paths:
            if path not in tsdb_map:
                continue
            tsdb = tsdb_map[path]
            idx = self.file_paths.index(path) + 1

            rename_map = {}
            for key in list(tsdb.getm().keys()):
                if not _looks_like_user_var(key):
                    continue
                m = re.search(r"_f(\d+)$", key)
                if m and int(m.group(1)) == idx:
                    continue
                base = re.sub(r"_f\d+$", "", key)
                new_key = f"{base}_f{idx}"
                dup = 1
                while new_key in tsdb.getm() or new_key in rename_map.values():
                    new_key = f"{base}_f{idx}_{dup}"
                    dup += 1
                rename_map[key] = new_key

            for old, new in rename_map.items():
                ts = tsdb.getm().pop(old)
                ts.name = new
                tsdb.getm()[new] = ts

            for k in tsdb.getm():
                if _looks_like_user_var(k):
                    self.user_variables.add(k)

            self.tsdbs[self.file_paths.index(path)] = tsdb

        self.refresh_variable_tabs()

    def refresh_variable_tabs(self):
        """Rebuild all variable tabs and map checkboxes for later access."""
        # Remove existing tabs
        while self.tabs.count():
            self.tabs.removeTab(0)

        # Clear previous lookup tables
        self.var_checkboxes = {}
        self.var_offsets = {}

        user_vars = set(self.user_variables) if hasattr(self, "user_variables") else set()

        # ---- Common variables -------------------------------------------------
        if not self.tsdbs:
            common_keys = set()
        else:
            common_keys = set(self.tsdbs[0].getm().keys())
            for tsdb in self.tsdbs[1:]:
                common_keys &= set(tsdb.getm().keys())

        common_tab = VariableTab("Common", common_keys - user_vars)
        self.tabs.addTab(common_tab, "Common")
        self.common_tab_widget = common_tab
        for key, cb in common_tab.checkboxes.items():
            self.var_checkboxes[key] = cb
            self.var_offsets[key] = common_tab.inputs.get(key)

        # ---- Per-file variables ---------------------------------------------
        for i, (tsdb, path) in enumerate(zip(self.tsdbs, self.file_paths), start=1):
            label = f"File {i}: {os.path.basename(path)}"
            var_keys = tsdb.getm().keys()
            tab = VariableTab(label, var_keys, user_var_set=user_vars)
            self.tabs.addTab(tab, label)

            prefix = os.path.basename(path)
            for var, cb in tab.checkboxes.items():
                u_key = f"{prefix}::{var}"
                self.var_checkboxes[u_key] = cb
                self.var_offsets[u_key] = tab.inputs.get(var)

        # ---- User variables --------------------------------------------------
        user_tab = VariableTab(
            "User Variables",
            user_vars,
            allow_rename=True,
            rename_callback=self.rename_user_variable,
        )
        self.tabs.addTab(user_tab, "User Variables")
        self.user_tab_widget = user_tab
        for key, cb in user_tab.checkboxes.items():
            self.var_checkboxes[key] = cb
            self.var_offsets[key] = user_tab.inputs.get(key)

        # Update lookup whenever tabs rebuild
        for tab in (common_tab, *[
            self.tabs.widget(i) for i in range(1, self.tabs.count() - 1)
        ], user_tab):
            if hasattr(tab, "checklist_updated"):
                tab.checklist_updated.connect(self.rebuild_var_lookup)

        # initial build
        self.rebuild_var_lookup()
        self._build_calc_variable_list()
        self._update_orcaflex_buttons()

    def rebuild_var_lookup(self):
        """Reconstruct the checkbox lookup after a tab refresh/search."""
        self.var_checkboxes = {}
        self.var_offsets = {}
        if not self.tabs.count():
            return

        # Common tab (index 0)
        common = self.tabs.widget(0)
        if hasattr(common, "checkboxes"):
            for k, cb in common.checkboxes.items():
                self.var_checkboxes[k] = cb
                if hasattr(common, "inputs"):
                    self.var_offsets[k] = common.inputs.get(k)

        # Per-file tabs
        for idx, path in enumerate(self.file_paths, start=1):
            if idx >= self.tabs.count():
                break
            tab = self.tabs.widget(idx)
            if not hasattr(tab, "checkboxes"):
                continue
            prefix = os.path.basename(path)
            for k, cb in tab.checkboxes.items():
                u_key = f"{prefix}::{k}"
                self.var_checkboxes[u_key] = cb
                if hasattr(tab, "inputs"):
                    self.var_offsets[u_key] = tab.inputs.get(k)

        # User variables tab (last)
        last = self.tabs.widget(self.tabs.count() - 1)
        if hasattr(last, "checkboxes"):
            for k, cb in last.checkboxes.items():
                self.var_checkboxes[k] = cb
                if hasattr(last, "inputs"):
                    self.var_offsets[k] = last.inputs.get(k)


    # ------------------------------------------------------------------
    # Compatibility helper -------------------------------------------------
    def _populate_variables(self, *_):
        """Backward‑compatible wrapper used by older callbacks."""
        self.refresh_variable_tabs()

    def highlight_file_tab(self, row):
        if row >= 0 and row+1 < self.tabs.count():
            self.tabs.setCurrentIndex(row+1)

    def update_progressbar(self, value, maximum=None):
        """Update the progress bar during lengthy operations."""
        if maximum is not None:
            self.progress.setMaximum(maximum)
        self.progress.setValue(value)
        QApplication.processEvents()

    def _unselect_all_variables(self):
        """Uncheck every variable checkbox in all tabs."""
        for cb in self.var_checkboxes.values():
            cb.setChecked(False)

    def _select_all_by_list_pos(self):

        """Select variables in all per-file tabs based on list positions."""

        idx = self.tabs.currentIndex()
        # Valid per-file tabs live between the common tab (0) and the user tab (last)

        if idx <= 0 or idx >= self.tabs.count() - 1:
            return

        current_tab = self.tabs.widget(idx)
        if not hasattr(current_tab, "all_vars"):
            return


        # Build filtered variable list for the active tab
        terms = _parse_search_terms(current_tab.search_box.text())
        if not terms:
            src_vars = current_tab.all_vars
        else:
            src_vars = [v for v in current_tab.all_vars if _matches_terms(v, terms)]

        positions = [i for i, var in enumerate(src_vars)

                     if current_tab.checkboxes.get(var) and current_tab.checkboxes[var].isChecked()]
        if not positions:
            return


        # Apply the same positions to every other per-file tab assuming the same filter

        for j in range(1, self.tabs.count() - 1):
            if j == idx:
                continue
            tab = self.tabs.widget(j)
            if not hasattr(tab, "all_vars"):
                continue


            # Determine which variables would be visible with the same search terms
            if not terms:
                tgt_vars = tab.all_vars
            else:
                tgt_vars = [v for v in tab.all_vars if _matches_terms(v, terms)]

            for pos in positions:
                if pos < len(tgt_vars):
                    var = tgt_vars[pos]

                    cb = tab.checkboxes.get(var)
                    if cb:
                        cb.setChecked(True)

    def _update_orcaflex_buttons(self):
        """Show or hide OrcaFlex-specific buttons based on loaded files."""
        has_sim = any(fp.lower().endswith(".sim") for fp in self.file_paths)
        self.clear_orcaflex_btn.setVisible(has_sim)
        self.reselect_orcaflex_btn.setVisible(has_sim)

    def rename_user_variable(self, old_name: str, new_name: str):

        """Rename ``old_name`` to ``new_name`` across all loaded files."""

        if not new_name:
            return

        new_name = new_name.strip()
        if not new_name:
            return

        exists = any(new_name in tsdb.getm() for tsdb in self.tsdbs)
        if exists or new_name in self.user_variables:
            QMessageBox.warning(self, "Name exists", f"Variable '{new_name}' already exists.")
            return

        was_checked = False
        if old_name in self.var_checkboxes:
            was_checked = self.var_checkboxes[old_name].isChecked()


        renamed = False
        for tsdb in self.tsdbs:
            if old_name in tsdb.getm():
                ts = tsdb.getm().pop(old_name)
                ts.name = new_name
                tsdb.getm()[new_name] = ts
                renamed = True


        if not renamed:
            return

        if old_name in self.user_variables:
            self.user_variables.remove(old_name)
        self.user_variables.add(new_name)

        self.refresh_variable_tabs()

        if was_checked and new_name in self.var_checkboxes:
            self.var_checkboxes[new_name].setChecked(True)


    def _trim_label(self, label, left_chars, right_chars):
        try:
            left = int(left_chars)
            right = int(right_chars)
        except Exception:
            left, right = 10, 50  # fallback defaults
        if left <= 0 and right <= 0:
            return label
        if left <= 0:
            return label if len(label) <= right else label[-right:]
        if right <= 0:
            return label if len(label) <= left else label[:left]
        if len(label) <= left + right + 3:
            return label
        return f"{label[:left]}...{label[-right:]}"

    def show_stats(self):
        """Show descriptive statistics for all selected variables."""

        import os
        import numpy as np
        from PySide6.QtWidgets import QMessageBox as mb

        # Refresh user variables (labels loaded from disk)
        self.user_variables = getattr(self, "user_variables", set())
        for tsdb in self.tsdbs:
            for k in tsdb.getm():
                if "[User]" in k:
                    self.user_variables.add(k)

        self.rebuild_var_lookup()

        def _uniq(paths):
            names = [os.path.basename(p) for p in paths]
            if len(names) <= 1:
                return [""] * len(names)
            pre = os.path.commonprefix(names)
            suf = os.path.commonprefix([n[::-1] for n in names])[::-1]
            out = []
            for n in names:
                u = n[len(pre):] if pre else n
                u = u[:-len(suf)] if suf and u.endswith(suf) else u
                out.append(u or "(all)")
            return out

        fnames = [os.path.basename(p) for p in self.file_paths]
        uniq_map = dict(zip(fnames, _uniq(self.file_paths)))

        series_info = []
        for file_idx, (tsdb, fp) in enumerate(zip(self.tsdbs, self.file_paths), start=1):
            fname = os.path.basename(fp)
            for uk, ck in self.var_checkboxes.items():
                if not ck.isChecked():
                    continue
                if uk.startswith(f"{fname}::"):
                    key = uk.split("::", 1)[1]
                elif uk.startswith(f"{fname}:"):
                    key = uk.split(":", 1)[1]
                else:
                    key = uk
                    if key not in tsdb.getm():
                        alt = f"{key}_f{file_idx}"
                        if alt in tsdb.getm():
                            key = alt
                        else:
                            continue
                ts = tsdb.getm().get(key)
                if ts is None:
                    continue
                mask = self.get_time_window(ts)
                t_win = ts.t[mask]
                x_win = ts.x[mask]
                series_info.append({
                    "file": fname,
                    "uniq_file": uniq_map.get(fname, ""),
                    "file_idx": file_idx,
                    "var": key,
                    "t": t_win,
                    "x": x_win,
                })

        if not series_info:
            mb.warning(self, "No selection", "Select variables then retry.")
            return

        dlg = StatsDialog(series_info, self)
        dlg.exec()

    def plot_selected_side_by_side(self, checked: bool = False):
        """Plot all selected series in a grid of subplots.

        This wrapper slot is used for the "Plot Selected (side-by-side)" button
        to ensure the ``grid`` argument is always passed with ``True`` even
        though ``QPushButton.clicked`` emits a boolean ``checked`` parameter.
        """
        # Forward the call to ``plot_selected`` with ``grid`` enabled.
        self.plot_selected(grid=True)

    def plot_selected(self, *, mode: str = "time", grid: bool = False):
        """
        Plot all ticked variables.

        Parameters
        ----------
        mode : {"time", "psd", "cycle", "cycle_rm", "cycle_rm3d", "rolling"}
            * time       – original raw / LP / HP line plot
            * psd        – TimeSeries.plot_psd()
            * cycle      – TimeSeries.plot_cycle_range()
            * cycle_rm   – TimeSeries.plot_cycle_rangemean()
            * cycle_rm3d – TimeSeries.plot_cycle_rangemean3d()
            * rolling    – time plot using rolling mean
        """


        self.rebuild_var_lookup()

        mark_extrema = (
            hasattr(self, "plot_extrema_cb") and self.plot_extrema_cb.isChecked()
        )

        import numpy as np, anyqats as qats, os
        from PySide6.QtWidgets import QMessageBox
        import matplotlib.pyplot as plt
        from anyqats import TimeSeries

        roll_window = 1
        if hasattr(self, "rolling_window"):
            try:
                roll_window = max(1, int(self.rolling_window.value()))
            except Exception:
                roll_window = 1

        # ---------- sanity for raw / LP / HP check-boxes (time-plot only) -------
        want_raw = self.plot_raw_cb.isChecked()
        want_lp = self.plot_lowpass_cb.isChecked()
        want_hp = self.plot_highpass_cb.isChecked()

        if mode == "time" and not (want_raw or want_lp or want_hp):
            QMessageBox.warning(
                self,
                "Nothing to plot",
                "Tick at least one of Raw / Low-pass / High-pass.",
            )
            return

        # keep a Figure per file (except for time-domain where we merge)
        fig_per_file = {}

        # =======================================================================
        #  MAIN LOOP   (file ⨯ selected key)
        # =======================================================================
        traces = []  # for the time-domain case
        # ``grid_traces`` keeps the original, untrimmed label as key to avoid
        # accidental merging when two trimmed labels become identical.  Each
        # value stores the display label and the collected curve data.
        grid_traces = {}
        left, right = self.label_trim_left.value(), self.label_trim_right.value()

        from collections import Counter

        fname_counts = Counter(os.path.basename(p) for p in self.file_paths)

        for file_idx, (tsdb, fp) in enumerate(
            zip(self.tsdbs, self.file_paths), start=1
        ):
            fname = os.path.basename(fp)
            fname_disp = fname if fname_counts[fname] == 1 else f"{fname} ({file_idx})"
            tsdb_name = os.path.splitext(fname)[0]

            for key, sel in self.var_checkboxes.items():
                if not sel.isChecked():
                    continue

                # 1) resolve key → variable inside *this* tsdb
                if key.startswith(f"{fname}::"):
                    var = key.split("::", 1)[1]
                elif key.startswith(f"{fname}:"):
                    var = key.split(":", 1)[1]
                elif key in tsdb.getm():
                    var = key
                else:
                    continue

                ts = tsdb.getm().get(var)
                if ts is None:
                    continue

                # 2) apply current time window
                mask = self.get_time_window(ts)
                if isinstance(mask, slice):
                    ts_win = TimeSeries(ts.name, ts.t[mask], ts.x[mask])
                else:
                    if not mask.any():
                        continue
                    ts_win = TimeSeries(ts.name, ts.t[mask], ts.x[mask])

                # 3) optional pre-filtering for time-domain plot
                if mode == "time":
                    dt = np.median(np.diff(ts.t))
                    raw_label = f"{fname_disp}: {var}"
                    disp_label = self._trim_label(raw_label, left, right)
                    entry = grid_traces.setdefault(
                        raw_label, {"label": disp_label, "curves": []}
                    )
                    curves = entry["curves"]
                    if want_raw:
                        tr = dict(
                            t=ts_win.t,
                            y=ts_win.x,
                            label=disp_label + " [raw]",
                            alpha=1.0,
                        )
                        traces.append(tr)
                        curves.append(dict(t=ts_win.t, y=ts_win.x, label="Raw", alpha=1.0))
                    if want_lp:
                        fc = float(self.lowpass_cutoff.text() or 0)
                        if fc > 0:
                            y_lp = qats.signal.lowpass(ts_win.x, dt, fc)
                            tr = dict(
                                t=ts_win.t,
                                y=y_lp,
                                label=disp_label + f" [LP {fc} Hz]",
                                alpha=1.0,
                            )
                            traces.append(tr)
                            curves.append(
                                dict(t=ts_win.t, y=y_lp, label=f"LP {fc} Hz", alpha=1.0)
                            )
                    if want_hp:
                        fc = float(self.highpass_cutoff.text() or 0)
                        if fc > 0:
                            y_hp = qats.signal.highpass(ts_win.x, dt, fc)
                            tr = dict(
                                t=ts_win.t,
                                y=y_hp,
                                label=disp_label + f" [HP {fc} Hz]",
                                alpha=1.0,
                            )
                            traces.append(tr)
                            curves.append(
                                dict(t=ts_win.t, y=y_hp, label=f"HP {fc} Hz", alpha=1.0)
                            )
                    continue  # nothing else to do for time-domain loop
                elif mode == "rolling":
                    y_roll = pd.Series(ts_win.x).rolling(window=roll_window, min_periods=1).mean().to_numpy()
                    traces.append(
                        dict(
                            t=ts_win.t,
                            y=y_roll,
                            label=self._trim_label(f"{fname_disp}: {var}", left, right),
                            alpha=1.0,
                        )
                    )
                    continue

                # -----------------------------------------------------------------
                #  All other modes → call the corresponding TimeSeries.plot_* once
                # -----------------------------------------------------------------
                # inside the loop, after ts_win has been prepared
                if mode == "psd":
                    dt_arr = np.diff(ts_win.t)
                    if dt_arr.size:
                        dt = np.median(dt_arr)
                        if dt > 0:
                            var_ratio = np.max(np.abs(dt_arr - dt)) / dt
                            if var_ratio > 0.01:
                                t_r, x_r = self._resample(ts_win.t, ts_win.x, dt)
                                ts_win = TimeSeries(ts_win.name, t_r, x_r)
                    fig = ts_win.plot_psd(show=False)  # store=False is NOT valid
                elif mode == "cycle":
                    fig = ts_win.plot_cycle_range(show=False)
                elif mode == "cycle_rm":
                    fig = ts_win.plot_cycle_rangemean(show=False)
                elif mode == "cycle_rm3d":
                    # Matplotlib >= 3.7 removed the 'projection' keyword from
                    # Figure.gca().  Older versions of qats still call
                    # ``fig.gca(projection='3d')`` which raises a TypeError.
                    # To maintain compatibility, temporarily patch ``gca`` to
                    # support the projection argument if it's missing.
                    import inspect
                    import matplotlib.figure as mpl_fig

                    orig_gca = mpl_fig.Figure.gca
                    needs_patch = (
                        "projection" not in inspect.signature(orig_gca).parameters
                    )

                    def _gca_with_projection(self, *args, **kwargs):
                        if "projection" in kwargs:
                            proj = kwargs.pop("projection")
                            if not args:
                                args = (111,)
                            return self.add_subplot(*args, projection=proj, **kwargs)
                        return orig_gca(self, *args, **kwargs)

                    if needs_patch:
                        mpl_fig.Figure.gca = _gca_with_projection
                    try:
                        fig = ts_win.plot_cycle_rangemean3d(show=False)
                    finally:
                        if needs_patch:
                            mpl_fig.Figure.gca = orig_gca
                else:
                    QMessageBox.critical(self, "Unknown plot mode", mode)
                    return

                # NEW – recover the figure if the helper returned None
                if fig is None:
                    fig = plt.gcf()

                fig_per_file.setdefault(fname_disp, []).append(fig)

        # ======================================================================
        #  DISPLAY
        # ======================================================================
        import matplotlib.pyplot as plt  # make sure this import is at top

        if mode == "time" and grid:
            if not grid_traces:
                QMessageBox.warning(
                    self,
                    "Nothing to plot",
                    "No series matched the selection.",
                )
                return

            engine = (
                self.plot_engine_combo.currentText()
                if hasattr(self, "plot_engine_combo")
                else ""
            ).lower()

            n = len(grid_traces)
            ncols = int(np.ceil(np.sqrt(n)))
            nrows = int(np.ceil(n / ncols))

            same_axes = (
                hasattr(self, "plot_same_axes_cb")
                and self.plot_same_axes_cb.isChecked()
            )
            if same_axes:
                x_min = min(
                    min(c["t"]) for v in grid_traces.values() for c in v["curves"]
                )
                x_max = max(
                    max(c["t"]) for v in grid_traces.values() for c in v["curves"]
                )
                y_min = min(
                    np.min(c["y"]) for v in grid_traces.values() for c in v["curves"]
                )
                y_max = max(
                    np.max(c["y"]) for v in grid_traces.values() for c in v["curves"]
                )

            items = list(grid_traces.items())

            # ───────────────────────── 1.  Bokeh branch ──────────────────────────
            if engine == "bokeh":
                from bokeh.plotting import figure, show
                from bokeh.layouts import gridplot
                from bokeh.models import HoverTool, ColumnDataSource, Range1d
                from bokeh.palettes import Category10_10
                from bokeh.io import curdoc
                from bokeh.embed import file_html
                from bokeh.resources import INLINE
                import itertools, tempfile
                import numpy as np

                curdoc().theme = (
                    "dark_minimal" if self.theme_switch.isChecked() else "light_minimal"
                )

                figs = []
                color_cycle = itertools.cycle(Category10_10)
                for _, data in items:
                    lbl = data["label"]
                    curves = data["curves"]
                    p = figure(
                        width=450,
                        height=300,
                        title=lbl,
                        x_axis_label="Time",
                        y_axis_label=self.yaxis_label.text() or "Value",
                        tools="pan,wheel_zoom,box_zoom,reset,save",
                        sizing_mode="stretch_both",
                    )
                    if self.theme_switch.isChecked():
                        p.background_fill_color = "#2b2b2b"
                        p.border_fill_color = "#2b2b2b"
                    hover = HoverTool(
                        tooltips=[("Series", "@label"), ("Time", "@x"), ("Value", "@y")]
                    )
                    p.add_tools(hover)
                    for c in curves:
                        color = next(color_cycle)
                        cds = ColumnDataSource(
                            dict(x=c["t"], y=c["y"], label=[c["label"]] * len(c["t"]))
                        )
                        p.line(
                            "x",
                            "y",
                            source=cds,
                            line_alpha=c.get("alpha", 1.0),
                            color=color,
                            legend_label=c["label"],
                            muted_alpha=0.0,
                        )
                    if mark_extrema and curves:
                        all_t = np.concatenate([np.asarray(c["t"]) for c in curves])
                        all_y = np.concatenate([np.asarray(c["y"]) for c in curves])
                        max_idx = np.argmax(all_y)
                        min_idx = np.argmin(all_y)
                        p.circle([all_t[max_idx]], [all_y[max_idx]], size=6, color="red")
                        p.circle([all_t[min_idx]], [all_y[min_idx]], size=6, color="blue")
                    if same_axes:
                        p.x_range = Range1d(x_min, x_max)
                        p.y_range = Range1d(y_min, y_max)
                    p.legend.click_policy = "mute"
                    p.add_layout(p.legend[0], "right")
                    figs.append(p)

                layout = gridplot(figs, ncols=ncols, sizing_mode="stretch_both")
                if self.theme_switch.isChecked():
                    layout.background = "#2b2b2b"

                if getattr(self, "embed_plot_cb", None) and self.embed_plot_cb.isChecked():
                    html = file_html(layout, INLINE, "Time-series Grid", theme=curdoc().theme)
                    if self.theme_switch.isChecked():
                        html = html.replace(
                            "<body>",
                            "<body style=\"background-color:#2b2b2b;\">",
                        )
                    if self._temp_plot_file and os.path.exists(self._temp_plot_file):
                        try:
                            os.remove(self._temp_plot_file)
                        except Exception:
                            pass
                    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
                    with open(tmp.name, "w", encoding="utf-8") as fh:
                        fh.write(html)
                    self._temp_plot_file = tmp.name
                    self.plot_view.load(QUrl.fromLocalFile(tmp.name))
                    self.plot_view.show()
                else:
                    self.plot_view.hide()
                    show(layout)
                return

            # ───────────────────────── 2.  Plotly branch ─────────────────────────
            if engine == "plotly":
                import plotly.graph_objects as go
                from plotly.subplots import make_subplots
                from plotly.io import to_html
                import tempfile
                import numpy as np

                fig = make_subplots(
                    rows=nrows,
                    cols=ncols,
                    subplot_titles=[data["label"] for _, data in items],
                )
                for idx, (_, data) in enumerate(items, start=1):
                    curves = data["curves"]
                    r = (idx - 1) // ncols + 1
                    c = (idx - 1) % ncols + 1
                    for curve in curves:
                        fig.add_trace(
                            go.Scatter(
                                x=curve["t"],
                                y=curve["y"],
                                mode="lines",
                                name=curve["label"],
                                opacity=curve.get("alpha", 1.0),
                            ),
                            row=r,
                            col=c,
                        )
                    if mark_extrema and curves:
                        all_t = np.concatenate([np.asarray(curve["t"]) for curve in curves])
                        all_y = np.concatenate([np.asarray(curve["y"]) for curve in curves])
                        max_idx = np.argmax(all_y)
                        min_idx = np.argmin(all_y)
                        fig.add_trace(
                            go.Scatter(
                                x=[all_t[max_idx]],
                                y=[all_y[max_idx]],
                                mode="markers",
                                marker=dict(color="red", size=8),
                                showlegend=False,
                            ),
                            row=r,
                            col=c,
                        )
                        fig.add_trace(
                            go.Scatter(
                                x=[all_t[min_idx]],
                                y=[all_y[min_idx]],
                                mode="markers",
                                marker=dict(color="blue", size=8),
                                showlegend=False,
                            ),
                            row=r,
                            col=c,
                        )

                if same_axes:
                    fig.update_xaxes(range=[x_min, x_max])
                    fig.update_yaxes(range=[y_min, y_max])
                fig.update_layout(
                    title="Time-series Grid",
                    showlegend=True,
                    template="plotly_dark" if self.theme_switch.isChecked() else "plotly",
                )
                if self.theme_switch.isChecked():
                    fig.update_layout(
                        paper_bgcolor="#2b2b2b",
                        plot_bgcolor="#2b2b2b",
                    )

                if getattr(self, "embed_plot_cb", None) and self.embed_plot_cb.isChecked():
                    if self._temp_plot_file and os.path.exists(self._temp_plot_file):
                        try:
                            os.remove(self._temp_plot_file)
                        except Exception:
                            pass
                    html = to_html(fig, include_plotlyjs=True, full_html=True)
                    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
                    with open(tmp.name, "w", encoding="utf-8") as fh:
                        fh.write(html)
                    self._temp_plot_file = tmp.name
                    self.plot_view.load(QUrl.fromLocalFile(tmp.name))
                    self.plot_view.show()
                else:
                    self.plot_view.hide()
                    fig.show(renderer="browser")
                return

            # ───────────────────────── 3.  Matplotlib branch ─────────────────────
            import matplotlib.pyplot as plt
            import numpy as np
            fig, axes = plt.subplots(nrows, ncols, squeeze=False)
            for ax, (_, data) in zip(axes.flat, items):
                lbl = data["label"]
                curves = data["curves"]
                for c in curves:
                    ax.plot(c["t"], c["y"], alpha=c.get("alpha", 1.0), label=c["label"])
                if mark_extrema and curves:
                    all_t = np.concatenate([np.asarray(c["t"]) for c in curves])
                    all_y = np.concatenate([np.asarray(c["y"]) for c in curves])
                    max_idx = np.argmax(all_y)
                    min_idx = np.argmin(all_y)
                    ax.scatter(all_t[max_idx], all_y[max_idx], color="red", label="Max")
                    ax.scatter(all_t[min_idx], all_y[min_idx], color="blue", label="Min")
                ax.set_title(lbl)
                ax.legend()
                if same_axes:
                    ax.set_xlim(x_min, x_max)
                    ax.set_ylim(y_min, y_max)
            for ax in axes.flat[n:]:
                ax.set_visible(False)
            fig.suptitle("Time-series Grid")
            fig.tight_layout()

            if getattr(self, "embed_plot_cb", None) and self.embed_plot_cb.isChecked():
                if self._mpl_canvas is not None:
                    self.right_outer_layout.removeWidget(self._mpl_canvas)
                    self._mpl_canvas.setParent(None)
                from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

                self._mpl_canvas = FigureCanvasQTAgg(fig)
                self._mpl_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
                self.right_outer_layout.addWidget(self._mpl_canvas)
                self._mpl_canvas.show()
                self.plot_view.hide()
            else:
                if self._mpl_canvas is not None:
                    self.right_outer_layout.removeWidget(self._mpl_canvas)
                    self._mpl_canvas.setParent(None)
                    self._mpl_canvas = None
                self.plot_view.hide()
                fig.show()
            return
        if mode in ("time", "rolling"):
            if not traces:
                QMessageBox.warning(
                    self,
                    "Nothing to plot",
                    "No series matched the selection.",
                )
                return
            self._plot_lines(
                traces,
                title="Rolling Mean" if mode == "rolling" else "Time-series Plot",
                y_label=self.yaxis_label.text() or "Value",
                mark_extrema=mark_extrema,
            )
            return

        # ----------------------------------------------------------------------
        # All non-time modes (PSD / cycle-range / …)
        # ----------------------------------------------------------------------
        if not fig_per_file:
            QMessageBox.warning(
                self,
                "Nothing to plot",
                "No series matched the selection.",
            )
            return

        # --- show every collected Matplotlib figure ---
        for figs in fig_per_file.values():
            for fig in figs:
                if fig is None:  # QATS returned None → active fig
                    fig = plt.gcf()
                # Optional: give window a nicer title
                try:
                    fname = (
                        fig.canvas.get_window_title()
                    )  # may fail in headless back-ends
                    if "<Figure" in fname:
                        fig.canvas.manager.set_window_title("AnyTimeSeries plot")
                except Exception:
                    pass
                fig.show()

    @staticmethod
    def _resample(t, y, dt, *, start=None, stop=None):
        """Return ``(t_resampled, y_resampled)`` on a uniform grid.

        ``start`` and ``stop`` may be provided to explicitly set the limits of
        the resampled signal.  If omitted, the limits of ``t`` are used.  The
        function falls back to a NumPy-only implementation when ``qats`` is not
        available.
        """
        if start is None:
            start = t[0]
        if stop is None:
            stop = t[-1]
        if stop < start:
            start, stop = stop, start

        try:
            import anyqats as qats, numpy as _np

            try:
                # Preferred when available
                t_r, y_r = qats.signal.resample(y, t, dt)
                sel = (t_r >= start) & (t_r <= stop)
                t_r, y_r = t_r[sel], y_r[sel]
                if t_r.size == 0 or t_r[0] > start or t_r[-1] < stop:
                    raise ValueError
            except Exception:
                # Fallback to TimeSeries.resample or manual interpolation
                try:
                    ts_tmp = qats.TimeSeries("tmp", t, y)
                    y_r = ts_tmp.resample(dt=dt, t_min=start, t_max=stop)
                    t_r = _np.arange(start, stop + 0.5 * dt, dt)
                except Exception:
                    raise
            return t_r, y_r
        except Exception:
            import numpy as _np
            t_r = _np.arange(start, stop + 0.5 * dt, dt)
            y_r = _np.interp(t_r, t, y)
            return t_r, y_r


    def animate_xyz_scatter_many(self, *, dt_resample: float = 0.1):
        """
        Build an animated 3-D scatter for all (x,y,z) triplets found among the
        *checked* variables.

        Workflow
        --------
        1.  All checked keys are grouped per file.  “Common-tab” keys belong to
            every file.
        2.  Inside each file `_find_xyz_triples()` is used to discover unique
            (x,y,z) triplets.  If no perfect match is found the user is warned
            and that file is skipped.
        3.  Every component is filtered (according to the current GUI settings),
            resampled to **dt = 0.1 s** (default) and clipped to the active
            time-window.
        4.  The resulting DataFrame is fed to Plotly Express for an animated
            3-D scatter, one colour per triplet.
        """
        self.rebuild_var_lookup()
        import os, itertools, warnings
        import numpy as np
        import pandas as pd
        import plotly.express as px
        from PySide6.QtWidgets import QMessageBox as mb
        from anyqats import TimeSeries

        # ──────────────────────────────────────────────────────────────────
        # helper: filter + resample ONE series and return a *new* TimeSeries
        # ──────────────────────────────────────────────────────────────────
        def _prep(ts_src: TimeSeries, dt: float) -> TimeSeries:
            """filter → resample → wrap into fresh TimeSeries"""
            x_filt = self.apply_filters(ts_src)  # same length as original
            t_grid, x_res = self._resample(ts_src.t, x_filt, dt)
            return TimeSeries(f"{ts_src.name}_r{dt}", t_grid, x_res)

        # ──────────────────────────────────────────────────────────────────
        # 1) gather the checked keys for every file
        # ──────────────────────────────────────────────────────────────────
        per_file = {os.path.basename(fp): [] for fp in self.file_paths}

        for uk, chk in self.var_checkboxes.items():
            if not chk.isChecked():
                continue
            placed = False
            for fname in per_file:  # “File::<var>”?
                if uk.startswith(f"{fname}::"):
                    per_file[fname].append(uk.split("::", 1)[1])
                    placed = True
                    break
            if not placed:  # common / user tab
                for fname in per_file:
                    per_file[fname].append(uk)

        # ──────────────────────────────────────────────────────────────────
        # 2) for every file build DataFrame rows
        # ──────────────────────────────────────────────────────────────────
        rows = []
        skipped_any = False

        for fp, tsdb in zip(self.file_paths, self.tsdbs):
            fname = os.path.basename(fp)
            cand = list(dict.fromkeys(per_file[fname]))  # keep unique order
            if len(cand) < 3:
                continue

            triplets = _find_xyz_triples(cand)
            if not triplets:
                skipped_any = True
                continue

            tsdb_m = tsdb.getm()

            for tri in triplets:  # tri = (x_key, y_key, z_key)
                ts_x = tsdb_m.get(tri[0])
                ts_y = tsdb_m.get(tri[1])
                ts_z = tsdb_m.get(tri[2])
                if None in (ts_x, ts_y, ts_z):
                    continue

                # resample & filter
                ts_xr = _prep(ts_x, dt_resample)
                ts_yr = _prep(ts_y, dt_resample)
                ts_zr = _prep(ts_z, dt_resample)

                # common time-window mask (on the resampled grid!)
                mask = self.get_time_window(ts_xr)
                if isinstance(mask, slice):
                    t_win = ts_xr.t[mask]
                    x_val, y_val, z_val = ts_xr.x[mask], ts_yr.x[mask], ts_zr.x[mask]
                else:
                    if not mask.any():
                        continue
                    t_win = ts_xr.t[mask]
                    x_val, y_val, z_val = ts_xr.x[mask], ts_yr.x[mask], ts_zr.x[mask]

                # one “point” label = file name + compact triple for legend clarity
                base_lbl = "|".join(os.path.basename(v) for v in tri)
                rows.append(
                    pd.DataFrame(
                        dict(
                            time=t_win,
                            x=x_val,
                            y=y_val,
                            z=z_val,
                            point=f"{fname}:{base_lbl}",
                        )
                    )
                )

        if not rows:
            mb.warning(
                self,
                "No triplets",
                "Could not find any valid (x,y,z) triplets among the checked variables.",
            )
            return

        if skipped_any:
            mb.information(
                self,
                "Some files skipped",
                "One or more files yielded no unambiguous (x,y,z) triplet and were ignored.  See console output for details.",
            )

        df_all = pd.concat(rows, ignore_index=True)

        # ──────────────────────────────────────────────────────────────────
        # 3) Plotly Express animation
        # ──────────────────────────────────────────────────────────────────
        warnings.filterwarnings("ignore", category=FutureWarning)  # clean log

        fig = px.scatter_3d(
            df_all,
            x="x",
            y="y",
            z="z",
            color="point",
            animation_frame="time",
            animation_group="point",
            opacity=0.9,
            title="Animated 3-D Coordinate Scatter",
            template="plotly_dark" if self.theme_switch.isChecked() else "plotly",
        )
        fig.update_layout(
            scene_aspectmode="data",
            legend_title_text="Point / Triplet",
            margin=dict(l=0, r=0, t=30, b=0),
        )
        if self.theme_switch.isChecked():
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",

                plot_bgcolor="#2b2b2b",

            )

        try:
            if getattr(self, "embed_plot_cb", None) and self.embed_plot_cb.isChecked():
                from plotly.io import to_html

                import tempfile, os
                if self._temp_plot_file and os.path.exists(self._temp_plot_file):
                    try:
                        os.remove(self._temp_plot_file)
                    except Exception:
                        pass
                html = to_html(fig, include_plotlyjs=True, full_html=True)
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
                with open(tmp.name, "w", encoding="utf-8") as fh:
                    fh.write(html)
                self._temp_plot_file = tmp.name
                self.plot_view.load(QUrl.fromLocalFile(tmp.name))

                self.plot_view.show()
            else:
                self.plot_view.hide()
                # Ensure Plotly opens in the system browser when not embedding
                fig.show(renderer="browser")
        except Exception:
            # fallback: dump to temp HTML
            import tempfile, pathlib, webbrowser

            tmp = pathlib.Path(tempfile.gettempdir()) / "xyz_anim.html"
            fig.write_html(tmp)
            webbrowser.open(str(tmp))

    def _plot_lines(self, traces, title, y_label, *, mark_extrema=False):
        """
        traces → list of dicts with keys
                 't', 'y', 'label', 'alpha', 'is_mean'
        """
        engine = (
            self.plot_engine_combo.currentText()
            if hasattr(self, "plot_engine_combo")
            else ""
        ).lower()

        if engine != "default" and self._mpl_canvas is not None:
            self.right_outer_layout.removeWidget(self._mpl_canvas)
            self._mpl_canvas.setParent(None)
            self._mpl_canvas = None

        # ───────────────────────── 1.  Bokeh branch ──────────────────────────
        if engine == "bokeh":
            from bokeh.plotting import figure, show
            from bokeh.models import Button, CustomJS, ColumnDataSource, HoverTool
            from bokeh.layouts import column
            from bokeh.palettes import Category10_10
            from bokeh.embed import file_html
            from bokeh.resources import INLINE
            from bokeh.io import curdoc

            import itertools, tempfile

            curdoc().theme = (
                "dark_minimal" if self.theme_switch.isChecked() else "light_minimal"
            )


            p = figure(
                width=900,
                height=450,
                title=title,
                x_axis_label="Time",
                y_axis_label=y_label,
                tools="pan,wheel_zoom,box_zoom,reset,save",
                sizing_mode="stretch_both",
            )
            if self.theme_switch.isChecked():
                p.background_fill_color = "#2b2b2b"
                p.border_fill_color = "#2b2b2b"

            hover = HoverTool(
                tooltips=[("Series", "@label"), ("Time", "@x"), ("Value", "@y")]
            )
            p.add_tools(hover)

            renderers = []
            color_cycle = itertools.cycle(Category10_10)

            for tr in traces:
                color = next(color_cycle)
                cds = ColumnDataSource(
                    dict(x=tr["t"], y=tr["y"], label=[tr["label"]] * len(tr["t"]))
                )
                r = p.line(
                    "x",
                    "y",
                    source=cds,
                    line_width=2 if tr.get("is_mean") else 1,
                    line_alpha=tr.get("alpha", 1.0),
                    color=color,
                    legend_label=tr["label"],
                    muted_alpha=0.0,
                )
                renderers.append(r)

            if mark_extrema and traces:
                import numpy as np
                all_t = np.concatenate([np.asarray(tr["t"]) for tr in traces])
                all_y = np.concatenate([np.asarray(tr["y"]) for tr in traces])
                max_idx = np.argmax(all_y)
                min_idx = np.argmin(all_y)
                r_max = p.circle([all_t[max_idx]], [all_y[max_idx]], size=6, color="red", legend_label="Max")
                r_min = p.circle([all_t[min_idx]], [all_y[min_idx]], size=6, color="blue", legend_label="Min")
                renderers.extend([r_max, r_min])

            p.legend.click_policy = "mute"
            p.add_layout(p.legend[0], "right")

            btn = Button(label="Hide All Lines", width=150, button_type="success")
            btn.js_on_click(
                CustomJS(
                    args=dict(lines=renderers, button=btn),
                    code="""
                const hide = button.label === 'Hide All Lines';
                lines.forEach(r => r.muted = hide);
                button.label = hide ? 'Show All Lines' : 'Hide All Lines';
            """,
                )
            )
            layout = column(btn, p, sizing_mode="stretch_both")
            if self.theme_switch.isChecked():
                layout.background = "#2b2b2b"
            if getattr(self, "embed_plot_cb", None) and self.embed_plot_cb.isChecked():

                html = file_html(layout, INLINE, title, theme=curdoc().theme)

                if self.theme_switch.isChecked():
                    html = html.replace(
                        "<body>",
                        "<body style=\"background-color:#2b2b2b;\">",
                    )

                if self._temp_plot_file and os.path.exists(self._temp_plot_file):
                    try:
                        os.remove(self._temp_plot_file)
                    except Exception:
                        pass
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
                with open(tmp.name, "w", encoding="utf-8") as fh:
                    fh.write(html)
                self._temp_plot_file = tmp.name
                self.plot_view.load(QUrl.fromLocalFile(tmp.name))

                self.plot_view.show()
            else:
                self.plot_view.hide()
                show(layout)
            return

        # ───────────────────────── 2.  Plotly branch ─────────────────────────
        if engine == "plotly":
            import plotly.graph_objects as go

            fig = go.Figure()
            for tr in traces:
                fig.add_trace(
                    go.Scatter(
                        x=tr["t"],
                        y=tr["y"],
                        mode="lines",
                        name=tr["label"],
                        line=dict(width=2 if tr.get("is_mean") else 1),
                        opacity=tr.get("alpha", 1.0),
                    )
                )
            if mark_extrema and traces:
                import numpy as np
                all_t = np.concatenate([np.asarray(tr["t"]) for tr in traces])
                all_y = np.concatenate([np.asarray(tr["y"]) for tr in traces])
                max_idx = np.argmax(all_y)
                min_idx = np.argmin(all_y)
                fig.add_trace(
                    go.Scatter(
                        x=[all_t[max_idx]],
                        y=[all_y[max_idx]],
                        mode="markers",
                        marker=dict(color="red", size=8),
                        name="Max",
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=[all_t[min_idx]],
                        y=[all_y[min_idx]],
                        mode="markers",
                        marker=dict(color="blue", size=8),
                        name="Min",
                    )
                )
            fig.update_layout(
                title=title,
                xaxis_title="Time",
                yaxis_title=y_label,
                showlegend=True,
                template="plotly_dark" if self.theme_switch.isChecked() else "plotly",
            )
            if self.theme_switch.isChecked():
                fig.update_layout(
                    paper_bgcolor="#2b2b2b",
                    plot_bgcolor="#2b2b2b",
                    margin=dict(t=0, b=0, l = 0, r=0)
                )
            if getattr(self, "embed_plot_cb", None) and self.embed_plot_cb.isChecked():
                from plotly.io import to_html

                import tempfile
                if self._temp_plot_file and os.path.exists(self._temp_plot_file):
                    try:
                        os.remove(self._temp_plot_file)
                    except Exception:
                        pass
                html = to_html(fig, include_plotlyjs=True, full_html=True)
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
                with open(tmp.name, "w", encoding="utf-8") as fh:
                    fh.write(html)
                self._temp_plot_file = tmp.name
                self.plot_view.load(QUrl.fromLocalFile(tmp.name))

                self.plot_view.show()
            else:
                self.plot_view.hide()
                # Ensure Plotly opens in the system browser when not embedding
                fig.show(renderer="browser")
            return

        # ───────────────────────── 3.  Matplotlib fallback ────────────────────
        import matplotlib.pyplot as plt
        from itertools import cycle
        import numpy as np

        fig, ax = plt.subplots(figsize=(10, 5))
        palette = cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])
        for tr in traces:
            color = next(palette)
            ax.plot(
                tr["t"],
                tr["y"],
                label=tr["label"],
                linewidth=2 if tr.get("is_mean") else 1,
                alpha=tr.get("alpha", 1.0),
                color=color,
            )
        if mark_extrema and traces:
            all_t = np.concatenate([np.asarray(tr["t"]) for tr in traces])
            all_y = np.concatenate([np.asarray(tr["y"]) for tr in traces])
            max_idx = np.argmax(all_y)
            min_idx = np.argmin(all_y)
            ax.scatter(all_t[max_idx], all_y[max_idx], color="red", label="Max")
            ax.scatter(all_t[min_idx], all_y[min_idx], color="blue", label="Min")

        ax.set_title(title)
        ax.set_xlabel("Time")
        ax.set_ylabel(y_label)
        ax.legend(loc="best")
        fig.tight_layout()

        if getattr(self, "embed_plot_cb", None) and self.embed_plot_cb.isChecked():
            # Use a native Matplotlib canvas instead of the HTML viewer
            if self._mpl_canvas is not None:
                self.right_outer_layout.removeWidget(self._mpl_canvas)
                self._mpl_canvas.setParent(None)
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

            self._mpl_canvas = FigureCanvasQTAgg(fig)
            self._mpl_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            self.right_outer_layout.addWidget(self._mpl_canvas)

            self._mpl_canvas.show()
            self.plot_view.hide()
        else:
            if self._mpl_canvas is not None:
                self.right_outer_layout.removeWidget(self._mpl_canvas)
                self._mpl_canvas.setParent(None)
                self._mpl_canvas = None
            self.plot_view.hide()
            plt.show()

    def plot_mean(self):
        self.rebuild_var_lookup()
        import numpy as np

        traces = []
        sel = [k for k, v in self.var_checkboxes.items() if v.isChecked()]
        if not sel:
            QMessageBox.warning(self, "Nothing selected", "Select variables first.")
            return

        from collections import Counter

        fname_counts = Counter(os.path.basename(p) for p in self.file_paths)

        common_t = None
        stacks = []

        for unique_key in sel:
            ts, fname, disp = None, None, None
            # resolve exactly as in plot_selected:
            for file_idx, (tsdb, fp) in enumerate(
                zip(self.tsdbs, self.file_paths), start=1
            ):
                fname_ = os.path.basename(fp)
                if unique_key.startswith(f"{fname_}::"):
                    real = unique_key.split("::", 1)[1]
                    ts = tsdb.getm().get(real)
                    fname = fname_
                    disp = real
                elif unique_key.startswith(f"{fname_}:"):
                    real = unique_key.split(":", 1)[1]
                    ts = tsdb.getm().get(real)
                    fname = fname_
                    disp = real
                elif unique_key in tsdb.getm():
                    ts = tsdb.getm()[unique_key]
                    fname = fname_
                    disp = unique_key
                if ts:
                    break
            if not ts:
                continue

            fname_disp = fname if fname_counts[fname] == 1 else f"{fname} ({file_idx})"

            m = self.get_time_window(ts)
            t, y = ts.t[m], self.apply_filters(ts)[m]
            if common_t is None:
                common_t = t
            elif not np.array_equal(t, common_t):
                y = qats.TimeSeries("", t, y).resample(t=common_t)
            stacks.append(y)
            if self.include_raw_mean_cb.isChecked():
                traces.append(
                    dict(
                        t=common_t,
                        y=y,
                        label=f"{fname_disp}: {disp}",
                        alpha=0.4,
                    )
                )

        if not stacks:
            QMessageBox.warning(self, "Nothing to plot", "No valid data.")
            return

        mean_y = np.nanmean(np.vstack(stacks), axis=0)
        traces.append(dict(t=common_t, y=mean_y, label="Mean", is_mean=True))

        self._plot_lines(
            traces, "Mean of Selected Series", self.yaxis_label.text() or "Value"
        )

    def get_time_window(self, ts):
        """Return a boolean mask or slice for the user-specified time window."""
        t = ts.t
        if t.size == 0:
            return np.zeros(0, dtype=bool)

        def _safe_float(txt, default):
            try:
                return float(txt.strip()) if txt.strip() else default
            except Exception:
                return default

        tmin = _safe_float(self.time_start.text(), t[0])
        tmax = _safe_float(self.time_end.text(), t[-1])
        if tmax < tmin:
            tmin, tmax = tmax, tmin

        i0 = np.searchsorted(t, tmin, side="left")
        i1 = np.searchsorted(t, tmax, side="right")
        if i0 == 0 and i1 == len(t):
            return slice(None)
        if np.all(np.diff(t[i0:i1]) > 0):
            return slice(i0, i1)
        return (t >= tmin) & (t <= tmax)

    def apply_filters(self, ts):
        """Apply frequency filters according to the current settings."""
        mode = "none"
        if self.filter_lowpass_rb.isChecked():
            mode = "lowpass"
        elif self.filter_highpass_rb.isChecked():
            mode = "highpass"
        elif self.filter_bandpass_rb.isChecked():
            mode = "bandpass"
        elif self.filter_bandblock_rb.isChecked():
            mode = "bandblock"

        x = ts.x.copy()
        t = ts.t
        nanmask = ~np.isnan(x)
        if not np.any(nanmask):
            return x
        valid_idx = np.where(nanmask)[0]
        x_valid = x[valid_idx]
        t_valid = t[valid_idx]
        x_filt = x_valid
        try:
            dt = np.median(np.diff(t_valid))
            if mode == "lowpass":
                fc = float(self.lowpass_cutoff.text() or 0)
                if fc > 0 and len(x_valid) > 1:
                    x_filt = qats.signal.lowpass(x_valid, dt, fc)
            elif mode == "highpass":
                fc = float(self.highpass_cutoff.text() or 0)
                if fc > 0 and len(x_valid) > 1:
                    x_filt = qats.signal.highpass(x_valid, dt, fc)
            elif mode == "bandpass":
                flow = float(self.bandpass_low.text() or 0)
                fupp = float(self.bandpass_high.text() or 0)
                if flow > 0 and fupp > flow and len(x_valid) > 1:
                    x_filt = qats.signal.bandpass(x_valid, dt, flow, fupp)
            elif mode == "bandblock":
                flow = float(self.bandblock_low.text() or 0)
                fupp = float(self.bandblock_high.text() or 0)
                if flow > 0 and fupp > flow and len(x_valid) > 1:
                    x_filt = qats.signal.bandblock(x_valid, dt, flow, fupp)
        except Exception:
            pass

        x_out = np.full_like(x, np.nan)
        x_out[valid_idx] = x_filt
        return x_out

    def _filter_tag(self) -> str:
        """Return short text tag describing the active frequency filter."""
        if self.filter_lowpass_rb.isChecked():
            val = self.lowpass_cutoff.text().strip()
            return f"LF{val.replace('.', '_')}" if val else ""
        if self.filter_highpass_rb.isChecked():
            val = self.highpass_cutoff.text().strip()
            return f"HF{val.replace('.', '_')}" if val else ""
        if self.filter_bandpass_rb.isChecked():
            low = self.bandpass_low.text().strip()
            high = self.bandpass_high.text().strip()
            if low and high:
                return f"BAND_{low.replace('.', '_')}to{high.replace('.', '_')}"
        if self.filter_bandblock_rb.isChecked():
            low = self.bandblock_low.text().strip()
            high = self.bandblock_high.text().strip()
            if low and high:
                return f"BLOCK_{low.replace('.', '_')}to{high.replace('.', '_')}"
        return ""

    def _gather_entry_values(self):
        values = {}
        for key, entry in self.var_offsets.items():
            try:
                val = float(entry.text())
                if val != 0.0:
                    values[key] = val
            except ValueError:
                continue
        return values

    def save_files(self):
        if not getattr(self, "work_dir", None):
            self.work_dir = QFileDialog.getExistingDirectory(self, "Select Folder to Save .ts Files")
            if not self.work_dir:
                return
        for tsdb, path in zip(self.tsdbs, self.file_paths):
            name = os.path.splitext(os.path.basename(path))[0] + ".ts"
            save_path = os.path.join(self.work_dir, name)
            tsdb.export(save_path, names=list(tsdb.getm().keys()), force_common_time=True)
        QMessageBox.information(self, "Saved", "Files exported.")

    def save_entry_values(self):
        data = self._gather_entry_values()
        if not data:
            QMessageBox.information(self, "Nothing to save", "All entry-boxes are zero – nothing to save.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save entry values", "", "JSON files (*.json)")
        if not path:
            return
        with open(path, "w", encoding="utf-8") as fp:
            json.dump(data, fp, indent=2)
        QMessageBox.information(self, "Saved", f"Saved {len(data)} value(s) to\n{os.path.basename(path)}")

    def load_entry_values(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "Load entry values", "", "JSON files (*.json)")
        if not paths:
            return
        applied = 0
        skipped = 0
        for path in paths:
            try:
                with open(path, "r", encoding="utf-8") as fp:
                    data = json.load(fp)
            except Exception as e:
                QMessageBox.warning(self, "Load error", f"Could not read {os.path.basename(path)}:\n{e}")
                continue
            for key, val in data.items():
                targets = []
                if key in self.var_offsets:
                    targets = [key]
                else:
                    for k in self.var_offsets:
                        if k.endswith(f"::{key}") or k.endswith(f":{key}") or k == key:
                            targets.append(k)
                if not targets:
                    skipped += 1
                    continue
                for tkey in targets:
                    self.var_offsets[tkey].setText(str(val))
                    applied += 1
        QMessageBox.information(self, "Loaded", f"Applied {applied} value(s) (skipped {skipped}).")

    def export_selected_to_csv(self):
        """Export all checked variables to a single CSV file."""
        self.rebuild_var_lookup()
        sel_keys = [k for k, ck in self.var_checkboxes.items() if ck.isChecked()]
        if not sel_keys:
            QMessageBox.warning(self, "No selection", "Select variables to export.")
            return

        try:
            dt = float(self.export_dt_input.text())
        except ValueError:
            dt = 0.0


        def _parse_f(txt):
            try:
                return float(txt.strip()) if txt.strip() else None
            except Exception:
                return None

        t_start = _parse_f(self.time_start.text())
        t_stop = _parse_f(self.time_end.text())
        if t_start is not None and t_stop is not None and t_stop < t_start:
            t_start, t_stop = t_stop, t_start


        path, _ = QFileDialog.getSaveFileName(self, "Export selected to CSV", "", "CSV files (*.csv)")
        if not path:
            return

        series_list = []
        for tsdb, fp in zip(self.tsdbs, self.file_paths):
            fname = os.path.basename(fp)
            tsdb_map = tsdb.getm()
            for key in sel_keys:
                if key.startswith(f"{fname}::"):
                    var = key.split("::", 1)[1]
                elif key.startswith(f"{fname}:"):
                    var = key.split(":", 1)[1]
                elif key in tsdb_map:
                    var = key
                else:
                    continue
                ts = tsdb_map.get(var)
                if ts is None:
                    continue
                mask = self.get_time_window(ts)
                if isinstance(mask, slice):
                    t = ts.t[mask]
                    y = self.apply_filters(ts)[mask]
                else:
                    if not mask.any():
                        continue
                    t = ts.t[mask]
                    y = self.apply_filters(ts)[mask]

                if dt > 0:

                    start = t_start if t_start is not None else t[0]
                    stop = t_stop if t_stop is not None else t[-1]
                    t, y = self._resample(t, y, dt, start=start, stop=stop)


                series_list.append(pd.Series(t, name=f"{key}_t"))
                series_list.append(pd.Series(y, name=key))

        if not series_list:
            QMessageBox.warning(self, "No data", "No data found for the selected variables.")
            return

        df = pd.concat(series_list, axis=1)
        df.to_csv(path, index=False)
        QMessageBox.information(self, "Exported", f"Exported {len(sel_keys)} series to\n{os.path.basename(path)}")

    def launch_qats(self):
        if not getattr(self, "work_dir", None):
            self.work_dir = QFileDialog.getExistingDirectory(self, "Select Work Folder for AnyQATS Export")
            if not self.work_dir:
                return
        ts_paths = []
        for i, (tsdb, original_path) in enumerate(zip(self.tsdbs, self.file_paths)):
            filename = f"temp_{i + 1}.ts"
            ts_path = os.path.join(self.work_dir, filename)
            tsdb.export(ts_path, names=list(tsdb.getm().keys()), force_common_time=True)
            ts_paths.append(ts_path)
        try:
            cmd = [sys.executable, "-m", "anyqats.cli", "app", "-f"] + ts_paths
            subprocess.Popen(cmd)
        except FileNotFoundError:
            QMessageBox.critical(self, "Error", "AnyQATS could not be launched using the current Python environment.")

    def open_evm_tool(self):
        """Launch the Extreme Value Analysis tool for the first checked variable."""

        # Build list of checked variable keys using the unique lookup table
        self.rebuild_var_lookup()
        selected_keys = [k for k, cb in self.var_checkboxes.items() if cb.isChecked()]

        if not selected_keys:
            QMessageBox.warning(self, "No Variables", "Please select at least one variable.")
            return

        if len(selected_keys) > 1:
            QMessageBox.information(self, "Multiple Variables", "Only the first selected variable will be used for EVA.")

        selected = selected_keys[0]

        index = None
        raw_key = selected

        for i, fp in enumerate(self.file_paths):
            fname = os.path.basename(fp)
            if selected.startswith(f"{fname}::"):
                raw_key = selected.split("::", 1)[1]
                index = i
                break
            if selected.startswith(f"{fname}:"):
                raw_key = selected.split(":", 1)[1]
                index = i
                break

        if index is None:
            if ":" in selected or "::" in selected:
                QMessageBox.critical(self, "EVA Error", f"Could not locate the file for: {selected}")
                return
            index = 0

        if index >= len(self.tsdbs):
            QMessageBox.critical(self, "EVA Error", f"Could not locate the file for: {selected}")
            return

        tsdb = self.tsdbs[index]
        ts = tsdb.getm().get(raw_key)
        if ts is None:
            QMessageBox.critical(self, "EVA Error", f"Variable not found in file:\n{raw_key}")
            return
        mask = self.get_time_window(ts)
        if mask is not None and np.any(mask):
            x = self.apply_filters(ts)[mask]
            t = ts.t[mask]
            ts_for_evm = TimeSeries(ts.name, t, x)
            local_db = TsDB()
            local_db.add(ts_for_evm)
        else:
            local_db = tsdb
        dlg = EVMWindow(local_db, ts.name, self)
        dlg.exec()

    def apply_dark_palette(self):

        app = QApplication.instance()
        # Reuse the stored Fusion style to avoid Qt owning temporary objects
        app.setStyle(self._fusion_style)
        # Apply to this window as well so existing widgets refresh
        self.setStyle(self._fusion_style)


        dark_palette = QPalette()
        dark_palette.setColor(QPalette.Window, QColor("#31363b"))
        dark_palette.setColor(QPalette.WindowText, QColor("#eff0f1"))
        dark_palette.setColor(QPalette.Base, QColor("#232629"))
        dark_palette.setColor(QPalette.AlternateBase, QColor("#31363b"))
        dark_palette.setColor(QPalette.ToolTipBase, QColor("#eff0f1"))
        dark_palette.setColor(QPalette.ToolTipText, QColor("#31363b"))
        dark_palette.setColor(QPalette.Text, QColor("#eff0f1"))
        dark_palette.setColor(QPalette.Button, QColor("#31363b"))
        dark_palette.setColor(QPalette.ButtonText, QColor("#eff0f1"))
        dark_palette.setColor(QPalette.BrightText, Qt.white)
        dark_palette.setColor(QPalette.Link, QColor("#3daee9"))
        dark_palette.setColor(QPalette.Highlight, QColor("#3daee9"))
        dark_palette.setColor(QPalette.HighlightedText, QColor("#31363b"))



        app.setPalette(dark_palette)
        self.setPalette(dark_palette)
        app.setStyleSheet(
            "QToolTip { color: #31363b; background-color: #3daee9; border: 1px solid #31363b; }"
        )
        
        import matplotlib.pyplot as plt
        plt.style.use("dark_background")

        # Keep the embedded Plotly background dark to avoid a light border
        self.plot_view.page().setBackgroundColor(QColor("#31363b"))
        self.plot_view.setStyleSheet("background-color:#31363b;border:0px;")

    def apply_light_palette(self):
        app = QApplication.instance()
        if app is None:  # safety net
            raise RuntimeError("No QApplication running")

        app.setStyle(self._fusion_style)
        self.setStyle(self._fusion_style)


        light_palette = QPalette()
        light_palette.setColor(QPalette.Window, QColor("#eff0f1"))
        light_palette.setColor(QPalette.WindowText, QColor("#31363b"))
        light_palette.setColor(QPalette.Base, QColor("#fcfcfc"))
        light_palette.setColor(QPalette.AlternateBase, QColor("#e5e5e5"))
        light_palette.setColor(QPalette.ToolTipBase, QColor("#31363b"))
        light_palette.setColor(QPalette.ToolTipText, QColor("#eff0f1"))
        light_palette.setColor(QPalette.Text, QColor("#31363b"))
        light_palette.setColor(QPalette.Button, QColor("#e5e5e5"))
        light_palette.setColor(QPalette.ButtonText, QColor("#31363b"))
        light_palette.setColor(QPalette.BrightText, Qt.white)
        light_palette.setColor(QPalette.Link, QColor("#2a82da"))
        light_palette.setColor(QPalette.Highlight, QColor("#2a82da"))
        light_palette.setColor(QPalette.HighlightedText, QColor("#ffffff"))

        app.setPalette(light_palette)
        self.setPalette(light_palette)
        app.setStyleSheet(
            "QToolTip { color: #31363b; background-color: #2a82da; border: 1px solid #31363b; }"
        )

        # And if you also use matplotlib in the same process:
        import matplotlib.pyplot as plt
        plt.style.use("default")

        # Restore the web view background for the light theme
        self.plot_view.page().setBackgroundColor(QColor("#eff0f1"))
        self.plot_view.setStyleSheet("background-color:#eff0f1;border:0px;")


    def toggle_dark_theme(self, state):

        # ``state`` comes from the checkbox signal but using ``isChecked`` is
        # more robust across Qt bindings.
        if self.theme_switch.isChecked():
            self.apply_dark_palette()
        else:
            self.apply_light_palette()
        # Refresh any open Matplotlib canvases so the new palette is used
        for canvas in self.findChildren(FigureCanvasQTAgg):
            canvas.draw()

    def _on_engine_changed(self, text):
        """Update layout when the plotting engine selection changes."""
        engine = text.lower()
        if engine != "default" and self._mpl_canvas is not None:
            self.right_outer_layout.removeWidget(self._mpl_canvas)
            self._mpl_canvas.setParent(None)
            self._mpl_canvas = None
        if self.embed_plot_cb.isChecked():
            # Refresh layout so the appropriate widget is shown
            self.toggle_embed_layout(True)

    def toggle_embed_layout(self, state):
        """Re-arrange layout when the embed checkbox is toggled."""
        checked = self.embed_plot_cb.isChecked()

        # Widgets that are moved between the main controls column and the
        # additional column when the plot is embedded.
        extra_groups = [
            self.calc_group,
            self.freq_group,
            self.tools_group,
        ]

        if checked:
            if self.extra_widget.parent() is None:
                self.top_row_layout.addWidget(self.extra_widget)

            if self.progress.parent() is self.controls_widget:
                self.controls_layout.removeWidget(self.progress)
            if self.progress.parent() is self.progress_transform_row:
                self.progress_transform_row.removeWidget(self.progress)
            if self.file_ctrls_layout.indexOf(self.progress) == -1:

                idx = self.file_ctrls_layout.indexOf(self.theme_embed_widget)
                if idx == -1:
                    self.file_ctrls_layout.addWidget(self.progress)
                else:
                    self.file_ctrls_layout.insertWidget(idx, self.progress)


            if self.transform_group.parent() is self.controls_widget:
                self.controls_layout.removeWidget(self.transform_group)
            if self.progress_transform_row.indexOf(self.transform_group) == -1:
                self.progress_transform_row.addWidget(self.transform_group)
            if self.extra_layout.indexOf(self.progress_transform_row) == -1:
                self.extra_layout.insertLayout(0, self.progress_transform_row)

            if self.extra_layout.indexOf(self.extra_stretch) != -1:
                self.extra_layout.removeItem(self.extra_stretch)

            idx_freq = self.controls_layout.indexOf(self.freq_group)
            idx_tools = self.controls_layout.indexOf(self.tools_group)

            for g in extra_groups:
                if g.parent() is self.controls_widget:
                    self.controls_layout.removeWidget(g)
                    self.extra_layout.addWidget(g)

            if self.analysis_group.parent() is self.extra_widget:
                self.extra_layout.removeWidget(self.analysis_group)
            if idx_freq == -1:
                idx_freq = self.controls_layout.count()
            self.controls_layout.insertWidget(idx_freq, self.analysis_group)

            if self.plot_group.parent() is self.extra_widget:
                self.extra_layout.removeWidget(self.plot_group)
            if idx_tools == -1:
                idx_tools = self.controls_layout.count()
            self.controls_layout.insertWidget(idx_tools, self.plot_group)




            self.extra_layout.addItem(self.extra_stretch)
            if self.plot_engine_combo.currentText().lower() == "default" and self._mpl_canvas is not None:
                self._mpl_canvas.show()
                self.plot_view.hide()
            else:
                self.plot_view.show()
                if self._mpl_canvas is not None:
                    self._mpl_canvas.hide()
        else:
            self.plot_view.hide()
            if self._mpl_canvas is not None:
                self._mpl_canvas.hide()
            if self.plot_view.parent() is self.extra_widget:
                self.extra_layout.removeWidget(self.plot_view)
                self.right_outer_layout.addWidget(self.plot_view)
            if self.extra_widget.parent() is not None:
                self.top_row_layout.removeWidget(self.extra_widget)
                self.extra_widget.setParent(None)


            if self.extra_layout.indexOf(self.progress_transform_row) != -1:
                self.extra_layout.removeItem(self.progress_transform_row)
            if self.progress_transform_row.indexOf(self.transform_group) != -1:
                self.progress_transform_row.removeWidget(self.transform_group)

            if self.file_ctrls_layout.indexOf(self.progress) != -1:
                self.file_ctrls_layout.removeWidget(self.progress)
            self.controls_layout.insertWidget(1, self.progress)

            for g in [self.freq_group, self.tools_group, self.transform_group, self.calc_group]:
                if g.parent() is self.extra_widget:
                    self.extra_layout.removeWidget(g)
                    g.setParent(self.controls_widget)
                    self.controls_layout.addWidget(g)

            if self.controls_layout.indexOf(self.analysis_group) != -1:
                self.controls_layout.removeWidget(self.analysis_group)
            self.controls_layout.addWidget(self.analysis_group)

            if self.controls_layout.indexOf(self.plot_group) != -1:
                self.controls_layout.removeWidget(self.plot_group)
            self.controls_layout.addWidget(self.plot_group)


            if self.extra_layout.indexOf(self.extra_stretch) != -1:
                self.extra_layout.removeItem(self.extra_stretch)
            self.extra_layout.addItem(self.extra_stretch)

class VariableRowWidget(QWidget):
    rename_requested = Signal(str, str)

    def __init__(self, varname, allow_rename=False, parent=None):
        super().__init__(parent)
        self._name = varname

        layout = QHBoxLayout(self)
        self.checkbox = QCheckBox()
        self.input = QLineEdit()
        self.input.setFixedWidth(70)
        self.label = QLabel(varname)

        layout.addWidget(self.checkbox)
        layout.addWidget(self.input)
        layout.addWidget(self.label)

        if allow_rename:
            self.rename_btn = QPushButton("Rename")
            self.rename_btn.setFixedWidth(70)
            self.rename_btn.clicked.connect(self._prompt_rename)
            layout.addWidget(self.rename_btn)

        layout.addStretch(1)
        layout.setContentsMargins(2, 2, 2, 2)
        self.setLayout(layout)

    def _prompt_rename(self):
        new_name, ok = QInputDialog.getText(
            self,
            "Rename Variable",
            "New variable name",
            text=self._name,
        )
        if ok and new_name and new_name != self._name:
            self.rename_requested.emit(self._name, new_name)

class VariableTab(QWidget):
    """VariableTab with search and select-all functionality."""
    checklist_updated = Signal()

    def __init__(self, label, variables, user_var_set=None, allow_rename=False, rename_callback=None):
        super().__init__()
        self.all_vars = sorted(list(variables))
        self.user_var_set = user_var_set or set()
        self.allow_rename = allow_rename
        self.rename_callback = rename_callback
        self.checkboxes = {}
        self.inputs = {}
        self._checked_state = {}
        self._input_state = {}
        layout = QVBoxLayout(self)
        # -- Search and Select All row --
        top_row = QHBoxLayout()
        self.search_box = QLineEdit()
        self.search_box.setPlaceholderText("Search…")
        top_row.addWidget(self.search_box)
        self.select_all_btn = QPushButton("Select All")
        self.unselect_all_btn = QPushButton("Unselect All")
        top_row.addWidget(self.select_all_btn)
        top_row.addWidget(self.unselect_all_btn)
        layout.addLayout(top_row)
        # -- Scrollable area for variable checkboxes --
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self.inner = QWidget()
        self.inner_layout = QVBoxLayout(self.inner)
        self._populate_checkboxes(self.all_vars)
        scroll.setWidget(self.inner)
        layout.addWidget(scroll)
        # Connections
        self.select_all_btn.clicked.connect(lambda: self.set_all(True))
        self.unselect_all_btn.clicked.connect(lambda: self.set_all(False))
        self.search_box.textChanged.connect(self._search_update)
    def _populate_checkboxes(self, vars_to_show):
        """Populate the scroll area with VariableRowWidget entries."""
        # Preserve existing states
        for var, cb in self.checkboxes.items():
            self._checked_state[var] = cb.isChecked()
            self._input_state[var] = self.inputs[var].text()

        # Clear layout
        for i in reversed(range(self.inner_layout.count())):
            widget = self.inner_layout.takeAt(i).widget()
            if widget:
                widget.deleteLater()
        self.checkboxes.clear()
        self.inputs.clear()

        for var in vars_to_show:
            row = VariableRowWidget(var, allow_rename=self.allow_rename)
            if var in self.user_var_set:
                row.label.setStyleSheet("color: #2277bb;")
            if self.allow_rename and self.rename_callback:
                row.rename_requested.connect(self.rename_callback)
            self.inner_layout.addWidget(row)
            self.checkboxes[var] = row.checkbox
            self.inputs[var] = row.input
            if var in self._checked_state:
                row.checkbox.setChecked(self._checked_state[var])
            if var in self._input_state:
                row.input.setText(self._input_state[var])

        self.inner_layout.addStretch()
        self.checklist_updated.emit()
    def _search_update(self, text):
        terms = _parse_search_terms(text)
        if not terms:
            vars_to_show = self.all_vars
        else:
            vars_to_show = [v for v in self.all_vars if _matches_terms(v, terms)]
        self._populate_checkboxes(vars_to_show)
    def selected_variables(self):
        return [var for var, cb in self.checkboxes.items() if cb.isChecked()]
    def set_all(self, value):
        for cb in self.checkboxes.values():
            cb.setChecked(value)

class StatsDialog(QDialog):
    """Qt table dialog with copy and plotting features."""

    def __init__(self, series_info, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Statistics Table")
        self.setWindowFlag(Qt.Window)
        # allow maximizing the statistics window
        self.setWindowFlags(self.windowFlags() | Qt.WindowMaximizeButtonHint)
        self.resize(900, 600)

        self.series_info = series_info
        self.ts_dict: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.selected_columns: set[int] = set()

        main_layout = QVBoxLayout(self)

        order_layout = QHBoxLayout()
        order_layout.addWidget(QLabel("Load order:"))
        self.order_combo = QComboBox()
        self.order_combo.addItems(["Files → Variables", "Variables → Files"])
        order_layout.addWidget(self.order_combo)
        order_layout.addStretch()
        main_layout.addLayout(order_layout)

        # Frequency filter controls
        freq_group = QGroupBox("Frequency Filter")
        freq_layout = QGridLayout(freq_group)
        self.filter_none_rb = QRadioButton("None")
        self.filter_lowpass_rb = QRadioButton("Low-pass")
        self.filter_highpass_rb = QRadioButton("High-pass")
        self.filter_bandpass_rb = QRadioButton("Band-pass")
        self.filter_bandblock_rb = QRadioButton("Band-block")
        self.filter_none_rb.setChecked(True)
        self.lowpass_cutoff = QLineEdit("0.01")
        self.highpass_cutoff = QLineEdit("0.1")
        self.bandpass_low = QLineEdit("0.0")
        self.bandpass_high = QLineEdit("0.0")
        self.bandblock_low = QLineEdit("0.0")
        self.bandblock_high = QLineEdit("0.0")
        row = 0
        freq_layout.addWidget(self.filter_none_rb, row, 0, 1, 2)
        row += 1
        freq_layout.addWidget(self.filter_lowpass_rb, row, 0)
        freq_layout.addWidget(QLabel("below"), row, 1)
        freq_layout.addWidget(self.lowpass_cutoff, row, 2)
        freq_layout.addWidget(QLabel("Hz"), row, 3)
        row += 1
        freq_layout.addWidget(self.filter_highpass_rb, row, 0)
        freq_layout.addWidget(QLabel("above"), row, 1)
        freq_layout.addWidget(self.highpass_cutoff, row, 2)
        freq_layout.addWidget(QLabel("Hz"), row, 3)
        row += 1
        freq_layout.addWidget(self.filter_bandpass_rb, row, 0)
        freq_layout.addWidget(QLabel("between"), row, 1)
        freq_layout.addWidget(self.bandpass_low, row, 2)
        freq_layout.addWidget(QLabel("Hz and"), row, 3)
        freq_layout.addWidget(self.bandpass_high, row, 4)
        freq_layout.addWidget(QLabel("Hz"), row, 5)
        row += 1
        freq_layout.addWidget(self.filter_bandblock_rb, row, 0)
        freq_layout.addWidget(QLabel("between"), row, 1)
        freq_layout.addWidget(self.bandblock_low, row, 2)
        freq_layout.addWidget(QLabel("Hz and"), row, 3)
        freq_layout.addWidget(self.bandblock_high, row, 4)
        freq_layout.addWidget(QLabel("Hz"), row, 5)
        main_layout.addWidget(freq_group)

        hline_layout = QHBoxLayout()
        hline_layout.addWidget(QLabel("Histogram lines:"))
        self.hist_lines_edit = QLineEdit()
        self.hist_lines_edit.setPlaceholderText("e.g. 1.0, 2.5")
        hline_layout.addWidget(self.hist_lines_edit)
        self.hist_show_text_cb = QCheckBox("Show bar text")
        self.hist_show_text_cb.setChecked(True)
        hline_layout.addWidget(self.hist_show_text_cb)
        hline_layout.addStretch()
        main_layout.addLayout(hline_layout)

        self.table = QTableWidget()
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        # Sorting is enabled, but will be temporarily disabled while
        # populating the table to avoid row mixing
        self.table.setSortingEnabled(True)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Interactive)
        header.setStretchLastSection(True)
        header.setContextMenuPolicy(Qt.CustomContextMenu)
        header.customContextMenuRequested.connect(self._header_right_click)
        main_layout.addWidget(self.table, stretch=2)

        plot_layout = QVBoxLayout()
        self.line_fig = Figure(figsize=(5, 3))
        self.line_canvas = FigureCanvasQTAgg(self.line_fig)
        self.psd_fig = Figure(figsize=(5, 3))
        self.psd_canvas = FigureCanvasQTAgg(self.psd_fig)

        for canvas in (self.line_canvas, self.psd_canvas):
            canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        ts_layout = QHBoxLayout()
        ts_layout.addWidget(self.line_canvas)
        ts_layout.addWidget(self.psd_canvas)
        hist_layout = QHBoxLayout()
        self.hist_fig_rows = Figure(figsize=(4, 3))
        self.hist_canvas_rows = FigureCanvasQTAgg(self.hist_fig_rows)
        self.hist_fig_cols = Figure(figsize=(4, 3))
        self.hist_canvas_cols = FigureCanvasQTAgg(self.hist_fig_cols)

        for canvas in (self.hist_canvas_rows, self.hist_canvas_cols):
            canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        plot_layout.addLayout(ts_layout)
        hist_layout.addWidget(self.hist_canvas_rows)
        hist_layout.addWidget(self.hist_canvas_cols)
        plot_layout.addLayout(hist_layout)
        main_layout.addLayout(plot_layout, stretch=3)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self.copy_btn = QPushButton("Copy as TSV")
        self.copy_btn.clicked.connect(self.copy_as_tsv)
        btn_row.addWidget(self.copy_btn)
        main_layout.addLayout(btn_row)


        self._connect_signals()
        self.update_data()

    def showEvent(self, event: QEvent) -> None:
        super().showEvent(event)
        QTimer.singleShot(0, self._refresh_plots)

    def _refresh_plots(self) -> None:
        """Process pending events before drawing initial plots."""
        QApplication.processEvents()
        self.update_plots()

    def _connect_signals(self):
        for w in [self.filter_none_rb, self.filter_lowpass_rb, self.filter_highpass_rb,
                  self.filter_bandpass_rb, self.filter_bandblock_rb]:
            w.toggled.connect(self.update_data)
        for e in [self.lowpass_cutoff, self.highpass_cutoff,

                   self.bandpass_low, self.bandpass_high,
                   self.bandblock_low, self.bandblock_high]:
            e.editingFinished.connect(self.update_data)
        self.hist_lines_edit.editingFinished.connect(self.update_plots)
        self.hist_show_text_cb.toggled.connect(self.update_plots)
        self.order_combo.currentIndexChanged.connect(self.update_data)
        self.table.selectionModel().selectionChanged.connect(self.update_plots)

    @staticmethod
    def _uniq(names: list[str]) -> list[str]:
        if len(names) <= 1:
            return [""] * len(names)
        pre = os.path.commonprefix(names)
        suf = os.path.commonprefix([n[::-1] for n in names])[::-1]
        out = []
        for n in names:
            u = n[len(pre):] if pre else n
            u = u[:-len(suf)] if suf and u.endswith(suf) else u
            out.append(u or "(all)")
        return out

    def _apply_filter(self, t: np.ndarray, x: np.ndarray) -> np.ndarray:
        mode = "none"
        if self.filter_lowpass_rb.isChecked():
            mode = "lowpass"
        elif self.filter_highpass_rb.isChecked():
            mode = "highpass"
        elif self.filter_bandpass_rb.isChecked():
            mode = "bandpass"
        elif self.filter_bandblock_rb.isChecked():
            mode = "bandblock"
        nanmask = ~np.isnan(x)
        if not np.any(nanmask):
            return x
        valid_idx = np.where(nanmask)[0]
        x_valid = x[valid_idx]
        t_valid = t[valid_idx]
        x_filt = x_valid
        try:
            dt = np.median(np.diff(t_valid))
            if mode == "lowpass":
                fc = float(self.lowpass_cutoff.text() or 0)
                if fc > 0 and len(x_valid) > 1:
                    x_filt = qats.signal.lowpass(x_valid, dt, fc)
            elif mode == "highpass":
                fc = float(self.highpass_cutoff.text() or 0)
                if fc > 0 and len(x_valid) > 1:
                    x_filt = qats.signal.highpass(x_valid, dt, fc)
            elif mode == "bandpass":
                flow = float(self.bandpass_low.text() or 0)
                fupp = float(self.bandpass_high.text() or 0)
                if flow > 0 and fupp > flow and len(x_valid) > 1:
                    x_filt = qats.signal.bandpass(x_valid, dt, flow, fupp)
            elif mode == "bandblock":
                flow = float(self.bandblock_low.text() or 0)
                fupp = float(self.bandblock_high.text() or 0)
                if flow > 0 and fupp > flow and len(x_valid) > 1:
                    x_filt = qats.signal.bandblock(x_valid, dt, flow, fupp)
        except Exception:
            pass
        x_out = np.full_like(x, np.nan)
        x_out[valid_idx] = x_filt
        return x_out

    @staticmethod
    def _tight_draw(fig, canvas) -> None:
        """Redraw canvas with a tight layout.

        Matplotlib requires a draw call before ``tight_layout`` can correctly
        calculate text bounding boxes when embedded in Qt.  Without this the
        axes may be misaligned or labels can be clipped.  Drawing once before
        and after ``tight_layout`` ensures a stable layout across all plots.
        """

        canvas.draw()
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="Tight layout not applied*", category=UserWarning
            )
            fig.tight_layout()
        canvas.draw()

    @staticmethod
    def _tight_draw(fig, canvas) -> None:
        """Redraw canvas with a tight layout.

        Matplotlib requires a draw call before ``tight_layout`` can correctly
        calculate text bounding boxes when embedded in Qt.  Without this the
        axes may be misaligned or labels can be clipped.  Drawing once before
        and after ``tight_layout`` ensures a stable layout across all plots.
        """

        canvas.draw()
        fig.tight_layout()
        canvas.draw()

    def update_data(self):
        stats_rows = []
        stat_cols = []
        self.ts_dict = {}

        # Temporarily disable sorting while populating the table to avoid
        # rows being rearranged mid-update. Sorting will be re-enabled at
        # the end of this method.
        sorting_was_enabled = self.table.isSortingEnabled()
        if sorting_was_enabled:
            self.table.setSortingEnabled(False)

        if self.filter_lowpass_rb.isChecked():
            f_lbl = f"Low-pass ({self.lowpass_cutoff.text()} Hz)"
        elif self.filter_highpass_rb.isChecked():
            f_lbl = f"High-pass ({self.highpass_cutoff.text()} Hz)"
        elif self.filter_bandpass_rb.isChecked():
            f_lbl = f"Band-pass ({self.bandpass_low.text()}-{self.bandpass_high.text()} Hz)"
        elif self.filter_bandblock_rb.isChecked():
            f_lbl = f"Band-block ({self.bandblock_low.text()}-{self.bandblock_high.text()} Hz)"
        else:
            f_lbl = "None"

        series_info = self.series_info
        if self.order_combo.currentIndex() == 1:

            # Variables → Files: preserve file list order using ``file_idx``
            series_info = sorted(series_info, key=lambda i: (i["var"], i["file_idx"]))
        else:
            # Files → Variables: maintain order of files as loaded
            series_info = sorted(series_info, key=lambda i: (i["file_idx"], i["var"]))


        for info in series_info:
            t = info["t"]
            x = info["x"]
            y = self._apply_filter(t, x)
            ts_tmp = TimeSeries("tmp", t, y)
            stats = ts_tmp.stats()
            if not stat_cols:
                stat_cols = list(stats.keys())
            row = [info["file"], info["uniq_file"], info["var"], "", f_lbl]
            for c in stat_cols:
                v = stats[c]
                if c.lower() == "start" and len(t):
                    v = t[0]
                elif c.lower() == "end" and len(t):
                    v = t[-1]
                if isinstance(v, float):
                    v = float(np.format_float_positional(v, precision=4, unique=False, trim="k"))
                row.append(v)
            stats_rows.append(row)
            sid = f"{info['file']}::{info['var']}"
            self.ts_dict[sid] = (t, y)

        var_uniq = self._uniq([info["var"] for info in series_info])
        for row, vu in zip(stats_rows, var_uniq):
            row[3] = vu

        headers = ["File", "Uniqueness", "Variable", "VarUniqueness", "Filter"] + stat_cols
        self.table.setRowCount(len(stats_rows))
        self.table.setColumnCount(len(headers))
        self.table.setHorizontalHeaderLabels(headers)
        for i, row in enumerate(stats_rows):
            for j, val in enumerate(row):

                text = str(val)
                if isinstance(val, (int, float)):
                    item = SortableTableWidgetItem(text)
                    item.setData(Qt.ItemDataRole.UserRole, float(val))
                else:
                    item = QTableWidgetItem(text)

                self.table.setItem(i, j, item)

        self.selected_columns.clear()
        max_col = next((i for i, h in enumerate(headers) if h.lower() == "max"), None)
        if max_col is not None:
            self.selected_columns.add(max_col)
        else:
            for col in range(5, self.table.columnCount()):
                self.selected_columns.add(col)
                break
        self.update_plots()

        # Restore previous sorting state after table population
        if sorting_was_enabled:
            self.table.setSortingEnabled(True)

    def _header_right_click(self, pos):
        header = self.table.horizontalHeader()
        section = header.logicalIndexAt(pos)
        if section >= 5:
            self.toggle_column(section)

    def toggle_column(self, section: int):
        if section < 5:
            return
        if section in self.selected_columns:
            self.selected_columns.remove(section)
        else:
            self.selected_columns.add(section)
        self.update_plots()

    def keyPressEvent(self, event):
        if event.matches(QKeySequence.Copy):
            self.copy_as_tsv()
            event.accept()
        else:
            super().keyPressEvent(event)

    def copy_as_tsv(self):
        selected = sorted({idx.row() for idx in self.table.selectionModel().selectedRows()})
        if selected:
            rows = selected
        else:
            rows = range(self.table.rowCount())
        lines = ["\t".join([self.table.horizontalHeaderItem(c).text() for c in range(self.table.columnCount())])]
        for r in rows:
            vals = [self.table.item(r, c).text() for c in range(self.table.columnCount())]
            lines.append("\t".join(vals))
        QGuiApplication.clipboard().setText("\n".join(lines))

    def update_plots(self):
        sel_rows = [idx.row() for idx in self.table.selectionModel().selectedRows()]
        if not sel_rows:
            sel_rows = [0] if self.table.rowCount() else []
        if not sel_rows:
            return

        show_text = self.hist_show_text_cb.isChecked()
        import matplotlib.pyplot as plt
        from matplotlib import colors as mcolors

        all_rows = list(range(self.table.rowCount()))

        self.line_fig.clear()
        ax = self.line_fig.add_subplot(111)
        self.psd_fig.clear()
        axp = self.psd_fig.add_subplot(111)
        for r in sel_rows:
            file = self.table.item(r, 0).text()
            var = self.table.item(r, 2).text()
            sid = f"{file}::{var}"
            data = self.ts_dict.get(sid)
            if not data:
                continue
            t, y = data
            label = var
            if file and len(self.ts_dict) > 1:
                label = f"{file}::{var}"
            ax.plot(t, y, label=label)
            if len(t) > 1:
                dt = np.median(np.diff(t))
                fs = 1.0 / dt if dt > 0 else 1.0
                axp.psd(y, Fs=fs, label=label)

        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        ax.legend()
        ax.grid(True)

        self._tight_draw(self.line_fig, self.line_canvas)

        axp.set_xlabel("Frequency [Hz]")
        axp.set_ylabel("PSD")
        axp.legend()
        axp.grid(True)

        self._tight_draw(self.psd_fig, self.psd_canvas)


        self.hist_fig_rows.clear()
        axh = self.hist_fig_rows.add_subplot(111)
        for r in sel_rows:
            file = self.table.item(r, 0).text()
            var = self.table.item(r, 2).text()
            sid = f"{file}::{var}"

            data = self.ts_dict.get(sid)
            if data:
                _, y = data
                counts, bins, patches = axh.hist(
                    y, bins=30, alpha=0.5, label=var
                )
                if show_text:
                    for count, patch in zip(counts, patches):
                        axh.text(
                            patch.get_x() + patch.get_width() / 2,
                            patch.get_height() / 2,
                            str(int(count)),
                            ha="center",
                            va="center",
                            fontsize=8,
                            color="black",
                        )

        axh.set_xlabel("Value")
        axh.set_ylabel("Frequency")
        axh.legend()
        axh.grid(True)
        self._tight_draw(self.hist_fig_rows, self.hist_canvas_rows)


        self.hist_fig_cols.clear()
        axc = self.hist_fig_cols.add_subplot(111)
        max_y = 0
        rows_idx = np.arange(len(all_rows))
        ncols = len(self.selected_columns) if self.selected_columns else 1
        bar_w = 0.8 / ncols
        bars_by_col = []
        for i, c in enumerate(sorted(self.selected_columns)):
            vals = []
            uniq_labels = []
            for r in all_rows:
                item = self.table.item(r, c)
                if item is None:
                    continue
                try:
                    vals.append(float(item.text()))
                except ValueError:
                    vals.append(np.nan)
                u_file = self.table.item(r, 1).text()
                u_var = self.table.item(r, 3).text()
                label = u_file
                if u_var:
                    label = f"{label}\n{u_var}" if label else u_var
                uniq_labels.append(label)
            if not any(np.isfinite(vals)):
                continue
            offset = (i - (ncols - 1) / 2) * bar_w
            bars = axc.bar(rows_idx + offset, vals, width=bar_w, alpha=0.7, label=self.table.horizontalHeaderItem(c).text())
            bars_by_col.append(bars)
            max_y = max(max_y, np.nanmax(vals))
            if show_text:
                for bar, label in zip(bars, uniq_labels):
                    axc.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() / 2,
                        label,
                        ha="center",
                        va="center",
                        rotation=90,
                        fontsize=8,
                        color="black",
                    )

        # Highlight selected rows with a color not already used

        used_colors = {mcolors.to_hex(bar.get_facecolor()) for bc in bars_by_col for bar in bc}
        candidates = ["red", "magenta", "cyan", "yellow", "black"]
        highlight = next((c for c in candidates if mcolors.to_hex(c) not in used_colors), "red")

        for bc in bars_by_col:
            for r in sel_rows:
                if r < len(bc):
                    bc[r].set_facecolor(highlight)
        lines_text = self.hist_lines_edit.text()
        hvals = []
        for token in re.split(r'[ ,]+', lines_text.strip()):
            if not token:
                continue
            try:
                hvals.append(float(token))
            except ValueError:
                pass

        for v in hvals:
            axc.axhline(v, color="red", linestyle="--")

        ylim_top = max([max_y] + hvals) if (max_y or hvals) else None

        axc.set_xlabel("Row")
        axc.set_ylabel("Value")
        axc.set_xticks(rows_idx)
        axc.set_xticklabels([self.table.item(r, 0).text() for r in all_rows], rotation=90)
        if self.selected_columns:
            axc.legend()
        if ylim_top is not None:
            axc.set_ylim(top=ylim_top * 1.1)
        axc.grid(True, axis="y")
        self._tight_draw(self.hist_fig_cols, self.hist_canvas_cols)


class EVMWindow(QDialog):
    def __init__(self, tsdb, var_name, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Extreme Value Analysis - {var_name}")
        self.resize(800, 600)

        self.ts = tsdb.getm()[var_name]
        self.x = self.ts.x
        self.t = self.ts.t

        self.tail_combo = QComboBox()
        self.tail_combo.addItems(["upper", "lower"])
        self.tail_combo.setCurrentText("upper")

        suggested = 0.8 * np.max(self.x)
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setMaximum(10000000000)
        self.threshold_spin.setMinimum(float(np.min(self.x)))
        self.threshold_spin.setDecimals(4)
        self.threshold_spin.setValue(round(suggested, 4))
        self.threshold_spin.setKeyboardTracking(False)

        # Determine an initial reasonable threshold
        threshold = self._auto_threshold(suggested, self.tail_combo.currentText())
        self.threshold_spin.setValue(round(threshold, 4))
        self.threshold_spin.editingFinished.connect(self.on_manual_threshold)

        self._manual_threshold = threshold

        #

        main_layout = QVBoxLayout(self)

        self.inputs_widget = QWidget()
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.plot_area = QWidget()
        self.plot_layout = QVBoxLayout(self.plot_area)
        self.fig = Figure(figsize=(6, 4))
        self.fig_canvas = FigureCanvasQTAgg(self.fig)
        self.plot_layout.addWidget(self.fig_canvas)
        self._evm_ran = False


        self.build_inputs()


        main_layout.addWidget(self.inputs_widget)
        main_layout.addWidget(self.result_text, stretch=1)
        main_layout.addWidget(self.plot_area, stretch=2)

        # Show initial time series with the suggested threshold
        self.plot_timeseries_with_threshold(threshold)

    def _auto_threshold(self, start_thresh, tail):
        x = self.x
        threshold = start_thresh
        attempts = 0

        mean_val = np.mean(x)
        cross_type = np.greater if tail == "upper" else np.less
        cross_indices = np.where(np.diff(cross_type(x, mean_val)))[0]
        if cross_indices.size == 0 or cross_indices[-1] != len(x) - 1:
            cross_indices = np.append(cross_indices, len(x) - 1)

        while attempts < 10:
            clustered_peaks = []
            for i in range(len(cross_indices) - 1):
                segment = x[cross_indices[i] : cross_indices[i + 1]]
                peak = np.max(segment) if tail == "upper" else np.min(segment)
                if (tail == "upper" and peak > threshold) or (
                    tail == "lower" and peak < threshold
                ):
                    clustered_peaks.append(peak)
            if len(clustered_peaks) >= 10:
                break
            threshold *= 0.95 if tail == "upper" else 1.05
            attempts += 1

        return threshold

    def plot_timeseries_with_threshold(self, threshold):

        self.fig.clear()
        ax = self.fig.add_subplot(111)

        ax.plot(self.t, self.x, label="Time series")
        ax.axhline(threshold, color="red", linestyle="--", label="Threshold")
        ax.set_title("Time Series")
        ax.set_xlabel("Time")
        ax.set_ylabel(self.ts.name)
        ax.grid(True)
        ax.legend()
        self.fig_canvas.draw()

    def _cluster_exceedances(self, threshold, tail):
        x = self.x

        mean_val = np.mean(x)
        cross_type = np.greater if tail == "upper" else np.less
        cross_indices = np.where(np.diff(cross_type(x, mean_val)))[0]
        if cross_indices.size == 0 or cross_indices[-1] != len(x) - 1:
            cross_indices = np.append(cross_indices, len(x) - 1)

        clustered_peaks = []
        for i in range(len(cross_indices) - 1):
            segment = x[cross_indices[i] : cross_indices[i + 1]]
            peak = np.max(segment) if tail == "upper" else np.min(segment)
            if (tail == "upper" and peak > threshold) or (
                tail == "lower" and peak < threshold
            ):
                clustered_peaks.append(peak)
        return clustered_peaks, cross_indices

    def on_manual_threshold(self):
        self.threshold_spin.interpretText()
        threshold = self.threshold_spin.value()

        # ensure the spin box keeps the manually entered value
        self.threshold_spin.setValue(threshold)

        self._manual_threshold = threshold

        self.plot_timeseries_with_threshold(threshold)
        peaks, _ = self._cluster_exceedances(threshold, self.tail_combo.currentText())
        self.result_text.setPlainText(f"Exceedances used: {len(peaks)}")
        self._evm_ran = False

    def on_calc_threshold(self):
        tail = self.tail_combo.currentText()
        suggested = 0.8 * np.max(self.x) if tail == "upper" else 0.8 * np.min(self.x)
        threshold = self._auto_threshold(suggested, tail)
        self.threshold_spin.setValue(round(threshold, 4))
        self.plot_timeseries_with_threshold(threshold)

    def build_inputs(self):
        layout = QGridLayout(self.inputs_widget)

        layout.addWidget(QLabel("Distribution: Generalized Pareto"), 0, 0, 1, 3)

        layout.addWidget(QLabel("Threshold:"), 1, 0)
        layout.addWidget(self.threshold_spin, 1, 1)
        self.calc_threshold_btn = QPushButton("Calc Threshold")
        self.calc_threshold_btn.clicked.connect(self.on_calc_threshold)
        layout.addWidget(self.calc_threshold_btn, 1, 2)

        self.ci_spin = QDoubleSpinBox()
        self.ci_spin.setDecimals(1)
        self.ci_spin.setValue(95.0)
        layout.addWidget(QLabel("Extremes to analyse:"), 2, 0)
        layout.addWidget(self.tail_combo, 2, 1)

        layout.addWidget(QLabel("Confidence level (%):"), 3, 0)
        layout.addWidget(self.ci_spin, 3, 1)
        self.ci_spin.valueChanged.connect(self.on_ci_changed)

        run_btn = QPushButton("Run EVM")
        run_btn.clicked.connect(self.run_evm)
        layout.addWidget(run_btn, 4, 0, 1, 3)

    def run_evm(self):

        from scipy.stats import genpareto

        x = self.x
        t = self.t

        tail = self.tail_combo.currentText()

        threshold = self.threshold_spin.value()

        clustered_peaks, cross_indices = self._cluster_exceedances(threshold, tail)

        if len(clustered_peaks) < 10:
            QMessageBox.warning(
                self,
                "Too Few Points",
                f"Threshold {threshold:.3f} resulted in only {len(clustered_peaks)} clustered exceedances.",
            )
            return

        excesses = np.array(clustered_peaks) - threshold
        c, loc, scale = genpareto.fit(excesses, floc=0)

        # Diagnostic: warn if shape is too extreme
        if abs(c) > 1:
            print(
                f"Warning: large shape parameter detected (xi = {c:.4f}). Return levels may be unstable."
            )
        if c < -1e-6:
            print("Note: fitted GPD shape xi < 0 indicates a bounded tail.")

        exceed_prob = len(clustered_peaks) / (t[-1] - t[0])

        return_periods = np.array([0.1, 0.5, 1, 3, 5])  # hours
        return_secs = return_periods * 3600
        rl = threshold + (scale / c) * ((exceed_prob * return_secs) ** c - 1)

        n_bootstrap = 500
        boot_levels = []
        rs = np.random.default_rng()

        for _ in range(n_bootstrap):
            sample = rs.choice(excesses, size=len(excesses), replace=True)
            try:
                bc, _, bscale = genpareto.fit(sample, floc=0)
                boot_level = threshold + (bscale / bc) * (
                    (exceed_prob * return_secs) ** bc - 1
                )
                boot_levels.append(boot_level)
            except Exception:
                continue

        boot_levels = np.array(boot_levels)
        boot_levels = boot_levels[~np.isnan(boot_levels).any(axis=1)]
        boot_levels = boot_levels[
            (boot_levels > -1e6).all(axis=1) & (boot_levels < 1e6).all(axis=1)
        ]

        if boot_levels.shape[0] > 0:
            ci_alpha = 100 - self.ci_spin.value()
            lower_bounds = np.percentile(boot_levels, ci_alpha / 2, axis=0)
            upper_bounds = np.percentile(boot_levels, 100 - ci_alpha / 2, axis=0)
        else:
            lower_bounds = upper_bounds = [np.nan] * len(return_secs)

        units = ""
        max_val = np.max(x) if tail == "upper" else np.min(x)

        result = f"Extreme value statistics: {self.ts.name}\n\n"
        result += f"The {return_periods[-2]:.1f} hour return level is\n{rl[-2]:.5f} {units}\n\n"
        result += f"Fitted GPD parameters:\nSigma: {scale:.4f}\nXi: {c:.4f}\nExceedances used: {len(excesses)}\n"
        result += f"Total crossings/clusters found: {len(cross_indices) - 1}\n"
        result += f"Observed maximum value: {max_val:.4f} {units}\n"
        result += f"Return level unit: {units or 'same as input'}\n\n"
        result += f"{self.ci_spin.value():.0f}% Confidence Interval:\n"
        for rp, lo, up in zip(return_periods, lower_bounds, upper_bounds):
            result += f"{rp:.1f} hr: {lo:.3f} – {up:.3f}\n"

        self.result_text.setPlainText(result)

        self.plot_diagnostics(
            return_secs,
            rl,
            excesses,
            c,
            scale,
            threshold,
            lower_bounds,
            upper_bounds,
        )
        self._evm_ran = True

    def on_ci_changed(self, value):
        if self._evm_ran:
            self.run_evm()


    def plot_diagnostics(
        self,
        durations,
        levels,
        excesses,
        c,
        scale,
        threshold,
        lower_bounds=None,
        upper_bounds=None,
    ):
        from scipy.stats import genpareto


        self.fig.clear()
        self.fig.set_size_inches(14, 4)
        ts_ax = self.fig.add_subplot(1, 3, 1)
        ax = self.fig.add_subplot(1, 3, 2)
        qax = self.fig.add_subplot(1, 3, 3)


        ts_ax.plot(self.t, self.x, label="Time series")
        ts_ax.axhline(threshold, color="red", linestyle="--", label="Threshold")
        ts_ax.set_title("Time Series")
        ts_ax.set_xlabel("Time")
        ts_ax.set_ylabel(self.ts.name)
        ts_ax.grid(True)
        ts_ax.legend()
        ax.plot(durations / 3600, levels, marker="o", label="Return Level")

        if lower_bounds is not None and upper_bounds is not None:
            ax.fill_between(
                durations / 3600,
                lower_bounds,
                upper_bounds,
                color="gray",
                alpha=0.3,
                label="Confidence Interval",
            )

        ax.set_title("Return level plot")
        ax.set_xlabel("Storm duration (hours)")
        ax.set_ylabel("Return level")
        ax.grid(True)
        ax.legend()

        sorted_empirical = np.sort(excesses)
        probs = (np.arange(1, len(sorted_empirical) + 1) - 0.5) / len(sorted_empirical)
        model_quantiles = genpareto.ppf(probs, c, scale=scale)

        qax.scatter(model_quantiles, sorted_empirical, alpha=0.6, label="Data")
        qax.plot(model_quantiles, model_quantiles, color="red", label="1:1 line")
        qax.set_title("Quantile Plot")
        qax.set_xlabel("Theoretical Quantiles")
        qax.set_ylabel("Empirical Quantiles")
        qax.grid(True)
        qax.legend()

        self.fig.tight_layout()

        self.fig_canvas.draw()

class OrcaflexVariableSelector(QDialog):
    def __init__(self, model, orcaflex_varmap=None, parent=None, previous_selection=None, allow_reuse=False):
        super().__init__(parent)
        self.setWindowTitle("Select OrcaFlex Variables")
        self.resize(1100, 800)
        self.model = model
        self.orcaflex_varmap = ORCAFLEX_VARIABLE_MAP or {}
        self.selected = []
        self.reuse_all = False

        main = QVBoxLayout(self)
        obj_var_layout = QHBoxLayout()
        main.addLayout(obj_var_layout)

        left_layout = QVBoxLayout()
        right_layout = QVBoxLayout()
        obj_var_layout.addLayout(left_layout, stretch=1)
        obj_var_layout.addLayout(right_layout, stretch=1)

        left_layout.addWidget(QLabel("Objects"))
        self.obj_filter = QLineEdit()
        self.obj_filter.setPlaceholderText("Search objects")
        left_layout.addWidget(self.obj_filter)
        self.obj_list = QListWidget()
        self.obj_list.setSelectionMode(QAbstractItemView.MultiSelection)
        left_layout.addWidget(self.obj_list)

        right_layout.addWidget(QLabel("Variables"))
        self.var_filter = QLineEdit()
        self.var_filter.setPlaceholderText("Search variables")
        right_layout.addWidget(self.var_filter)
        self.var_list = QListWidget()
        self.var_list.setSelectionMode(QAbstractItemView.MultiSelection)
        right_layout.addWidget(self.var_list)

        # Only objects for which we have variable types in the varmap
        self.object_map = {
            obj.Name: obj
            for obj in self.model.objects
            if obj.typeName in self.orcaflex_varmap
        }
        self.all_objects = sorted(self.object_map.keys())
        self._update_object_list("")

        self.extra_entry = QLineEdit()
        self.extra_entry.setPlaceholderText("Arc Lengths / Extra (e.g. EndA, 10.0, EndB)")
        obj_var_layout.addWidget(QLabel("Arc Length/Extra:"))
        obj_var_layout.addWidget(self.extra_entry)

        self.redundant_entry = QLineEdit()
        self.redundant_entry.setPlaceholderText("Redundant substrings (comma-separated)")
        main.addWidget(QLabel("Remove Redundant Substrings from Labels:"))
        main.addWidget(self.redundant_entry)

        self.reuse_cb = QCheckBox("Use this selection for all future OrcaFlex files")
        self.reuse_cb.setChecked(allow_reuse)
        main.addWidget(self.reuse_cb)

        btns = QHBoxLayout()
        self.ok_btn = QPushButton("Load")
        self.cancel_btn = QPushButton("Cancel")
        btns.addWidget(self.ok_btn)
        btns.addWidget(self.cancel_btn)
        main.addLayout(btns)

        self.obj_list.itemSelectionChanged.connect(self.update_var_list)
        self.obj_filter.textChanged.connect(self._update_object_list)
        self.var_filter.textChanged.connect(self.update_var_list)
        self.ok_btn.clicked.connect(self.on_ok)
        self.cancel_btn.clicked.connect(self.reject)

    def _update_object_list(self, text):
        terms = _parse_search_terms(text)
        selected = {item.text() for item in self.obj_list.selectedItems()}
        self.obj_list.clear()
        for name in self.all_objects:

            if not terms or _matches_terms(name, terms):

                item = QListWidgetItem(name)
                self.obj_list.addItem(item)
                if name in selected:
                    item.setSelected(True)
        self.update_var_list()

    def update_var_list(self, *_):

        self.var_list.clear()
        selected_objs = self.obj_list.selectedItems()
        if not selected_objs:
            return
        last_obj = self.object_map[selected_objs[-1].text()]
        variables = get_object_available_vars(last_obj, self.orcaflex_varmap)
        terms = _parse_search_terms(self.var_filter.text())
        for var in variables:

            if not terms or _matches_terms(var, terms):

                self.var_list.addItem(var)

    def on_ok(self):
        objects = [item.text() for item in self.obj_list.selectedItems()]
        variables = [item.text() for item in self.var_list.selectedItems()]
        if not objects or not variables:
            QMessageBox.warning(self, "Selection required", "Select at least one object and one variable.")
            return
        extras = [x.strip() for x in self.extra_entry.text().split(",") if x.strip()]
        redundant = [x.strip() for x in self.redundant_entry.text().split(",") if x.strip()]
        self.selected = []
        for obj in objects:
            for var in variables:
                if extras:
                    for extra in extras:
                        self.selected.append((obj, var, extra))
                else:
                    self.selected.append((obj, var, None))
        self.redundant = redundant
        self.reuse_all = self.reuse_cb.isChecked()
        self.accept()

    @staticmethod
    def get_selection(model, orcaflex_varmap=None, parent=None, previous_selection=None, allow_reuse=False):
        dlg = OrcaflexVariableSelector(model, orcaflex_varmap, parent, previous_selection, allow_reuse)
        result = dlg.exec()
        if result == QDialog.Accepted:
            return dlg.selected, dlg.redundant, dlg.reuse_all
        else:
            return None, None, None

class FileLoader:
    def __init__(self, orcaflex_varmap=None, parent_gui=None):
        self.orcaflex_varmap = orcaflex_varmap or {}
        self.parent_gui = parent_gui
        self._last_orcaflex_selection = None
        self._reuse_orcaflex_selection = False
        self.loaded_sim_models = {}
        self.orcaflex_redundant_subs = []
        self.progress_callback = None  # called while pre-loading
        self._last_diffraction_dir = None
        self._diffraction_cache = {}

    @property
    def reuse_orcaflex_selection(self):
        """The radius property."""
        return self._reuse_orcaflex_selection

    @reuse_orcaflex_selection.setter
    def reuse_orcaflex_selection(self, value):
        print("Set radius")
        self._reuse_orcaflex_selection = value

    def preload_sim_models(self, filepaths):
        try:
            import OrcFxAPI
        except ImportError:
            print("OrcFxAPI not available. Cannot preload .sim files.")
            return
        total_files = len(filepaths)
        for idx, path in enumerate(filepaths):
            if path not in self.loaded_sim_models:
                try:
                    model = OrcFxAPI.Model(path)
                    self.loaded_sim_models[path] = model
                    print(f"✅ Loaded OrcaFlex model: {os.path.basename(path)}")
                except Exception as e:
                    print(f"❌ Failed to load OrcaFlex model {os.path.basename(path)}:\n{e}")

            if self.progress_callback:
                self.progress_callback(idx + 1, total_files)
            if self.parent_gui:
                QApplication.processEvents()

    def load_files(self, file_paths):
        tsdbs = []
        errors = []
        sim_files = [fp for fp in file_paths if fp.lower().endswith(".sim")]
        other_files = [fp for fp in file_paths if fp not in sim_files]

        tsdb_by_path = {}
        # --- OrcaFlex handling ---
        if sim_files:
            try:

                picked = self.open_orcaflex_picker(sim_files)
                tsdb_by_path.update(picked)

            except Exception as e:
                for fp in sim_files:
                    errors.append((fp, str(e)))
        # --- Other files ---
        for fp in other_files:
            try:
                tsdb_by_path[fp] = self._load_generic_file(fp)
            except Exception as e:
                errors.append((fp, str(e)))

        for fp in file_paths:
            if fp in tsdb_by_path:
                tsdbs.append(tsdb_by_path[fp])

        return tsdbs, errors

    def _load_orcaflex_file(self, filepath):
        import OrcFxAPI
        # Preload model on first use
        if filepath not in self.loaded_sim_models:
            self.loaded_sim_models[filepath] = OrcFxAPI.Model(filepath)
        model = self.loaded_sim_models[filepath]

        # Reuse previous selection?
        if self._reuse_orcaflex_selection and self._last_orcaflex_selection:
            self.orcaflex_redundant_subs = getattr(
                self, "orcaflex_redundant_subs", []
            )
            return self._load_orcaflex_data_from_specs(
                model, self._last_orcaflex_selection
            )

        # Variable/object selection dialog
        selected, redundant, reuse_all = OrcaflexVariableSelector.get_selection(
            model, self.orcaflex_varmap, self.parent_gui
        )
        if not selected:
            return None

        # Convert extras to OrcFxAPI objects
        specs = []
        for obj_name, var, extra_str in selected:
            obj = model[obj_name]
            for extra, label in self._parse_extras(obj, extra_str or ""):
                specs.append((obj, obj_name, var, extra, label))

        self.orcaflex_redundant_subs = redundant or []
        if reuse_all:
            self._last_orcaflex_selection = specs.copy()
            self._reuse_orcaflex_selection = True

        return self._load_orcaflex_data_from_specs(model, specs)

    def open_orcaflex_picker(self, file_paths):
        """Qt version of the OrcaFlex variable picker."""
        import OrcFxAPI

        missing = [fp for fp in file_paths if fp not in self.loaded_sim_models]
        if missing:

            # Lazily preload missing models so the picker can proceed
            self.preload_sim_models(missing)
            remaining = [fp for fp in missing if fp not in self.loaded_sim_models]
            if remaining:
                raise RuntimeError(
                    "Models not preloaded: "
                    + ", ".join(os.path.basename(m) for m in remaining)
                )


        if self._reuse_orcaflex_selection and self._last_orcaflex_selection:
            result = {}
            for fp in file_paths:
                tsdb = self._load_orcaflex_data_from_specs(
                    self.loaded_sim_models[fp],
                    self._last_orcaflex_selection,
                )
                if tsdb:
                    result[fp] = tsdb
            return result

        dialog = QDialog(self.parent_gui)
        dialog.setWindowTitle("Pick OrcaFlex Variables")
        dialog.setWindowFlags(dialog.windowFlags() | Qt.WindowMaximizeButtonHint)
        dialog.resize(1150, 820)
        main_layout = QHBoxLayout(dialog)

        file_side = QVBoxLayout()
        right_side = QVBoxLayout()
        main_layout.addLayout(file_side)
        main_layout.addLayout(right_side)

        red_layout = QHBoxLayout()
        red_layout.addWidget(QLabel("Remove (comma-separated):"))
        redundant_entry = QLineEdit()
        red_layout.addWidget(redundant_entry)
        strip_cb = QCheckBox("Strip '_nodes_(x,y)' suffix")
        red_layout.addWidget(strip_cb)
        right_side.addLayout(red_layout)

        wc_layout = QHBoxLayout()
        wc_layout.addWidget(QLabel("Strip rule:"))
        wc_entry = QLineEdit()
        wc_layout.addWidget(wc_entry)
        regex_cb = QCheckBox("Use as REGEX")
        wc_layout.addWidget(regex_cb)
        right_side.addLayout(wc_layout)

        # Default extras for each object type
        default_group = QGroupBox("Default Extra Input")
        def_layout = QGridLayout(default_group)
        default_inputs = {}
        types = ["Line", "Vessel", "Buoy", "Constraint", "Environment"]
        defaults = {
            "Line": "EndA, mid, EndB",
            "Vessel": "0,0,0",
            "Buoy": "0,0,0",
            "Constraint": "0,0,0",
            "Environment": "0,0,0",
        }
        for row, typ in enumerate(types):
            def_layout.addWidget(QLabel(f"{typ}:"), row, 0)
            e = QLineEdit(defaults[typ])
            default_inputs[typ] = e
            def_layout.addWidget(e, row, 1)
        right_side.addWidget(default_group)

        tabs = QTabWidget()
        right_side.addWidget(tabs)

        per_file_state = {}
        panel_pressure_by_file = {}
        for fp in file_paths:
            model = self.loaded_sim_models[fp]
            obj_map = {
                o.Name: (o, self.orcaflex_varmap[o.typeName])
                for o in model.objects
                if o.typeName in self.orcaflex_varmap
            }

            tab = QWidget()
            tabs.addTab(tab, os.path.basename(fp))
            tab_layout = QHBoxLayout(tab)
            left = QVBoxLayout()
            right = QVBoxLayout()
            tab_layout.addLayout(left)
            tab_layout.addLayout(right)

            left.addWidget(QLabel("Objects"))
            obj_filter = QLineEdit()

            obj_filter.setPlaceholderText("Filter objects")

            left.addWidget(obj_filter)
            obj_scroll = QScrollArea()
            obj_widget = QWidget()
            obj_vbox = QVBoxLayout(obj_widget)
            obj_scroll.setWidgetResizable(True)
            obj_scroll.setWidget(obj_widget)
            left.addWidget(obj_scroll)
            obj_vars = {}
            for name in sorted(obj_map):
                cb = QCheckBox(name)
                obj_vbox.addWidget(cb)
                obj_vars[name] = cb

            obj_btns = QHBoxLayout()
            btn_obj_all = QPushButton("Select All Objects")
            btn_obj_none = QPushButton("Unselect All Objects")
            obj_btns.addWidget(btn_obj_all)
            obj_btns.addWidget(btn_obj_none)
            left.addLayout(obj_btns)


            obj_show_cb = QCheckBox("Show only selected")
            left.addWidget(obj_show_cb)


            var_column = QVBoxLayout()
            var_column.addWidget(QLabel("Variables"))
            var_filter = QLineEdit()

            var_filter.setPlaceholderText("Filter variables")

            var_column.addWidget(var_filter)
            var_scroll = QScrollArea()
            var_widget = QWidget()
            var_vbox = QVBoxLayout(var_widget)
            var_scroll.setWidgetResizable(True)
            var_scroll.setWidget(var_widget)
            var_column.addWidget(var_scroll)
            var_vars = {}

            var_btns = QHBoxLayout()
            btn_var_all = QPushButton("Select All Variables")
            btn_var_none = QPushButton("Unselect All Variables")
            var_btns.addWidget(btn_var_all)
            var_btns.addWidget(btn_var_none)
            var_column.addLayout(var_btns)


            var_show_cb = QCheckBox("Show only selected")
            var_column.addWidget(var_show_cb)


            extra_group = QGroupBox("Position / Arc Length")
            extra_layout = QVBoxLayout(extra_group)
            extra_entry = QLineEdit()

            extra_entry.setPlaceholderText("Arc length / position")
            extra_layout.addWidget(extra_entry)
            extra_layout.addWidget(QLabel("Find Closest:"))
            coord_entry = QLineEdit()
            coord_entry.setPlaceholderText("x,y,z; x,y,z ...")

            extra_layout.addWidget(coord_entry)
            find_btn = QPushButton("Find Closest (Selected)")
            extra_layout.addWidget(find_btn)
            skip_entry = QLineEdit()
            skip_entry.setPlaceholderText("Skip names (comma-separated)")
            extra_layout.addWidget(skip_entry)
            find_all_btn = QPushButton("Find Closest (All)")
            extra_layout.addWidget(find_all_btn)
            pressure_btn = QPushButton("Extract Surface Pressures")
            extra_layout.addWidget(pressure_btn)
            result_table = QTableWidget()
            result_table.setColumnCount(4)
            result_table.setHorizontalHeaderLabels(
                ["Coordinate", "Object", "Node / Panel", "Distance"]
            )
            extra_layout.addWidget(result_table)
            copy_btn = QPushButton("Copy Table")

            def _copy_table(*_, table=result_table):

                lines = [
                    "\t".join(
                        table.horizontalHeaderItem(c).text()
                        for c in range(table.columnCount())
                    )
                ]
                for r in range(table.rowCount()):
                    vals = [
                        table.item(r, c).text() if table.item(r, c) else ""
                        for c in range(table.columnCount())
                    ]
                    lines.append("\t".join(vals))
                QGuiApplication.clipboard().setText("\n".join(lines))

            copy_btn.clicked.connect(_copy_table)
            extra_layout.addWidget(copy_btn)

            right_split = QHBoxLayout()
            right_split.addLayout(var_column)
            right_split.addWidget(extra_group)
            right.addLayout(right_split)


            def rebuild_lists(
                *_,

                obj_filter=obj_filter,
                obj_vars=obj_vars,
                var_filter=var_filter,
                var_vbox=var_vbox,
                var_vars=var_vars,
                obj_map=obj_map,
                extra_entry=extra_entry,
                default_inputs=default_inputs,

                obj_show_cb=obj_show_cb,
                var_show_cb=var_show_cb,

            ):

                def update_var_visibility(*_):
                    terms_var_vis = _parse_search_terms(var_filter.text())
                    for name, cb in var_vars.items():
                        visible = True

                        if terms_var_vis and not _matches_terms(name, terms_var_vis):
                            visible = False

                        if var_show_cb.isChecked() and not cb.isChecked():

                            visible = False
                        cb.setVisible(visible)

                terms_obj = _parse_search_terms(obj_filter.text())
                for name, cb in obj_vars.items():
                    visible = True
                    if terms_obj and not _matches_terms(name, terms_obj):
                        visible = False

                    if obj_show_cb.isChecked() and not cb.isChecked():

                        visible = False
                    cb.setVisible(visible)


                selected = [n for n, cb in obj_vars.items() if cb.isChecked()]
                prev_states = {name: cb.isChecked() for name, cb in var_vars.items()}
                for i in range(var_vbox.count() - 1, -1, -1):
                    w = var_vbox.itemAt(i).widget()
                    if w is not None:
                        w.deleteLater()
                var_vars.clear()

                extra_entry.clear()

                if selected:
                    first_type = obj_map[selected[0]][0].typeName
                    same_type = all(
                        obj_map[n][0].typeName == first_type for n in selected
                    )
                else:
                    same_type = False

                if selected and same_type:
                    otype = first_type
                    terms_var = _parse_search_terms(var_filter.text())
                    for vname in self.orcaflex_varmap.get(otype, []):

                        if not terms_var or _matches_terms(vname, terms_var):

                            cbv = QCheckBox(vname)
                            cbv.setChecked(prev_states.get(vname, False))
                            cbv.toggled.connect(update_var_visibility)
                            var_vbox.addWidget(cbv)
                            var_vars[vname] = cbv

                    if otype == "Line":
                        default_val = default_inputs.get("Line").text().strip()
                        extra_entry.setText(default_val or "EndA, mid, EndB")
                    elif otype in (
                        "Vessel",
                        "Buoy",
                        "Constraint",
                        "Environment",
                    ):
                        default_val = default_inputs.get(otype).text().strip()
                        extra_entry.setText(default_val or "0,0,0")
                else:
                    if not selected:
                        msg = "Select object(s) to see variables"
                    elif not same_type:
                        msg = "Selected objects are not the same type"
                    else:
                        msg = "Select object(s) to see variables"
                    var_vbox.addWidget(QLabel(msg))

                update_var_visibility()

            obj_filter.textChanged.connect(rebuild_lists)
            var_filter.textChanged.connect(rebuild_lists)
            for cb in obj_vars.values():
                cb.toggled.connect(rebuild_lists)
            rebuild_lists()


            def select_all_objects(*_, obj_vars=obj_vars):


                # Block signals while toggling to avoid repeated rebuilds

                for cb in obj_vars.values():
                    if cb.isVisible():
                        cb.blockSignals(True)
                        cb.setChecked(True)
                        cb.blockSignals(False)


                # Rebuild lists once after bulk toggle
                rebuild_lists()



            def unselect_all_objects(*_, obj_vars=obj_vars):

                for cb in obj_vars.values():
                    if cb.isVisible():
                        cb.blockSignals(True)
                        cb.setChecked(False)
                        cb.blockSignals(False)


                rebuild_lists()


            btn_obj_all.clicked.connect(select_all_objects)
            btn_obj_none.clicked.connect(unselect_all_objects)

            def select_all_vars(*_, var_vars=var_vars):

                for cb in var_vars.values():
                    if cb.isVisible():
                        cb.blockSignals(True)
                        cb.setChecked(True)
                        cb.blockSignals(False)


                rebuild_lists()



            def unselect_all_vars(*_, var_vars=var_vars):

                for cb in var_vars.values():
                    if cb.isVisible():
                        cb.blockSignals(True)
                        cb.setChecked(False)
                        cb.blockSignals(False)


                rebuild_lists()


            btn_var_all.clicked.connect(select_all_vars)
            btn_var_none.clicked.connect(unselect_all_vars)


            obj_show_cb.toggled.connect(rebuild_lists)
            var_show_cb.toggled.connect(rebuild_lists)


            def _update_table(coords, info, table=result_table):
                table.setRowCount(len(coords))
                for row, (coord, (name, node, dist)) in enumerate(zip(coords, info)):
                    coord_str = f"{coord[0]:.2f}, {coord[1]:.2f}, {coord[2]:.2f}"
                    table.setItem(row, 0, QTableWidgetItem(coord_str))
                    table.setItem(row, 1, QTableWidgetItem(name or ""))
                    node_txt = "" if node is None else str(node)
                    table.setItem(row, 2, QTableWidgetItem(node_txt))
                    dist_txt = "" if dist is None else f"{dist:.3f}"
                    table.setItem(row, 3, QTableWidgetItem(dist_txt))
                table.resizeColumnsToContents()

            def find_closest(
                *_,
                obj_vars=obj_vars,
                obj_map=obj_map,
                coord_entry=coord_entry,
                update_table=_update_table,
            ):
                coords = self._parse_xyz_list(coord_entry.text())
                if not coords:
                    return
                selected = [
                    obj_map[n][0] for n, cb in obj_vars.items() if cb.isChecked()
                ]
                if not selected:
                    return

                closest_info = self._get_closest_objects(coords, selected)
                chosen = {name for name, _, _ in closest_info}

                for name, cb in obj_vars.items():
                    cb.setChecked(name in chosen)
                rebuild_lists()
                update_table(coords, closest_info)

            find_btn.clicked.connect(find_closest)

            def find_closest_all(
                *_,
                obj_vars=obj_vars,
                obj_map=obj_map,
                coord_entry=coord_entry,
                skip_entry=skip_entry,
                update_table=_update_table,
            ):
                coords = self._parse_xyz_list(coord_entry.text())
                if not coords:
                    return
                skip_terms = [
                    s.strip().lower()
                    for s in skip_entry.text().split(',')
                    if s.strip()
                ]
                selected = [
                    pair[0]
                    for name, pair in obj_map.items()
                    if not any(term in name.lower() for term in skip_terms)
                ]
                closest_info = self._get_closest_objects(coords, selected)
                chosen = {name for name, _, _ in closest_info}

                for name, cb in obj_vars.items():
                    cb.setChecked(name in chosen)
                rebuild_lists()
                update_table(coords, closest_info)

            find_all_btn.clicked.connect(find_closest_all)

            def extract_surface_pressures(
                *_,
                coord_entry=coord_entry,
                model=model,
                fp=fp,
                update_table=_update_table,
                table=result_table,
            ):
                coords = self._parse_xyz_list(coord_entry.text())
                if not coords:
                    QMessageBox.warning(
                        dialog,
                        "No Coordinates",
                        "Enter at least one coordinate in the form x,y,z.",
                    )
                    panel_pressure_by_file.pop(fp, None)
                    table.setRowCount(0)
                    return
                coord_tuples = [tuple(float(v) for v in c) for c in coords]
                if not coord_tuples:
                    QMessageBox.warning(
                        dialog,
                        "No Coordinates",
                        "Could not interpret the provided coordinates.",
                    )
                    panel_pressure_by_file.pop(fp, None)
                    table.setRowCount(0)
                    return

                start_dir = self._last_diffraction_dir or os.path.dirname(fp)
                if not start_dir or not os.path.isdir(start_dir):
                    start_dir = os.path.dirname(fp)
                file_path, _ = QFileDialog.getOpenFileName(
                    dialog,
                    "Select Diffraction Model (.dat)",
                    start_dir,
                    "Diffraction data (*.dat);;All Files (*)",
                )
                if not file_path:
                    return

                directory = os.path.dirname(file_path)
                if directory:
                    self._last_diffraction_dir = directory

                try:
                    import OrcFxAPI
                except ImportError:
                    QMessageBox.critical(
                        dialog,
                        "OrcaFlex Not Available",
                        "OrcFxAPI is required to extract surface pressures.",
                    )
                    return

                diffraction_model = self._diffraction_cache.get(file_path)
                if diffraction_model is None:
                    try:
                        diffraction_model = OrcFxAPI.Diffraction(file_path)
                    except Exception as exc:
                        QMessageBox.critical(
                            dialog,
                            "Diffraction Load Error",
                            f"Failed to load diffraction model:\n{exc}",
                        )
                        return
                    self._diffraction_cache[file_path] = diffraction_model

                try:
                    pressure_df, panel_info = self._get_panel_pressure(
                        model=model,
                        diffraction_model=diffraction_model,
                        panel_coords=tuple(coord_tuples),
                    )
                except Exception as exc:
                    QMessageBox.critical(
                        dialog,
                        "Surface Pressure Error",
                        f"Failed to extract surface pressures:\n{exc}",
                    )
                    return

                if (
                    pressure_df is None
                    or pressure_df.empty
                    or ("Time" in pressure_df.columns and pressure_df.shape[1] <= 1)
                    or ("Time" not in pressure_df.columns and pressure_df.shape[1] == 0)
                ):
                    QMessageBox.warning(
                        dialog,
                        "No Surface Pressures",
                        "No surface pressure data was returned for the provided coordinates.",
                    )
                    panel_pressure_by_file.pop(fp, None)
                    table.setRowCount(0)
                    return

                if panel_info is None:
                    panel_info = pd.DataFrame()
                else:
                    panel_info = panel_info.copy().reset_index(drop=True)

                panel_pressure_by_file[fp] = [
                    {
                        "data": pressure_df.copy(),
                        "info": panel_info.copy(),
                        "diffraction_path": file_path,
                    }
                ]

                if not panel_info.empty:
                    coords_for_table = []
                    info_for_table = []
                    for _, row in panel_info.iterrows():
                        coord_val = row.get("input_coord")
                        if isinstance(coord_val, (tuple, list, np.ndarray)) and len(coord_val) == 3:
                            coords_for_table.append(np.array(coord_val, dtype=float))
                        else:
                            coords_for_table.append(
                                np.array(
                                    [
                                        float(row.get("X", 0.0)),
                                        float(row.get("Y", 0.0)),
                                        float(row.get("Z", 0.0)),
                                    ],
                                    dtype=float,
                                )
                            )
                        node_val = row.get("pidx")
                        node_disp = None if pd.isna(node_val) else int(node_val) + 1
                        dist_val = row.get("distance")
                        if pd.isna(dist_val):
                            dist_val = None
                        info_for_table.append((row.get("name"), node_disp, dist_val))
                    update_table(coords_for_table, info_for_table)
                else:
                    table.setRowCount(0)

                time_col = "Time" if "Time" in pressure_df.columns else None
                n_series = pressure_df.shape[1] - (1 if time_col else 0)
                QMessageBox.information(
                    dialog,
                    "Surface Pressures Extracted",
                    f"Loaded {n_series} surface pressure series.",
                )

            pressure_btn.clicked.connect(extract_surface_pressures)

            per_file_state[fp] = {
                "obj_vars": obj_vars,
                "var_vars": var_vars,
                "extra_entry": extra_entry,
                "model": model,
                "obj_map": obj_map,
                "rebuild": rebuild_lists,
                "table": result_table,
            }

        apply_specs = {}

        file_group = QGroupBox("Select Sims for this selection")
        file_layout = QVBoxLayout(file_group)
        file_checks = {}
        select_all_btn = QPushButton("Select All Sims")
        file_layout.addWidget(select_all_btn)
        status_label = QLabel()

        def select_all_sims():
            for cb in file_checks.values():
                cb.setChecked(True)

        select_all_btn.clicked.connect(select_all_sims)

        def check_files(*_):
            selected = [fp for fp, cb in file_checks.items() if cb.isChecked()]

            # Disable tabs not part of the selection
            if selected:
                for idx, fp in enumerate(file_paths):
                    tabs.setTabEnabled(idx, fp in selected)
                reuse_cb.setEnabled(False)
            else:
                for idx in range(tabs.count()):
                    tabs.setTabEnabled(idx, True)
                reuse_cb.setEnabled(True)

            if len(selected) < 2:
                status_label.setText("")
                apply_btn.setEnabled(bool(selected))
                return
            base = set(per_file_state[selected[0]]["obj_map"].keys())
            identical = all(
                set(per_file_state[f]["obj_map"].keys()) == base for f in selected[1:]
            )
            if identical:
                status_label.setText("")
                apply_btn.setEnabled(True)
            else:
                status_label.setText("objects not identical")
                apply_btn.setEnabled(False)

        for fp in file_paths:
            cb = QCheckBox(os.path.basename(fp))
            file_layout.addWidget(cb)
            file_checks[fp] = cb
            cb.toggled.connect(check_files)

        file_side.addWidget(file_group)
        file_side.addWidget(status_label)

        apply_btn = QPushButton("Apply Selection to Checked Sims")
        apply_btn.setEnabled(False)
        file_side.addWidget(apply_btn)

        def apply_selection():
            selected = [fp for fp, cb in file_checks.items() if cb.isChecked()]
            if not selected:
                return
            base_fp = file_paths[tabs.currentIndex()] if tabs.count() else selected[0]
            names_base = set(per_file_state[base_fp]["obj_map"].keys())
            if any(set(per_file_state[f]["obj_map"].keys()) != names_base for f in selected):
                status_label.setText("objects not identical")
                return
            st = per_file_state[base_fp]
            sel_objs = [n for n, cb in st["obj_vars"].items() if cb.isChecked()]
            if not sel_objs:
                QMessageBox.warning(dialog, "No Objects", "Select objects first")
                return
            sel_types = {st["obj_map"][n][0].typeName for n in sel_objs}
            if len(sel_types) != 1:
                QMessageBox.warning(dialog, "Type mismatch", "Selected objects are not the same type")
                return
            sel_vars = [v for v, cb in st["var_vars"].items() if cb.isChecked()]
            if not sel_vars:
                QMessageBox.warning(dialog, "No Variables", "Select variables first")
                return
            specs = []
            for obj_name in sel_objs:
                obj = st["obj_map"][obj_name][0]
                for var in sel_vars:
                    for ex, label in self._parse_extras(obj, st["extra_entry"].text()):
                        specs.append((obj_name, var, ex, label))
            for fp in selected:
                apply_specs[fp] = specs.copy()

                st_target = per_file_state[fp]
                for name, cb in st_target["obj_vars"].items():
                    cb.setChecked(name in sel_objs)
                st_target["rebuild"]()
                for var, cb in st_target["var_vars"].items():
                    cb.setChecked(var in sel_vars)
                st_target["extra_entry"].setText(st["extra_entry"].text())

                file_checks[fp].setChecked(False)
            status_label.setText(f"Stored selection for {len(selected)} file(s)")
            apply_btn.setEnabled(False)

        apply_btn.clicked.connect(apply_selection)

        reuse_cb = QCheckBox("Use this selection for all future OrcaFlex files")
        right_side.addWidget(reuse_cb)
        check_files()

        btn_layout = QHBoxLayout()
        load_btn = QPushButton("Load")
        cancel_btn = QPushButton("Cancel")
        btn_layout.addStretch()
        btn_layout.addWidget(load_btn)
        btn_layout.addWidget(cancel_btn)
        right_side.addLayout(btn_layout)

        out_specs = {}

        def on_load():
            self.orcaflex_redundant_subs = [
                s.strip() for s in redundant_entry.text().split(",") if s.strip()
            ]
            self._strip_coord_in_labels = strip_cb.isChecked()
            self._try_coord_wildcard = strip_cb.isChecked()
            rule_text = wc_entry.text().strip()
            if rule_text:
                if regex_cb.isChecked():
                    try:
                        pattern = re.compile(rule_text)
                        self._strip_rule = lambda n, p=pattern: p.sub("", n)
                    except re.error:
                        self._strip_rule = None
                else:
                    self._strip_rule = lambda n, t=rule_text: n.replace(t, "")
            else:
                self._strip_rule = None

            out_specs.update(apply_specs)

            for fp, st in per_file_state.items():
                if fp in out_specs:
                    continue
                sel_objs = [n for n, cb in st["obj_vars"].items() if cb.isChecked()]
                if not sel_objs:
                    continue
                sel_types = {st["obj_map"][n][0].typeName for n in sel_objs}
                if len(sel_types) != 1:
                    continue
                sel_vars = [v for v, cb in st["var_vars"].items() if cb.isChecked()]
                if not sel_vars:
                    continue
                specs = []
                for obj_name in sel_objs:
                    obj = st["obj_map"][obj_name][0]
                    for var in sel_vars:
                        for ex, label in self._parse_extras(obj, st["extra_entry"].text()):
                            specs.append((obj_name, var, ex, label))
                out_specs[fp] = specs

            if reuse_cb.isChecked() and file_paths:
                active_fp = file_paths[tabs.currentIndex()] if tabs.count() else None
                if active_fp in out_specs:
                    self._last_orcaflex_selection = out_specs[active_fp].copy()
                    self._reuse_orcaflex_selection = True

                    base_specs = self._last_orcaflex_selection
                    for fp in file_paths:
                        if fp == active_fp or fp in out_specs:
                            continue
                        st = per_file_state[fp]
                        mapped = []
                        obj_names = st["obj_map"].keys()
                        for obj_name, var, ex, label in base_specs:
                            target_name = obj_name
                            if target_name not in obj_names and getattr(self, "_strip_rule", None):
                                stripped = self._strip_rule(obj_name)
                                for cand in obj_names:
                                    if self._strip_rule(cand) == stripped:
                                        target_name = cand
                                        break
                            if target_name in obj_names:
                                mapped.append((target_name, var, ex, label))
                        if mapped:
                            out_specs[fp] = mapped

            missing_files = [fp for fp in file_paths if fp not in out_specs]
            if missing_files:
                resp = QMessageBox.question(
                    dialog,
                    "No Selection",
                    f"{len(missing_files)} file(s) have no selection. Continue?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No,
                )
                if resp != QMessageBox.Yes:
                    return

            dialog.accept()

        load_btn.clicked.connect(on_load)
        cancel_btn.clicked.connect(dialog.reject)

        if dialog.exec() != QDialog.Accepted:
            return {}

        result = {}
        for fp in file_paths:
            specs = out_specs.get(fp)
            tsdb = None
            if specs:
                tsdb = self._load_orcaflex_data_from_specs(
                    self.loaded_sim_models[fp], specs
                )
            if tsdb is None:
                tsdb = TsDB()

            entries = panel_pressure_by_file.get(fp, [])
            for entry in entries:
                tsdb = self._merge_panel_pressures(
                    tsdb, entry.get("data"), entry.get("info")
                )

            result[fp] = tsdb
        return result


    def _merge_panel_pressures(self, tsdb, pressures_df, panel_info):
        if tsdb is None:
            tsdb = TsDB()
        if pressures_df is None or pressures_df.empty:
            return tsdb
        if "Time" not in pressures_df.columns:
            return tsdb

        time = np.asarray(pressures_df["Time"].to_numpy())
        mapping = {}
        if panel_info is not None and hasattr(panel_info, "empty") and not panel_info.empty:
            for _, row in panel_info.iterrows():
                pidx = row.get("pidx")
                if pd.isna(pidx):
                    continue
                try:
                    key = str(int(pidx) + 1)
                except (TypeError, ValueError):
                    continue
                name = row.get("name")
                coord_val = row.get("input_coord")
                coord_txt = None
                if isinstance(coord_val, (tuple, list, np.ndarray)) and len(coord_val) == 3:
                    coord_txt = f"({coord_val[0]:.2f}, {coord_val[1]:.2f}, {coord_val[2]:.2f})"
                if name:
                    label = f"{name} Panel {int(pidx) + 1}"
                else:
                    label = f"Panel {int(pidx) + 1}"
                if coord_txt:
                    label = f"{label} Surface Pressure @ {coord_txt}"
                else:
                    label = f"{label} Surface Pressure"
                mapping[key] = label

        redundant = getattr(self, "orcaflex_redundant_subs", [])
        existing_names = {ts.name for ts in getattr(tsdb, "register", {}).values()}

        for col in pressures_df.columns:
            if col == "Time":
                continue
            data = pressures_df[col].to_numpy(dtype=float, copy=True)
            label = mapping.get(col, f"Panel {col} Surface Pressure")
            if redundant:
                label = self._strip_redundant(label, redundant)
            base_label = label
            suffix = 1
            while True:
                try:
                    tsdb.add(TimeSeries(label, time, data))
                    break
                except KeyError:
                    suffix += 1
                    label = f"{base_label} ({suffix})"
            existing_names.add(label)

        return tsdb


    def _strip_redundant(self, label, subs):
        for s in subs:
            label = label.replace(s, "")
        label = label.replace("__", "_").replace("::", ":").replace("  ", " ").strip("_:- ")
        return label

    def _resolve_orcaflex_line_end(self, end):
        try:
            import OrcFxAPI
        except ImportError:
            return None
        if end == "EndA":
            return OrcFxAPI.oeEndA
        elif end == "EndB":
            return OrcFxAPI.oeEndB
        elif isinstance(end, (float, int)):
            return OrcFxAPI.oeArcLength(end)
        else:
            return None

    def _parse_xyz_list(self, text: str):
        """Return list of xyz numpy arrays parsed from *text*."""
        import re
        coords = []
        pattern = re.compile(
            r"\(?\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)?"
        )
        for match in pattern.finditer(text):
            try:
                coords.append(
                    np.array(
                        [
                            float(match.group(1)),
                            float(match.group(2)),
                            float(match.group(3)),
                        ]
                    )
                )
            except ValueError:
                pass
        return coords

    def _get_closest_objects(self, coords, objects):
        """Return info on closest objects for each coordinate."""
        try:
            import OrcFxAPI
        except ImportError:
            return []

        import numpy as np

        if not coords or not objects:
            return []

        specs = []
        info = []
        for obj in objects:
            if obj.typeName == "Constraint":
                for v in ["In-frame X", "In-frame Y", "In-frame Z"]:
                    specs.append(OrcFxAPI.TimeHistorySpecification(obj, v, None))
                info.append((obj.Name, 1, "Constraint"))
            elif obj.typeName == "Line":
                n_nodes = len(obj.NodeArclengths)
                for n in range(n_nodes):
                    for v in ["X", "Y", "Z"]:
                        specs.append(
                            OrcFxAPI.TimeHistorySpecification(
                                obj, v, OrcFxAPI.oeNodeNum(n + 1)
                            )
                        )
                info.append((obj.Name, n_nodes, "Line"))

        if not specs:
            return []

        data = OrcFxAPI.GetMultipleTimeHistories(specs, OrcFxAPI.pnStaticState)[0]

        obj_data = {}
        idx = 0
        for name, count, typ in info:
            if typ == "Constraint":
                arr = np.array([data[idx : idx + 3]])
                idx += 3
            else:
                arr = np.array([
                    data[idx + i * 3 : idx + i * 3 + 3] for i in range(count)
                ])
                idx += 3 * count
            obj_data[name] = arr

        results = []
        for c in coords:
            best_name = None
            best_dist = None
            best_node = None
            for name, arr in obj_data.items():
                if arr.shape[0] == 1:
                    dist = np.linalg.norm(arr[0] - c)
                    node = None
                else:
                    dists = np.linalg.norm(arr - c, axis=1)
                    node_idx = int(np.argmin(dists))
                    dist = dists[node_idx]
                    node = node_idx + 1
                if best_dist is None or dist < best_dist:
                    best_dist = dist
                    best_name = name
                    best_node = node
                elif dist == best_dist:
                    if node is not None and best_node is not None:
                        if node < best_node:
                            best_name = name
                            best_node = node
            results.append((best_name, best_node, best_dist))

        return results

    def _get_panel_pressure(
        self,
        model,
        diffraction_model,
        time_window=None,
        kpa_to_pa=True,
        panel_coords=None,
        include_hydrostatic=True,
        simidx=0,
        work_folder=None,
        use_existing=True,
        ow_element_nth_resolution: int = 1,
    ):
        try:
            import OrcFxAPI
        except ImportError as exc:
            raise RuntimeError("OrcFxAPI not available") from exc

        if model is None:
            raise ValueError("model is required")
        if diffraction_model is None:
            raise ValueError("diffraction_model is required")

        if time_window is None:
            time_window = OrcFxAPI.SpecifiedPeriod(
                model.simulationStartTime, model.simulationStopTime
            )
        else:
            time_window = OrcFxAPI.SpecifiedPeriod(time_window[0], time_window[1])

        geometry = getattr(diffraction_model, "panelGeometry", None)
        if geometry is None:
            return pd.DataFrame(), pd.DataFrame()

        panel_geometry = [
            (res[0], res[1], res[3][0], res[3][1], res[3][2]) for res in geometry
        ]
        if not panel_geometry:
            return pd.DataFrame(), pd.DataFrame()

        pd_geo = pd.DataFrame(
            panel_geometry, columns=("idx_id", "name", "X", "Y", "Z")
        )
        pd_geo["pidx"] = pd_geo.index.values

        panel_array = pd.DataFrame()
        if panel_coords is None:
            panel_array = pd_geo.loc[pd_geo.idx_id != -1].copy()
            if panel_array.empty:
                time = model["General"].TimeHistory("Time", time_window)
                return pd.DataFrame({"Time": time}), panel_array
            panel_array["input_coord"] = list(
                zip(panel_array["X"], panel_array["Y"], panel_array["Z"])
            )
            panel_array["distance"] = np.nan
        else:
            coords_list = []
            for coord in panel_coords:
                if isinstance(coord, np.ndarray):
                    coord = coord.tolist()
                if coord is None:
                    continue
                try:
                    coords_list.append(tuple(float(v) for v in coord))
                except (TypeError, ValueError):
                    continue

            if not coords_list:
                time = model["General"].TimeHistory("Time", time_window)
                return pd.DataFrame({"Time": time}), pd.DataFrame()

            available = pd_geo.loc[pd_geo.idx_id != -1].copy()
            if available.empty:
                time = model["General"].TimeHistory("Time", time_window)
                return pd.DataFrame({"Time": time}), pd.DataFrame()

            rows = []
            for idx, pt in enumerate(coords_list):
                x, y, z = pt
                this_geo = available.copy()
                this_geo["distance"] = np.sqrt(
                    np.power(this_geo.X.values - x, 2)
                    + np.power(this_geo.Y.values - y, 2)
                    + np.power(this_geo.Z.values - z, 2)
                )
                best = this_geo.sort_values("distance").iloc[0]
                row_dict = best.to_dict()
                row_dict["distance"] = float(row_dict.get("distance", np.nan))
                row_dict["input_coord"] = tuple(pt)
                row_dict["input_index"] = idx
                rows.append(row_dict)

            if rows:
                panel_array = pd.DataFrame(rows)
                if ow_element_nth_resolution and ow_element_nth_resolution > 1:
                    panel_array = panel_array.iloc[::ow_element_nth_resolution]
                panel_array = panel_array.reset_index(drop=True)

        if panel_array is None or panel_array.empty:
            time = model["General"].TimeHistory("Time", time_window)
            return pd.DataFrame({"Time": time}), panel_array

        if "input_index" in panel_array.columns:
            panel_array = panel_array.sort_values("input_index").reset_index(drop=True)

        panel_array = panel_array.dropna(subset=["name"]).reset_index(drop=True)
        if panel_array.empty:
            time = model["General"].TimeHistory("Time", time_window)
            return pd.DataFrame({"Time": time}), panel_array

        if "pidx" in panel_array.columns:
            panel_array["pidx"] = panel_array["pidx"].astype(int)

        all_press = []
        unique_bodies = np.unique(panel_array["name"].values)

        for this_body in unique_bodies:
            press_df = None
            found_this_body_press = False
            if (
                use_existing
                and work_folder
                and isinstance(work_folder, str)
                and os.path.isdir(work_folder)
            ):
                for run_file in os.listdir(work_folder):
                    if (
                        f"of_pressures_sim_idx_{simidx}" in run_file
                        and this_body in run_file
                    ):
                        try:
                            press_df = pd.read_feather(
                                os.path.join(work_folder, run_file)
                            )
                            found_this_body_press = True
                            break
                        except Exception:
                            press_df = None
                            found_this_body_press = False
            if not found_this_body_press:
                panel_ids = panel_array.loc[panel_array.name == this_body, "pidx"].values
                if panel_ids.size == 0:
                    continue
                panel_press = model[this_body].PanelPressureTimeHistory(
                    diffraction=diffraction_model,
                    resultPanels=panel_ids,
                    period=time_window,
                    parameters=OrcFxAPI.PanelPressureParameters(
                        IncludeHydrostaticPressure=include_hydrostatic,
                        IncludeDiffractionPressure=True,
                        IncludeRadiationPressure=True,
                    ),
                )
                press_df = pd.DataFrame(
                    panel_press,
                    columns=np.int32(panel_ids + 1).astype(str),
                )
                if (
                    work_folder
                    and isinstance(work_folder, str)
                    and os.path.isdir(work_folder)
                ):
                    filename = (
                        f"of_pressures_sim_idx_{simidx}_body_{this_body}"
                        f"_resolution_{ow_element_nth_resolution}_"
                        f"{datetime.datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.feather"
                    )
                    try:
                        press_df.to_feather(os.path.join(work_folder, filename))
                    except Exception:
                        pass
            if press_df is not None:
                all_press.append(press_df)

        time = model["General"].TimeHistory("Time", time_window)
        if not all_press:
            return pd.DataFrame({"Time": time}), panel_array

        all_press = pd.concat(all_press, axis=1)
        if kpa_to_pa:
            all_press = all_press * 1000
        all_press["Time"] = time
        return all_press, panel_array

    def _parse_extras(self, obj, entry_val: str):
        """Interpret extra strings for an OrcaFlex object."""
        import OrcFxAPI

        entry_val = entry_val.strip()

        if obj.typeName == "Line":
            if not entry_val:
                return [(None, None)]
            tokens = [t.strip() for t in entry_val.split(",") if t.strip()]
            total = obj.NodeArclengths[-1]
            extras = []
            for t in tokens:
                low = t.lower().strip()
                m_node = re.match(r"node\s*(\d+)", low)
                m_arc = re.match(r"arc\s*(\d+(?:\.\d+)?)", low)
                if low == "enda":
                    extras.append((OrcFxAPI.oeArcLength(obj.NodeArclengths[0]), "EndA"))
                elif low == "endb":
                    extras.append((OrcFxAPI.oeArcLength(total), "EndB"))
                elif low in ("mid", "middle"):
                    extras.append((OrcFxAPI.oeArcLength(total / 2), "mid"))
                elif m_node:
                    num = int(m_node.group(1))
                    extras.append((OrcFxAPI.oeNodeNum(num), f"Node {num}"))
                elif m_arc:
                    val = float(m_arc.group(1))
                    extras.append((OrcFxAPI.oeArcLength(val), f"Arc {val}"))
                else:
                    try:
                        val = float(t)
                    except ValueError:
                        continue
                    extras.append((OrcFxAPI.oeArcLength(val), f"Arc {val}"))
            return extras

        if obj.typeName in ("Vessel", "Buoy", "Constraint", "Environment"):
            groups = [g.strip() for g in entry_val.split(";") if g.strip()]
            if not groups:
                groups = ["0,0,0"]
            extras = []
            for grp in groups:
                txt = grp
                if txt.lower().startswith("pos"):
                    txt = txt[3:].strip()
                xyz = [s.strip() for s in txt.strip("() ").split(",") if s.strip()]
                if len(xyz) != 3:
                    continue
                try:
                    coords = [float(v) for v in xyz]
                except ValueError:
                    continue
                label = f"Pos ({coords[0]:.2f}, {coords[1]:.2f}, {coords[2]:.2f})"
                if obj.typeName == "Vessel":
                    extras.append((OrcFxAPI.oeVessel(coords), label))
                elif obj.typeName == "Buoy":
                    extras.append((OrcFxAPI.oeBuoy(coords), label))
                elif obj.typeName == "Constraint":
                    extras.append((OrcFxAPI.oeConstraint(coords), label))
                elif obj.typeName == "Environment":
                    extras.append((OrcFxAPI.oeEnvironment(coords), label))
            return extras if extras else [(None, None)]

        return [(None, None)]

    def _load_orcaflex_data_from_specs(self, model, selection_specs):
        try:
            import OrcFxAPI
        except ImportError:
            print("OrcFxAPI not available. Cannot preload .sim files.")
            return
        tsdb = TsDB()
        time_spec = OrcFxAPI.SpecifiedPeriod(0, model.simulationTimeStatus.CurrentTime)
        time = model["General"].TimeHistory("Time", time_spec)
        object_var_map = {obj.Name: obj for obj in model.objects}
        def _match_obj(name):
            obj = object_var_map.get(name)
            if obj is None and getattr(self, "_strip_rule", None):
                stripped = self._strip_rule(name)
                for cand, o in object_var_map.items():
                    if self._strip_rule(cand) == stripped:
                        obj = o
                        break
            return obj
        resolved_specs = []
        names = []
        redundant_subs = getattr(self, "orcaflex_redundant_subs", [])
        for spec in selection_specs:
            try:
                label_override = None
                if len(spec) == 5:
                    obj, obj_name, var, end, label_override = spec
                elif len(spec) == 4:
                    if isinstance(spec[0], str):
                        obj_name, var, end, label_override = spec
                        obj = _match_obj(obj_name)
                    else:
                        obj, obj_name, var, end = spec
                elif len(spec) == 3:
                    obj_name, var, end = spec
                    obj = _match_obj(obj_name)
                else:
                    continue
                if obj is None:
                    continue
                short_obj = self._strip_redundant(obj_name, redundant_subs)
                short_var = self._strip_redundant(var, redundant_subs)
                if obj.typeName == "Line":
                    if end == "EndA" or (isinstance(end, str) and end.lower() == "enda"):
                        end_enum = self._resolve_orcaflex_line_end("EndA")
                        try:
                            spec_obj = OrcFxAPI.TimeHistorySpecification(model[obj_name], var, end_enum)
                            label = (
                                f"{short_obj}:{short_var} ({label_override})"
                                if label_override
                                else f"{short_obj}:{short_var} (EndA)"
                            )
                            resolved_specs.append(spec_obj)
                            names.append(label)
                        except Exception:
                            continue
                    elif end == "EndB" or (isinstance(end, str) and end.lower() == "endb"):
                        end_enum = self._resolve_orcaflex_line_end("EndB")
                        try:
                            spec_obj = OrcFxAPI.TimeHistorySpecification(model[obj_name], var, end_enum)
                            label = (
                                f"{short_obj}:{short_var} ({label_override})"
                                if label_override
                                else f"{short_obj}:{short_var} (EndB)"
                            )
                            resolved_specs.append(spec_obj)
                            names.append(label)
                        except Exception:
                            continue
                    elif hasattr(OrcFxAPI, "ObjectExtra") and isinstance(end, OrcFxAPI.ObjectExtra):
                        arc_val = getattr(end, 'ArcLength', None)
                        if label_override:
                            label = f"{short_obj}:{short_var} ({label_override})"
                        elif arc_val is not None:
                            label = f"{short_obj}:{short_var} (Arc {arc_val:.2f})"
                        else:
                            label = f"{short_obj}:{short_var} (extra {end})"
                        try:
                            spec_obj = OrcFxAPI.TimeHistorySpecification(model[obj_name], var, end)
                            resolved_specs.append(spec_obj)
                            names.append(label)
                        except Exception:
                            continue
                    elif isinstance(end, (float, int)):
                        try:
                            extra = OrcFxAPI.oeArcLength(end)
                            label = (
                                f"{short_obj}:{short_var} ({label_override})"
                                if label_override
                                else f"{short_obj}:{short_var} (Arc {end:.2f})"
                            )
                            spec_obj = OrcFxAPI.TimeHistorySpecification(model[obj_name], var, extra)
                            resolved_specs.append(spec_obj)
                            names.append(label)
                        except Exception:
                            continue
                    elif isinstance(end, str):
                        try:
                            end_val = float(end)
                            extra = OrcFxAPI.oeArcLength(end_val)
                            label = (
                                f"{short_obj}:{short_var} ({label_override})"
                                if label_override
                                else f"{short_obj}:{short_var} (Arc {end_val:.2f})"
                            )
                            spec_obj = OrcFxAPI.TimeHistorySpecification(model[obj_name], var, extra)
                            resolved_specs.append(spec_obj)
                            names.append(label)
                        except Exception:
                            continue
                    else:
                        continue
                elif obj.typeName in ("Vessel", "Buoy", "Constraint", "Environment") and hasattr(OrcFxAPI, "ObjectExtra") and isinstance(end, OrcFxAPI.ObjectExtra):
                    try:
                        spec_obj = OrcFxAPI.TimeHistorySpecification(model[obj_name], var, end)
                        label = (
                            f"{short_obj}:{short_var} ({label_override})"
                            if label_override
                            else f"{short_obj}:{short_var} (pos {end})"
                        )
                        resolved_specs.append(spec_obj)
                        names.append(label)
                    except Exception:
                        continue
                else:
                    try:
                        spec_obj = OrcFxAPI.TimeHistorySpecification(model[obj_name], var)
                        label = f"{short_obj}:{short_var}"
                        resolved_specs.append(spec_obj)
                        names.append(label)
                    except Exception:
                        continue
            except Exception:
                continue
        try:
            results = OrcFxAPI.GetMultipleTimeHistories(resolved_specs, time_spec)
            for i, name in enumerate(names):
                tsdb.add(TimeSeries(name, time, results[:, i]))
            return tsdb
        except Exception as e:
            QMessageBox.critical(self.parent_gui, "OrcaFlex Read Error", f"Could not read variables:\n{e}")
            return None

    def _load_generic_file(self, filepath):
        ext = os.path.splitext(filepath)[-1].lower().lstrip(".")
        if ext in ["csv", 'mat', 'dat', 'ts',  'h5', 'pickle', 'tda', 'asc', 'tdms', 'pkl', 'bin']:
            return TsDB.fromfile(filepath)
        elif ext == "xlsx":
            df = pd.read_excel(filepath)
        elif ext == "json":
            df = pd.read_json(filepath)
        elif ext == "feather":
            df = pd.read_feather(filepath)
        elif ext == "parquet":
            df = pd.read_parquet(filepath)
        else:
            raise NotImplementedError(f"No loader for extension: {ext}")
        # Detect time column
        time_col = next((c for c in df.columns if c.lower() in ["time", "t"]), df.columns[0])
        time = df[time_col].values
        tsdb = TsDB()
        skipped = set()

        # Detect potential identifier columns with string values
        id_col = None

        string_cols = [
            c
            for c in df.columns
            if c != time_col
            and pd.api.types.is_string_dtype(df[c])
            and df[c].map(
                lambda x: isinstance(x, str)
                or (not hasattr(x, "__iter__") and pd.isna(x))
            ).all()
        ]

        for sc in string_cols:
            resp = QMessageBox.question(
                self.parent_gui,
                "Identifier Column?",
                f"Column '{sc}' contains strings. Use as identifier?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if resp == QMessageBox.Yes:
                id_col = sc
                break
            else:
                skipped.add(sc)


        if id_col:
            split_columns = {}
            for ident, subdf in df.groupby(id_col):
                time_vals = subdf[time_col].values
                for col in df.columns:

                    if col in (time_col, id_col):
                        continue
                    # ensure any pyarrow/extension values are converted to
                    # regular Python objects before further inspection
                    values = []

                    for v in subdf[col].tolist():
                        if hasattr(v, "to_pylist"):
                            v = v.to_pylist()
                        elif isinstance(v, array):
                            v = list(v)
                        elif isinstance(v, np.ndarray):
                            v = v.tolist()
                        values.append(v)

                    if col not in split_columns:
                        # consider only non-null entries when checking for list-like values
                        non_null = []
                        for v in values:
                            if isinstance(v, Sequence) and not isinstance(v, (str, bytes)):
                                non_null.append(v)
                            elif not pd.isna(v):
                                non_null.append(v)
                        if non_null and all(
                            isinstance(v, Sequence) and not isinstance(v, (str, bytes))
                            for v in non_null
                        ):
                            lengths = {len(v) for v in non_null}
                            if len(lengths) == 1:
                                n = lengths.pop()
                                resp = QMessageBox.question(
                                    self.parent_gui,
                                    "Split Column?",
                                    f"Column '{col}' contains list/tuple values of length {n}.\nSplit into {n} columns?",
                                    QMessageBox.Yes | QMessageBox.No,
                                    QMessageBox.Yes,
                                )
                                if resp == QMessageBox.Yes:
                                    name_str, ok = QInputDialog.getText(
                                        self.parent_gui,
                                        "Column Names",
                                        f"Enter {n} comma-separated names for '{col}':",
                                    )
                                    if ok:
                                        names = [s.strip() for s in name_str.split(",") if s.strip()]
                                    else:
                                        names = []
                                    if len(names) != n:
                                        names = [f"{col}_{i+1}" for i in range(n)]
                                    split_columns[col] = names
                                else:
                                    split_columns[col] = None
                            else:
                                split_columns[col] = None
                        else:
                            split_columns[col] = None
                    names = split_columns.get(col)
                    if names:
                        for i in range(len(names)):
                            row_vals = []
                            for row in values:
                                if isinstance(row, Sequence) and not isinstance(row, (str, bytes)):
                                    try:
                                        row_vals.append(row[i])
                                    except Exception:
                                        row_vals.append(np.nan)
                                else:
                                    row_vals.append(np.nan)
                            try:
                                data = np.array(row_vals, dtype=float)
                            except Exception:
                                skipped.add(f"{col}_{i}")
                                continue

                            tsdb.add(TimeSeries(f"{names[i]}_{ident}", time_vals, data))
                        continue
                    try:
                        numeric_values = np.array(values, dtype=float)
                        tsdb.add(TimeSeries(f"{col}_{ident}", time_vals, numeric_values))
                    except Exception:
                        skipped.add(col)

        else:
            for col in df.columns:
                if col == time_col:
                    continue
                # Convert potential extension array values to regular Python
                values = []
                for v in df[col].tolist():
                    if hasattr(v, "to_pylist"):
                        v = v.to_pylist()
                    elif isinstance(v, array):
                        v = list(v)
                    elif isinstance(v, np.ndarray):
                        v = v.tolist()
                    values.append(v)
                # Consider only non-null entries when checking for list-like values
                non_null = []
                for v in values:
                    if isinstance(v, Sequence) and not isinstance(v, (str, bytes)):
                        non_null.append(v)
                    elif not pd.isna(v):
                        non_null.append(v)
                if non_null and all(
                    isinstance(v, Sequence) and not isinstance(v, (str, bytes))
                    for v in non_null
                ):
                    lengths = {len(v) for v in non_null}
                    if len(lengths) == 1:
                        n = lengths.pop()
                        resp = QMessageBox.question(
                            self.parent_gui,
                            "Split Column?",
                            f"Column '{col}' contains list/tuple values of length {n}.\nSplit into {n} columns?",
                            QMessageBox.Yes | QMessageBox.No,
                            QMessageBox.Yes,
                        )
                        if resp == QMessageBox.Yes:
                            name_str, ok = QInputDialog.getText(
                                self.parent_gui,
                                "Column Names",
                                f"Enter {n} comma-separated names for '{col}':",
                            )
                            if ok:
                                names = [s.strip() for s in name_str.split(",") if s.strip()]
                            else:
                                names = []
                            if len(names) != n:
                                names = [f"{col}_{i+1}" for i in range(n)]
                            for i in range(n):
                                row_vals = []
                                for row in values:
                                    if isinstance(row, Sequence) and not isinstance(row, (str, bytes)):
                                        try:
                                            row_vals.append(row[i])
                                        except Exception:
                                            row_vals.append(np.nan)
                                    else:
                                        row_vals.append(np.nan)
                                try:
                                    data = np.array(row_vals, dtype=float)
                                except Exception:
                                    skipped.add(f"{col}_{i}")
                                    continue

                                tsdb.add(TimeSeries(names[i], time, data))
                            continue
                try:
                    numeric_values = np.array(values, dtype=float)
                    tsdb.add(TimeSeries(col, time, numeric_values))
                except Exception:
                    skipped.add(col)

        if len(tsdb.getm()) == 0:
            if 'time' in df.columns or 't' in df.columns:
                time_col = next((c for c in df.columns if c.lower() in ["time", "t"]), df.columns[0])
                time = df[time_col].values
            else:
                time = np.arange(len(df))
            tsdb.add(TimeSeries("NO_DATA", time, np.full_like(time, np.nan, dtype=float)))
        if skipped:
            print(
                f"Skipped non-numeric columns in {os.path.basename(filepath)}: {', '.join(sorted(skipped))}"
            )
        return tsdb

def get_object_available_vars(obj, orcaflex_varmap=None):
    if orcaflex_varmap is not None:
        return orcaflex_varmap.get(obj.typeName, [])
    for attr in ["AvailableTimeHistories", "AvailableDerivedVariables"]:
        if hasattr(obj, attr):
            try:
                vals = getattr(obj, attr)
                if isinstance(vals, (list, tuple)):
                    return list(vals)
                elif hasattr(vals, "__iter__"):
                    return list(vals)
            except Exception:
                continue
    if hasattr(obj, "AvailableVariables"):
        try:
            vals = obj.AvailableVariables
            if isinstance(vals, (list, tuple)):
                return list(vals)
            elif hasattr(vals, "__iter__"):
                return list(vals)
        except Exception:
            pass
    return [k for k in dir(obj) if not k.startswith("__")]



def main():
    """Launch the AnytimeSeries GUI."""
    app = QApplication(sys.argv)
    window = TimeSeriesEditorQt()
    window.resize(1400, 800)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
