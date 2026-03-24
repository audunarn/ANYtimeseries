"""Main Qt window for the AnytimeSeries application."""
from __future__ import annotations

import datetime
import json
import multiprocessing
import os
import re
import subprocess
import sys
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import nullcontext
from array import array
from collections.abc import Callable, Sequence

import anyqats as qats
import numpy as np
import pandas as pd
import scipy.io
from tqdm.auto import tqdm
from anyqats import TimeSeries, TsDB
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PySide6.QtCore import QEvent, QTimer, Qt, QUrl, Signal, Slot
from PySide6.QtGui import (
    QColor,
    QGuiApplication,
    QKeyEvent,
    QKeySequence,
    QPalette,
    QTextCursor,
)
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QStyleFactory,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QHeaderView,
    QSizePolicy,
    QSpacerItem,
)

from .file_loader import FileLoader
from .layout_utils import apply_initial_size
from .stats_dialog import StatsDialog
from .evm_window import EVMWindow
from .rao_dialog import RAODialog
from ..fatigue import FatigueSeries
from .fatigue_dialog import FatigueDialog
from .sortable_table_widget_item import SortableTableWidgetItem
from .variable_tab import VariableRowWidget, VariableTab


_TRANSFORM_SPEC = None




def _tqdm_progress(iterable, total, desc):
    """Return a tqdm progress iterator when possible."""
    if total <= 0:
        return nullcontext(iterable)
    return tqdm(iterable, total=total, desc=desc, leave=False)

def _evaluate_calculator_task(file_idx, task):
    """Evaluate one calculator expression for a single file payload."""
    time_window = np.asarray(task["time_window"], dtype=float)
    ctx = dict(task.get("shared_ctx") or {})
    ctx.update(task.get("file_ctx") or {})
    ctx["time"] = time_window
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

    exec(task["exec_expr"], ctx)
    y = np.asarray(ctx[task["base_output"]], dtype=float)
    if y.ndim == 0:
        y = np.full_like(time_window, y, dtype=float)
    if len(y) != len(time_window):
        raise ValueError("Result length mismatch with time vector")
    return file_idx, y


def _init_transform_worker(transform_spec):
    """Initialize transformation worker state for multiprocessing."""
    global _TRANSFORM_SPEC
    _TRANSFORM_SPEC = transform_spec


def _rolling_mean(y_values, window):
    """Return a trailing rolling mean with min_periods=1."""
    y = np.asarray(y_values, dtype=float)
    if window <= 1 or y.size <= 1:
        return y.copy()
    cumsum = np.cumsum(np.insert(y, 0, 0.0))
    counts = np.minimum(np.arange(1, y.size + 1), window)
    totals = cumsum[window:] - cumsum[:-window]
    prefix = np.cumsum(y[: window - 1])
    return np.concatenate((prefix / counts[: window - 1], totals / counts[window - 1 :]))


def _apply_transform_spec(y_values, transform_spec):
    """Apply a serializable quick-transformation specification."""
    y = np.asarray(y_values, dtype=float)
    kind = transform_spec["kind"]
    if kind == "abs":
        return np.abs(y)
    if kind == "scale":
        return y * float(transform_spec["factor"])
    if kind == "offset":
        return y + float(transform_spec["value"])
    if kind == "radians":
        return np.radians(y)
    if kind == "degrees":
        return np.degrees(y)
    if kind == "rolling_mean":
        return _rolling_mean(y, int(transform_spec["window"]))
    if kind == "trig_scale":
        return y * float(transform_spec["factor"])
    if kind == "shift_min_to_zero":
        lower = np.sort(y)[int(len(y) * 0.01)] if transform_spec.get("ignore_anomalies") and len(y) else np.min(y)
        return y if lower >= 0 else y - lower
    if kind == "shift_repeated_neg_min":
        if y.size == 0:
            return y
        ymin = y.min()
        if ymin >= 0:
            return y
        tol_abs = abs(ymin) * float(transform_spec["tol_pct"])
        plate_cnt = np.count_nonzero(np.abs(y - ymin) <= tol_abs)
        return y - ymin if plate_cnt >= int(transform_spec["min_count"]) else y
    if kind == "shift_mean_to_zero":
        if transform_spec.get("ignore_anomalies") and y.size:
            p01, p99 = np.percentile(y, [1, 99])
            mask = (y >= p01) & (y <= p99)
            mean_value = np.mean(y[mask]) if np.any(mask) else np.mean(y)
        else:
            mean_value = np.mean(y)
        return y - mean_value
    raise ValueError(f"Unknown transformation kind: {kind}")


def _evaluate_transform_payload(task_idx, y_values):
    """Evaluate one quick transformation payload."""
    return task_idx, _apply_transform_spec(y_values, _TRANSFORM_SPEC)

from .utils import (
    MATH_FUNCTIONS,
    ORCAFLEX_VARIABLE_MAP,
    _find_xyz_triples,
    _looks_like_user_var,
    _matches_terms,
    _parse_search_terms,
    _safe,
)


class TimeSeriesEditorQt(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AnytimeSeries - time series editor (Qt/PySide6)")

        self._min_left_panel = 320
        self._min_right_panel = 360
        self._splitter_ratio = 0.52

        self._updating_splitter = False

        # Palette and style for theme switching
        app = QApplication.instance()
        self.default_palette = app.palette()
        self.default_style = app.style().objectName()
        # Reuse a single style instance when toggling themes to avoid
        # crashes from Python garbage-collecting temporary QStyle objects
        self._fusion_style = QStyleFactory.create("Fusion")

        # Track the latest embedded plot so theme toggles can refresh it for
        # non-matplotlib engines without reloading the entire UI.
        self._last_plot_call: tuple[Callable[..., None], tuple, dict] | None = None
        self._refreshing_plot = False
        self._marker_input_auto_value = ""

        # =======================
        # DATA STRUCTURES
        # =======================
        self.tsdbs = []  # List of anyqats.TsDB instances (one per file)
        self.file_paths = []  # List of file paths (order matches tsdbs)
        self.user_variables = set()  # User-defined/calculated variables
        self.common_lookup = {}  # Map safe common names -> per-file variable names

        self.var_checkboxes = {}  # key: variable key → QCheckBox
        self.var_offsets = {}  # key: variable key → QLineEdit for numeric offset

        # These lists must be filled before refresh_variable_tabs()
        self.common_var_keys = []  # e.g. ["Heave", "Surge"]
        self.file_var_keys = {}  # dict: file name → [var1, var2, ...]
        self.user_var_keys = []  # e.g. ["result_var1", ...]
        self.var_labels = {}  # Optional: key → display label

        self.file_loader = FileLoader(
            orcaflex_varmap=ORCAFLEX_VARIABLE_MAP,
            parent_gui=self,
        )
        # Progress updates while loading files
        self.file_loader.progress_callback = self.update_progressbar

        # =======================
        # LAYOUT: MAIN SPLITTER
        # =======================
        self.main_splitter = QSplitter(Qt.Horizontal)
        self.main_splitter.setChildrenCollapsible(False)
        self.main_splitter.setHandleWidth(6)

        # -----------------------
        # LEFT: Variable Tabs
        # -----------------------
        left_widget = QWidget()
        left_widget.setMinimumWidth(self._min_left_panel)
        # Allow the variable panel to grow when the splitter handle is dragged.
        # ``Preferred`` prevented the widget from expanding even though the
        # splitter reported the new size, resulting in the left pane snapping
        # back to its original width. ``Expanding`` makes the panel honour the
        # splitter geometry updates while keeping the existing minimum width.
        left_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        left_layout = QVBoxLayout(left_widget)

        # Quick navigation buttons
        btn_row = QHBoxLayout()
        btn_row.setContentsMargins(0, 0, 0, 0)
        btn_row.setSpacing(6)
        self.goto_common_btn = QPushButton("Go to Common")
        self.goto_user_btn = QPushButton("Go to User Variables")
        self.unselect_all_btn = QPushButton("Unselect All")
        self.select_pos_btn = QPushButton("Select all by list pos.")

        nav_buttons = (
            self.goto_common_btn,
            self.goto_user_btn,
            self.unselect_all_btn,
            self.select_pos_btn,
        )

        for btn in nav_buttons:
            btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            btn.setMinimumHeight(28)
            btn_row.addWidget(btn)
        left_layout.addLayout(btn_row)

        # Tab widget for variables (common, per-file, user)
        self.tabs = QTabWidget()
        self.tabs.setMinimumWidth(self._min_left_panel)
        self.tabs.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        left_layout.addWidget(self.tabs)

        self.main_splitter.addWidget(left_widget)

        # -----------------------
        # RIGHT: Controls and Analysis
        # -----------------------
        right_widget = QWidget()
        right_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        # Use a vertical layout so an optional embedded plot can span

        # the full width below the control sections when embedded

        self.right_outer_layout = QVBoxLayout(right_widget)
        self.top_row_layout = QHBoxLayout()
        self.right_outer_layout.addLayout(self.top_row_layout)

        self.controls_widget = QWidget()
        self.controls_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
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

        row_trig = QHBoxLayout()
        row_trig.addWidget(QLabel("Trig:"))
        self.trig_combo = QComboBox()
        self.trig_combo.addItems(["sin", "cos", "tan"])
        row_trig.addWidget(self.trig_combo)
        row_trig.addWidget(QLabel("Angle [deg]:"))
        self.trig_angle_entry = QLineEdit()
        self.trig_angle_entry.setPlaceholderText("0")
        self.trig_angle_entry.setFixedWidth(80)
        row_trig.addWidget(self.trig_angle_entry)
        self.trig_calc_btn = QPushButton("Calculate")
        row_trig.addWidget(self.trig_calc_btn)
        self.reduction_pct_entry = QLineEdit("100")
        self.reduction_pct_entry.setFixedWidth(70)
        self.reduction_pct_entry.setToolTip(
            "Percentage of points to keep (0 = no points, 100 = all points)."
        )
        row_trig.addWidget(self.reduction_pct_entry)
        row_trig.addWidget(QLabel("Bias:"))
        self.reduction_bias_combo = QComboBox()
        self.reduction_bias_combo.addItems(["Mean", "Upper", "Lower"])
        self.reduction_bias_combo.setToolTip(
            "Mean uses local averages, Upper uses local maxima, and Lower uses local minima."
        )
        row_trig.addWidget(self.reduction_bias_combo)
        self.reduce_points_btn = QPushButton("Reduce Points")
        row_trig.addWidget(self.reduce_points_btn)
        row_trig.addStretch(1)
        transform_layout.addLayout(row_trig)

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
        self.merge_selected_btn = QPushButton("Merge Selected")
        row4.addWidget(self.sqrt_sum_btn)
        row4.addWidget(self.mean_of_sel_btn)
        row4.addWidget(self.abs_btn)
        row4.addWidget(self.rolling_avg_btn)
        row4.addWidget(self.merge_selected_btn)
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
            "Shift Min -> 0"
        )
        self.shift_common_max_btn = QPushButton(
            "Common Shift Min -> 0"
        )
        row5.addWidget(self.shift_min_nz_btn)
        row5.addWidget(self.shift_common_max_btn)
        transform_layout.addLayout(row5)

        row6 = QHBoxLayout()
        self.shift_x_start_zero_btn = QPushButton("Shift X Start → 0")
        self.shift_x_start_zero_btn.setToolTip(
            "Create a new series where the x-axis starts at zero by subtracting the initial x value."
        )
        row6.addWidget(self.shift_x_start_zero_btn)
        row6.addStretch(1)
        transform_layout.addLayout(row6)


        # Progress bar is shown by itself unless the plot is embedded
        self.controls_layout.addWidget(self.progress)
        # Row used when embedding the plot to move transformations next to the
        # progress bar
        self.progress_transform_row = QHBoxLayout()

        # ---- Offset Group ----
        offset_group = QGroupBox("Apply operation from variable input fields")
        offset_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        offset_layout = QVBoxLayout(offset_group)
        offset_examples = QLabel('Examples: add "+1 / 1" substract "-1" divide "/2" multiply "*2"   '
                                 'To plot 2D scatter input "x" and "y" in the fields. For 3D scatter also input "z".')
        offset_examples.setWordWrap(True)
        offset_layout.addWidget(offset_examples)
        self.apply_value_user_var_cb = QCheckBox("Create user variable instead of overwriting?")
        offset_layout.addWidget(self.apply_value_user_var_cb)
        self.apply_values_btn = QPushButton("Apply Values")
        self.plot_marked_axes_btn = QPushButton("Plot X/Y(/Z)")
        self.animate_marked_axes_btn = QPushButton("Animate X/Y(/Z)")
        apply_plot_row = QHBoxLayout()
        apply_plot_row.addWidget(self.apply_values_btn)
        apply_plot_row.addWidget(self.plot_marked_axes_btn)
        apply_plot_row.addWidget(self.animate_marked_axes_btn)
        offset_layout.addLayout(apply_plot_row)
        self.controls_layout.addWidget(offset_group)

        # ---- File list group ----
        file_group = QGroupBox("Loaded Files")
        file_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        file_list_layout = QVBoxLayout(file_group)
        self.file_list = QListWidget()
        self.file_list.setMinimumWidth(160)
        self.remove_file_btn = QPushButton("Remove File")
        file_list_layout.addWidget(self.file_list)
        file_list_layout.addWidget(self.remove_file_btn)
        self.controls_layout.addWidget(file_group)

        # ---- Time window controls ----
        time_group = QGroupBox("Time Window (for Plot/Stats/Transform)")
        time_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
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
        self.freq_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        freq_layout = QGridLayout(self.freq_group)
        self.filter_none_rb = QRadioButton("None")
        self.filter_lowpass_rb = QRadioButton("Low-pass")
        self.filter_highpass_rb = QRadioButton("High-pass")
        self.filter_bandpass_rb = QRadioButton("Band-pass")
        self.filter_bandblock_rb = QRadioButton("Band-block")
        self.filter_none_rb.setChecked(True)
        self.lowpass_cutoff = QLineEdit("0.04")
        self.highpass_cutoff = QLineEdit("0.04")
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
        self.tools_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        tools_layout = QHBoxLayout(self.tools_group)
        self.launch_qats_btn = QPushButton("Open in AnyQATS")
        self.evm_tool_btn = QPushButton("Open Extreme Value Statistics Tool")
        self.rao_tool_btn = QPushButton("Generate RAO from Selected Time Series")
        tools_layout.addWidget(self.launch_qats_btn)
        tools_layout.addWidget(self.evm_tool_btn)
        tools_layout.addWidget(self.rao_tool_btn)
        self.controls_layout.addWidget(self.tools_group)

        # ---- Plot controls ----
        self.plot_group = QGroupBox("Plot Controls")
        self.plot_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

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
        self.plot_datetime_x_cb = QCheckBox("Datetime x-axis (if possible)")
        plot_btn_row.addWidget(self.plot_raw_cb)
        plot_btn_row.addWidget(self.plot_lowpass_cb)
        plot_btn_row.addWidget(self.plot_highpass_cb)
        plot_btn_row.addWidget(self.plot_datetime_x_cb)
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

        # Rolling mean window + x-axis marker
        rolling_row = QHBoxLayout()
        rolling_row.addWidget(QLabel("Rolling mean window:"))
        self.rolling_window = QSpinBox()
        self.rolling_window.setMinimum(1)
        self.rolling_window.setMaximum(1000000)

        self.rolling_window.setValue(1)
        rolling_row.addWidget(self.rolling_window)
        rolling_row.addWidget(QLabel("X-axis marker:"))
        self.x_axis_marker_input = QLineEdit()
        rolling_row.addWidget(self.x_axis_marker_input)
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
        self.analysis_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
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
        self._mpl_toolbar = None
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
        self.main_splitter.addWidget(right_widget)
        self.main_splitter.setStretchFactor(0, 1)
        self.main_splitter.setStretchFactor(1, 2)

        # ---- Set main container ----
        container = QWidget()
        container.setAutoFillBackground(True)
        container_layout = QHBoxLayout(container)
        container_layout.addWidget(self.main_splitter)
        self.setCentralWidget(container)
        self.setAutoFillBackground(True)

        self.main_splitter.splitterMoved.connect(self._on_splitter_moved)
        self._configure_initial_geometry()

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
        self.plot_marked_axes_btn.clicked.connect(self.plot_marked_axes)
        self.animate_marked_axes_btn.clicked.connect(self.animate_marked_axes)
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
        self.merge_selected_btn.clicked.connect(self.merge_selected_series)
        self.radians_btn.clicked.connect(self.to_radians)
        self.degrees_btn.clicked.connect(self.to_degrees)
        self.trig_calc_btn.clicked.connect(self.apply_trig_from_degrees)
        self.reduce_points_btn.clicked.connect(self.reduce_selected_points)
        self.shift_min0_btn.clicked.connect(self.shift_min_to_zero)
        self.shift_mean0_btn.clicked.connect(self.shift_mean_to_zero)
        self.save_btn.clicked.connect(self.save_files)
        self.save_values_btn.clicked.connect(self.save_entry_values)
        self.load_values_btn.clicked.connect(self.load_entry_values)
        self.export_csv_btn.clicked.connect(self.export_selected_to_csv)
        self.shift_min_nz_btn.clicked.connect(self.shift_repeated_neg_min)
        self.shift_common_max_btn.clicked.connect(self.shift_common_max)
        self.shift_x_start_zero_btn.clicked.connect(self.shift_x_start_to_zero)
        self.launch_qats_btn.clicked.connect(self.launch_qats)
        self.evm_tool_btn.clicked.connect(self.open_evm_tool)
        self.rao_tool_btn.clicked.connect(self.open_rao_tool)
        self.reselect_orcaflex_btn.clicked.connect(self.reselect_orcaflex_variables)
        self.psd_btn.clicked.connect(lambda: self.plot_selected(mode="psd"))
        self.cycle_range_btn.clicked.connect(lambda: self.plot_selected(mode="cycle"))
        self.cycle_mean_btn.clicked.connect(lambda: self.plot_selected(mode="cycle_rm"))
        self.cycle_mean3d_btn.clicked.connect(lambda: self.plot_selected(mode="cycle_rm3d"))
        self.plot_rolling_btn.clicked.connect(lambda: self.plot_selected(mode="rolling"))

        self.theme_switch.stateChanged.connect(self.toggle_dark_theme)
        self.embed_plot_cb.stateChanged.connect(self.toggle_embed_layout)
        self.plot_engine_combo.currentTextChanged.connect(self._on_engine_changed)
        self.plot_datetime_x_cb.stateChanged.connect(self._refresh_marker_input_defaults)
        self.time_start.textChanged.connect(self._refresh_marker_input_defaults)
        self.time_end.textChanged.connect(self._refresh_marker_input_defaults)

        # ==== Populate variable tabs on startup ====
        self.refresh_variable_tabs()
        self._refresh_marker_input_defaults()
        # Apply the light palette by default
        self.apply_light_palette()
        #self.apply_dark_palette()
        #self.theme_switch.setChecked(True)
        self.toggle_embed_layout('')
        self.embed_plot_cb.setChecked(True)

    def _configure_initial_geometry(self) -> None:
        """Size the window and splitter based on the current screen."""

        apply_initial_size(
            self,
            desired_width=1400,
            desired_height=900,
            min_width=880,
            min_height=640,
            width_ratio=0.92,
            height_ratio=0.9,
        )

        self._apply_splitter_ratio()
        QTimer.singleShot(0, self._apply_splitter_ratio)

    def _apply_splitter_ratio(self) -> None:
        """Keep the main splitter proportions responsive when resizing."""

        if not hasattr(self, "main_splitter"):
            return

        total = self.main_splitter.size().width()
        if total < 2:
            return

        left = int(total * self._splitter_ratio)
        left = max(self._min_left_panel, left)

        if total - left < self._min_right_panel:
            left = max(self._min_left_panel, total - self._min_right_panel)

        left = max(1, min(left, total - 1))
        right = max(1, total - left)

        self._updating_splitter = True
        try:
            self.main_splitter.setSizes([left, right])
        finally:
            self._updating_splitter = False

    def _on_splitter_moved(self, _pos: int, _index: int) -> None:
        if self._updating_splitter:
            return

        sizes = self.main_splitter.sizes()
        total = sum(sizes)
        if not total:
            return

        ratio = sizes[0] / total
        self._splitter_ratio = max(0.15, min(0.85, ratio))

    def resizeEvent(self, event):  # type: ignore[override]
        super().resizeEvent(event)
        self._apply_splitter_ratio()

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
        if self.common_lookup:
            for key in sorted(self.common_lookup):
                safe = f"c_{key}"
                if safe not in self.calc_variables:
                    self.calc_variables.append(safe)
                    self.calc_var_filemap[safe] = "common"
        for key in getattr(self, "user_variables", set()):
            safe = f"u_{_safe(key)}"
            if safe not in self.calc_variables:
                filename = next(
                    (os.path.basename(fp) for tsdb, fp in zip(self.tsdbs, self.file_paths) if key in tsdb.getm()),
                    "user variable")
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

    def _format_calculator_equation(self, expr: str, output_names: Sequence[str] | None = None) -> str:
        """Return readable equation lines for the created calculator output(s)."""
        lhs, sep, rhs = expr.partition("=")
        if not sep:
            return expr.strip()

        lhs = lhs.strip()
        rhs_lines = [line.strip() for line in rhs.splitlines() if line.strip()]
        if not rhs_lines:
            rhs_lines = [""]

        names = list(output_names) if output_names else [lhs]
        formatted = []
        for name in names:
            if len(rhs_lines) == 1:
                formatted.append(f"{name} = {rhs_lines[0]}")
            else:
                formatted.append(f"{name} =\n    " + "\n    ".join(rhs_lines))
        return "\n\n".join(formatted)

    def _auto_calculator_output_name(self, expr: str) -> str:
        """Create a compact unique user-variable name from a bare calculator expression."""
        stem_src = expr
        stem_src = re.sub(r"\bc_", "cc_", stem_src)
        for old, new in (
            ("**", " pow "),
            ("+", " p "),
            ("-", " m "),
            ("*", " x "),
            ("/", " d "),
            ("%", " mod "),
        ):
            stem_src = stem_src.replace(old, new)

        stem = _safe(stem_src)
        for old, new in (
            ("radians", "rad"),
            ("degrees", "deg"),
            ("common", "cc"),
        ):
            stem = re.sub(rf"(?<![A-Za-z0-9]){old}(?![A-Za-z0-9])", new, stem)

        stem = re.sub(r"_+", "_", stem).strip("_") or "result"
        stem = stem[:96].rstrip("_") or "result"
        candidate = f"calc_{stem}"

        existing = set(getattr(self, "user_variables", set()))
        for tsdb in self.tsdbs:
            existing.update(tsdb.getm())

        if candidate not in existing:
            return candidate

        idx = 2
        while f"{candidate}_{idx}" in existing:
            idx += 1
        return f"{candidate}_{idx}"

    def calculate_series(self):
        """Evaluate the Calculator expression and create new series."""
        import traceback

        self.progress.setFormat("Calculating %v/%m files")
        self.update_progressbar(0, max(len(self.tsdbs), 1))

        expr = self.calc_entry.toPlainText().strip()
        if not expr:
            self.progress.reset()
            self.progress.setFormat("%p%")
            QMessageBox.warning(self, "No Formula", "Please enter a formula.")
            return

        m_out = re.match(r"\s*([A-Za-z_]\w*)\s*=", expr)
        if m_out:
            base_output = m_out.group(1)
            exec_expr = expr
            display_expr = expr
        else:
            base_output = self._auto_calculator_output_name(expr)
            exec_expr = f"{base_output} = {expr}"
            display_expr = exec_expr

        file_windows = []
        file_window_coords = []
        file_window_dtg_refs = []

        def _time_coordinates(ts):
            """Return alignment coordinates for a series.

            Uses absolute datetimes when available, otherwise falls back to
            the native numeric time axis.
            """
            if ts.dtg_time is not None:
                return np.array(ts.dtg_time, dtype="datetime64[us]")
            return np.asarray(ts.t)

        def _coord_to_numeric(coord):
            coord = np.asarray(coord)
            if np.issubdtype(coord.dtype, np.datetime64):
                return coord.astype("datetime64[us]").astype(np.int64).astype(float)
            return coord.astype(float)

        def _align_to_window(ts, x_values, target_time, target_coord):
            coord = _time_coordinates(ts)
            idx = (coord >= target_coord[0]) & (coord <= target_coord[-1])
            if not np.any(idx):
                return np.full_like(target_time, np.nan, dtype=float)

            coord_part = coord[idx]
            x_part = np.asarray(x_values[idx], dtype=float)
            if np.array_equal(coord_part, target_coord):
                return x_part

            overlap = (target_coord >= coord_part[0]) & (target_coord <= coord_part[-1])
            if not np.any(overlap):
                return np.full_like(target_time, np.nan, dtype=float)

            full = np.full_like(target_time, np.nan, dtype=float)
            target_numeric = _coord_to_numeric(target_coord[overlap])
            source_numeric = _coord_to_numeric(coord_part)

            if source_numeric.size == 1:
                full[overlap] = x_part[0]
            else:
                full[overlap] = np.interp(target_numeric, source_numeric, x_part)
            return full

        for tsdb in self.tsdbs:
            file_t_window = None
            file_t_window_coord = None
            file_t_window_dtg_ref = None
            for ts in tsdb.getm().values():
                mask = self.get_time_window(ts)
                if mask is not None and np.any(mask):
                    file_t_window = ts.t[mask]
                    file_t_window_coord = _time_coordinates(ts)[mask]
                    file_t_window_dtg_ref = ts.dtg_ref
                    break
            if file_t_window is None:
                self.progress.reset()
                self.progress.setFormat("%p%")
                QMessageBox.critical(self, "No Time Window", "Could not infer a valid time window for one or more files.")
                return
            file_windows.append(file_t_window)
            file_window_coords.append(file_t_window_coord)
            file_window_dtg_refs.append(file_t_window_dtg_ref)

        common_tokens = {m.group(1) for m in re.finditer(r"\bc_([\w\- ]+)\b", exec_expr)}
        user_tokens = {m.group(1) for m in re.finditer(r"\bu_([\w\- ]+)", exec_expr)}
        explicit_file_refs = list(re.finditer(r"\bf(\d+)_([A-Za-z_]\w*)\b", exec_expr))
        explicit_file_tags = {int(m.group(1)) for m in explicit_file_refs}
        file_tags_used = explicit_file_tags or set(range(1, len(self.tsdbs) + 1))

        u_global = {u for u in user_tokens if not re.search(r"_f\d+$", u)}
        u_perfile = {u for u in user_tokens if re.search(r"_f\d+$", u)}

        known_user = getattr(self, "user_variables", set())
        missing = u_global - known_user
        if missing:
            self.progress.reset()
            self.progress.setFormat("%p%")
            QMessageBox.critical(self, "Unknown user variable", ", ".join(sorted(missing)))
            return

        def _resolve_series(file_index, name, name_by_file=None):
            tsdb = self.tsdbs[file_index]
            lookup = None
            if name_by_file and file_index < len(name_by_file):
                lookup = name_by_file[file_index]
            ts = tsdb.getm().get(lookup or name)
            if ts is None and not name_by_file:
                alt = next(
                    (key for key in tsdb.getm() if re.sub(r"^f\d+_", "", key) == name),
                    None,
                )
                if alt:
                    ts = tsdb.getm().get(alt)
                    lookup = alt
            return ts, lookup or name

        explicit_var_names: dict[int, set[str]] = {}
        if explicit_file_refs:
            db_safe_name_maps = []
            for db in self.tsdbs:
                safe_name_map = {}
                for key in db.getm().keys():
                    safe_name_map.setdefault(_safe(key), set()).add(key)
                db_safe_name_maps.append(safe_name_map)

            for match in explicit_file_refs:
                src_idx = int(match.group(1)) - 1
                if not (0 <= src_idx < len(self.tsdbs)):
                    self.progress.reset()
                    self.progress.setFormat("%p%")
                    QMessageBox.critical(self, "Calculation Error", f"File #{match.group(1)} does not exist.")
                    return
                safe_var = match.group(2)
                matches = db_safe_name_maps[src_idx].get(safe_var)
                if not matches:
                    self.progress.reset()
                    self.progress.setFormat("%p%")
                    QMessageBox.critical(
                        self,
                        "Calculation Error",
                        f"Variable reference 'f{match.group(1)}_{safe_var}' was not found in {os.path.basename(self.file_paths[src_idx])}.",
                    )
                    return
                explicit_var_names.setdefault(src_idx, set()).update(matches)

        filtered_series_cache = []
        for db in self.tsdbs:
            cache = {}
            for key, ts in db.getm().items():
                cache[key] = np.asarray(self.apply_filters(ts), dtype=float)
            filtered_series_cache.append(cache)

        calculator_tasks = []
        current_file_idx = 0
        for file_idx, tsdb in enumerate(self.tsdbs):
            target_time = file_windows[file_idx]
            target_coord = file_window_coords[file_idx]
            shared_ctx = {}
            for src_idx, db in enumerate(self.tsdbs):
                tag = f"f{src_idx + 1}"
                source_names = explicit_var_names.get(src_idx)
                items = (
                    ((name, db.getm()[name]) for name in source_names)
                    if source_names is not None
                    else db.getm().items()
                )
                for key, ts in items:
                    x_part = _align_to_window(ts, filtered_series_cache[src_idx][key], target_time, target_coord)
                    if np.all(np.isnan(x_part)):
                        continue
                    shared_ctx[f"{tag}_{_safe(key)}"] = x_part.astype(float)

            file_ctx = {}
            for k in common_tokens:
                names = None
                if self.common_lookup and k in self.common_lookup:
                    names = self.common_lookup[k]
                    if len(names) != len(self.tsdbs):
                        names = None
                ts, missing_name = _resolve_series(file_idx, k, name_by_file=names)
                if ts is None:
                    self.progress.reset()
                    self.progress.setFormat("%p%")
                    QMessageBox.critical(self, "Common variable error", f"'{missing_name}' not in {os.path.basename(self.file_paths[file_idx])}")
                    return
                file_ctx[f"c_{_safe(k)}"] = _align_to_window(ts, ts.x, target_time, target_coord)

            for k in u_global:
                ts, missing_name = _resolve_series(file_idx, k)
                if ts is None:
                    self.progress.reset()
                    self.progress.setFormat("%p%")
                    QMessageBox.critical(self, "User variable error", f"'{missing_name}' not in {os.path.basename(self.file_paths[file_idx])}")
                    return
                file_ctx[f"u_{_safe(k)}"] = _align_to_window(ts, ts.x, target_time, target_coord)

            for tok in u_perfile:
                m = re.match(r"(.+)_f(\d+)$", tok)
                if not m:
                    continue
                src_idx = int(m.group(2)) - 1
                if src_idx >= len(self.tsdbs):
                    self.progress.reset()
                    self.progress.setFormat("%p%")
                    QMessageBox.critical(self, "User variable error", f"File #{m.group(2)} does not exist.")
                    return
                ts = self.tsdbs[src_idx].getm().get(tok)
                if ts is None:
                    self.progress.reset()
                    self.progress.setFormat("%p%")
                    QMessageBox.critical(self, "User variable error", f"Variable '{tok}' not found in {os.path.basename(self.file_paths[src_idx])}")
                    return
                file_ctx[f"u_{tok}"] = _align_to_window(ts, ts.x, target_time, target_coord)

            calculator_tasks.append({
                "time_window": target_time,
                "dtg_ref": file_window_dtg_refs[file_idx],
                "shared_ctx": shared_ctx,
                "file_ctx": file_ctx,
                "exec_expr": exec_expr,
                "base_output": base_output,
            })

        evaluated_results = {}
        completed = 0
        use_multiprocessing = len(self.tsdbs) > 1

        current_file_idx = 0
        try:
            if use_multiprocessing:
                max_workers = min(len(self.tsdbs), max(1, multiprocessing.cpu_count()))
                with ProcessPoolExecutor(max_workers=max_workers) as executor:
                    futures = {
                        executor.submit(_evaluate_calculator_task, file_idx, task): file_idx
                        for file_idx, task in enumerate(calculator_tasks)
                    }
                    with _tqdm_progress(as_completed(futures), len(self.tsdbs), "Calculating") as progress_iter:
                        for future in progress_iter:
                            file_idx = futures[future]
                            current_file_idx = file_idx
                            evaluated_results[file_idx] = future.result()[1]
                            completed += 1
                            self.update_progressbar(completed, len(self.tsdbs))
            else:
                for file_idx, task in enumerate(calculator_tasks):
                    current_file_idx = file_idx
                    evaluated_results[file_idx] = _evaluate_calculator_task(file_idx, task)[1]
                    completed += 1
                    self.update_progressbar(completed, len(self.tsdbs))
        except Exception as e:
            self.progress.reset()
            self.progress.setFormat("%p%")
            QMessageBox.critical(
                self,
                "Calculation Error",
                f"{os.path.basename(self.file_paths[current_file_idx])}:\n{e}\n\n{traceback.format_exc()}",
            )
            return

        create_common_output = len(explicit_file_tags) >= 2
        results = []
        for file_idx, tsdb in enumerate(self.tsdbs):
            f_no = file_idx + 1
            y = evaluated_results[file_idx]
            must_write_here = (create_common_output and f_no == min(file_tags_used)) or (not create_common_output and f_no in file_tags_used)
            if not must_write_here:
                continue

            filt_tag = self._filter_tag()
            suffix = "" if create_common_output else f"_f{f_no}"
            out_name = base_output
            if filt_tag:
                out_name += f"_{filt_tag}"
            out_name += suffix
            time_window = calculator_tasks[file_idx]["time_window"]
            dtg_ref = calculator_tasks[file_idx]["dtg_ref"]
            ts_new = qats.TimeSeries(out_name, time_window, y, dtg_ref=dtg_ref)

            tsdb.add(ts_new)

            if create_common_output:
                for other_db in self.tsdbs:
                    if out_name not in other_db.getm():
                        other_db.add(ts_new.copy())

            self.user_variables = getattr(self, "user_variables", set())
            self.user_variables.add(out_name)
            results.append((tsdb, ts_new))

        self.progress.setFormat("%p%")
        self.refresh_variable_tabs()

        if create_common_output:
            msg = base_output
            output_names = [base_output]
        else:
            output_names = [f"{base_output}_f{n}" for n in sorted(file_tags_used)]
            msg = ", ".join(output_names)
        equation_text = self._format_calculator_equation(display_expr, output_names=output_names)
        QMessageBox.information(
            self,
            "Success",
            f"New variable(s): {msg}\n\nEquation used:\n{equation_text}",
        )

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
            "     sin(radians(60)) + f1_AccX * 2",
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
                "  •  You can give the left-hand side any name you like, or omit it",
                "     and let the Calculator create an automatic name from the equation.",
                "  •  Assigned or generated names become new user variables and appear",
                "     under the 'User Variables' tab.",
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
        min_width = getattr(self, "_min_left_panel", 240)
        var_list_widget.setMinimumWidth(min_width + 60)

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
                    new_ts = TimeSeries(name, ts.t.copy(), data, dtg_ref=ts.dtg_ref)
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

    def plot_marked_axes(self):
        """Scatter-plot variables marked as x/y/z in the variable input fields."""
        import os

        self._clear_last_plot_call()
        role_entries = {"x": [], "y": [], "z": [], "color": []}
        for key, entry in self.var_offsets.items():
            if entry is None:
                continue
            role = entry.text().strip().lower()
            if role == "c":
                role = "color"
            if role in role_entries:
                role_entries[role].append(key)

        if not role_entries["x"] or not role_entries["y"]:
            QMessageBox.warning(
                self,
                "Plot X/Y(/Z)",
                'Mark one variable as "x" and one as "y" in the input fields.',
            )
            return

        def _expand_key(series_key: str):
            expanded = []
            if "::" in series_key:
                fname, var = series_key.split("::", 1)
                for file_idx, fp in enumerate(self.file_paths):
                    if os.path.basename(fp) == fname:
                        expanded.append((file_idx, var))
            elif ":" in series_key:
                fname, var = series_key.split(":", 1)
                for file_idx, fp in enumerate(self.file_paths):
                    if os.path.basename(fp) == fname:
                        expanded.append((file_idx, var))
            else:
                for file_idx, tsdb in enumerate(self.tsdbs):
                    if series_key in tsdb.getm():
                        expanded.append((file_idx, series_key))
            return expanded

        role_per_file: dict[int, dict[str, str]] = {}
        conflicts = []
        for role, keys in role_entries.items():
            for key in keys:
                for file_idx, var_name in _expand_key(key):
                    bucket = role_per_file.setdefault(file_idx, {})
                    if role in bucket and bucket[role] != var_name:
                        conflicts.append(
                            f'File {file_idx + 1}: multiple "{role}" variables '
                            f'({bucket[role]}, {var_name})'
                        )
                    else:
                        bucket[role] = var_name

        if conflicts:
            QMessageBox.warning(
                self,
                "Plot X/Y(/Z)",
                "Resolve marked-axis conflicts before plotting:\n\n" + "\n".join(conflicts),
            )
            return

        traces = []
        use_3d = False
        for file_idx, roles in sorted(role_per_file.items()):
            if "x" not in roles or "y" not in roles:
                continue

            tsdb = self.tsdbs[file_idx]
            x_ts = tsdb.getm().get(roles["x"])
            y_ts = tsdb.getm().get(roles["y"])
            z_ts = tsdb.getm().get(roles["z"]) if "z" in roles else None
            c_ts = tsdb.getm().get(roles["color"]) if "color" in roles else None
            if x_ts is None or y_ts is None:
                continue

            x_vals = np.asarray(x_ts.x)
            y_vals = np.asarray(y_ts.x)
            min_len = min(len(x_vals), len(y_vals))
            if c_ts is not None:
                min_len = min(min_len, len(c_ts.x))
            if min_len <= 0:
                continue
            x_vals = x_vals[:min_len]
            y_vals = y_vals[:min_len]
            t_vals = np.asarray(x_ts.t)[:min_len]

            trace = {
                "file_label": os.path.basename(self.file_paths[file_idx]),
                "x_var": roles["x"],
                "y_var": roles["y"],
                "t": t_vals,
                "x": x_vals,
                "y": y_vals,
            }
            if c_ts is not None:
                trace["c"] = np.asarray(c_ts.x)[:min_len]
                trace["c_var"] = roles["color"]
            if z_ts is not None:
                z_vals = np.asarray(z_ts.x)
                min_len = min(min_len, len(z_vals))
                if min_len <= 0:
                    continue
                trace["t"] = trace["t"][:min_len]
                trace["x"] = trace["x"][:min_len]
                trace["y"] = trace["y"][:min_len]
                if "c" in trace:
                    trace["c"] = trace["c"][:min_len]
                trace["z"] = z_vals[:min_len]
                trace["z_var"] = roles["z"]
                use_3d = True

            # Respect the active time window (if any start/end input is given).
            ts_for_window = TimeSeries(
                "__xy_plot_window__",
                trace["t"],
                np.zeros(len(trace["t"])),
                dtg_ref=getattr(x_ts, "dtg_ref", None),
            )
            mask = self.get_time_window(ts_for_window)
            if isinstance(mask, slice):
                trace["t"] = trace["t"][mask]
                trace["x"] = trace["x"][mask]
                trace["y"] = trace["y"][mask]
                if "c" in trace:
                    trace["c"] = trace["c"][mask]
                if "z" in trace:
                    trace["z"] = trace["z"][mask]
            else:
                if not np.asarray(mask).any():
                    continue
                trace["t"] = trace["t"][mask]
                trace["x"] = trace["x"][mask]
                trace["y"] = trace["y"][mask]
                if "c" in trace:
                    trace["c"] = trace["c"][mask]
                if "z" in trace:
                    trace["z"] = trace["z"][mask]
            if len(trace["x"]) == 0:
                continue
            traces.append(trace)

        if not traces:
            QMessageBox.warning(
                self,
                "Plot X/Y(/Z)",
                "No matching file with marked x/y variables could be plotted.",
            )
            return

        engine = (
            self.plot_engine_combo.currentText()
            if hasattr(self, "plot_engine_combo")
            else "plotly"
        ).lower()
        title = "3D scatter plot (x, y, z)" if use_3d else "2D scatter plot (x, y)"
        axis_labels = traces[0]
        if engine == "bokeh" and use_3d:
            QMessageBox.information(
                self,
                "Plot X/Y(/Z)",
                "Bokeh does not support native 3D scatter here. Falling back to Matplotlib for 3D.",
            )
            engine = "default"

        if engine == "bokeh":
            from bokeh.embed import file_html
            from bokeh.models import ColumnDataSource, HoverTool, LinearColorMapper, ColorBar
            from bokeh.palettes import Category10_10
            from bokeh.plotting import figure, show
            from bokeh.resources import INLINE
            from bokeh.transform import linear_cmap
            import itertools
            import tempfile

            fig = figure(
                width=900,
                height=450,
                title=title,
                x_axis_label=axis_labels["x_var"],
                y_axis_label=axis_labels["y_var"],
                tools="pan,wheel_zoom,box_zoom,reset,save",
                sizing_mode="stretch_both",
            )
            if self.theme_switch.isChecked():
                fig.background_fill_color = "#2b2b2b"
                fig.border_fill_color = "#2b2b2b"
            has_color = any("c" in tr for tr in traces)
            if has_color:
                hover_tips = [("Series", "@label"), ("x", "@x"), ("y", "@y"), ("color", "@c")]
            else:
                hover_tips = [("Series", "@label"), ("x", "@x"), ("y", "@y")]
            fig.add_tools(HoverTool(tooltips=hover_tips))

            mapper = None
            if has_color:
                all_c = np.concatenate([np.asarray(tr["c"]) for tr in traces if "c" in tr])
                if len(all_c):
                    c_min = float(np.min(all_c))
                    c_max = float(np.max(all_c))
                    if abs(c_max - c_min) < 1e-12:
                        c_max = c_min + 1.0
                    mapper = LinearColorMapper(palette="Viridis256", low=c_min, high=c_max)
                    fig.add_layout(ColorBar(color_mapper=mapper, title=traces[0].get("c_var", "color")), "right")

            color_cycle = itertools.cycle(Category10_10)
            for trace in traces:
                color = next(color_cycle)
                data = dict(
                    x=trace["x"],
                    y=trace["y"],
                    label=[trace["file_label"]] * len(trace["x"]),
                )
                if "c" in trace:
                    data["c"] = trace["c"]
                src = ColumnDataSource(data=data)
                if mapper is not None and "c" in trace:
                    fig.circle(
                        x="x",
                        y="y",
                        source=src,
                        size=5,
                        alpha=0.8,
                        color=linear_cmap("c", "Viridis256", mapper.low, mapper.high),
                        legend_label=trace["file_label"],
                        muted_alpha=0.1,
                    )
                else:
                    fig.circle(
                        x="x",
                        y="y",
                        source=src,
                        size=5,
                        alpha=0.8,
                        color=color,
                        legend_label=trace["file_label"],
                        muted_alpha=0.1,
                    )
            fig.legend.click_policy = "mute"

            if getattr(self, "embed_plot_cb", None) and self.embed_plot_cb.isChecked():
                if self._temp_plot_file and os.path.exists(self._temp_plot_file):
                    try:
                        os.remove(self._temp_plot_file)
                    except OSError:
                        pass
                html = file_html(fig, INLINE, title)
                with tempfile.NamedTemporaryFile("w", delete=False, suffix=".html", encoding="utf-8") as tmp:
                    tmp.write(html)
                    tmp.flush()
                    self._temp_plot_file = tmp.name
                self.plot_view.load(QUrl.fromLocalFile(self._temp_plot_file))
                self.plot_view.show()
                self._remember_plot_call(self.plot_marked_axes)
            else:
                self.plot_view.hide()
                show(fig)
            return

        if engine == "default":
            import matplotlib.pyplot as plt

            if use_3d:
                fig = plt.figure(figsize=(10, 6))
                ax = fig.add_subplot(111, projection="3d")
                for trace in traces:
                    z_vals = trace.get("z")
                    if z_vals is None:
                        continue
                    if "c" in trace:
                        sc = ax.scatter(
                            trace["x"], trace["y"], z_vals, c=trace["c"], cmap="viridis", s=12, label=trace["file_label"]
                        )
                    else:
                        sc = ax.scatter(trace["x"], trace["y"], z_vals, s=10, label=trace["file_label"])
                ax.set_zlabel(axis_labels.get("z_var", "z"))
            else:
                fig, ax = plt.subplots(figsize=(10, 6))
                for trace in traces:
                    if "c" in trace:
                        sc = ax.scatter(
                            trace["x"], trace["y"], c=trace["c"], cmap="viridis", s=18, alpha=0.8, label=trace["file_label"]
                        )
                    else:
                        sc = ax.scatter(trace["x"], trace["y"], s=14, alpha=0.8, label=trace["file_label"])
            ax.set_title(title)
            ax.set_xlabel(axis_labels["x_var"])
            ax.set_ylabel(axis_labels["y_var"])
            if any("c" in tr for tr in traces):
                fig.colorbar(sc, ax=ax, label=axis_labels.get("c_var", "color"))
            ax.legend(loc="best")
            ax.grid(True, alpha=0.25)
            fig.tight_layout()

            if getattr(self, "embed_plot_cb", None) and self.embed_plot_cb.isChecked():
                self._show_embedded_mpl_figure(fig)
                self._remember_plot_call(self.plot_marked_axes)
            else:
                self.plot_view.hide()
                fig.show()
            return

        # Plotly branch (default for non-"default"/non-"bokeh" engines)
        import plotly.graph_objects as go
        from plotly.io import to_html
        import tempfile

        fig = go.Figure()
        for trace in traces:
            label = trace["file_label"]
            if use_3d and "z" in trace:
                fig.add_trace(
                    go.Scatter3d(
                        x=trace["x"],
                        y=trace["y"],
                        z=trace["z"],
                        mode="markers",
                        name=label,
                        marker=dict(
                            color=trace.get("c"),
                            colorscale="Viridis",
                            showscale=("c" in trace),
                            colorbar=dict(title=trace.get("c_var", "color")),
                        ),
                    )
                )
            else:
                marker = dict(size=7)
                if "c" in trace:
                    marker.update(
                        color=trace["c"],
                        colorscale="Viridis",
                        showscale=True,
                        colorbar=dict(title=trace.get("c_var", "color")),
                    )
                fig.add_trace(go.Scatter(x=trace["x"], y=trace["y"], mode="markers", name=label, marker=marker))

        layout_kwargs = {
            "title": title,
            "xaxis_title": axis_labels["x_var"],
            "yaxis_title": axis_labels["y_var"],
            "template": "plotly_dark" if self.theme_switch.isChecked() else "plotly",
        }
        if use_3d:
            layout_kwargs["scene"] = dict(
                xaxis_title=axis_labels["x_var"],
                yaxis_title=axis_labels["y_var"],
                zaxis_title=axis_labels.get("z_var", "z"),
            )
        fig.update_layout(**layout_kwargs)

        if getattr(self, "embed_plot_cb", None) and self.embed_plot_cb.isChecked():
            if self._temp_plot_file and os.path.exists(self._temp_plot_file):
                try:
                    os.remove(self._temp_plot_file)
                except OSError:
                    pass
            html = to_html(fig, include_plotlyjs=True, full_html=True)
            with tempfile.NamedTemporaryFile("w", delete=False, suffix=".html", encoding="utf-8") as tmp:
                tmp.write(html)
                tmp.flush()
                self._temp_plot_file = tmp.name
            self.plot_view.load(QUrl.fromLocalFile(self._temp_plot_file))
            self.plot_view.show()
            self._remember_plot_call(self.plot_marked_axes)
        else:
            self.plot_view.hide()
            fig.show(renderer="browser")

    def animate_marked_axes(self):
        """Animate scatter points over time for variables marked as x/y(/z)."""
        import os
        import tempfile
        import plotly.express as px
        from plotly.io import to_html

        role_entries = {"x": [], "y": [], "z": [], "color": []}
        for key, entry in self.var_offsets.items():
            if entry is None:
                continue
            role = entry.text().strip().lower()
            if role == "c":
                role = "color"
            if role in role_entries:
                role_entries[role].append(key)

        if not role_entries["x"] or not role_entries["y"]:
            QMessageBox.warning(
                self,
                "Animate X/Y(/Z)",
                'Mark one variable as "x" and one as "y" in the input fields.',
            )
            return

        def _expand_key(series_key: str):
            expanded = []
            if "::" in series_key:
                fname, var = series_key.split("::", 1)
                for file_idx, fp in enumerate(self.file_paths):
                    if os.path.basename(fp) == fname:
                        expanded.append((file_idx, var))
            elif ":" in series_key:
                fname, var = series_key.split(":", 1)
                for file_idx, fp in enumerate(self.file_paths):
                    if os.path.basename(fp) == fname:
                        expanded.append((file_idx, var))
            else:
                for file_idx, tsdb in enumerate(self.tsdbs):
                    if series_key in tsdb.getm():
                        expanded.append((file_idx, series_key))
            return expanded

        role_per_file: dict[int, dict[str, str]] = {}
        conflicts = []
        for role, keys in role_entries.items():
            for key in keys:
                for file_idx, var_name in _expand_key(key):
                    bucket = role_per_file.setdefault(file_idx, {})
                    if role in bucket and bucket[role] != var_name:
                        conflicts.append(
                            f'File {file_idx + 1}: multiple "{role}" variables '
                            f'({bucket[role]}, {var_name})'
                        )
                    else:
                        bucket[role] = var_name

        if conflicts:
            QMessageBox.warning(
                self,
                "Animate X/Y(/Z)",
                "Resolve marked-axis conflicts before animating:\n\n" + "\n".join(conflicts),
            )
            return

        rows = []
        use_3d = False
        for file_idx, roles in sorted(role_per_file.items()):
            if "x" not in roles or "y" not in roles:
                continue
            tsdb = self.tsdbs[file_idx]
            x_ts = tsdb.getm().get(roles["x"])
            y_ts = tsdb.getm().get(roles["y"])
            z_ts = tsdb.getm().get(roles["z"]) if "z" in roles else None
            c_ts = tsdb.getm().get(roles["color"]) if "color" in roles else None
            if x_ts is None or y_ts is None:
                continue

            x_vals = np.asarray(x_ts.x)
            y_vals = np.asarray(y_ts.x)
            min_len = min(len(x_vals), len(y_vals))
            if z_ts is not None:
                min_len = min(min_len, len(z_ts.x))
            if c_ts is not None:
                min_len = min(min_len, len(c_ts.x))
            if min_len <= 0:
                continue

            t_vals = np.asarray(x_ts.t)[:min_len]
            x_vals = x_vals[:min_len]
            y_vals = y_vals[:min_len]
            z_vals = np.asarray(z_ts.x)[:min_len] if z_ts is not None else None
            c_vals = np.asarray(c_ts.x)[:min_len] if c_ts is not None else None
            if z_vals is not None:
                use_3d = True

            ts_for_window = TimeSeries(
                "__xy_anim_window__",
                t_vals,
                np.zeros(len(t_vals)),
                dtg_ref=getattr(x_ts, "dtg_ref", None),
            )
            mask = self.get_time_window(ts_for_window)
            if isinstance(mask, slice):
                t_vals, x_vals, y_vals = t_vals[mask], x_vals[mask], y_vals[mask]
                if z_vals is not None:
                    z_vals = z_vals[mask]
                if c_vals is not None:
                    c_vals = c_vals[mask]
            else:
                mask_arr = np.asarray(mask)
                if not mask_arr.any():
                    continue
                t_vals, x_vals, y_vals = t_vals[mask_arr], x_vals[mask_arr], y_vals[mask_arr]
                if z_vals is not None:
                    z_vals = z_vals[mask_arr]
                if c_vals is not None:
                    c_vals = c_vals[mask_arr]
            if len(x_vals) == 0:
                continue

            time_labels = []
            for t in t_vals:
                if isinstance(t, (np.datetime64, pd.Timestamp)):
                    time_labels.append(str(pd.Timestamp(t)))
                else:
                    time_labels.append(f"{float(t):.6g}")

            data = {
                "time": time_labels,
                "x": x_vals,
                "y": y_vals,
                "series": [os.path.basename(self.file_paths[file_idx])] * len(x_vals),
            }
            if z_vals is not None:
                data["z"] = z_vals
            if c_vals is not None:
                data["color"] = c_vals
            rows.append(pd.DataFrame(data))

        if not rows:
            QMessageBox.warning(
                self,
                "Animate X/Y(/Z)",
                "No matching file with marked x/y variables could be animated.",
            )
            return

        df = pd.concat(rows, ignore_index=True)
        base_kwargs = dict(
            data_frame=df,
            x="x",
            y="y",
            animation_frame="time",
            animation_group="series",
            hover_name="series",
            title="Animated scatter plot from marked axes",
            template="plotly_dark" if self.theme_switch.isChecked() else "plotly",
        )
        if "color" in df.columns:
            base_kwargs["color"] = "color"
            base_kwargs["color_continuous_scale"] = "Viridis"
        else:
            base_kwargs["color"] = "series"

        if use_3d and "z" in df.columns:
            fig = px.scatter_3d(**base_kwargs, z="z")
        else:
            fig = px.scatter(**base_kwargs)

        # Keep axis limits fixed for all animation frames.
        x_min, x_max = float(df["x"].min()), float(df["x"].max())
        y_min, y_max = float(df["y"].min()), float(df["y"].max())
        if abs(x_max - x_min) < 1e-12:
            x_min, x_max = x_min - 0.5, x_max + 0.5
        if abs(y_max - y_min) < 1e-12:
            y_min, y_max = y_min - 0.5, y_max + 0.5

        fig.update_layout(
            xaxis_title=role_per_file[next(iter(role_per_file))].get("x", "x"),
            yaxis_title=role_per_file[next(iter(role_per_file))].get("y", "y"),
        )
        fig.update_xaxes(range=[x_min, x_max])
        fig.update_yaxes(range=[y_min, y_max])
        if use_3d and "z" in df.columns:
            z_min, z_max = float(df["z"].min()), float(df["z"].max())
            if abs(z_max - z_min) < 1e-12:
                z_min, z_max = z_min - 0.5, z_max + 0.5
            fig.update_layout(
                scene=dict(
                    xaxis_title=role_per_file[next(iter(role_per_file))].get("x", "x"),
                    yaxis_title=role_per_file[next(iter(role_per_file))].get("y", "y"),
                    zaxis_title=role_per_file[next(iter(role_per_file))].get("z", "z"),
                    xaxis=dict(range=[x_min, x_max], autorange=False),
                    yaxis=dict(range=[y_min, y_max], autorange=False),
                    zaxis=dict(range=[z_min, z_max], autorange=False),
                    aspectmode="data",
                )
            )

        if getattr(self, "embed_plot_cb", None) and self.embed_plot_cb.isChecked():
            if self._temp_plot_file and os.path.exists(self._temp_plot_file):
                try:
                    os.remove(self._temp_plot_file)
                except OSError:
                    pass
            html = to_html(fig, include_plotlyjs=True, full_html=True)
            with tempfile.NamedTemporaryFile("w", delete=False, suffix=".html", encoding="utf-8") as tmp:
                tmp.write(html)
                tmp.flush()
                self._temp_plot_file = tmp.name
            self.plot_view.load(QUrl.fromLocalFile(self._temp_plot_file))
            self.plot_view.show()
            self._remember_plot_call(self.animate_marked_axes)
        else:
            self.plot_view.hide()
            fig.show(renderer="browser")

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

    def _apply_transformation(self, func, suffix, announce=True, transform_spec=None):
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
        tasks = []
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

                tasks.append({
                    "file_idx": f_idx - 1,
                    "tsdb": tsdb,
                    "new_name": new_name,
                    "t_win": t_win,
                    "dtg_ref": ts.dtg_ref,
                    "y_src": np.asarray(y_src, dtype=float),
                })

        self.progress.setFormat("Transforming %v/%m series")
        self.update_progressbar(0, max(len(tasks), 1))

        use_multiprocessing = len(tasks) > 1 and transform_spec is not None
        completed = 0
        current_task = None
        try:
            if use_multiprocessing:
                max_workers = min(len(tasks), max(1, multiprocessing.cpu_count()))
                with ProcessPoolExecutor(
                    max_workers=max_workers,
                    initializer=_init_transform_worker,
                    initargs=(transform_spec,),
                ) as executor:
                    futures = {
                        executor.submit(_evaluate_transform_payload, idx, task["y_src"]): idx
                        for idx, task in enumerate(tasks)
                    }
                    with _tqdm_progress(as_completed(futures), len(tasks), "Transforming") as progress_iter:
                        for future in progress_iter:
                            current_task = tasks[futures[future]]
                            task_idx, y_new = future.result()
                            task = tasks[task_idx]
                            task["y_new"] = y_new
                            completed += 1
                            self.update_progressbar(completed, len(tasks))
            else:
                for idx, task in enumerate(tasks):
                    current_task = task
                    y_new = _apply_transform_spec(task["y_src"], transform_spec) if transform_spec is not None else func(task["y_src"])
                    task["y_new"] = y_new
                    completed += 1
                    self.update_progressbar(completed, len(tasks))
        except Exception as exc:
            self.progress.reset()
            self.progress.setFormat("%p%")
            failing_name = current_task["new_name"] if current_task else suffix
            QMessageBox.critical(
                self,
                "Transformation Error",
                f"{failing_name}:\n{exc}\n\n{traceback.format_exc()}",
            )
            return

        for task in tasks:
            task["tsdb"].add(TimeSeries(task["new_name"], task["t_win"], task["y_new"], dtg_ref=task["dtg_ref"]))
            made.append(task["new_name"])
            self.user_variables = getattr(self, "user_variables", set())
            self.user_variables.add(task["new_name"])

        self.progress.setFormat("%p%")

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

        self._apply_transformation(lambda y: np.abs(y), "abs", True, transform_spec={"kind": "abs"})

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
        self._apply_transformation(func, "rollMean", True, transform_spec={"kind": "rolling_mean", "window": window})

    def reduce_selected_points(self):
        """Create reduced-resolution copies of selected series."""
        import os
        from PySide6.QtCore import QTimer
        from anyqats import TimeSeries

        try:
            keep_percent = float(self.reduction_pct_entry.text().strip())
        except ValueError:
            QMessageBox.warning(
                self,
                "Invalid value",
                "Reduction percentage must be a number between 0 and 100.",
            )
            return

        if not 0 <= keep_percent <= 100:
            QMessageBox.warning(
                self,
                "Invalid value",
                "Reduction percentage must be between 0 and 100.",
            )
            return

        bias_mode = self.reduction_bias_combo.currentText().strip().lower()
        if bias_mode not in {"mean", "upper", "lower"}:
            bias_mode = "mean"

        self.rebuild_var_lookup()
        made = []
        fnames = [os.path.basename(p) for p in self.file_paths]

        def _has_file_prefix(key: str) -> bool:
            for name in fnames:
                if key.startswith(f"{name}::") or key.startswith(f"{name}:"):
                    return True
            return False

        def _reduce_points(t_values, y_values, percent, mode):
            t_arr = np.asarray(t_values)
            y_arr = np.asarray(y_values)
            n_points = len(y_arr)
            if n_points == 0:
                return t_arr, y_arr

            keep_points = int(round(n_points * percent / 100.0))
            keep_points = max(0, min(n_points, keep_points))

            if keep_points == n_points:
                return t_arr.copy(), y_arr.copy()

            if keep_points == 0:
                return t_arr[:0], y_arr[:0]

            idx_bins = np.array_split(np.arange(n_points), keep_points)
            t_new = np.array([t_arr[idx].mean() for idx in idx_bins])

            if mode == "upper":
                y_new = np.array([np.max(y_arr[idx]) for idx in idx_bins])
            elif mode == "lower":
                y_new = np.array([np.min(y_arr[idx]) for idx in idx_bins])
            else:
                y_new = np.array([y_arr[idx].mean() for idx in idx_bins])
            return t_new, y_new

        for f_idx, (tsdb, path) in enumerate(zip(self.tsdbs, self.file_paths), start=1):
            fname = os.path.basename(path)

            for u_key, chk in self.var_checkboxes.items():
                if not chk.isChecked():
                    continue

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
                filtered = self.apply_filters(ts)
                if isinstance(mask, slice):
                    t_win = ts.t[mask]
                    y_src = filtered[mask]
                else:
                    if not mask.any():
                        continue
                    t_win = ts.t[mask]
                    y_src = filtered[mask]

                t_new, y_new = _reduce_points(t_win, y_src, keep_percent, bias_mode)

                filt_tag = self._filter_tag()
                pct_tag = (
                    str(int(keep_percent))
                    if keep_percent.is_integer()
                    else f"{keep_percent:g}"
                )
                base = f"{ts.name}_red{pct_tag}pct"
                if bias_mode != "mean":
                    base += f"_{bias_mode}"
                if filt_tag:
                    base += f"_{filt_tag}"
                base += f"_f{f_idx}"
                new_name = base
                k = 1
                while new_name in tsdb.getm():
                    new_name = f"{base}_{k}"
                    k += 1

                tsdb.add(TimeSeries(new_name, t_new, y_new, dtg_ref=ts.dtg_ref))
                made.append(new_name)
                self.user_variables = getattr(self, "user_variables", set())
                self.user_variables.add(new_name)

        if made:
            QTimer.singleShot(0, lambda: self._populate_variables(None))

            def _ok():
                show = 10
                if len(made) <= show:
                    msg = "\n".join(sorted(made))
                else:
                    msg = "\n".join(sorted(made)[:show]) + f"\n… and {len(made) - show} more"
                QMessageBox.information(self, "Transformation complete", msg)

            QTimer.singleShot(0, _ok)
        else:
            QTimer.singleShot(
                0,
                lambda: QMessageBox.warning(
                    self,
                    "Nothing new",
                    "No series were selected or the selected time window was empty.",
                ),
            )

    def merge_selected_series(self):
        """Merge selected time series end-to-end into a new user variable."""
        import os
        import re
        from PySide6.QtCore import QTimer
        from PySide6.QtWidgets import QMessageBox
        from anyqats import TimeSeries

        self.rebuild_var_lookup()

        selected_keys = [k for k, ck in self.var_checkboxes.items() if ck.isChecked()]
        if not selected_keys:
            QMessageBox.warning(
                self, "No selection", "Select variables to merge into a new series."
            )
            return

        filenames = [os.path.basename(path) for path in self.file_paths]
        filename_set = set(filenames)

        def _normalize_key(key: str) -> str:
            """Map legacy "file:var" selections to "file::var" when possible."""
            if "::" in key:
                return key
            if ":" in key:
                prefix, rest = key.split(":", 1)
                candidate = f"{prefix}::{rest}"
                if candidate in self.var_checkboxes:
                    return candidate
            return key

        normalized_keys = [_normalize_key(key) for key in selected_keys]

        def _has_file_prefix(key: str) -> bool:
            if "::" in key:
                prefix = key.split("::", 1)[0]
                return prefix in filename_set
            if ":" in key:
                prefix, rest = key.split(":", 1)
                if prefix in filename_set:
                    candidate = f"{prefix}::{rest}"
                    return candidate in self.var_checkboxes
            return False

        per_file_mode = any(_has_file_prefix(k) for k in normalized_keys)
        if per_file_mode and not all(_has_file_prefix(k) for k in normalized_keys):
            QMessageBox.critical(
                self,
                "Mixed selection",
                "Pick either only common-tab variables or only per-file keys when merging.",
            )
            return

        per_file_map = {name: [] for name in filenames}
        if per_file_mode:
            for original, normalized in zip(selected_keys, normalized_keys):
                if "::" not in normalized:
                    continue
                prefix, varname = normalized.split("::", 1)
                if prefix in per_file_map:
                    per_file_map[prefix].append((original, varname))

        created = []
        filt_tag = self._filter_tag()
        multi_file = len(self.tsdbs) > 1
        self.user_variables = getattr(self, "user_variables", set())
        re_suffix = re.compile(r"_f\d+$")

        def _clean_label(label: str) -> str:
            label = label.split("::", 1)[-1]
            label = label.split(":", 1)[-1]
            return re_suffix.sub("", label)

        merged_dtg_ref = None

        def _append_segment(ts, offset, last_dt, merged_segments, merged_time_parts):
            nonlocal merged_dtg_ref
            if ts is None:
                return offset, last_dt

            if merged_dtg_ref is None and getattr(ts, "dtg_ref", None) is not None:
                merged_dtg_ref = ts.dtg_ref

            data = self.apply_filters(ts)
            mask = self.get_time_window(ts)
            if isinstance(mask, slice):
                y_segment = data[mask]
                t_segment = ts.t[mask]
            else:
                if not mask.any():
                    return offset, last_dt
                y_segment = data[mask]
                t_segment = ts.t[mask]

            if y_segment.size == 0:
                return offset, last_dt

            y_segment = np.asarray(y_segment)
            raw_time = np.asarray(t_segment)

            if raw_time.dtype.kind == "O":
                try:
                    raw_time = raw_time.astype("datetime64[ns]")
                except (TypeError, ValueError):
                    raw_time = raw_time.astype(float)

            if np.issubdtype(raw_time.dtype, np.datetime64):
                raw_time = raw_time.astype("datetime64[ns]").astype("int64") / 1e9
            elif np.issubdtype(raw_time.dtype, np.timedelta64):
                raw_time = raw_time.astype("timedelta64[ns]").astype("int64") / 1e9
            else:
                raw_time = raw_time.astype(float, copy=False)

            if raw_time.size:
                local_time = raw_time - raw_time[0]
            else:
                local_time = np.zeros_like(raw_time, dtype=float)

            local_time = np.asarray(local_time, dtype=float)

            dt_value = getattr(ts, "dt", None)
            if dt_value not in (None, 0):
                dt_value = float(dt_value)
            else:
                dt_value = None

            diffs = np.diff(local_time)
            if dt_value in (None, 0):
                if diffs.size:
                    dt_value = float(np.median(diffs))
                elif last_dt not in (None, 0):
                    dt_value = float(last_dt)
                else:
                    dt_value = 0.0

            merged_segments.append(y_segment)
            merged_time_parts.append(local_time + offset)

            if dt_value not in (None, 0):
                last_dt = float(dt_value)

            if local_time.size:
                if dt_value not in (None, 0):
                    offset = offset + local_time[-1] + float(dt_value)
                else:
                    offset = offset + local_time[-1]
            elif dt_value not in (None, 0):
                offset = offset + float(dt_value)

            return offset, last_dt

        if not per_file_mode:
            merged_segments = []
            merged_time_parts = []
            offset = 0.0
            last_dt = None

            for _, varname in zip(selected_keys, normalized_keys):
                for tsdb in self.tsdbs:
                    ts = tsdb.getm().get(varname)
                    offset, last_dt = _append_segment(
                        ts, offset, last_dt, merged_segments, merged_time_parts
                    )

            if merged_segments:
                merged_x = np.concatenate(merged_segments)
                merged_t = np.concatenate(merged_time_parts)

                cleaned_labels = [_clean_label(label) for label in selected_keys]
                name_base = f"merge({'+'.join(cleaned_labels)})"
                if filt_tag:
                    name_base += f"_{filt_tag}"

                name = name_base
                counter = 1
                while any(name in tsdb.getm() for tsdb in self.tsdbs):
                    name = f"{name_base}_{counter}"
                    counter += 1

                merged_ts = TimeSeries(name, merged_t, merged_x, dtg_ref=merged_dtg_ref)
                if self.tsdbs:
                    # The merged result should behave like a single user variable.
                    # Adding duplicates to every file leads to repeated plots and
                    # duplicated entries.  Keep a single authoritative copy in the
                    # first database so downstream features (plotting, stats, …)
                    # only see one series.
                    self.tsdbs[0].add(merged_ts)
                self.user_variables.add(name)
                created.append(name)
        else:
            for f_idx, (tsdb, path) in enumerate(zip(self.tsdbs, self.file_paths), start=1):
                fname = os.path.basename(path)
                entries = per_file_map.get(fname, [])
                if not entries:
                    continue
                source_labels = [orig for orig, _ in entries]
                varnames = [var for _, var in entries]

                merged_segments = []
                merged_time_parts = []
                offset = 0.0
                last_dt = None

                for _, varname in zip(source_labels, varnames):
                    ts = tsdb.getm().get(varname)
                    offset, last_dt = _append_segment(
                        ts, offset, last_dt, merged_segments, merged_time_parts
                    )

                if not merged_segments:
                    continue

                merged_x = np.concatenate(merged_segments)
                merged_t = np.concatenate(merged_time_parts)

                cleaned_labels = [_clean_label(label) for label in source_labels]
                name_base = f"merge({'+'.join(cleaned_labels)})"
                if not per_file_mode and multi_file:
                    name_base += f"_f{f_idx}"
                if filt_tag:
                    name_base += f"_{filt_tag}"

                name = name_base
                counter = 1
                while name in tsdb.getm():
                    name = f"{name_base}_{counter}"
                    counter += 1

                tsdb.add(TimeSeries(name, merged_t, merged_x, dtg_ref=merged_dtg_ref))
                self.user_variables.add(name)
                created.append(name)

        if created:
            QTimer.singleShot(0, lambda: self._populate_variables(None))

            def _ok():
                show = 10
                if len(created) <= show:
                    msg = "\n".join(sorted(created))
                else:
                    msg = "\n".join(sorted(created)[:show]) + f"\n… and {len(created) - show} more"
                QMessageBox.information(self, "Merge complete", msg)

            QTimer.singleShot(0, _ok)
        else:
            QTimer.singleShot(
                0,
                lambda: QMessageBox.warning(
                    self,
                    "No data",
                    "No merged series were created. Ensure the selected series exist in the chosen files.",
                ),
            )

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
                values, t_ref, t_ref_dtg = [], None, None
                for k in sel_keys:
                    ts = tsdb.getm().get(k)
                    if ts is None:
                        continue
                    if t_ref is None:
                        t_ref = ts.t
                        t_ref_dtg = ts.dtg_ref
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

                tsdb.add(TimeSeries(name, t_ref, y, dtg_ref=t_ref_dtg))
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

                values, t_ref, t_ref_dtg = [], None, None
                for v in per_file[fname]:
                    ts = tsdb.getm().get(v)
                    if ts is None:
                        continue
                    if t_ref is None:
                        t_ref = ts.t
                        t_ref_dtg = ts.dtg_ref
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

                tsdb.add(TimeSeries(name, t_ref, y, dtg_ref=t_ref_dtg))
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
        def _store(tsdb, name_base, t_ref, vals, t_ref_dtg):
            """Add a new TimeSeries, ensuring uniqueness inside *tsdb*."""
            y = np.mean(np.vstack(vals), axis=0)
            new = name_base
            n = 1
            while new in tsdb.getm():
                new = f"{name_base}_{n}"
                n += 1
            tsdb.add(TimeSeries(new, t_ref, y, dtg_ref=t_ref_dtg))
            self.user_variables.add(new)
            created.append(new)

        # ─────────────────── COMMON-TAB BRANCH ────────────────────────
        if common_pick:
            clean_keys = [_clean(k) for k in sel_keys]

            for f_idx, (tsdb, fp) in enumerate(zip(self.tsdbs, self.file_paths), 1):
                vals, t_ref, t_ref_dtg = [], None, None
                for k in sel_keys:
                    ts = tsdb.getm().get(k)
                    if ts is None:
                        continue
                    if t_ref is None:
                        t_ref = ts.t
                        t_ref_dtg = ts.dtg_ref
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
                _store(tsdb, namebase, t_ref, vals, t_ref_dtg)

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

                vals, t_ref, t_ref_dtg = [], None, None
                for v in vars_here:
                    ts = tsdb.getm().get(v)
                    if ts is None:
                        continue
                    if t_ref is None:
                        t_ref = ts.t
                        t_ref_dtg = ts.dtg_ref
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
                _store(tsdb, namebase, t_ref, vals, t_ref_dtg)

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
        self._apply_transformation(lambda y: y * 1000, "×1000", True, transform_spec={"kind": "scale", "factor": 1000})

    def divide_by_1000(self):
        self._apply_transformation(lambda y: y / 1000, "÷1000", True, transform_spec={"kind": "scale", "factor": 1 / 1000})

    def multiply_by_10(self):
        self._apply_transformation(lambda y: y * 10, "×10", True, transform_spec={"kind": "scale", "factor": 10})

    def divide_by_10(self):
        self._apply_transformation(lambda y: y / 10, "÷10", True, transform_spec={"kind": "scale", "factor": 0.1})

    def multiply_by_2(self):
        self._apply_transformation(lambda y: y * 2, "×2", True, transform_spec={"kind": "scale", "factor": 2})

    def divide_by_2(self):
        self._apply_transformation(lambda y: y / 2, "÷2", True, transform_spec={"kind": "scale", "factor": 0.5})

    def multiply_by_neg1(self):
        self._apply_transformation(lambda y: y * -1, "×-1", True, transform_spec={"kind": "scale", "factor": -1})

    def to_radians(self):
        import numpy as np

        self._apply_transformation(lambda y: np.radians(y), "rad", True, transform_spec={"kind": "radians"})

    def to_degrees(self):
        import numpy as np

        self._apply_transformation(lambda y: np.degrees(y), "deg", True, transform_spec={"kind": "degrees"})

    def apply_trig_from_degrees(self):
        import numpy as np
        from PySide6.QtWidgets import QMessageBox

        func_name = self.trig_combo.currentText()
        func_map = {"sin": np.sin, "cos": np.cos, "tan": np.tan}
        trig_func = func_map.get(func_name)

        try:
            angle_deg = float(self.trig_angle_entry.text())
        except ValueError:
            QMessageBox.critical(
                self, "Invalid angle", "Enter a numeric angle in degrees."
            )
            return

        trig_value = trig_func(np.deg2rad(angle_deg))

        def _apply_trig(y: np.ndarray, factor: float = trig_value) -> np.ndarray:
            return np.asarray(y, dtype=float) * factor

        suffix = f"*{func_name}({angle_deg:g})"
        self._apply_transformation(_apply_trig, suffix, True, transform_spec={"kind": "trig_scale", "factor": trig_value})

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
        self._apply_transformation(shift, "shift0", True, transform_spec={"kind": "shift_min_to_zero", "ignore_anomalies": self.ignore_anomalies_cb.isChecked()})

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
        self._apply_transformation(_shift_if_plateau, "shiftNZ", True, transform_spec={"kind": "shift_repeated_neg_min", "tol_pct": tol_pct, "min_count": min_count})

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
                lambda y: y + max_shift,
                "shiftCommon",
                announce=False,
                transform_spec={"kind": "offset", "value": max_shift},
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
        self._apply_transformation(_demean, "shiftMean0", True, transform_spec={"kind": "shift_mean_to_zero", "ignore_anomalies": self.ignore_anomalies_cb.isChecked()})

    def shift_x_start_to_zero(self):
        """Create copies where x starts at zero by subtracting the initial x value."""

        self.rebuild_var_lookup()
        created = []
        skipped_datetime = []
        fnames = [os.path.basename(p) for p in self.file_paths]

        def _has_file_prefix(key: str) -> bool:
            for name in fnames:
                if key.startswith(f"{name}::") or key.startswith(f"{name}:"):
                    return True
            return False

        for f_idx, (tsdb, path) in enumerate(zip(self.tsdbs, self.file_paths), start=1):
            fname = os.path.basename(path)

            for u_key, chk in self.var_checkboxes.items():
                if not chk.isChecked():
                    continue

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
                if isinstance(mask, slice):
                    t_win = np.asarray(ts.t[mask])
                    y_win = self.apply_filters(ts)[mask]
                else:
                    if not mask.any():
                        continue
                    t_win = np.asarray(ts.t[mask])
                    y_win = self.apply_filters(ts)[mask]

                if t_win.size == 0:
                    continue

                if np.issubdtype(t_win.dtype, np.datetime64) or ts.dtg_ref is not None:
                    skipped_datetime.append(ts.name)
                    continue

                t_new = np.asarray(t_win, dtype=float) - float(t_win[0])

                filt_tag = self._filter_tag()
                base = f"{ts.name}_x0"
                if filt_tag:
                    base += f"_{filt_tag}"
                base += f"_f{f_idx}"
                new_name = base
                k = 1
                while new_name in tsdb.getm():
                    new_name = f"{base}_{k}"
                    k += 1

                tsdb.add(TimeSeries(new_name, t_new, y_win, dtg_ref=None))
                created.append(new_name)
                self.user_variables = getattr(self, "user_variables", set())
                self.user_variables.add(new_name)

        self._populate_variables(None)

        if created:
            msg = [f"Created {len(created)} x-shifted series."]
            if skipped_datetime:
                msg.append(f"Skipped {len(skipped_datetime)} datetime series (not applicable).")
            QMessageBox.information(self, "X-axis shift complete", "\n".join(msg))
        elif skipped_datetime:
            QMessageBox.information(
                self,
                "No eligible series",
                "Selected series use datetime x-axis; this operation only applies to numeric x values.",
            )
        else:
            QMessageBox.warning(
                self,
                "Nothing new",
                "No eligible selected series found to shift.",
            )

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
            # print(f"Loaded {path}: variables = {list(tsdb.getm().keys())}")
        if errors:
            QMessageBox.warning(self, "Errors occurred", "\n".join([f"{f}: {e}" for f, e in errors]))
        self.refresh_variable_tabs()

    def _build_common_lookup(self):
        """Map safe common names to the corresponding variable in each file."""

        self.common_lookup = {}
        if not self.tsdbs:
            return

        per_file_maps = []
        for tsdb in self.tsdbs:
            canonical_map = {}
            for key in tsdb.getm().keys():
                canonical = re.sub(r"^f\d+_", "", key)
                safe_name = _safe(canonical)
                canonical_map.setdefault(safe_name, set()).add(key)
            per_file_maps.append(canonical_map)

        shared_keys = set(per_file_maps[0].keys())
        for mapping in per_file_maps[1:]:
            shared_keys &= set(mapping.keys())

        for safe_name in shared_keys:
            resolved = []
            for mapping in per_file_maps:
                choices = sorted(mapping[safe_name], key=len)
                resolved.append(choices[0])
            self.common_lookup[safe_name] = resolved

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
        self.user_variables.clear()
        self.work_dir = None
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

        self._build_common_lookup()

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

        self._connect_marker_input_refresh()
        self._refresh_marker_input_defaults()

    def _connect_marker_input_refresh(self):
        for cb in self.var_checkboxes.values():
            if cb.property("_marker_refresh_connected"):
                continue
            cb.toggled.connect(self._refresh_marker_input_defaults)
            cb.setProperty("_marker_refresh_connected", True)

    def _selected_series_marker_context(self):
        for selected_key, checkbox in self.var_checkboxes.items():
            if not checkbox.isChecked():
                continue
            for file_idx, (tsdb, fp) in enumerate(zip(self.tsdbs, self.file_paths), start=1):
                fname = os.path.basename(fp)
                if selected_key.startswith(f"{fname}::"):
                    var = selected_key.split("::", 1)[1]
                elif selected_key.startswith(f"{fname}:"):
                    var = selected_key.split(":", 1)[1]
                elif selected_key in tsdb.getm():
                    var = selected_key
                else:
                    continue

                ts = tsdb.getm().get(var)
                if ts is None:
                    continue

                mask = self.get_time_window(ts)
                dtg_ref = getattr(ts, "dtg_ref", None)
                if isinstance(mask, slice):
                    t_window = ts.t[mask]
                    x_window = ts.x[mask]
                else:
                    if not mask.any():
                        continue
                    t_window = ts.t[mask]
                    x_window = ts.x[mask]
                if len(t_window) == 0:
                    continue

                ts_window = TimeSeries(ts.name, t_window, x_window, dtg_ref=dtg_ref)
                t_plot = self._time_values_for_plot(ts_window)
                if len(t_plot) == 0:
                    continue
                return t_plot[0]
        return None

    def _format_marker_example(self, marker_start):
        if marker_start is None:
            if getattr(self, "plot_datetime_x_cb", None) and self.plot_datetime_x_cb.isChecked():
                return "2024-01-01 00:00:00"
            return "0.0"

        if isinstance(marker_start, pd.Timestamp):
            return marker_start.strftime("%Y-%m-%d %H:%M:%S")

        if isinstance(marker_start, np.datetime64):
            return pd.Timestamp(marker_start).strftime("%Y-%m-%d %H:%M:%S")

        try:
            numeric_value = float(marker_start)
        except (TypeError, ValueError):
            parsed_dt = pd.to_datetime(marker_start, errors="coerce")
            if pd.notna(parsed_dt):
                return parsed_dt.strftime("%Y-%m-%d %H:%M:%S")
            return str(marker_start)

        return f"{numeric_value:g}"

    def _refresh_marker_input_defaults(self, *_args):
        marker_input = getattr(self, "x_axis_marker_input", None)
        if marker_input is None:
            return

        marker_start = self._selected_series_marker_context()
        default_value = self._format_marker_example(marker_start)
        if getattr(self, "plot_datetime_x_cb", None) and self.plot_datetime_x_cb.isChecked():
            marker_input.setPlaceholderText(
                f"Start: {default_value} (example format: YYYY-MM-DD HH:MM:SS)"
            )
        else:
            marker_input.setPlaceholderText(f"Start: {default_value} (numeric x-axis value)")

        current_value = marker_input.text().strip()
        if not current_value or current_value == self._marker_input_auto_value:
            marker_input.setText(default_value)
            self._marker_input_auto_value = default_value
        elif current_value == default_value:
            self._marker_input_auto_value = default_value


    # ------------------------------------------------------------------
    # Compatibility helper -------------------------------------------------
    def _populate_variables(self, *_):
        """Backward‑compatible wrapper used by older callbacks."""
        self.refresh_variable_tabs()

    def highlight_file_tab(self, row):
        if row >= 0 and row + 1 < self.tabs.count():
            self.tabs.setCurrentIndex(row + 1)

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

        # Clear any cached plot state; a new successful render will store it
        # again if embedding is active.
        self._clear_last_plot_call()

        mark_extrema = (
                hasattr(self, "plot_extrema_cb") and self.plot_extrema_cb.isChecked()
        )
        marker_x = self._marker_x_value()

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
                dtg_ref = getattr(ts, "dtg_ref", None)
                if isinstance(mask, slice):
                    ts_win = TimeSeries(ts.name, ts.t[mask], ts.x[mask], dtg_ref=dtg_ref)
                else:
                    if not mask.any():
                        continue
                    ts_win = TimeSeries(ts.name, ts.t[mask], ts.x[mask], dtg_ref=dtg_ref)

                # 3) optional pre-filtering for time-domain plot
                t_plot = self._time_values_for_plot(ts_win)
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
                            t=t_plot,
                            y=ts_win.x,
                            label=disp_label + " [raw]",
                            alpha=1.0,
                        )
                        traces.append(tr)
                        curves.append(dict(t=t_plot, y=ts_win.x, label="Raw", alpha=1.0))
                    if want_lp:
                        fc = float(self.lowpass_cutoff.text() or 0)
                        if fc > 0:
                            y_lp = qats.signal.lowpass(ts_win.x, dt, fc)
                            tr = dict(
                                t=t_plot,
                                y=y_lp,
                                label=disp_label + f" [LP {fc} Hz]",
                                alpha=1.0,
                            )
                            traces.append(tr)
                            curves.append(
                                dict(t=t_plot, y=y_lp, label=f"LP {fc} Hz", alpha=1.0)
                            )
                    if want_hp:
                        fc = float(self.highpass_cutoff.text() or 0)
                        if fc > 0:
                            y_hp = qats.signal.highpass(ts_win.x, dt, fc)
                            tr = dict(
                                t=t_plot,
                                y=y_hp,
                                label=disp_label + f" [HP {fc} Hz]",
                                alpha=1.0,
                            )
                            traces.append(tr)
                            curves.append(
                                dict(t=t_plot, y=y_hp, label=f"HP {fc} Hz", alpha=1.0)
                            )
                    continue  # nothing else to do for time-domain loop
                elif mode == "rolling":
                    y_roll = pd.Series(ts_win.x).rolling(window=roll_window, min_periods=1).mean().to_numpy()
                    traces.append(
                        dict(
                            t=t_plot,
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

        if mode in {"time", "rolling"}:
            traces = self._apply_datetime_xaxis_to_traces(traces)
            for entry in grid_traces.values():
                entry["curves"] = self._apply_datetime_xaxis_to_traces(entry["curves"])

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
                from bokeh.models import HoverTool, ColumnDataSource, Range1d, Span
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
                        x_axis_label=self._x_axis_label(),
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
                    if marker_x is not None:
                        p.add_layout(
                            Span(
                                location=marker_x,
                                dimension="height",
                                line_color="orange",
                                line_width=2,
                                line_dash="dashed",
                            )
                        )
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
                    self._remember_plot_call(
                        self.plot_selected, mode=mode, grid=grid
                    )
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

                if marker_x is not None:
                    fig.add_vline(
                        x=marker_x,
                        line_color="orange",
                        line_width=2,
                        line_dash="dash",
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
                    html = to_html(
                        fig,
                        include_plotlyjs=True,
                        full_html=True,
                        config={"displayModeBar": True},
                    )
                    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
                    with open(tmp.name, "w", encoding="utf-8") as fh:
                        fh.write(html)
                    self._temp_plot_file = tmp.name
                    self.plot_view.load(QUrl.fromLocalFile(tmp.name))
                    self.plot_view.show()
                    self._remember_plot_call(
                        self.plot_selected, mode=mode, grid=grid
                    )
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
                if marker_x is not None:
                    ax.axvline(marker_x, color="orange", linestyle="--", linewidth=2, label="Marker")
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
                self._show_embedded_mpl_figure(fig)
            else:
                self._clear_mpl_embed()
                self.plot_view.hide()
                self._show_mpl_figure_window(fig)
            return
        if mode in ("time", "rolling"):
            if not traces:
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
            self._plot_lines(
                traces,
                title="Rolling Mean" if mode == "rolling" else "Time-series Plot",
                y_label=self.yaxis_label.text() or "Value",
                mark_extrema=mark_extrema,
            )
            embed_cb = getattr(self, "embed_plot_cb", None)
            if (
                    embed_cb is not None
                    and embed_cb.isChecked()
                    and engine in {"plotly", "bokeh"}
            ):
                self._remember_plot_call(self.plot_selected, mode=mode, grid=grid)
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
                self._show_mpl_figure_window(fig)

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
        self._clear_last_plot_call()
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
                html = to_html(
                    fig,
                    include_plotlyjs=True,
                    full_html=True,
                    config={"displayModeBar": True},
                )
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
                with open(tmp.name, "w", encoding="utf-8") as fh:
                    fh.write(html)
                self._temp_plot_file = tmp.name
                self.plot_view.load(QUrl.fromLocalFile(tmp.name))

                self.plot_view.show()
                self._remember_plot_call(
                    self.animate_xyz_scatter_many, dt_resample=dt_resample
                )
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
        self._clear_last_plot_call()
        marker_x = self._marker_x_value()

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
            from bokeh.models import Button, CustomJS, ColumnDataSource, HoverTool, Span
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
                x_axis_label=self._x_axis_label(),
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
            if marker_x is not None:
                p.add_layout(
                    Span(
                        location=marker_x,
                        dimension="height",
                        line_color="orange",
                        line_width=2,
                        line_dash="dashed",
                    )
                )

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
                self._remember_plot_call(
                    self._plot_lines,
                    traces,
                    title,
                    y_label,
                    mark_extrema=mark_extrema,
                )
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
                xaxis_title=self._x_axis_label(),
                yaxis_title=y_label,
                showlegend=True,
                template="plotly_dark" if self.theme_switch.isChecked() else "plotly",
            )
            if marker_x is not None:
                fig.add_vline(
                    x=marker_x,
                    line_color="orange",
                    line_width=2,
                    line_dash="dash",
                )
            if self.theme_switch.isChecked():
                fig.update_layout(
                    paper_bgcolor="#2b2b2b",
                    plot_bgcolor="#2b2b2b",
                    margin=dict(t=0, b=0, l=0, r=0)
                )
            if getattr(self, "embed_plot_cb", None) and self.embed_plot_cb.isChecked():
                from plotly.io import to_html

                import tempfile
                if self._temp_plot_file and os.path.exists(self._temp_plot_file):
                    try:
                        os.remove(self._temp_plot_file)
                    except Exception:
                        pass
                html = to_html(
                    fig,
                    include_plotlyjs=True,
                    full_html=True,
                    config={"displayModeBar": True},
                )
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
                with open(tmp.name, "w", encoding="utf-8") as fh:
                    fh.write(html)
                self._temp_plot_file = tmp.name
                self.plot_view.load(QUrl.fromLocalFile(tmp.name))

                self.plot_view.show()
                self._remember_plot_call(
                    self._plot_lines,
                    traces,
                    title,
                    y_label,
                    mark_extrema=mark_extrema,
                )
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
        if marker_x is not None:
            ax.axvline(marker_x, color="orange", linestyle="--", linewidth=2, label="Marker")

        ax.set_title(title)
        ax.set_xlabel(self._x_axis_label())
        ax.set_ylabel(y_label)
        ax.legend(loc="best")
        fig.tight_layout()

        if getattr(self, "embed_plot_cb", None) and self.embed_plot_cb.isChecked():
            # Use a native Matplotlib canvas instead of the HTML viewer
            self._show_embedded_mpl_figure(fig)
        else:
            self._clear_mpl_embed()
            self.plot_view.hide()
            self._show_mpl_figure_window(fig)

    def _time_values_for_plot(self, ts: TimeSeries):
        """Return datetime values for plotting when enabled and available."""
        if not (getattr(self, "plot_datetime_x_cb", None) and self.plot_datetime_x_cb.isChecked()):
            return ts.t

        dtg_vals = getattr(ts, "dtg_time", None)
        if dtg_vals is not None:
            arr = np.asarray(dtg_vals)
            if arr.size:
                converted = pd.to_datetime(arr, errors="coerce")
                if converted.notna().any():
                    return converted

        return self._convert_to_datetime_if_possible(ts.t)

    def _marker_x_value(self):
        marker_input = getattr(self, "x_axis_marker_input", None)
        if marker_input is None:
            return None

        raw_value = marker_input.text().strip()
        if not raw_value:
            return None

        if getattr(self, "plot_datetime_x_cb", None) and self.plot_datetime_x_cb.isChecked():
            parsed_dt = pd.to_datetime(raw_value, errors="coerce")
            if pd.notna(parsed_dt):
                return parsed_dt

        try:
            return float(raw_value)
        except ValueError:
            parsed_dt = pd.to_datetime(raw_value, errors="coerce")
            if pd.notna(parsed_dt):
                return parsed_dt

        QMessageBox.warning(
            self,
            "Invalid x-axis marker",
            "Enter a numeric x-axis location or a datetime value that pandas can parse.",
        )
        return None

    def _x_axis_label(self) -> str:
        if getattr(self, "plot_datetime_x_cb", None) and self.plot_datetime_x_cb.isChecked():
            return "Datetime"
        return "Time"

    def _apply_datetime_xaxis_to_traces(self, traces):
        if not (getattr(self, "plot_datetime_x_cb", None) and self.plot_datetime_x_cb.isChecked()):
            return traces

        converted = []
        for trace in traces:
            tr = dict(trace)
            tr["t"] = self._convert_to_datetime_if_possible(trace.get("t", []))
            converted.append(tr)
        return converted

    @staticmethod
    def _convert_to_datetime_if_possible(time_values):
        arr = np.asarray(time_values)
        if arr.size == 0:
            return time_values

        if np.issubdtype(arr.dtype, np.datetime64):
            return pd.to_datetime(arr)

        if arr.dtype.kind in {"f", "i", "u"}:
            finite = arr[np.isfinite(arr)] if arr.dtype.kind == "f" else arr
            if finite.size == 0:
                return time_values
            magnitude = float(np.nanmedian(np.abs(finite.astype(float))))
            unit = None
            if 1e8 <= magnitude <= 1e11:
                unit = "s"
            elif 1e11 < magnitude <= 1e14:
                unit = "ms"
            elif 1e14 < magnitude <= 1e17:
                unit = "us"
            elif 1e17 < magnitude <= 1e20:
                unit = "ns"
            if unit is None:
                return time_values

            converted = pd.to_datetime(arr, unit=unit, errors="coerce")
            if converted.notna().any():
                return converted
            return time_values

        converted = pd.to_datetime(arr, errors="coerce")
        if converted.notna().any():
            return converted
        return time_values

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

    @staticmethod
    def _time_vectors_match(ref, other):
        """Return True if two time axes share identical samples."""

        ref = np.asarray(ref)
        other = np.asarray(other)

        if ref.shape != other.shape:
            return False
        if ref.size == 0:
            return True

        if np.array_equal(ref, other):
            return True

        try:
            return np.allclose(ref, other, rtol=1e-9, atol=1e-9, equal_nan=True)
        except TypeError:
            return False

    def _group_series_by_timebase(self, tsdb):
        """Partition time series into groups that share a common time base."""

        groups = []
        names = list(getattr(tsdb, "register_keys", []))
        if not names:
            return groups

        used = set()
        for name in names:
            if name in used:
                continue

            ts_ref = tsdb.get(name=name)
            if ts_ref is None:
                continue

            group = [name]
            used.add(name)
            t_ref = ts_ref.t

            for other_name in names:
                if other_name in used:
                    continue

                ts_other = tsdb.get(name=other_name)
                if ts_other is None:
                    continue

                if self._time_vectors_match(t_ref, ts_other.t):
                    group.append(other_name)
                    used.add(other_name)

            groups.append(group)

        return groups

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

        series_items = []
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

                series_items.append((key, np.asarray(t), np.asarray(y)))

        if not series_items:
            QMessageBox.warning(self, "No data", "No data found for the selected variables.")
            return

        shared_time = series_items[0][1]
        has_common_time = all(self._time_vectors_match(shared_time, t) for _, t, _ in series_items)

        if has_common_time:
            data = {"time": shared_time}
            data.update({key: y for key, _, y in series_items})
            df = pd.DataFrame(data)
        else:
            series_list = []
            for key, t, y in series_items:
                series_list.append(pd.Series(t, name=f"{key}_t"))
                series_list.append(pd.Series(y, name=key))
            df = pd.concat(series_list, axis=1)

        df.to_csv(path, index=False)
        QMessageBox.information(self, "Exported", f"Exported {len(sel_keys)} series to\n{os.path.basename(path)}")

    def launch_qats(self):
        if not getattr(self, "work_dir", None):
            self.work_dir = QFileDialog.getExistingDirectory(self, "Select Work Folder for AnyQATS Export")
            if not self.work_dir:
                return
        ts_paths = []
        for i, (tsdb, original_path) in enumerate(zip(self.tsdbs, self.file_paths), start=1):
            groups = self._group_series_by_timebase(tsdb)
            if not groups:
                continue

            base_label = os.path.splitext(os.path.basename(original_path))[0] or f"file_{i}"

            for group_idx, names in enumerate(groups, start=1):
                temp_db = TsDB()
                copied = []
                for key in names:
                    ts_obj = tsdb.get(name=key)
                    if ts_obj is None:
                        continue
                    clone = ts_obj.__copy__()
                    temp_db.add(clone)
                    copied.append(clone)

                if not copied:
                    continue

                is_user_group = all(ts.name in getattr(self, "user_variables", set()) for ts in copied)

                if len(groups) == 1:
                    filename = f"temp_{i}.ts"
                else:
                    suffix = "_user" if is_user_group else f"_part{group_idx}"
                    filename = f"temp_{i}_{base_label}{suffix}.ts"

                ts_path = os.path.join(self.work_dir, filename)

                # Group members share the same time base – enforce a shared grid per file only.
                temp_db.export(ts_path, names=list(temp_db.getm().keys()), force_common_time=True)
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
            QMessageBox.information(self, "Multiple Variables",
                                    "Only the first selected variable will be used for EVA.")

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
            matches = [
                (i, tsdb.getm().get(selected))
                for i, tsdb in enumerate(self.tsdbs)
                if selected in tsdb.getm()
            ]
            if matches:
                index, ts = matches[0]
                raw_key = ts.name
            else:
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

    def open_fatigue_tool(self) -> None:
        """Launch the fatigue dialog using all checked variables."""

        from collections import Counter

        self.rebuild_var_lookup()
        selected_keys = [k for k, cb in self.var_checkboxes.items() if cb.isChecked()]
        if not selected_keys:
            QMessageBox.warning(self, "No Variables", "Please select at least one variable.")
            return

        series_entries: list[FatigueSeries] = []
        fname_counts = Counter(os.path.basename(p) for p in self.file_paths)

        for file_idx, (tsdb, fp) in enumerate(zip(self.tsdbs, self.file_paths), start=1):
            fname = os.path.basename(fp)
            fname_disp = fname if fname_counts[fname] == 1 else f"{fname} ({file_idx})"

            for key in selected_keys:
                if key.startswith(f"{fname}::"):
                    var_name = key.split("::", 1)[1]
                elif key.startswith(f"{fname}:"):
                    var_name = key.split(":", 1)[1]
                elif key in tsdb.getm():
                    var_name = key
                else:
                    continue

                ts = tsdb.getm().get(var_name)
                if ts is None:
                    continue

                filtered = self.apply_filters(ts)
                mask = self.get_time_window(ts)
                if isinstance(mask, slice):
                    t_window = ts.t[mask]
                    x_window = filtered[mask]
                else:
                    if mask is None or not np.any(mask):
                        continue
                    t_window = ts.t[mask]
                    x_window = filtered[mask]

                if t_window.size == 0:
                    continue

                valid = np.isfinite(x_window)
                if not np.any(valid):
                    continue

                t_valid = t_window[valid]
                x_valid = x_window[valid]
                duration = float(t_valid[-1] - t_valid[0]) if t_valid.size > 1 else 0.0
                label = f"{fname_disp}: {var_name}"
                series_entries.append(FatigueSeries(label, x_valid, duration, fname, var_name))

        if not series_entries:
            QMessageBox.warning(
                self,
                "No valid data",
                "Could not find any valid samples for the selected variables.",
            )
            return

        dlg = FatigueDialog(series_entries, self)
        dlg.exec()

    def open_rao_tool(self) -> None:
        """Launch the RAO dialog for selected time series."""

        self.rebuild_var_lookup()
        selected_keys = [k for k, cb in self.var_checkboxes.items() if cb.isChecked()]
        if len(selected_keys) < 1:
            QMessageBox.warning(
                self,
                "No variables selected",
                "Please check at least one variable for RAO generation.",
            )
            return

        series_data: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        spectral_data: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for tsdb, fp in zip(self.tsdbs, self.file_paths):
            fname = os.path.basename(fp)
            tsdb_map = tsdb.getm()
            for key in selected_keys:
                if key in series_data:
                    continue
                if key.startswith(f"{fname}::"):
                    var_name = key.split("::", 1)[1]
                elif key.startswith(f"{fname}:"):
                    var_name = key.split(":", 1)[1]
                elif key in tsdb_map:
                    var_name = key
                else:
                    continue

                ts = tsdb_map.get(var_name)
                if ts is None:
                    continue

                mask = self.get_time_window(ts)
                if isinstance(mask, slice):
                    t = np.asarray(ts.t[mask], dtype=float)
                    y = np.asarray(self.apply_filters(ts)[mask], dtype=float)
                else:
                    if not mask.any():
                        continue
                    t = np.asarray(ts.t[mask], dtype=float)
                    y = np.asarray(self.apply_filters(ts)[mask], dtype=float)

                if t.size < 8:
                    continue
                if not np.all(np.isfinite(t)) or not np.all(np.isfinite(y)):
                    continue

                series_data[key] = (t, y)
                freq_hz = getattr(ts, "freq_hz", None)
                rao_amp = getattr(ts, "rao_amp", None)
                if freq_hz is not None and rao_amp is not None:
                    freq_hz_arr = np.asarray(freq_hz, dtype=float)
                    rao_amp_arr = np.asarray(rao_amp, dtype=float)
                    if freq_hz_arr.size and freq_hz_arr.size == rao_amp_arr.size:
                        spectral_data[key] = (freq_hz_arr, rao_amp_arr)

        if len(series_data) < 1:
            QMessageBox.warning(
                self,
                "No usable data",
                "Could not build valid series from the current selection/time window.",
            )
            return

        labels = [k for k in selected_keys if k in series_data]
        dlg = RAODialog(
            labels=labels,
            series_data=series_data,
            spectral_data=spectral_data,
            parent=self,
        )
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

    def _clear_last_plot_call(self) -> None:
        self._last_plot_call = None

    def _remember_plot_call(
            self, callback: Callable[..., None], /, *args, **kwargs
    ) -> None:
        self._last_plot_call = (callback, args, kwargs)

    def _refresh_embedded_plot(self) -> None:
        if self._last_plot_call is None:
            return
        embed_cb = getattr(self, "embed_plot_cb", None)
        if embed_cb is None or not embed_cb.isChecked():
            return
        if not self.plot_view.isVisible():
            return
        engine = (
            self.plot_engine_combo.currentText().lower()
            if hasattr(self, "plot_engine_combo")
            else ""
        )
        if engine not in {"plotly", "bokeh"}:
            return
        callback, args, kwargs = self._last_plot_call
        if callback is None or self._refreshing_plot:
            return
        self._refreshing_plot = True
        try:
            callback(*args, **kwargs)
        except Exception:
            traceback.print_exc()
        finally:
            self._refreshing_plot = False

    def toggle_dark_theme(self, state):

        # ``state`` comes from the checkbox signal but using ``isChecked`` is
        # more robust across Qt bindings.
        if self.theme_switch.isChecked():
            self.apply_dark_palette()
        else:
            self.apply_light_palette()
        # Refresh any open Matplotlib canvases so the new palette is used
        for canvas in self.findChildren(FigureCanvasQTAgg):
            # ``draw_idle`` schedules a redraw without blocking the UI thread,
            # keeping the theme toggle responsive even when large Matplotlib
            # figures are embedded. Fall back to ``draw`` for older backends
            # that might not provide the idle variant.
            draw_fn = getattr(canvas, "draw_idle", None)
            if callable(draw_fn):
                draw_fn()
            else:
                canvas.draw()

        # Refresh embedded HTML-based plots (Plotly/Bokeh) so their templates
        # follow the new palette.
        self._refresh_embedded_plot()

    def _on_engine_changed(self, text):
        """Update layout when the plotting engine selection changes."""
        engine = text.lower()
        if engine != "default" and self._mpl_canvas is not None:
            self._clear_mpl_embed()
        if self.embed_plot_cb.isChecked():
            # Refresh layout so the appropriate widget is shown
            self.toggle_embed_layout(True)

    def _show_mpl_figure_window(self, fig):
        """Show a Matplotlib figure in a desktop window when not embedding.

        ``Figure.show()`` can raise for figures not managed by pyplot.
        Prefer the canvas manager when available, then fall back to non-blocking
        pyplot display.
        """
        try:
            manager = getattr(fig.canvas, "manager", None)
            if manager is not None and hasattr(manager, "show"):
                manager.show()
                return
        except Exception:
            pass

        try:
            plt.figure(fig.number)
            plt.show(block=False)
        except Exception:
            pass

    def _clear_mpl_embed(self):
        """Remove any embedded Matplotlib canvas and toolbar."""
        if self._mpl_toolbar is not None:
            self.right_outer_layout.removeWidget(self._mpl_toolbar)
            self._mpl_toolbar.setParent(None)
            self._mpl_toolbar.deleteLater()
            self._mpl_toolbar = None
        if self._mpl_canvas is not None:
            self.right_outer_layout.removeWidget(self._mpl_canvas)
            self._mpl_canvas.setParent(None)
            self._mpl_canvas = None

    def _show_embedded_mpl_figure(self, fig):
        """Render ``fig`` with a Qt Matplotlib toolbar when embedding plots."""
        self._clear_mpl_embed()
        self._mpl_canvas = FigureCanvasQTAgg(fig)
        self._mpl_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._mpl_toolbar = NavigationToolbar2QT(self._mpl_canvas, self)
        self.right_outer_layout.addWidget(self._mpl_toolbar)
        self.right_outer_layout.addWidget(self._mpl_canvas)
        self._mpl_toolbar.show()
        self._mpl_canvas.show()
        self.plot_view.hide()

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
                if self._mpl_toolbar is not None:
                    self._mpl_toolbar.show()
                self._mpl_canvas.show()
                self.plot_view.hide()
            else:
                self.plot_view.show()
                if self._mpl_toolbar is not None:
                    self._mpl_toolbar.hide()
                if self._mpl_canvas is not None:
                    self._mpl_canvas.hide()
        else:
            self.plot_view.hide()
            if self._mpl_toolbar is not None:
                self._mpl_toolbar.hide()
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


__all__ = ['TimeSeriesEditorQt']
