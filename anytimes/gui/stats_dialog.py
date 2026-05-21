"""Dialog for displaying statistics with filtering and plotting."""
from __future__ import annotations

import os
import re
import tempfile
import warnings
from dataclasses import dataclass, field
from datetime import date, datetime

import anyqats as qats
import numpy as np
from anyqats import TimeSeries
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PySide6.QtCore import QEvent, Qt, QTimer, QUrl
from PySide6.QtGui import QGuiApplication, QKeySequence
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QRadioButton,
    QSplitter,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QWidget,
    QVBoxLayout,
    QHeaderView,
    QSizePolicy,

)

try:
    from PySide6.QtWebEngineWidgets import QWebEngineView
except ImportError:  # pragma: no cover - platform packaging decides this.
    QWebEngineView = None

from .sortable_table_widget_item import SortableTableWidgetItem
from .layout_utils import apply_initial_size
from .filename_parser import choose_parse_target, parse_embedded_values


@dataclass
class _StatsSeriesState:
    """Validated series state shared by table and plot renderers."""

    sid: str
    info: dict
    t: np.ndarray
    plot_t: np.ndarray
    y: np.ndarray
    time_is_datetime: bool
    qc_status: str
    qc_messages: list[str] = field(default_factory=list)
    psd_message: str = ""

class StatsDialog(QDialog):
    """Qt table dialog with copy and plotting features."""

    _PSD_CUMULATIVE_POWER_COVERAGE = 0.995
    _PSD_RELATIVE_LEVEL_THRESHOLD = 1.0e-3
    _PSD_XLIM_PADDING = 1.1
    _HISTOGRAM_BINS = 30
    _WEB_TIME_MAX_POINTS = 24000
    _WEB_PSD_MAX_POINTS = 12000
    _CORE_STATS_HEADERS = {
        "file",
        "uniqueness",
        "variable",
        "varuniqueness",
        "editor filter",
        "local filter",
        "qc",
        "qc messages",
        "samples",
        "finite samples",
        "nan fraction",
        "time step qc",
        "psd qc",
        "start",
        "end",
        "duration",
        "dtavg",
        "mean",
        "std",
        "min",
        "max",
        "skew",
        "kurt",
        "tz",
    }

    def __init__(self, series_info, parent=None, preferred_plot_engine: str = "default"):
        super().__init__(parent)
        self.setWindowTitle("Statistics Table")
        self.setWindowFlag(Qt.Window)
        # allow maximizing the statistics window
        self.setWindowFlags(self.windowFlags() | Qt.WindowMaximizeButtonHint)
        apply_initial_size(
            self,
            desired_width=1100,
            desired_height=720,
            min_width=820,
            min_height=560,
            width_ratio=0.9,
            height_ratio=0.9,
        )

        self.series_info = series_info
        self.ts_dict: dict[str, _StatsSeriesState] = {}
        self.selected_columns: set[int] = set()
        self._parsed_headers: list[str] = []
        self._fatigue_headers: list[str] = []
        self._table_headers: list[str] = []
        self._temp_plot_file: str | None = None
        self.preferred_plot_engine = self._normalize_plot_engine(preferred_plot_engine)

        main_layout = QVBoxLayout(self)

        source_group = QGroupBox("Source and filter context")
        source_layout = QVBoxLayout(source_group)
        self.source_context_label = QLabel()
        self.source_context_label.setWordWrap(True)
        source_layout.addWidget(self.source_context_label)

        order_layout = QHBoxLayout()
        order_layout.addWidget(QLabel("Load order:"))
        self.order_combo = QComboBox()
        self.order_combo.addItems(["Files → Variables", "Variables → Files"])
        order_layout.addWidget(self.order_combo)
        order_layout.addStretch()
        source_layout.addLayout(order_layout)
        main_layout.addWidget(source_group)

        # Frequency filter controls
        freq_group = QGroupBox("Local frequency filter")
        freq_layout = QGridLayout(freq_group)
        self.filter_none_rb = QRadioButton("None")
        self.filter_lowpass_rb = QRadioButton("Low-pass")
        self.filter_highpass_rb = QRadioButton("High-pass")
        self.filter_bandpass_rb = QRadioButton("Band-pass")
        self.filter_bandblock_rb = QRadioButton("Band-block")
        self.filter_none_rb.setChecked(True)
        self.fatigue_filter_cb = QCheckBox(
            "Add std and tz high pass and low pass filtered values in new columns"
        )
        self.fatigue_filter_cb.setChecked(False)
        self.lowpass_cutoff = QLineEdit("0.04")
        self.highpass_cutoff = QLineEdit("0.04")
        self.bandpass_low = QLineEdit("0.0")
        self.bandpass_high = QLineEdit("0.0")
        self.bandblock_low = QLineEdit("0.0")
        self.bandblock_high = QLineEdit("0.0")
        row = 0
        freq_layout.addWidget(self.filter_none_rb, row, 0)
        freq_layout.addWidget(self.fatigue_filter_cb, row, 1, 1, 5)
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
        row += 1
        metric_group = QGroupBox("Table metric controls")
        metric_layout = QGridLayout(metric_group)
        metric_layout.addWidget(QLabel("Visible columns:"), 0, 0)
        self.column_preset_combo = QComboBox()
        self.column_preset_combo.addItems(
            ["Core metrics", "Distribution metrics", "Fatigue metrics", "Filename metrics", "All metrics"]
        )
        metric_layout.addWidget(self.column_preset_combo, 0, 1)
        metric_layout.addWidget(QLabel("Comparison metrics:"), 1, 0, 1, 2)
        self.metric_list = QListWidget()
        self.metric_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.metric_list.setMinimumHeight(110)
        metric_layout.addWidget(self.metric_list, 2, 0, 1, 2)

        filter_metric_layout = QHBoxLayout()
        filter_metric_layout.addWidget(freq_group, stretch=3)
        filter_metric_layout.addWidget(metric_group, stretch=2)
        main_layout.addLayout(filter_metric_layout)

        plot_group = QGroupBox("Plot controls")
        hline_layout = QHBoxLayout(plot_group)
        hline_layout.addWidget(QLabel("Histogram lines:"))
        self.hist_lines_edit = QLineEdit()
        self.hist_lines_edit.setPlaceholderText("e.g. 1.0, 2.5")
        hline_layout.addWidget(self.hist_lines_edit)
        self.hist_show_text_cb = QCheckBox("Show bar text")
        self.hist_show_text_cb.setChecked(True)
        hline_layout.addWidget(self.hist_show_text_cb)
        self.parse_filename_cb = QCheckBox("Parse file name")
        self.parse_filename_cb.setChecked(False)
        hline_layout.addWidget(self.parse_filename_cb)
        self.plot_period_cb = QCheckBox("Period [s]")
        self.plot_period_cb.setChecked(False)
        hline_layout.addWidget(self.plot_period_cb)
        self.render_warning_label = QLabel()
        self.render_warning_label.setWordWrap(True)
        self.render_warning_label.setVisible(False)
        hline_layout.addWidget(self.render_warning_label, stretch=1)
        hline_layout.addStretch()
        main_layout.addWidget(plot_group)

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
        plot_layout = QVBoxLayout()
        self.line_fig = Figure(figsize=(5, 3))
        self.line_canvas = FigureCanvasQTAgg(self.line_fig)
        self.psd_fig = Figure(figsize=(5, 3))
        self.psd_canvas = FigureCanvasQTAgg(self.psd_fig)

        for canvas in (self.line_canvas, self.psd_canvas):
            canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        ts_layout = QHBoxLayout()
        ts_layout.addLayout(self._create_plot_panel(self.line_canvas))
        ts_layout.addLayout(self._create_plot_panel(self.psd_canvas))
        hist_layout = QHBoxLayout()
        self.hist_fig_rows = Figure(figsize=(4, 3))
        self.hist_canvas_rows = FigureCanvasQTAgg(self.hist_fig_rows)
        self.hist_fig_cols = Figure(figsize=(4, 3))
        self.hist_canvas_cols = FigureCanvasQTAgg(self.hist_fig_cols)

        for canvas in (self.hist_canvas_rows, self.hist_canvas_cols):
            canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        plot_layout.addLayout(ts_layout)
        hist_layout.addLayout(self._create_plot_panel(self.hist_canvas_rows))
        hist_layout.addLayout(self._create_plot_panel(self.hist_canvas_cols))
        plot_layout.addLayout(hist_layout)
        self.mpl_plot_widget = QWidget()
        self.mpl_plot_widget.setLayout(plot_layout)
        self.plot_stack = QStackedWidget()
        self.plot_stack.addWidget(self.mpl_plot_widget)
        self.web_plot_view = QWebEngineView() if QWebEngineView is not None else None
        if self.web_plot_view is not None:
            self.plot_stack.addWidget(self.web_plot_view)
        self.results_splitter = QSplitter(Qt.Vertical)
        self.results_splitter.addWidget(self.table)
        self.results_splitter.addWidget(self.plot_stack)
        self.results_splitter.setChildrenCollapsible(False)
        self.results_splitter.setStretchFactor(0, 2)
        self.results_splitter.setStretchFactor(1, 3)
        self.results_splitter.setSizes([230, 430])
        main_layout.addWidget(self.results_splitter, stretch=1)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self.copy_selected_btn = QPushButton("Copy selected as TSV")
        self.copy_selected_btn.clicked.connect(self.copy_selected_as_tsv)
        btn_row.addWidget(self.copy_selected_btn)

        self.copy_all_btn = QPushButton("Copy all as TSV")
        self.copy_all_btn.clicked.connect(self.copy_all_as_tsv)
        btn_row.addWidget(self.copy_all_btn)
        main_layout.addLayout(btn_row)


        self._connect_signals()
        self._init_inherited_filter()
        self._update_source_context()
        self.update_data()

    def showEvent(self, event: QEvent) -> None:
        super().showEvent(event)
        QTimer.singleShot(0, self._refresh_plots)

    def closeEvent(self, event: QEvent) -> None:
        self._remove_temp_plot_file()
        super().closeEvent(event)

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
        self.parse_filename_cb.toggled.connect(self.update_data)
        self.plot_period_cb.toggled.connect(self.update_plots)
        self.fatigue_filter_cb.toggled.connect(self.update_data)
        self.order_combo.currentIndexChanged.connect(self.update_data)
        self.column_preset_combo.currentIndexChanged.connect(self._apply_column_visibility)
        self.metric_list.itemSelectionChanged.connect(self._on_metric_selection_changed)
        self.table.selectionModel().selectionChanged.connect(self.update_plots)

    def _init_inherited_filter(self):
        """Start Statistics with the filter state currently active in the editor."""
        context = self.series_info[0].get("editor_filter", {}) if self.series_info else {}
        mode = context.get("mode", "none")
        self.lowpass_cutoff.setText(str(context.get("cutoff_high") or self.lowpass_cutoff.text()))
        self.highpass_cutoff.setText(str(context.get("cutoff_low") or self.highpass_cutoff.text()))
        if mode in {"bandpass", "bandblock"}:
            low_text = str(context.get("cutoff_low") or "")
            high_text = str(context.get("cutoff_high") or "")
            if mode == "bandpass":
                self.bandpass_low.setText(low_text)
                self.bandpass_high.setText(high_text)
            else:
                self.bandblock_low.setText(low_text)
                self.bandblock_high.setText(high_text)
        buttons = {
            "lowpass": self.filter_lowpass_rb,
            "highpass": self.filter_highpass_rb,
            "bandpass": self.filter_bandpass_rb,
            "bandblock": self.filter_bandblock_rb,
        }
        buttons.get(mode, self.filter_none_rb).setChecked(True)

    def _update_source_context(self):
        """Describe the inherited time window and editor filter in the dialog."""
        if not self.series_info:
            self.source_context_label.setText("No series context available.")
            return
        filter_descriptions = {
            str(info.get("editor_filter", {}).get("description") or "Editor filter: none")
            for info in self.series_info
        }
        filter_text = "; ".join(sorted(filter_descriptions))
        first_window = self.series_info[0].get("time_window", {})
        start = first_window.get("datetime_start", first_window.get("start"))
        end = first_window.get("datetime_end", first_window.get("end"))
        if start is not None and end is not None:
            window_text = f"Selected window: {self._format_time_value(start)} to {self._format_time_value(end)}"
        else:
            window_text = "Selected window: empty or unavailable"
        self.source_context_label.setText(
            f"{window_text}. {filter_text}. Local filter controls below start from that editor state."
        )

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

    @staticmethod
    def _normalize_plot_engine(engine: str | None) -> str:
        """Normalize configured plot engine names used by the stats tool."""
        value = (engine or "").strip().lower()
        if value == "ploly":
            value = "plotly"
        if value not in {"plotly", "bokeh", "default"}:
            value = "default"
        return value

    @staticmethod
    def _parse_from_filename(name: str) -> dict[str, float]:
        """Extract key-value pairs embedded in a filename."""

        if not name:
            return {}
        base = os.path.splitext(name)[0]
        pattern = re.compile(r"([A-Za-z]+)([-+]?(?:\d+(?:[._]\d+)*))")
        parsed: dict[str, float] = {}
        for key, val in pattern.findall(base):
            if not val:
                continue
            try:
                parsed[key] = float(val.replace("_", "."))
            except ValueError:
                continue
        return parsed

    def _parse_positive_float(self, text: str) -> float | None:
        try:
            value = float(text)
        except (TypeError, ValueError):
            return None
        return value if value > 0 else None

    @staticmethod
    def _is_datetime_axis(values: np.ndarray) -> bool:
        arr = np.asarray(values)
        if np.issubdtype(arr.dtype, np.datetime64):
            return True
        if not arr.size:
            return False
        return isinstance(arr.flat[0], (date, datetime, np.datetime64))

    @staticmethod
    def _format_time_value(value):
        if isinstance(value, np.datetime64):
            return np.datetime_as_string(value, unit="s")
        if isinstance(value, datetime):
            return value.isoformat(sep=" ")
        if isinstance(value, date):
            return value.isoformat()
        return StatsDialog._format_stat_value(value)

    def _plot_time_from_info(self, info: dict, t: np.ndarray) -> np.ndarray:
        dtg_time = info.get("dtg_time")
        if dtg_time is not None:
            dtg_arr = np.asarray(dtg_time)
            if dtg_arr.shape == t.shape:
                return dtg_arr
        return t

    def _current_filter_spec(self) -> dict[str, str | float | None]:
        mode = "none"
        low_text = high_text = ""
        if self.filter_lowpass_rb.isChecked():
            mode = "lowpass"
            high_text = self.lowpass_cutoff.text().strip()
        elif self.filter_highpass_rb.isChecked():
            mode = "highpass"
            low_text = self.highpass_cutoff.text().strip()
        elif self.filter_bandpass_rb.isChecked():
            mode = "bandpass"
            low_text = self.bandpass_low.text().strip()
            high_text = self.bandpass_high.text().strip()
        elif self.filter_bandblock_rb.isChecked():
            mode = "bandblock"
            low_text = self.bandblock_low.text().strip()
            high_text = self.bandblock_high.text().strip()
        return {
            "mode": mode,
            "low_text": low_text,
            "high_text": high_text,
            "cutoff_low": self._parse_positive_float(low_text),
            "cutoff_high": self._parse_positive_float(high_text),
        }

    @staticmethod
    def _filter_label(spec: dict[str, str | float | None]) -> str:
        mode = str(spec["mode"])
        low = str(spec["low_text"] or "n/a")
        high = str(spec["high_text"] or "n/a")
        if mode == "lowpass":
            return f"Low-pass ({high} Hz)"
        if mode == "highpass":
            return f"High-pass ({low} Hz)"
        if mode == "bandpass":
            return f"Band-pass ({low}-{high} Hz)"
        if mode == "bandblock":
            return f"Band-block ({low}-{high} Hz)"
        return "None"

    @staticmethod
    def _filter_spec_messages(spec: dict[str, str | float | None]) -> list[str]:
        mode = spec["mode"]
        low = spec["cutoff_low"]
        high = spec["cutoff_high"]
        if mode == "none":
            return []
        if mode == "lowpass" and high is None:
            return ["Local low-pass cutoff is invalid; unfiltered values are shown."]
        if mode == "highpass" and low is None:
            return ["Local high-pass cutoff is invalid; unfiltered values are shown."]
        if mode in {"bandpass", "bandblock"}:
            if low is None or high is None:
                return [f"Local {mode} cutoffs are invalid; unfiltered values are shown."]
            if high <= low:
                return [f"Local {mode} high cutoff must exceed low cutoff; unfiltered values are shown."]
        return []

    @staticmethod
    def _time_qc(t: np.ndarray, finite_mask: np.ndarray) -> tuple[str, list[str]]:
        messages: list[str] = []
        t_valid = np.asarray(t, dtype=float)[finite_mask]
        if t_valid.size < 2:
            return "Unavailable", ["Time-step quality unavailable with fewer than two finite samples."]
        diffs = np.diff(t_valid)
        finite_diffs = diffs[np.isfinite(diffs)]
        if finite_diffs.size == 0 or np.any(finite_diffs <= 0):
            return "Invalid", ["Time values are non-increasing or contain invalid intervals."]
        if float(t_valid[-1] - t_valid[0]) <= 0:
            return "Invalid", ["Usable time span is not positive."]
        median_dt = float(np.median(finite_diffs))
        if not np.allclose(finite_diffs, median_dt, rtol=0.02, atol=max(abs(median_dt), 1.0) * 1e-9):
            messages.append("Sampling is irregular; PSD uses a resampling fallback where possible.")
            return "Irregular", messages
        return "Regular", messages

    def _filter_signal_with_qc(
        self,
        t: np.ndarray,
        x: np.ndarray,
        mode: str,
        cutoff_low: float | None = None,
        cutoff_high: float | None = None,
    ) -> tuple[np.ndarray, list[str]]:
        messages: list[str] = []
        x = np.asarray(x, dtype=float)
        nanmask = np.isfinite(x)
        if not np.any(nanmask):
            return x, ["No finite samples are available for filtering."]
        valid_idx = np.where(nanmask)[0]
        x_valid = x[valid_idx]
        t_valid = np.asarray(t, dtype=float)[valid_idx]
        x_filt = x_valid
        if mode == "none":
            return x.copy(), messages
        if len(x_valid) <= 1:
            return x.copy(), ["Filter needs at least two finite samples; unfiltered values are shown."]
        diffs = np.diff(t_valid)
        if not diffs.size or not np.all(np.isfinite(diffs)) or np.median(diffs) <= 0:
            return x.copy(), ["Filter needs a positive numeric time step; unfiltered values are shown."]
        try:
            dt = float(np.median(diffs))
            if mode == "lowpass" and cutoff_high is not None:
                x_filt = qats.signal.lowpass(x_valid, dt, cutoff_high)
            elif mode == "highpass" and cutoff_low is not None:
                x_filt = qats.signal.highpass(x_valid, dt, cutoff_low)
            elif mode == "bandpass" and cutoff_low is not None and cutoff_high is not None:
                x_filt = qats.signal.bandpass(x_valid, dt, cutoff_low, cutoff_high)
            elif mode == "bandblock" and cutoff_low is not None and cutoff_high is not None:
                x_filt = qats.signal.bandblock(x_valid, dt, cutoff_low, cutoff_high)
        except Exception as exc:
            messages.append(f"Local filter failed ({exc}); unfiltered values are shown.")
            x_filt = x_valid
        x_out = np.full_like(x, np.nan)
        x_out[valid_idx] = x_filt
        return x_out, messages

    def _filter_signal(
        self,
        t: np.ndarray,
        x: np.ndarray,
        mode: str,
        cutoff_low: float | None = None,
        cutoff_high: float | None = None,
    ) -> np.ndarray:
        filtered, _ = self._filter_signal_with_qc(
            t, x, mode, cutoff_low=cutoff_low, cutoff_high=cutoff_high
        )
        return filtered

    def _apply_filter_with_qc(self, t: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, list[str]]:
        spec = self._current_filter_spec()
        messages = self._filter_spec_messages(spec)
        if messages:
            return np.asarray(x, dtype=float).copy(), messages
        return self._filter_signal_with_qc(
            t,
            x,
            str(spec["mode"]),
            cutoff_low=spec["cutoff_low"],
            cutoff_high=spec["cutoff_high"],
        )

    def _apply_filter(self, t: np.ndarray, x: np.ndarray) -> np.ndarray:
        filtered, _ = self._apply_filter_with_qc(t, x)
        return filtered

    def _fatigue_filter_stats(self, t: np.ndarray, x: np.ndarray, mode: str, cutoff: float | None) -> tuple[float, float]:
        if cutoff is None:
            return np.nan, np.nan
        if mode == "highpass":
            y = self._filter_signal(t, x, mode, cutoff_low=cutoff)
        else:
            y = self._filter_signal(t, x, mode, cutoff_high=cutoff)
        if not np.any(np.isfinite(y)):
            return np.nan, np.nan
        ts_tmp = TimeSeries("tmp", t, y)
        stats = ts_tmp.stats()
        return stats.get("tz", np.nan), stats.get("std", np.nan)

    @staticmethod
    def _format_stat_value(value):
        if isinstance(value, float):
            if np.isnan(value) or np.isinf(value):
                return value
            return float(
                np.format_float_positional(value, precision=4, unique=False, trim="k")
            )
        return value

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
    def _short_plot_label(text: str, max_len: int = 42) -> str:
        """Return a compact label for legends and inline text."""
        if len(text) <= max_len:
            return text
        return f"{text[:max_len - 1]}…"

    @staticmethod
    def _format_plot_title(labels: list[str], max_len: int = 140) -> str:
        """Return a compact title containing full variable labels."""
        if not labels:
            return ""
        joined = " | ".join(labels)
        if len(joined) <= max_len:
            return joined
        return f"{joined[:max_len - 1]}…"

    def _create_plot_panel(self, canvas: FigureCanvasQTAgg) -> QVBoxLayout:
        """Create a canvas with navigation controls anchored at bottom-left."""
        panel = QVBoxLayout()
        panel.addWidget(canvas)
        toolbar = NavigationToolbar2QT(canvas, self)
        toolbar.setOrientation(Qt.Horizontal)
        panel.addWidget(toolbar, alignment=Qt.AlignLeft)
        return panel

    def _remove_temp_plot_file(self):
        if self._temp_plot_file and os.path.exists(self._temp_plot_file):
            try:
                os.remove(self._temp_plot_file)
            except OSError:
                pass
        self._temp_plot_file = None

    def _load_web_plot_html(self, html: str):
        """Load renderer HTML from disk so QWebEngine handles large inline assets."""
        if self.web_plot_view is None:
            return
        self._remove_temp_plot_file()
        with tempfile.NamedTemporaryFile(
            "w",
            delete=False,
            suffix=".html",
            encoding="utf-8",
        ) as tmp:
            tmp.write(html)
            self._temp_plot_file = tmp.name
        self.web_plot_view.load(QUrl.fromLocalFile(self._temp_plot_file))

    def update_data(self):
        records: list[dict] = []
        stat_cols: list[str] = []
        self.ts_dict = {}
        parse_enabled = self.parse_filename_cb.isChecked()
        parse_headers: list[str] = []

        # Temporarily disable sorting while populating the table to avoid
        # rows being rearranged mid-update. Sorting will be re-enabled at
        # the end of this method.
        sorting_was_enabled = self.table.isSortingEnabled()
        if sorting_was_enabled:
            self.table.setSortingEnabled(False)

        filter_spec = self._current_filter_spec()
        local_filter_label = self._filter_label(filter_spec)

        series_info = self.series_info
        if self.order_combo.currentIndex() == 1:

            # Variables → Files: preserve file list order using ``file_idx``
            series_info = sorted(series_info, key=lambda i: (i["var"], i["file_idx"]))
        else:
            # Files → Variables: maintain order of files as loaded
            series_info = sorted(series_info, key=lambda i: (i["file_idx"], i["var"]))


        fatigue_enabled = self.fatigue_filter_cb.isChecked()
        fatigue_headers: list[str] = []
        if fatigue_enabled:
            fatigue_headers = [
                "tz_high_pass",
                "tz_low_pass",
                "std_high_pass",
                "std_low_pass",
            ]
            hp_cutoff = self._parse_positive_float(self.highpass_cutoff.text())
            lp_cutoff = self._parse_positive_float(self.lowpass_cutoff.text())
        else:
            hp_cutoff = lp_cutoff = None

        for info in series_info:
            t = np.asarray(info["t"], dtype=float)
            x = np.asarray(info["x"], dtype=float)
            y, filter_messages = self._apply_filter_with_qc(t, x)
            finite_mask = np.isfinite(y) & np.isfinite(t)
            time_step_qc, time_messages = self._time_qc(t, finite_mask)
            n_samples = int(x.size)
            n_finite = int(np.count_nonzero(np.isfinite(y)))
            nan_fraction = float(1.0 - (n_finite / n_samples)) if n_samples else 1.0
            qc_messages = list(filter_messages) + list(time_messages)
            if n_samples == 0:
                qc_messages.append("Selected time window contains no samples.")
            elif n_finite == 0:
                qc_messages.append("No finite values are available.")
            elif nan_fraction >= 0.5:
                qc_messages.append(f"NaN coverage is high ({nan_fraction:.0%}).")
            elif nan_fraction > 0:
                qc_messages.append(f"NaN coverage is {nan_fraction:.0%}.")

            if n_finite < 4:
                psd_qc = "Unavailable: fewer than four finite samples."
            elif time_step_qc in {"Unavailable", "Invalid"}:
                psd_qc = f"Unavailable: {time_step_qc.lower()} time steps."
            elif time_step_qc == "Irregular":
                psd_qc = "Limited: irregular sampling."
            else:
                psd_qc = "Ready"

            stats: dict = {}
            if n_samples and n_finite:
                try:
                    stats = TimeSeries("tmp", t, y).stats()
                except Exception as exc:
                    qc_messages.append(f"Statistics calculation failed ({exc}).")
            else:
                qc_messages.append("Statistics calculation skipped.")
            for key in stats:
                if key not in stat_cols:
                    stat_cols.append(key)

            if fatigue_enabled:
                tz_hp, std_hp = self._fatigue_filter_stats(t, x, "highpass", hp_cutoff)
                tz_lp, std_lp = self._fatigue_filter_stats(t, x, "lowpass", lp_cutoff)
                fatigue_values = [tz_hp, tz_lp, std_hp, std_lp]
            else:
                fatigue_values = []

            sid = f"{info['file']}::{info['var']}"
            plot_t = self._plot_time_from_info(info, t)
            status = "OK" if not qc_messages and psd_qc == "Ready" else "Review"
            self.ts_dict[sid] = _StatsSeriesState(
                sid=sid,
                info=info,
                t=t,
                plot_t=plot_t,
                y=y,
                time_is_datetime=self._is_datetime_axis(plot_t),
                qc_status=status,
                qc_messages=qc_messages,
                psd_message=psd_qc,
            )

            if parse_enabled:
                target_name = choose_parse_target(
                    info.get("file"),
                    info.get("uniq_file"),
                )
                parsed = parse_embedded_values(target_name)
            else:
                parsed = {}
            for key in parsed:
                if key not in parse_headers:
                    parse_headers.append(key)
            records.append(
                {
                    "info": info,
                    "stats": stats,
                    "parsed": parsed,
                    "fatigue": fatigue_values,
                    "samples": n_samples,
                    "finite": n_finite,
                    "nan_fraction": self._format_stat_value(nan_fraction),
                    "time_qc": time_step_qc,
                    "psd_qc": psd_qc,
                    "qc_status": status,
                    "qc_messages": " ".join(qc_messages),
                    "local_filter": local_filter_label,
                }
            )

        headers = [
            "File",
            "Uniqueness",
            "Variable",
            "VarUniqueness",
            "Editor filter",
            "Local filter",
            "QC",
            "QC messages",
            "Samples",
            "Finite samples",
            "NaN fraction",
            "Time step QC",
            "PSD QC",
        ]
        headers += parse_headers + stat_cols + fatigue_headers
        self._parsed_headers = parse_headers
        self._fatigue_headers = fatigue_headers
        self._table_headers = headers
        self.table.setRowCount(len(records))
        self.table.setColumnCount(len(headers))
        self.table.setHorizontalHeaderLabels(headers)
        var_uniq = self._uniq([record["info"]["var"] for record in records])
        for i, (record, var_uniq_value) in enumerate(zip(records, var_uniq)):
            info = record["info"]
            row_by_header = {
                "File": info["file"],
                "Uniqueness": info["uniq_file"],
                "Variable": info["var"],
                "VarUniqueness": var_uniq_value,
                "Editor filter": info.get("editor_filter", {}).get("description", "Editor filter: none"),
                "Local filter": record["local_filter"],
                "QC": record["qc_status"],
                "QC messages": record["qc_messages"],
                "Samples": record["samples"],
                "Finite samples": record["finite"],
                "NaN fraction": record["nan_fraction"],
                "Time step QC": record["time_qc"],
                "PSD QC": record["psd_qc"],
            }
            for key, value in record["parsed"].items():
                row_by_header[key] = value
            for key in stat_cols:
                value = record["stats"].get(key, np.nan)
                if key.lower() == "start" and len(info["t"]):
                    value = self._format_time_value(
                        info.get("dtg_time")[0]
                        if info.get("dtg_time") is not None and len(info.get("dtg_time"))
                        else info["t"][0]
                    )
                elif key.lower() == "end" and len(info["t"]):
                    value = self._format_time_value(
                        info.get("dtg_time")[-1]
                        if info.get("dtg_time") is not None and len(info.get("dtg_time"))
                        else info["t"][-1]
                    )
                else:
                    value = self._format_stat_value(value)
                row_by_header[key] = value
            for key, value in zip(fatigue_headers, record["fatigue"]):
                row_by_header[key] = self._format_stat_value(value)
            row = [row_by_header.get(header, "") for header in headers]
            for j, val in enumerate(row):

                text = str(val)
                if isinstance(val, (int, float)):
                    item = SortableTableWidgetItem(text)
                    item.setData(Qt.ItemDataRole.UserRole, float(val))
                else:
                    item = QTableWidgetItem(text)

                self.table.setItem(i, j, item)

        prior_metric_headers = {
            self.table.horizontalHeaderItem(col).text()
            for col in self.selected_columns
            if col < self.table.columnCount() and self.table.horizontalHeaderItem(col) is not None
        }
        self._populate_metric_selector(headers, prior_metric_headers)
        self._apply_column_visibility()
        self.update_plots()

        # Restore previous sorting state after table population
        if sorting_was_enabled:
            self.table.setSortingEnabled(True)

    def _header_right_click(self, pos):
        header = self.table.horizontalHeader()
        section = header.logicalIndexAt(pos)
        if section in self._metric_columns():
            self.toggle_column(section)

    def toggle_column(self, section: int):
        if section not in self._metric_columns():
            return
        if section in self.selected_columns:
            self.selected_columns.remove(section)
        else:
            self.selected_columns.add(section)
        self._sync_metric_selector()
        self.update_plots()

    def _metric_columns(self) -> list[int]:
        cols: list[int] = []
        for col, header in enumerate(self._table_headers):
            if header.lower() in self._CORE_STATS_HEADERS and header.lower() not in {
                "duration",
                "dtavg",
                "mean",
                "std",
                "min",
                "max",
                "skew",
                "kurt",
                "tz",
            }:
                continue
            if header in self._parsed_headers:
                continue
            has_numeric = False
            for row in range(self.table.rowCount()):
                item = self.table.item(row, col)
                try:
                    has_numeric = item is not None and np.isfinite(float(item.text()))
                except ValueError:
                    has_numeric = False
                if has_numeric:
                    cols.append(col)
                    break
        return cols

    def _populate_metric_selector(self, headers: list[str], prior_headers: set[str]):
        self.metric_list.blockSignals(True)
        self.metric_list.clear()
        metric_cols = self._metric_columns()
        selected_headers = prior_headers or {"max"}
        self.selected_columns = set()
        for col in metric_cols:
            header = headers[col]
            item = QListWidgetItem(header)
            item.setData(Qt.ItemDataRole.UserRole, col)
            self.metric_list.addItem(item)
            if header.lower() in {name.lower() for name in selected_headers}:
                item.setSelected(True)
                self.selected_columns.add(col)
        if not self.selected_columns and self.metric_list.count():
            item = self.metric_list.item(0)
            item.setSelected(True)
            self.selected_columns.add(int(item.data(Qt.ItemDataRole.UserRole)))
        self.metric_list.blockSignals(False)

    def _sync_metric_selector(self):
        self.metric_list.blockSignals(True)
        for index in range(self.metric_list.count()):
            item = self.metric_list.item(index)
            item.setSelected(int(item.data(Qt.ItemDataRole.UserRole)) in self.selected_columns)
        self.metric_list.blockSignals(False)

    def _on_metric_selection_changed(self):
        self.selected_columns = {
            int(item.data(Qt.ItemDataRole.UserRole))
            for item in self.metric_list.selectedItems()
        }
        self.update_plots()

    @staticmethod
    def _is_distribution_header(header: str) -> bool:
        lower = header.lower()
        return any(token in lower for token in ("weibull", "gumbel", "quantile", "q0", "q1", "q5", "q9"))

    def _apply_column_visibility(self, *_args):
        preset = self.column_preset_combo.currentText().lower()
        for col, header in enumerate(self._table_headers):
            lower = header.lower()
            show = lower in self._CORE_STATS_HEADERS
            if preset == "all metrics":
                show = True
            elif preset == "distribution metrics":
                show = show or self._is_distribution_header(header)
            elif preset == "fatigue metrics":
                show = show or header in self._fatigue_headers
            elif preset == "filename metrics":
                show = show or header in self._parsed_headers
            self.table.setColumnHidden(col, not show)

    def keyPressEvent(self, event):
        if event.matches(QKeySequence.Copy):
            self.copy_selected_as_tsv()
            event.accept()
        else:
            super().keyPressEvent(event)

    def copy_selected_as_tsv(self):
        rows = sorted({idx.row() for idx in self.table.selectionModel().selectedRows()})
        if not rows:
            return
        self._copy_rows_as_tsv(rows)

    def copy_all_as_tsv(self):
        if not self.table.rowCount():
            return
        self._copy_rows_as_tsv(range(self.table.rowCount()))

    def _copy_rows_as_tsv(self, rows):
        lines = ["\t".join([self.table.horizontalHeaderItem(c).text() for c in range(self.table.columnCount())])]
        for r in rows:
            vals = [self.table.item(r, c).text() for c in range(self.table.columnCount())]
            lines.append("\t".join(vals))
        QGuiApplication.clipboard().setText("\n".join(lines))

    def _suggest_psd_frequency_limit(self, freqs: np.ndarray, psd_vals: np.ndarray) -> tuple[float, float] | None:
        """Return focused high/low frequency limits for the informative PSD region."""
        freqs = np.asarray(freqs, dtype=float)
        psd_vals = np.asarray(psd_vals, dtype=float)
        valid = np.isfinite(freqs) & np.isfinite(psd_vals)
        if not np.any(valid):
            return None

        freqs = freqs[valid]
        psd_vals = psd_vals[valid]
        order = np.argsort(freqs)
        freqs = freqs[order]
        psd_vals = psd_vals[order]

        non_negative = freqs >= 0.0
        freqs = freqs[non_negative]
        psd_vals = psd_vals[non_negative]
        positive_freqs = freqs[freqs > 0.0]
        if freqs.size < 2 or positive_freqs.size == 0 or np.all(psd_vals <= 0.0):
            return None

        cumulative_power = np.concatenate(
            ([0.0], np.cumsum(np.diff(freqs) * (psd_vals[1:] + psd_vals[:-1]) * 0.5))
        )
        total_power = cumulative_power[-1]
        if total_power <= 0.0:
            return None

        high_target_power = self._PSD_CUMULATIVE_POWER_COVERAGE * total_power
        low_target_power = (1.0 - self._PSD_CUMULATIVE_POWER_COVERAGE) * total_power
        high_cumulative_idx = int(np.searchsorted(cumulative_power, high_target_power, side="left"))
        high_cumulative_idx = min(high_cumulative_idx, freqs.size - 1)
        low_cumulative_idx = int(np.searchsorted(cumulative_power, low_target_power, side="left"))
        low_cumulative_idx = min(low_cumulative_idx, freqs.size - 1)

        prominent = np.flatnonzero(
            psd_vals >= self._PSD_RELATIVE_LEVEL_THRESHOLD * np.nanmax(psd_vals)
        )
        prominent_low_idx = int(prominent[0]) if prominent.size else 0
        prominent_high_idx = int(prominent[-1]) if prominent.size else 0

        high_idx = max(high_cumulative_idx, prominent_high_idx)
        low_idx = max(low_cumulative_idx, prominent_low_idx)

        min_positive_freq = float(positive_freqs[0])
        low_freq = float(np.clip(freqs[low_idx], min_positive_freq, freqs[-1]))
        high_freq = float(np.clip(freqs[high_idx] * self._PSD_XLIM_PADDING, low_freq, freqs[-1]))
        return high_freq, low_freq

    @staticmethod
    def _psd_plot_axis(
        freqs: np.ndarray,
        psd_vals: np.ndarray,
        use_period: bool,
        limit_freq: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray, str]:
        """Return PSD x-values, y-values, and x-axis label for the selected unit."""
        freqs = np.asarray(freqs, dtype=float)
        psd_vals = np.asarray(psd_vals, dtype=float)
        valid = np.isfinite(freqs) & np.isfinite(psd_vals)
        freqs = freqs[valid]
        psd_vals = psd_vals[valid]

        if limit_freq is not None:
            within_limit = freqs <= limit_freq
            freqs = freqs[within_limit]
            psd_vals = psd_vals[within_limit]

        if not use_period:
            return freqs, psd_vals, "Frequency [Hz]"

        positive = freqs > 0.0
        freqs = freqs[positive]
        psd_vals = psd_vals[positive]
        if freqs.size == 0:
            return freqs, psd_vals, "Period [s]"

        periods = 1.0 / freqs
        order = np.argsort(periods)
        return periods[order], psd_vals[order], "Period [s]"

    @staticmethod
    def _parse_histogram_lines(text: str) -> list[float]:
        values = []
        for token in re.split(r"[ ,]+", text.strip()):
            if not token:
                continue
            try:
                values.append(float(token))
            except ValueError:
                continue
        return values

    @staticmethod
    def _histogram_bins(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        counts, edges = np.histogram(np.asarray(values, dtype=float), bins=StatsDialog._HISTOGRAM_BINS)
        return counts.astype(float), edges.astype(float)

    @staticmethod
    def _downsample_line(
        x: np.ndarray,
        y: np.ndarray,
        max_points: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Keep browser line payloads compact while retaining end points."""
        x_arr = np.asarray(x)
        y_arr = np.asarray(y)
        if y_arr.size <= max_points:
            return x_arr, y_arr
        indices = np.unique(
            np.linspace(0, y_arr.size - 1, num=max_points, dtype=int)
        )
        return x_arr[indices], y_arr[indices]

    def _web_prepared_plot_data(self, prepared: dict) -> dict:
        """Return renderer data sized for embedded browser plots."""
        web_prepared = dict(prepared)
        web_prepared["time"] = []
        web_prepared["psd"] = []
        for trace in prepared["time"]:
            x_vals, y_vals = self._downsample_line(
                trace["x"],
                trace["y"],
                self._WEB_TIME_MAX_POINTS,
            )
            web_prepared["time"].append({**trace, "x": x_vals, "y": y_vals})
        for trace in prepared["psd"]:
            x_vals, y_vals = self._downsample_line(
                trace["x"],
                trace["y"],
                self._WEB_PSD_MAX_POINTS,
            )
            web_prepared["psd"].append({**trace, "x": x_vals, "y": y_vals})
        return web_prepared

    @staticmethod
    def _axis_message(ax, title: str, message: str):
        ax.set_title(title)
        ax.text(
            0.5,
            0.5,
            message,
            ha="center",
            va="center",
            wrap=True,
            transform=ax.transAxes,
        )
        ax.set_xticks([])
        ax.set_yticks([])

    def _prepare_psd_trace(self, state: _StatsSeriesState, use_period: bool) -> tuple[dict | None, str]:
        finite = np.isfinite(state.t) & np.isfinite(state.y)
        if np.count_nonzero(finite) < 4:
            return None, state.psd_message
        t = state.t[finite]
        y = state.y[finite]
        ts_tmp = TimeSeries("tmp", t, y)
        try:
            freqs, psd_vals = ts_tmp.psd(resample=ts_tmp.dt)
        except Exception:
            diffs = np.diff(t)
            dt = float(np.median(diffs)) if diffs.size else np.nan
            if not np.isfinite(dt) or dt <= 0:
                return None, "PSD unavailable: no positive resampling interval."
            try:
                freqs, psd_vals = ts_tmp.psd(resample=dt)
            except Exception as exc:
                return None, f"PSD unavailable ({exc})."
        if not freqs.size or not psd_vals.size:
            return None, "PSD unavailable: calculation returned no values."
        limit_info = self._suggest_psd_frequency_limit(freqs, psd_vals)
        limit_freq = limit_info[0] if limit_info is not None else None
        x_vals, y_vals, x_label = self._psd_plot_axis(
            freqs,
            psd_vals,
            use_period,
            limit_freq=limit_freq,
        )
        if not x_vals.size or not y_vals.size:
            unit = "period" if use_period else "frequency"
            return None, f"PSD unavailable for {unit} display."
        trace = {
            "x": x_vals,
            "y": y_vals,
            "xlabel": x_label,
            "limit_freq": limit_freq,
            "limit_info": limit_info,
        }
        return trace, ""

    def _prepare_plot_data(self) -> dict:
        sel_rows = [idx.row() for idx in self.table.selectionModel().selectedRows()]
        if not sel_rows:
            sel_rows = [0] if self.table.rowCount() else []
        use_period = self.plot_period_cb.isChecked()
        all_rows = list(range(self.table.rowCount()))
        prepared = {
            "selected_rows": sel_rows,
            "all_rows": all_rows,
            "time": [],
            "psd": [],
            "hist": [],
            "metrics": [],
            "hist_lines": self._parse_histogram_lines(self.hist_lines_edit.text()),
            "psd_xlabel": "Period [s]" if use_period else "Frequency [Hz]",
            "psd_limits": [],
            "psd_period_limits": [],
            "messages": {
                "time": "Select a row with finite values to view a time plot.",
                "psd": "PSD is unavailable for the selected rows.",
                "hist": "Histogram is unavailable for the selected rows.",
                "metrics": "Select one or more numeric comparison metrics.",
            },
        }
        psd_messages: list[str] = []
        for r in sel_rows:
            file_item = self.table.item(r, 0)
            var_item = self.table.item(r, 2)
            if file_item is None or var_item is None:
                continue
            file = file_item.text()
            var = var_item.text()
            sid = f"{file}::{var}"
            state = self.ts_dict.get(sid)
            if state is None:
                continue
            full_label = var
            if file and len(self.ts_dict) > 1:
                full_label = f"{file}::{var}"
            label = self._short_plot_label(full_label)
            finite_y = np.isfinite(state.y)
            if np.any(finite_y):
                prepared["time"].append(
                    {
                        "x": state.plot_t,
                        "y": state.y,
                        "label": label,
                        "title_label": full_label,
                        "datetime": state.time_is_datetime,
                    }
                )
                prepared["hist"].append(
                    {
                        "values": state.y[finite_y],
                        "label": self._short_plot_label(var),
                        "title_label": full_label,
                    }
                )
            psd_trace, psd_message = self._prepare_psd_trace(state, use_period)
            if psd_trace is not None:
                psd_trace.update(label=label, title_label=full_label)
                prepared["psd"].append(psd_trace)
                prepared["psd_xlabel"] = psd_trace["xlabel"]
                if psd_trace["limit_info"] is not None:
                    limit_freq, low_freq = psd_trace["limit_info"]
                    prepared["psd_limits"].append(limit_freq)
                    prepared["psd_period_limits"].append((1.0 / limit_freq, 1.0 / low_freq))
            elif psd_message:
                psd_messages.append(psd_message)

        if prepared["time"]:
            prepared["messages"]["time"] = ""
        if prepared["hist"]:
            prepared["messages"]["hist"] = ""
        if prepared["psd"]:
            prepared["messages"]["psd"] = ""
        elif psd_messages:
            prepared["messages"]["psd"] = " ".join(dict.fromkeys(psd_messages))

        row_labels = []
        selected_set = set(sel_rows)
        for r in all_rows:
            file = self.table.item(r, 0).text() if self.table.item(r, 0) is not None else f"Row {r + 1}"
            uniq_file = self.table.item(r, 1).text() if self.table.item(r, 1) is not None else ""
            uniq_var = self.table.item(r, 3).text() if self.table.item(r, 3) is not None else ""
            label = uniq_file or file or f"Row {r + 1}"
            if uniq_var:
                label = f"{label} / {uniq_var}"
            row_labels.append(label)
        for col in sorted(self.selected_columns):
            if col >= self.table.columnCount():
                continue
            vals = []
            for r in all_rows:
                item = self.table.item(r, col)
                try:
                    vals.append(float(item.text()))
                except (AttributeError, TypeError, ValueError):
                    vals.append(np.nan)
            if any(np.isfinite(vals)):
                prepared["metrics"].append(
                    {
                        "header": self.table.horizontalHeaderItem(col).text(),
                        "values": np.asarray(vals, dtype=float),
                        "row_labels": row_labels,
                        "selected": [r in selected_set for r in all_rows],
                    }
                )
        if prepared["metrics"]:
            prepared["messages"]["metrics"] = ""
        return prepared

    def _render_matplotlib(self, prepared: dict):
        from matplotlib import colors as mcolors

        self.line_fig.clear()
        ax = self.line_fig.add_subplot(111)
        self.psd_fig.clear()
        axp = self.psd_fig.add_subplot(111)
        if prepared["time"]:
            for trace in prepared["time"]:
                ax.plot(trace["x"], trace["y"], label=trace["label"])
            ax.set_xlabel("Time")
            ax.set_ylabel("Value")
            ax.set_title(self._format_plot_title([trace["title_label"] for trace in prepared["time"]]))
            ax.legend()
            ax.grid(True)
        else:
            self._axis_message(ax, "Time plot", prepared["messages"]["time"])

        self._tight_draw(self.line_fig, self.line_canvas)

        if prepared["psd"]:
            for trace in prepared["psd"]:
                axp.plot(trace["x"], trace["y"], label=trace["label"])
            axp.set_xlabel(prepared["psd_xlabel"])
            axp.set_ylabel("Power spectral density")
            axp.set_title(self._format_plot_title([trace["title_label"] for trace in prepared["psd"]]))
            if self.plot_period_cb.isChecked() and prepared["psd_period_limits"]:
                axp.set_xlim(
                    left=min(limit[0] for limit in prepared["psd_period_limits"]),
                    right=max(limit[1] for limit in prepared["psd_period_limits"]),
                )
            elif prepared["psd_limits"]:
                axp.set_xlim(left=0.0, right=max(prepared["psd_limits"]))
            axp.legend()
            axp.grid(True)
        else:
            self._axis_message(axp, "PSD / period plot", prepared["messages"]["psd"])

        self._tight_draw(self.psd_fig, self.psd_canvas)


        self.hist_fig_rows.clear()
        axh = self.hist_fig_rows.add_subplot(111)
        show_text = self.hist_show_text_cb.isChecked()
        if prepared["hist"]:
            for trace in prepared["hist"]:
                counts, bins, patches = axh.hist(
                    trace["values"], bins=self._HISTOGRAM_BINS, alpha=0.5, label=trace["label"]
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
            axh.set_title(self._format_plot_title([trace["title_label"] for trace in prepared["hist"]]))
            axh.legend()
            axh.grid(True)
        else:
            self._axis_message(axh, "Value histogram", prepared["messages"]["hist"])
        self._tight_draw(self.hist_fig_rows, self.hist_canvas_rows)


        self.hist_fig_cols.clear()
        axc = self.hist_fig_cols.add_subplot(111)
        max_y = 0
        rows_idx = np.arange(len(prepared["all_rows"]))
        ncols = len(prepared["metrics"]) if prepared["metrics"] else 1
        bar_w = 0.8 / ncols
        bars_by_col = []
        if prepared["metrics"]:
            for i, metric in enumerate(prepared["metrics"]):
                offset = (i - (ncols - 1) / 2) * bar_w
                bars = axc.bar(
                    rows_idx + offset,
                    metric["values"],
                    width=bar_w,
                    alpha=0.7,
                    label=metric["header"],
                )
                bars_by_col.append(bars)
                if np.any(np.isfinite(metric["values"])):
                    max_y = max(max_y, float(np.nanmax(metric["values"])))
                if show_text:
                    for bar, label in zip(bars, metric["row_labels"]):
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

            used_colors = {mcolors.to_hex(bar.get_facecolor()) for bc in bars_by_col for bar in bc}
            candidates = ["red", "magenta", "cyan", "yellow", "black"]
            highlight = next((c for c in candidates if mcolors.to_hex(c) not in used_colors), "red")
            for metric, bars in zip(prepared["metrics"], bars_by_col):
                for idx, selected in enumerate(metric["selected"]):
                    if selected and idx < len(bars):
                        bars[idx].set_facecolor(highlight)
            for value in prepared["hist_lines"]:
                axc.axhline(value, color="red", linestyle="--")
            ylim_top = max([max_y] + prepared["hist_lines"]) if (max_y or prepared["hist_lines"]) else None
            axc.set_xlabel("Row")
            axc.set_ylabel("Value")
            axc.set_xticks(rows_idx)
            axc.set_xticklabels(
                [self.table.item(r, 0).text() for r in prepared["all_rows"]],
                rotation=90,
            )
            axc.legend()
            if ylim_top is not None and np.isfinite(ylim_top):
                axc.set_ylim(top=ylim_top * 1.1 if ylim_top else 1.0)
            axc.grid(True, axis="y")
        else:
            self._axis_message(axc, "Metric comparison", prepared["messages"]["metrics"])
        self._tight_draw(self.hist_fig_cols, self.hist_canvas_cols)

    @staticmethod
    def _plotly_message(fig, row: int, col: int, message: str):
        fig.add_annotation(
            text=message,
            row=row,
            col=col,
            showarrow=False,
            x=0.5,
            y=0.5,
            xref="x domain",
            yref="y domain",
            align="center",
        )

    def _render_plotly_html(self, prepared: dict) -> str:
        import plotly.graph_objects as go
        from plotly.io import to_html
        from plotly.subplots import make_subplots

        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=("Time plot", "PSD / period plot", "Value histogram", "Metric comparison"),
        )
        for trace in prepared["time"]:
            fig.add_trace(
                go.Scatter(x=trace["x"], y=trace["y"], mode="lines", name=trace["label"]),
                row=1,
                col=1,
            )
        if not prepared["time"]:
            self._plotly_message(fig, 1, 1, prepared["messages"]["time"])
        for trace in prepared["psd"]:
            fig.add_trace(
                go.Scatter(x=trace["x"], y=trace["y"], mode="lines", name=trace["label"], showlegend=False),
                row=1,
                col=2,
            )
        if not prepared["psd"]:
            self._plotly_message(fig, 1, 2, prepared["messages"]["psd"])
        for trace in prepared["hist"]:
            counts, edges = self._histogram_bins(trace["values"])
            centers = (edges[:-1] + edges[1:]) / 2.0
            widths = np.diff(edges)
            fig.add_trace(
                go.Bar(
                    x=centers,
                    y=counts,
                    width=widths,
                    opacity=0.55,
                    name=trace["label"],
                    showlegend=False,
                ),
                row=2,
                col=1,
            )
        if not prepared["hist"]:
            self._plotly_message(fig, 2, 1, prepared["messages"]["hist"])
        for metric in prepared["metrics"]:
            colors = ["#d94f2a" if selected else "#3b82b6" for selected in metric["selected"]]
            fig.add_trace(
                go.Bar(
                    x=metric["row_labels"],
                    y=metric["values"],
                    marker_color=colors,
                    name=metric["header"],
                    showlegend=True,
                ),
                row=2,
                col=2,
            )
        if not prepared["metrics"]:
            self._plotly_message(fig, 2, 2, prepared["messages"]["metrics"])
        for value in prepared["hist_lines"]:
            fig.add_hline(y=value, line_dash="dash", line_color="#d94f2a", row=2, col=2)
        fig.update_xaxes(title_text="Time", row=1, col=1)
        fig.update_yaxes(title_text="Value", row=1, col=1)
        fig.update_xaxes(title_text=prepared["psd_xlabel"], row=1, col=2)
        fig.update_yaxes(title_text="Power spectral density", row=1, col=2)
        fig.update_xaxes(title_text="Value", row=2, col=1)
        fig.update_yaxes(title_text="Frequency", row=2, col=1)
        fig.update_xaxes(title_text="Row", tickangle=-55, row=2, col=2)
        fig.update_yaxes(title_text="Value", row=2, col=2)
        fig.update_layout(
            barmode="group",
            hovermode="closest",
            height=760,
            margin={"l": 58, "r": 24, "t": 64, "b": 92},
        )
        return to_html(fig, include_plotlyjs=True, full_html=True)

    @staticmethod
    def _bokeh_message_plot(title: str, message: str):
        from bokeh.models import Label
        from bokeh.plotting import figure

        plot = figure(title=title, width=480, height=300, toolbar_location="above")
        plot.add_layout(Label(x=0, y=0, text=message, text_align="center"))
        plot.xaxis.visible = False
        plot.yaxis.visible = False
        plot.grid.visible = False
        return plot

    def _render_bokeh_html(self, prepared: dict) -> str:
        from bokeh.embed import file_html
        from bokeh.layouts import gridplot
        from bokeh.models import FactorRange, Span
        from bokeh.plotting import figure
        from bokeh.resources import INLINE

        if prepared["time"]:
            x_axis_type = "datetime" if any(trace["datetime"] for trace in prepared["time"]) else "linear"
            p_time = figure(title="Time plot", width=480, height=300, x_axis_type=x_axis_type)
            for trace in prepared["time"]:
                p_time.line(trace["x"], trace["y"], legend_label=trace["label"], line_width=2)
            p_time.xaxis.axis_label = "Time"
            p_time.yaxis.axis_label = "Value"
        else:
            p_time = self._bokeh_message_plot("Time plot", prepared["messages"]["time"])

        if prepared["psd"]:
            p_psd = figure(title="PSD / period plot", width=480, height=300)
            for trace in prepared["psd"]:
                p_psd.line(trace["x"], trace["y"], legend_label=trace["label"], line_width=2)
            p_psd.xaxis.axis_label = prepared["psd_xlabel"]
            p_psd.yaxis.axis_label = "Power spectral density"
        else:
            p_psd = self._bokeh_message_plot("PSD / period plot", prepared["messages"]["psd"])

        if prepared["hist"]:
            p_hist = figure(title="Value histogram", width=480, height=300)
            for trace in prepared["hist"]:
                counts, edges = self._histogram_bins(trace["values"])
                p_hist.quad(
                    top=counts,
                    bottom=0,
                    left=edges[:-1],
                    right=edges[1:],
                    fill_alpha=0.45,
                    line_alpha=0.8,
                    legend_label=trace["label"],
                )
            p_hist.xaxis.axis_label = "Value"
            p_hist.yaxis.axis_label = "Frequency"
        else:
            p_hist = self._bokeh_message_plot("Value histogram", prepared["messages"]["hist"])

        if prepared["metrics"]:
            factors = []
            vals = []
            colors = []
            for metric in prepared["metrics"]:
                for idx, (label, value) in enumerate(zip(metric["row_labels"], metric["values"])):
                    factors.append((f"{idx + 1}: {label}", metric["header"]))
                    vals.append(value)
                    colors.append("#d94f2a" if metric["selected"][idx] else "#3b82b6")
            p_metric = figure(
                title="Metric comparison",
                width=480,
                height=300,
                x_range=FactorRange(*factors),
            )
            p_metric.vbar(x=factors, top=vals, width=0.8, color=colors)
            for value in prepared["hist_lines"]:
                p_metric.add_layout(
                    Span(location=value, dimension="width", line_dash="dashed", line_color="#d94f2a")
                )
            p_metric.xaxis.axis_label = "Row"
            p_metric.yaxis.axis_label = "Value"
            p_metric.xaxis.major_label_orientation = 1.1
        else:
            p_metric = self._bokeh_message_plot("Metric comparison", prepared["messages"]["metrics"])

        return file_html(gridplot([[p_time, p_psd], [p_hist, p_metric]]), INLINE, "Statistics")

    def update_plots(self):
        if not self.table.rowCount():
            return
        prepared = self._prepare_plot_data()
        if self.preferred_plot_engine == "default":
            self.render_warning_label.setVisible(False)
            self.plot_stack.setCurrentWidget(self.mpl_plot_widget)
            self._render_matplotlib(prepared)
            return
        if self.web_plot_view is None:
            self.render_warning_label.setText(
                f"{self.preferred_plot_engine.title()} renderer is unavailable; showing Matplotlib."
            )
            self.render_warning_label.setVisible(True)
            self.plot_stack.setCurrentWidget(self.mpl_plot_widget)
            self._render_matplotlib(prepared)
            return
        try:
            web_prepared = self._web_prepared_plot_data(prepared)
            if self.preferred_plot_engine == "plotly":
                html = self._render_plotly_html(web_prepared)
            else:
                html = self._render_bokeh_html(web_prepared)
            self._load_web_plot_html(html)
            self.render_warning_label.setVisible(False)
            self.plot_stack.setCurrentWidget(self.web_plot_view)
        except Exception as exc:
            self.render_warning_label.setText(
                f"{self.preferred_plot_engine.title()} renderer failed ({exc}); showing Matplotlib."
            )
            self.render_warning_label.setVisible(True)
            self.plot_stack.setCurrentWidget(self.mpl_plot_widget)
            self._render_matplotlib(prepared)

__all__ = ['StatsDialog']
