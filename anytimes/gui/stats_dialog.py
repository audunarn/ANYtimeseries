"""Dialog for displaying statistics with filtering and plotting."""
from __future__ import annotations

import os
import re
import warnings

import anyqats as qats
import numpy as np
from anyqats import TimeSeries
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import QEvent, Qt, QTimer
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QAbstractItemView,
    QHeaderView,
    QSizePolicy,
)

from .sortable_table_widget_item import SortableTableWidgetItem
from .layout_utils import apply_initial_size
from .filename_parser import parse_embedded_values

class StatsDialog(QDialog):
    """Qt table dialog with copy and plotting features."""

    def __init__(self, series_info, parent=None):
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
        self.parse_filename_cb = QCheckBox("Parse file name")
        self.parse_filename_cb.setChecked(False)
        hline_layout.addWidget(self.parse_filename_cb)
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
        self.copy_selected_btn = QPushButton("Copy selected as TSV")
        self.copy_selected_btn.clicked.connect(self.copy_selected_as_tsv)
        btn_row.addWidget(self.copy_selected_btn)

        self.copy_all_btn = QPushButton("Copy all as TSV")
        self.copy_all_btn.clicked.connect(self.copy_all_as_tsv)
        btn_row.addWidget(self.copy_all_btn)
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
        self.parse_filename_cb.toggled.connect(self.update_data)
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

    def update_data(self):
        stats_rows = []
        stat_cols = []
        self.ts_dict = {}
        parse_enabled = self.parse_filename_cb.isChecked()
        parse_data_list: list[dict[str, float]] = []
        parse_headers: list[str] = []

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

            if parse_enabled:
                target_name = info.get("uniq_file") or info.get("file", "")

                parsed = parse_embedded_values(target_name)

            else:
                parsed = {}
            parse_data_list.append(parsed)
            for key in parsed:
                if key not in parse_headers:
                    parse_headers.append(key)

        var_uniq = self._uniq([info["var"] for info in series_info])
        for row, vu, pdata in zip(stats_rows, var_uniq, parse_data_list):
            row[3] = vu
            if parse_headers:
                insert_at = 5  # after the Filter column
                for key in parse_headers:
                    row.insert(insert_at, pdata.get(key, ""))
                    insert_at += 1

        headers = ["File", "Uniqueness", "Variable", "VarUniqueness", "Filter"]
        if parse_headers:
            headers.extend(parse_headers)
        headers += stat_cols
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
                ts_tmp = TimeSeries("tmp", t, y)
                try:
                    freqs, psd_vals = ts_tmp.psd(resample=ts_tmp.dt)
                except ValueError:
                    dt = float(np.median(np.diff(t)))
                    if dt <= 0:
                        continue
                    freqs, psd_vals = ts_tmp.psd(resample=dt)
                if freqs.size and psd_vals.size:
                    axp.plot(freqs, psd_vals, label=label)

        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        ax.legend()
        ax.grid(True)

        self._tight_draw(self.line_fig, self.line_canvas)

        axp.set_xlabel("Frequency [Hz]")
        axp.set_ylabel("Power spectral density")
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

__all__ = ['StatsDialog']

