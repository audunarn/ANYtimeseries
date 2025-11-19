"""Fatigue calculation dialog for AnytimeSeries."""
"""Qt dialog that performs rainflow-based fatigue calculations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from anyqats.fatigue.rainflow import count_cycles
from anyqats.fatigue.sn import SNCurve, minersum
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QHeaderView,
)

from .fatigue_curves import FatigueCurveTemplate, curve_templates, find_template


@dataclass
class FatigueSeries:
    """Container for prepared series data used in the fatigue dialog."""

    label: str
    values: np.ndarray
    duration: float


class FatigueDialog(QDialog):
    """Compute fatigue damage for selected time series using rainflow counting."""

    _DURATION_UNITS: list[tuple[str, float]] = [
        ("Seconds", 1.0),
        ("Minutes", 60.0),
        ("Hours", 3600.0),
        ("Days", 86400.0),
        ("Years", 365.2425 * 86400.0),
    ]

    def __init__(self, series: Sequence[FatigueSeries], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Fatigue Calculation Tool")
        self.series = list(series)

        self.resize(820, 620)

        main_layout = QVBoxLayout(self)
        desc = QLabel(
            "Compute fatigue damage for the selected time series. "
            "Provide S-N or T-N curve parameters before running the calculation."
        )
        desc.setWordWrap(True)
        main_layout.addWidget(desc)

        # Curve definition block
        curve_box = QGroupBox("Fatigue Curve")
        curve_layout = QGridLayout(curve_box)

        self.curve_type_combo = QComboBox()
        self.curve_type_combo.addItem("S-N curve", "sn")
        self.curve_type_combo.addItem("T-N curve", "tn")
        self.curve_name = QLineEdit()
        self.curve_name.setPlaceholderText("Optional name")

        self.template_combo = QComboBox()
        self.template_combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self.template_combo.addItem("Custom parameters", None)

        self.m1_edit = QLineEdit()
        self.m1_edit.setPlaceholderText("e.g. 3.0")
        self.a1_edit = QLineEdit()
        self.a1_edit.setPlaceholderText("Optional if log10(a1) is given")
        self.loga1_edit = QLineEdit()
        self.loga1_edit.setPlaceholderText("e.g. 12.0")
        self.m2_edit = QLineEdit()
        self.nswitch_edit = QLineEdit()
        self.t_exp_edit = QLineEdit()
        self.t_ref_edit = QLineEdit()
        self.mean_tension_ratio_label = QLabel("Mean tension ratio Lm")
        self.mean_tension_ratio_spin = QDoubleSpinBox()
        self.mean_tension_ratio_spin.setDecimals(3)
        self.mean_tension_ratio_spin.setRange(0.0, 2.0)
        self.mean_tension_ratio_spin.setSingleStep(0.01)
        self.mean_tension_ratio_spin.setValue(0.1)
        self.mean_tension_ratio_spin.setToolTip(
            "Ratio of mean tension to minimum breaking strength (Lm) used for ABS T-N curves."
        )

        curve_layout.addWidget(QLabel("Curve type:"), 0, 0)
        curve_layout.addWidget(self.curve_type_combo, 0, 1)
        curve_layout.addWidget(QLabel("Name:"), 0, 2)
        curve_layout.addWidget(self.curve_name, 0, 3)
        curve_layout.addWidget(QLabel("Preset:"), 1, 0)
        curve_layout.addWidget(self.template_combo, 1, 1, 1, 3)

        form = QFormLayout()
        form.addRow("m1 (slope)*", self.m1_edit)
        form.addRow("a1", self.a1_edit)
        form.addRow("log10(a1)", self.loga1_edit)
        form.addRow("m2 (optional)", self.m2_edit)
        form.addRow("nswitch (if m2)", self.nswitch_edit)
        form.addRow("thickness exponent", self.t_exp_edit)
        form.addRow("reference thickness [mm]", self.t_ref_edit)
        form.addRow(self.mean_tension_ratio_label, self.mean_tension_ratio_spin)
        curve_layout.addLayout(form, 2, 0, 1, 4)

        self.include_thickness_cb = QCheckBox("Apply thickness correction during damage calculation")
        curve_layout.addWidget(self.include_thickness_cb, 3, 0, 1, 4)

        self.tn_info_label = QLabel("The T-N curves are applicable for tension-tension fatigue assessment.")
        self.tn_info_label.setWordWrap(True)
        curve_layout.addWidget(self.tn_info_label, 4, 0, 1, 4)

        main_layout.addWidget(curve_box)

        # Calculation options
        opts_box = QGroupBox("Calculation Options")
        opts_layout = QGridLayout(opts_box)

        self.duration_value = QDoubleSpinBox()
        self.duration_value.setDecimals(3)
        self.duration_value.setRange(0.001, 1e9)
        default_duration = self._suggest_default_duration()
        self.duration_value.setValue(default_duration)

        self.duration_unit = QComboBox()
        for label, seconds in self._DURATION_UNITS:
            self.duration_unit.addItem(label, seconds)
        self.duration_unit.setCurrentIndex(2)  # hours

        self.scf_spin = QDoubleSpinBox()
        self.scf_spin.setDecimals(3)
        self.scf_spin.setRange(0.1, 1e3)
        self.scf_spin.setValue(1.0)

        self.thickness_spin = QDoubleSpinBox()
        self.thickness_spin.setDecimals(3)
        self.thickness_spin.setRange(0.001, 1e6)
        self.thickness_spin.setValue(25.0)
        self.thickness_spin.setEnabled(False)
        self.include_thickness_cb.toggled.connect(self.thickness_spin.setEnabled)

        opts_layout.addWidget(QLabel("Duration:"), 0, 0)
        opts_layout.addWidget(self.duration_value, 0, 1)
        opts_layout.addWidget(self.duration_unit, 0, 2)
        opts_layout.addWidget(QLabel("Stress concentration factor (SCF):"), 1, 0)
        opts_layout.addWidget(self.scf_spin, 1, 1)
        opts_layout.addWidget(QLabel("Thickness [mm]:"), 2, 0)
        opts_layout.addWidget(self.thickness_spin, 2, 1)

        main_layout.addWidget(opts_box)

        # Table with prepared series metadata
        self.series_table = QTableWidget(0, 3)
        self.series_table.setHorizontalHeaderLabels([
            "Series",
            "Signal length [s]",
            "Valid samples",
        ])
        header = self.series_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self._populate_series_table()
        main_layout.addWidget(self.series_table)

        # Results output
        self.results_table = QTableWidget(0, 4)
        self.results_table.setHorizontalHeaderLabels([
            "Series",
            "Total cycles",
            "Damage",
            "Max range",
        ])
        res_header = self.results_table.horizontalHeader()
        res_header.setSectionResizeMode(0, QHeaderView.Stretch)
        res_header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        res_header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        res_header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        main_layout.addWidget(self.results_table)

        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setPlaceholderText("Results and warnings will appear here.")
        main_layout.addWidget(self.log_output)

        # Buttons
        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        self.compute_btn = QPushButton("Run fatigue calculation")
        self.close_btn = QPushButton("Close")
        btn_row.addWidget(self.compute_btn)
        btn_row.addWidget(self.close_btn)
        main_layout.addLayout(btn_row)

        self.compute_btn.clicked.connect(self._compute_damage)
        self.close_btn.clicked.connect(self.accept)
        self.curve_type_combo.currentIndexChanged.connect(self._on_curve_type_changed)
        self.template_combo.currentIndexChanged.connect(self._on_template_selected)
        self.mean_tension_ratio_spin.valueChanged.connect(self._on_mean_tension_ratio_changed)

        self._populate_template_combo()
        self._update_curve_type_state()

    # ------------------------------------------------------------------
    # Helpers
    def _suggest_default_duration(self) -> float:
        if not self.series:
            return 1.0
        longest = max((s.duration for s in self.series if s.duration > 0), default=1.0)
        return max(longest, 1.0)

    def _populate_series_table(self) -> None:
        self.series_table.setRowCount(len(self.series))
        for row, series in enumerate(self.series):
            self.series_table.setItem(row, 0, QTableWidgetItem(series.label))
            self.series_table.setItem(row, 1, QTableWidgetItem(f"{series.duration:.3f}"))
            self.series_table.setItem(row, 2, QTableWidgetItem(str(series.values.size)))

    def _current_curve_type(self) -> str:
        data = self.curve_type_combo.currentData(Qt.UserRole)
        if data is None:
            return "sn"
        return str(data)

    def _populate_template_combo(self, *, selected_key: str | None = None) -> None:
        curve_type = self._current_curve_type()
        self.template_combo.blockSignals(True)
        self.template_combo.clear()
        self.template_combo.addItem("Custom parameters", None)
        index_to_select = 0
        for template in curve_templates(curve_type):
            label = f"{template.label} ({template.source})"
            self.template_combo.addItem(label, template.key)
            if template.key == selected_key:
                index_to_select = self.template_combo.count() - 1
        self.template_combo.setCurrentIndex(index_to_select)
        self.template_combo.blockSignals(False)

    def _on_curve_type_changed(self, _index: int) -> None:
        self._update_curve_type_state()
        self._populate_template_combo()

    def _on_template_selected(self, _index: int) -> None:
        key = self.template_combo.currentData(Qt.UserRole)
        if not key:
            return
        template = find_template(str(key))
        if template is None:
            return
        self._apply_template(template)

    def _apply_template(self, template: FatigueCurveTemplate) -> None:
        desired_type = template.curve_type
        current_index = self.curve_type_combo.findData(desired_type)
        if current_index != -1 and self.curve_type_combo.currentIndex() != current_index:
            self.curve_type_combo.blockSignals(True)
            self.curve_type_combo.setCurrentIndex(current_index)
            self.curve_type_combo.blockSignals(False)
            self._populate_template_combo(selected_key=template.key)
        self._update_curve_type_state()

        self.curve_name.setText(template.label)
        params = dict(template.parameters)
        loga1 = self._loga1_from_lm(template)
        if loga1 is not None:
            params.pop("a1", None)
            params["loga1"] = loga1
        self._set_line_value(self.m1_edit, params.get("m1"))
        self._set_line_value(self.a1_edit, params.get("a1"))
        self._set_line_value(self.loga1_edit, params.get("loga1"))
        self._set_line_value(self.m2_edit, params.get("m2"))
        self._set_line_value(self.nswitch_edit, params.get("nswitch"))
        self._set_line_value(self.t_exp_edit, params.get("t_exp"))
        self._set_line_value(self.t_ref_edit, params.get("t_ref"))

        include_thickness = params.get("t_exp") is not None and params.get("t_ref") is not None
        self.include_thickness_cb.setChecked(include_thickness)

        current_key = self.template_combo.currentData(Qt.UserRole)
        if current_key != template.key:
            index = self.template_combo.findData(template.key)
            if index != -1:
                self.template_combo.blockSignals(True)
                self.template_combo.setCurrentIndex(index)
                self.template_combo.blockSignals(False)

    def _set_line_value(self, widget: QLineEdit, value: float | None) -> None:
        if value is None:
            widget.clear()
        else:
            widget.setText(f"{value}")

    def _update_curve_type_state(self) -> None:
        is_tn = self._current_curve_type() == "tn"
        self.tn_info_label.setVisible(is_tn)
        self.mean_tension_ratio_label.setVisible(is_tn)
        self.mean_tension_ratio_spin.setVisible(is_tn)

    def _loga1_from_lm(self, template: FatigueCurveTemplate) -> float | None:
        if template.lm_formula is None:
            return None
        intercept, slope = template.lm_formula
        lm_value = float(self.mean_tension_ratio_spin.value())
        return intercept + slope * lm_value

    def _on_mean_tension_ratio_changed(self, _value: float) -> None:
        key = self.template_combo.currentData(Qt.UserRole)
        if not key:
            return
        template = find_template(str(key))
        if template is None or template.lm_formula is None:
            return
        loga1 = self._loga1_from_lm(template)
        if loga1 is None:
            return
        self._set_line_value(self.a1_edit, None)
        self._set_line_value(self.loga1_edit, loga1)

    def _parse_float(self, widget: QLineEdit, *, optional: bool = False) -> float | None:
        text = widget.text().strip()
        if not text:
            if optional:
                return None
            raise ValueError("Required field is empty")
        try:
            return float(text)
        except ValueError as exc:
            raise ValueError(f"Invalid number: '{text}'") from exc

    def _build_curve(self) -> SNCurve:
        m1 = self._parse_float(self.m1_edit)
        params: dict[str, float] = {}
        a1 = self.a1_edit.text().strip()
        loga1 = self.loga1_edit.text().strip()
        if a1:
            try:
                params["a1"] = float(a1)
            except ValueError as exc:
                raise ValueError(f"Invalid a1 value: '{a1}'") from exc
        elif loga1:
            try:
                params["loga1"] = float(loga1)
            except ValueError as exc:
                raise ValueError(f"Invalid log10(a1) value: '{loga1}'") from exc
        else:
            raise ValueError("Specify either a1 or log10(a1) for the fatigue curve.")

        m2 = self._parse_float(self.m2_edit, optional=True)
        nswitch = self._parse_float(self.nswitch_edit, optional=True)
        if m2 is not None:
            if nswitch is None:
                raise ValueError("nswitch must be provided when m2 is specified")
            params["m2"] = m2
            params["nswitch"] = nswitch

        t_exp = self._parse_float(self.t_exp_edit, optional=True)
        t_ref = self._parse_float(self.t_ref_edit, optional=True)
        if t_exp is not None or t_ref is not None:
            if t_exp is None or t_ref is None:
                raise ValueError("Both thickness exponent and reference thickness must be provided.")
            params["t_exp"] = t_exp
            params["t_ref"] = t_ref

        name = self.curve_name.text().strip() or self.curve_type_combo.currentText()
        return SNCurve(name, m1, **params)

    def _duration_seconds(self) -> float:
        factor = self.duration_unit.currentData(Qt.UserRole)
        if factor is None:
            factor = 1.0
        return float(self.duration_value.value()) * float(factor)

    def _thickness_value(self) -> float | None:
        if not self.include_thickness_cb.isChecked():
            return None
        return float(self.thickness_spin.value())

    def _log(self, message: str) -> None:
        self.log_output.append(message)

    # ------------------------------------------------------------------
    # Computation
    def _compute_damage(self) -> None:
        if not self.series:
            QMessageBox.warning(self, "No data", "Select at least one valid time series before running the fatigue tool.")
            return

        try:
            curve = self._build_curve()
        except ValueError as exc:
            QMessageBox.critical(self, "Curve definition error", str(exc))
            return

        duration = self._duration_seconds()
        if duration <= 0:
            QMessageBox.critical(self, "Invalid duration", "Duration must be positive.")
            return

        scf = float(self.scf_spin.value())
        thickness = self._thickness_value()

        self.results_table.setRowCount(0)
        self.log_output.clear()

        for series in self.series:
            valid = np.isfinite(series.values)
            values = series.values[valid]
            if values.size < 2:
                self._log(f"{series.label}: skipped (not enough valid samples)")
                continue

            cycles = count_cycles(values)
            if cycles.size == 0:
                self._log(f"{series.label}: skipped (no cycles detected)")
                continue

            try:
                damage = minersum(cycles[:, 0], cycles[:, 2], curve, td=duration, scf=scf, th=thickness)
            except ValueError as exc:
                QMessageBox.critical(self, "Fatigue calculation error", str(exc))
                return

            total_cycles = float(np.sum(cycles[:, 2]))
            max_range = float(np.max(cycles[:, 0]))

            row = self.results_table.rowCount()
            self.results_table.insertRow(row)
            self.results_table.setItem(row, 0, QTableWidgetItem(series.label))
            self.results_table.setItem(row, 1, QTableWidgetItem(f"{total_cycles:.3f}"))
            self.results_table.setItem(row, 2, QTableWidgetItem(f"{damage:.6g}"))
            self.results_table.setItem(row, 3, QTableWidgetItem(f"{max_range:.3f}"))

            self._log(
                f"{series.label}: damage={damage:.6g} over {total_cycles:.2f} rainflow cycles (max range {max_range:.3f})."
            )

