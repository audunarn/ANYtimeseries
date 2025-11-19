"""Fatigue calculation dialog for AnytimeSeries."""
"""Qt dialog that performs rainflow-based fatigue calculations."""

from __future__ import annotations

from typing import Sequence

from anyqats.fatigue.sn import SNCurve
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

from ..fatigue import (
    FatigueComputationError,
    FatigueSeries,
    compute_fatigue_damage,
    summarize_damage,
)
from .fatigue_curves import FatigueCurveTemplate, curve_templates, find_template
from .filename_parser import exposure_hours_from_name


class FatigueDialog(QDialog):
    """Compute fatigue damage for selected time series using rainflow counting."""

    _DURATION_UNITS: list[tuple[str, float]] = [
        ("Seconds", 1.0),
        ("Minutes", 60.0),
        ("Hours", 3600.0),
        ("Days", 86400.0),
        ("Years", 365.2425 * 86400.0),
    ]

    _LOAD_UNIT_OPTIONS: dict[str, list[tuple[str, float]]] = {
        "stress": [
            ("MPa", 1.0),
            ("kPa", 1e-3),
            ("Pa", 1e-6),
        ],
        "tension": [
            ("MN", 1.0),
            ("kN", 1e-3),
            ("N", 1e-6),
        ],
    }

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

        self.load_basis_combo = QComboBox()
        self.load_basis_combo.addItem("Stress-based (S-N)", "stress")
        self.load_basis_combo.addItem("Tension-based (T-N)", "tension")
        self.load_unit_combo = QComboBox()
        self.reference_strength_label = QLabel("Reference breaking strength [unit]:")
        self.reference_strength_edit = QLineEdit()
        self.reference_strength_edit.setPlaceholderText("Optional for T-N curves")
        self.reference_strength_label.setVisible(False)
        self.reference_strength_edit.setVisible(False)
        self._populate_load_units("stress")

        opts_layout.addWidget(QLabel("Duration:"), 0, 0)
        opts_layout.addWidget(self.duration_value, 0, 1)
        opts_layout.addWidget(self.duration_unit, 0, 2)
        opts_layout.addWidget(QLabel("Stress concentration factor (SCF):"), 1, 0)
        opts_layout.addWidget(self.scf_spin, 1, 1)
        opts_layout.addWidget(QLabel("Thickness [mm]:"), 2, 0)
        opts_layout.addWidget(self.thickness_spin, 2, 1)
        opts_layout.addWidget(QLabel("Load interpretation:"), 3, 0)
        opts_layout.addWidget(self.load_basis_combo, 3, 1)
        opts_layout.addWidget(self.load_unit_combo, 3, 2)
        opts_layout.addWidget(self.reference_strength_label, 4, 0)
        opts_layout.addWidget(self.reference_strength_edit, 4, 1)

        main_layout.addWidget(opts_box)

        exposure_box = QGroupBox("Exposure Time")
        exposure_layout = QGridLayout(exposure_box)

        self.design_life_spin = QDoubleSpinBox()
        self.design_life_spin.setDecimals(2)
        self.design_life_spin.setRange(0.0, 1e4)
        self.design_life_spin.setValue(20.0)

        self.auto_exposure_cb = QCheckBox("Auto-calculate exposure times from filenames")
        self.auto_exposure_cb.setChecked(False)

        exposure_layout.addWidget(QLabel("Design life [years]:"), 0, 0)
        exposure_layout.addWidget(self.design_life_spin, 0, 1)
        exposure_layout.addWidget(self.auto_exposure_cb, 1, 0, 1, 2)
        exposure_layout.addWidget(
            QLabel("Provide the exposure time (hours over lifetime) for each series below."),
            2,
            0,
            1,
            2,
        )
        main_layout.addWidget(exposure_box)

        # Table with prepared series metadata
        self.series_table = QTableWidget(0, 4)
        self.series_table.setHorizontalHeaderLabels([
            "Series",
            "Signal length [s]",
            "Valid samples",
            "Exposure time [h]",
        ])
        header = self.series_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
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

        summary_box = QGroupBox("Summary over all time series")
        summary_layout = QGridLayout(summary_box)
        self.summary_damage_value = QLabel("—")
        self.summary_exposure_value = QLabel("—")
        self.summary_life_value = QLabel("—")
        summary_layout.addWidget(QLabel("Total exposure [h]:"), 0, 0)
        summary_layout.addWidget(self.summary_exposure_value, 0, 1)
        summary_layout.addWidget(QLabel("Overall damage:"), 1, 0)
        summary_layout.addWidget(self.summary_damage_value, 1, 1)
        summary_layout.addWidget(QLabel("Estimated life [years]:"), 2, 0)
        summary_layout.addWidget(self.summary_life_value, 2, 1)
        main_layout.addWidget(summary_box)

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
        self.auto_exposure_cb.toggled.connect(self._on_auto_exposure_toggled)
        self.design_life_spin.valueChanged.connect(self._on_design_life_changed)
        self.load_basis_combo.currentIndexChanged.connect(self._on_load_basis_changed)
        self.load_unit_combo.currentIndexChanged.connect(self._on_load_unit_changed)

        self._populate_template_combo()
        self._update_curve_type_state()
        self._reset_summary()

    # ------------------------------------------------------------------
    # Helpers
    def _suggest_default_duration(self) -> float:
        if not self.series:
            return 1.0
        longest = max((s.duration for s in self.series if s.duration > 0), default=1.0)
        return max(longest, 1.0)

    def _populate_series_table(self) -> None:
        self.series_table.setRowCount(len(self.series))
        default_hours = max(self._duration_seconds() / 3600.0, 0.0)
        for row, series in enumerate(self.series):
            self.series_table.setItem(row, 0, QTableWidgetItem(series.label))
            self.series_table.setItem(row, 1, QTableWidgetItem(f"{series.duration:.3f}"))
            self.series_table.setItem(row, 2, QTableWidgetItem(str(series.values.size)))
            spin = self._create_exposure_spinbox()
            spin.setValue(default_hours)
            self.series_table.setCellWidget(row, 3, spin)

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
        recommended_basis = "tension" if is_tn else "stress"
        if self._current_load_basis() != recommended_basis:
            self._set_load_basis(recommended_basis)

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

    def _current_load_basis(self) -> str:
        data = self.load_basis_combo.currentData(Qt.UserRole)
        if not data:
            return "stress"
        return str(data)

    def _populate_load_units(self, basis: str) -> None:
        options = self._LOAD_UNIT_OPTIONS.get(basis, [])
        self.load_unit_combo.blockSignals(True)
        self.load_unit_combo.clear()
        for label, factor in options:
            self.load_unit_combo.addItem(label, factor)
        self.load_unit_combo.blockSignals(False)
        self._update_reference_strength_state()

    def _set_load_basis(self, basis: str) -> None:
        index = self.load_basis_combo.findData(basis)
        if index == -1:
            return
        self.load_basis_combo.blockSignals(True)
        self.load_basis_combo.setCurrentIndex(index)
        self.load_basis_combo.blockSignals(False)
        self._populate_load_units(basis)

    def _on_load_basis_changed(self, _index: int) -> None:
        self._populate_load_units(self._current_load_basis())

    def _on_load_unit_changed(self, _index: int) -> None:
        self._update_reference_strength_state()

    def _load_unit_factor(self) -> float:
        data = self.load_unit_combo.currentData(Qt.UserRole)
        if data is None:
            return 1.0
        return float(data)

    def _current_load_unit_label(self) -> str:
        text = self.load_unit_combo.currentText()
        if not text:
            return "unit"
        return text

    def _update_reference_strength_state(self) -> None:
        is_tension = self._current_load_basis() == "tension"
        self.reference_strength_label.setVisible(is_tension)
        self.reference_strength_edit.setVisible(is_tension)
        if is_tension:
            unit = self._current_load_unit_label()
            self.reference_strength_label.setText(f"Reference breaking strength [{unit}]:")

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

    def _reference_strength_value(self) -> float | None:
        if self._current_load_basis() != "tension":
            return None
        text = self.reference_strength_edit.text().strip()
        if not text:
            return None
        try:
            value = float(text)
        except ValueError as exc:
            raise ValueError(f"Invalid reference breaking strength: '{text}'") from exc
        if value <= 0:
            raise ValueError("Reference breaking strength must be positive.")
        return value * self._load_unit_factor()

    def _log(self, message: str) -> None:
        self.log_output.append(message)

    def _create_exposure_spinbox(self) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setDecimals(3)
        spin.setRange(0.0, 1e12)
        spin.setValue(0.0)
        spin.setSuffix(" h")
        spin.setKeyboardTracking(False)
        return spin

    def _reset_summary(self) -> None:
        self.summary_damage_value.setText("—")
        self.summary_exposure_value.setText("—")
        self.summary_life_value.setText("—")

    def _update_summary(self, results, exposures) -> None:
        summary = summarize_damage(results, exposures)
        self.summary_damage_value.setText(f"{summary.total_damage:.6g}")
        if summary.total_exposure_hours is None:
            self.summary_exposure_value.setText("—")
        else:
            self.summary_exposure_value.setText(f"{summary.total_exposure_hours:.3f}")
        if summary.estimated_life_years is None:
            self.summary_life_value.setText("—")
        else:
            self.summary_life_value.setText(f"{summary.estimated_life_years:.3f}")

    def _exposure_widget(self, row: int) -> QDoubleSpinBox | None:
        widget = self.series_table.cellWidget(row, 3)
        if isinstance(widget, QDoubleSpinBox):
            return widget
        return None

    def _set_exposure_value(self, row: int, hours: float) -> None:
        widget = self._exposure_widget(row)
        if widget is None:
            return
        widget.setValue(max(float(hours), 0.0))

    def _collect_exposure_hours(self) -> list[float] | None:
        missing: list[str] = []
        exposures: list[float] = []
        for row, series in enumerate(self.series):
            widget = self._exposure_widget(row)
            if widget is None:
                missing.append(series.label)
                continue
            value = float(widget.value())
            if value <= 0.0:
                missing.append(series.label)
                continue
            exposures.append(value)
        if missing:
            message = "\n".join(missing)
            QMessageBox.critical(
                self,
                "Missing exposure time",
                "Provide an exposure time (hours) for the following series before running the calculation:\n"
                f"{message}",
            )
            return None
        return exposures

    def _on_auto_exposure_toggled(self, checked: bool) -> None:
        if checked:
            self._auto_fill_exposure_times()

    def _on_design_life_changed(self, _value: float) -> None:
        if self.auto_exposure_cb.isChecked():
            self._auto_fill_exposure_times()

    def _auto_fill_exposure_times(self) -> None:
        if not self.series:
            return
        design_life = float(self.design_life_spin.value())
        unresolved: list[str] = []
        filled = 0
        for row, series in enumerate(self.series):
            hours = exposure_hours_from_name(series.source_file, design_life)
            if hours is None:
                unresolved.append(series.label)
                continue
            self._set_exposure_value(row, hours)
            filled += 1
        if unresolved:
            names = "\n".join(unresolved)
            QMessageBox.warning(
                self,
                "Exposure parsing failed",
                "Could not determine exposure time for the following series. "
                "Please enter the values manually:\n"
                f"{names}",
            )
        elif filled == 0:
            QMessageBox.warning(
                self,
                "Exposure parsing failed",
                "Could not determine exposure times from the filenames.",
            )

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

        exposures = self._collect_exposure_hours()
        if exposures is None:
            return

        scf = float(self.scf_spin.value())
        thickness = self._thickness_value()
        load_basis = self._current_load_basis()
        curve_type = self._current_curve_type()
        if curve_type == "sn" and load_basis != "stress":
            QMessageBox.critical(
                self,
                "Load interpretation error",
                "S-N curves require stress-based input. Please select the stress interpretation.",
            )
            return
        if curve_type == "tn" and load_basis != "tension":
            QMessageBox.critical(
                self,
                "Load interpretation error",
                "T-N curves require tension-based input. Please select the tension interpretation.",
            )
            return

        unit_factor = self._load_unit_factor()
        try:
            reference_strength = self._reference_strength_value()
        except ValueError as exc:
            QMessageBox.critical(self, "Reference strength error", str(exc))
            return

        self.results_table.setRowCount(0)
        self.log_output.clear()
        self._reset_summary()

        try:
            results, logs = compute_fatigue_damage(
                self.series,
                exposures,
                curve,
                scf=scf,
                thickness=thickness,
                load_basis=load_basis,
                curve_type=curve_type,
                unit_factor=unit_factor,
                reference_strength=reference_strength,
            )
        except (FatigueComputationError, ValueError) as exc:
            QMessageBox.critical(self, "Fatigue calculation error", str(exc))
            return

        for entry in logs:
            self._log(entry)

        for result in results:
            row = self.results_table.rowCount()
            self.results_table.insertRow(row)
            self.results_table.setItem(row, 0, QTableWidgetItem(result.label))
            self.results_table.setItem(row, 1, QTableWidgetItem(f"{result.total_cycles:.3f}"))
            self.results_table.setItem(row, 2, QTableWidgetItem(f"{result.damage:.6g}"))
            self.results_table.setItem(row, 3, QTableWidgetItem(f"{result.max_range:.3f}"))

        self._update_summary(results, exposures)

