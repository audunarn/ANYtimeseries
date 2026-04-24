"""Dialog for estimating RAOs from selected time series.

RAO workflow
------------
1. If the selected response has precomputed spectral/OrcaFlex RAO data, use it.
2. Otherwise, compute paired excitation/response RAO from time series.
3. As a fallback, allow a single-series approximation assuming unit excitation.

Important unit convention
-------------------------
The plot x-axis is always Period [s].

For precomputed OrcaFlex/spectral RAO data, the source x-data may be either:
    - frequency in Hz
    - period in seconds

The user selects this in the dialog.

For time-series based RAO estimates, the frequency axis comes from rao.py in Hz
and is automatically converted to period using:

    period = 1 / frequency_hz
"""

from __future__ import annotations

from enum import Enum

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
from matplotlib.figure import Figure
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
)

from anytimes.rao import compute_rao, compute_rao_from_timeseries


class RAOMode(str, Enum):
    """Available RAO calculation modes."""

    AUTO = "Automatic"
    PRECOMPUTED = "Use precomputed OrcaFlex / spectral RAO"
    PAIRED = "Compute from excitation + response time series"
    SINGLE = "Single time-series simplification"


class RAOXSourceUnit(str, Enum):
    """Source unit for precomputed RAO x-data."""

    HZ = "Hz"
    PERIOD_S = "Period [s]"


class RAODialog(QDialog):
    """Modal dialog that computes/plots RAO estimates."""

    def __init__(
        self,
        labels: list[str],
        series_data: dict[str, tuple[np.ndarray, np.ndarray]],
        spectral_data: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)

        self.setWindowTitle("RAO Estimator")
        self.resize(1050, 780)
        self.setMinimumSize(780, 560)

        self.setWindowFlags(
            self.windowFlags()
            | Qt.WindowMinimizeButtonHint
            | Qt.WindowMaximizeButtonHint
            | Qt.WindowCloseButtonHint
        )

        self._series_data = series_data
        self._spectral_data = spectral_data or {}

        response_labels = self._unique_preserve_order(
            list(labels) + list(self._series_data.keys()) + list(self._spectral_data.keys())
        )
        excitation_labels = self._unique_preserve_order(
            list(labels) + list(self._series_data.keys())
        )

        main_layout = QVBoxLayout(self)

        explanation_label = QLabel(
            "<b>RAO Estimator</b><br>"
            "Use this dialog to plot Response Amplitude Operators. "
            "If the selected response has precomputed OrcaFlex/spectral RAO data, "
            "that RAO is plotted directly. Otherwise, select both an excitation time series "
            "and a response time series to estimate the RAO. The single-series option is only "
            "a fallback approximation when no excitation signal is available.<br><br>"
            "The plot x-axis is always <b>Period [s]</b>. "
            "For precomputed RAO data, select whether the source x-data is <b>Hz</b> "
            "or already <b>Period [s]</b>."
        )
        explanation_label.setWordWrap(True)
        main_layout.addWidget(explanation_label)

        input_group = QGroupBox("Input selection")
        input_layout = QVBoxLayout(input_group)

        form = QFormLayout()

        self.mode_combo = QComboBox()
        self.mode_combo.addItems([mode.value for mode in RAOMode])
        self.mode_combo.setToolTip(
            "Choose how the RAO should be obtained. "
            "Automatic uses precomputed OrcaFlex/spectral RAO if available; otherwise it computes from time series."
        )

        self.response_combo = QComboBox()
        self.response_combo.addItems(response_labels)
        self.response_combo.setToolTip(
            "Select the response quantity. This may be a motion, force, acceleration, "
            "moment, or a precomputed OrcaFlex/spectral RAO."
        )

        self.excitation_combo = QComboBox()
        self.excitation_combo.addItems(excitation_labels)
        self.excitation_combo.setToolTip(
            "Select the excitation signal, normally wave elevation. "
            "This is required for a true paired excitation/response RAO."
        )

        if len(excitation_labels) > 1:
            self.excitation_combo.setCurrentIndex(1)

        self.nperseg_spin = QSpinBox()
        self.nperseg_spin.setRange(8, 1_000_000)
        self.nperseg_spin.setValue(1024)
        self.nperseg_spin.setToolTip(
            "Welch segment length used for paired time-series spectral estimation. "
            "Larger values give finer frequency resolution but noisier estimates. "
            "Smaller values give smoother estimates but coarser frequency resolution."
        )

        self.x_source_unit_combo = QComboBox()
        self.x_source_unit_combo.addItems([unit.value for unit in RAOXSourceUnit])
        self.x_source_unit_combo.setCurrentText(RAOXSourceUnit.HZ.value)
        self.x_source_unit_combo.setToolTip(
            "Source unit of the first array in precomputed/spectral RAO data. "
            "Use Hz if the source x-data is frequency. "
            "Use Period [s] if the source x-data is already period."
        )

        self.smoothing_checkbox = QCheckBox("Apply rolling mean smoothing")
        self.smoothing_checkbox.setChecked(False)
        self.smoothing_checkbox.setToolTip(
            "Apply a centered rolling mean to the plotted RAO amplitude. "
            "For paired RAOs, coherence is also smoothed. Phase is not smoothed."
        )

        self.smoothing_window_spin = QSpinBox()
        self.smoothing_window_spin.setRange(3, 501)
        self.smoothing_window_spin.setSingleStep(2)
        self.smoothing_window_spin.setValue(9)
        self.smoothing_window_spin.setToolTip(
            "Number of neighboring period points used for the rolling mean. "
            "Use an odd number such as 5, 9, 15, or 21. Larger values give smoother curves."
        )
        self.smoothing_window_spin.setEnabled(False)

        self.smoothing_checkbox.toggled.connect(self.smoothing_window_spin.setEnabled)

        form.addRow("RAO mode", self.mode_combo)
        form.addRow("Response series / RAO", self.response_combo)
        form.addRow("Excitation series", self.excitation_combo)
        form.addRow("Welch segment length", self.nperseg_spin)
        form.addRow("Precomputed RAO x-data source", self.x_source_unit_combo)
        form.addRow("Smooth RAO amplitude", self.smoothing_checkbox)
        form.addRow("Smoothing window", self.smoothing_window_spin)

        input_layout.addLayout(form)

        self.mode_help_label = QLabel()
        self.mode_help_label.setWordWrap(True)
        input_layout.addWidget(self.mode_help_label)

        self.response_help_label = QLabel()
        self.response_help_label.setWordWrap(True)
        input_layout.addWidget(self.response_help_label)

        self.excitation_help_label = QLabel()
        self.excitation_help_label.setWordWrap(True)
        input_layout.addWidget(self.excitation_help_label)

        self.x_source_unit_help_label = QLabel()
        self.x_source_unit_help_label.setWordWrap(True)
        input_layout.addWidget(self.x_source_unit_help_label)

        self.nperseg_help_label = QLabel(
            "<b>Welch segment length:</b> Used only when computing RAO from paired time series. "
            "The value is automatically limited by the signal length in rao.py. "
            "A typical value is 1024, but longer records may allow larger values."
        )
        self.nperseg_help_label.setWordWrap(True)
        input_layout.addWidget(self.nperseg_help_label)

        self.smoothing_help_label = QLabel(
            "<b>Smoothing:</b> Optional centered rolling mean applied to RAO amplitude. "
            "For paired RAOs, coherence is also smoothed. "
            "Phase is always plotted raw to avoid misleading smoothing across phase jumps."
        )
        self.smoothing_help_label.setWordWrap(True)
        input_layout.addWidget(self.smoothing_help_label)

        main_layout.addWidget(input_group)

        btn_row = QHBoxLayout()
        self.compute_btn = QPushButton("Compute / Plot RAO")
        self.compute_btn.setToolTip("Compute or plot the selected RAO based on the selected mode.")
        btn_row.addWidget(self.compute_btn)
        btn_row.addStretch(1)
        main_layout.addLayout(btn_row)

        self.summary_label = QLabel(
            "Select a response. If precomputed RAO data is available, it will be used automatically. "
            "Otherwise select an excitation time series."
        )
        self.summary_label.setWordWrap(True)
        main_layout.addWidget(self.summary_label)

        self.figure = Figure(figsize=(9, 6), tight_layout=True)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.updateGeometry()

        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        main_layout.addWidget(self.toolbar)
        main_layout.addWidget(self.canvas, stretch=1)

        self.compute_btn.clicked.connect(self._compute)
        self.mode_combo.currentTextChanged.connect(self._update_control_state)
        self.response_combo.currentTextChanged.connect(self._update_control_state)
        self.excitation_combo.currentTextChanged.connect(self._update_control_state)
        self.x_source_unit_combo.currentTextChanged.connect(self._update_control_state)

        self._update_control_state()

    @staticmethod
    def _unique_preserve_order(values: list[str]) -> list[str]:
        """Return unique non-empty strings while preserving order."""

        seen: set[str] = set()
        unique: list[str] = []

        for value in values:
            if not value:
                continue
            if value in seen:
                continue

            seen.add(value)
            unique.append(value)

        return unique

    def _rolling_mean_same(
        self,
        values: np.ndarray,
        window: int,
    ) -> np.ndarray:
        """Return centered rolling mean with the same length as input.

        NaN values are ignored. If all values in a window are NaN, the smoothed
        value is NaN.
        """

        values = np.asarray(values, dtype=float)

        if values.size == 0:
            return values.copy()

        window = int(window)

        if window < 3:
            return values.copy()

        if window % 2 == 0:
            window += 1

        window = min(window, values.size)

        if window < 3:
            return values.copy()

        if window % 2 == 0:
            window -= 1

        kernel = np.ones(window, dtype=float)

        finite = np.isfinite(values)
        filled = np.where(finite, values, 0.0)

        summed = np.convolve(filled, kernel, mode="same")
        counts = np.convolve(finite.astype(float), kernel, mode="same")

        with np.errstate(invalid="ignore", divide="ignore"):
            smoothed = summed / counts

        smoothed[counts == 0.0] = np.nan

        return smoothed

    def _smoothing_enabled(self) -> bool:
        """Return True if rolling mean smoothing should be applied."""

        return self.smoothing_checkbox.isChecked()

    def _smoothing_window(self) -> int:
        """Return smoothing window as an odd integer."""

        window = int(self.smoothing_window_spin.value())

        if window % 2 == 0:
            window += 1

        return window

    def _selected_mode(self) -> RAOMode:
        """Return selected RAO mode."""

        text = self.mode_combo.currentText()

        for mode in RAOMode:
            if mode.value == text:
                return mode

        return RAOMode.AUTO

    def _selected_x_source_unit(self) -> RAOXSourceUnit:
        """Return selected x-data source unit for precomputed RAO data."""

        text = self.x_source_unit_combo.currentText()

        for unit in RAOXSourceUnit:
            if unit.value == text:
                return unit

        return RAOXSourceUnit.HZ

    def _update_control_state(self) -> None:
        """Enable/disable controls and update input explanations."""

        mode = self._selected_mode()
        response_key = self.response_combo.currentText()
        excitation_key = self.excitation_combo.currentText()
        has_precomputed = response_key in self._spectral_data

        if mode == RAOMode.AUTO:
            using_precomputed = has_precomputed
            using_single = False
            using_paired = not has_precomputed
        elif mode == RAOMode.PRECOMPUTED:
            using_precomputed = True
            using_single = False
            using_paired = False
        elif mode == RAOMode.SINGLE:
            using_precomputed = False
            using_single = True
            using_paired = False
        else:
            using_precomputed = False
            using_single = False
            using_paired = True

        self.excitation_combo.setEnabled(using_paired)
        self.nperseg_spin.setEnabled(using_paired)

        # Keep this enabled so the user can select the expected precomputed
        # source x-data unit before plotting.
        self.x_source_unit_combo.setEnabled(True)

        if mode == RAOMode.AUTO:
            self.mode_help_label.setText(
                "<b>Mode:</b> Automatic. If the selected response has precomputed "
                "OrcaFlex/spectral RAO data, that RAO is plotted directly. "
                "If not, the dialog computes a paired time-series RAO using the selected excitation."
            )
        elif mode == RAOMode.PRECOMPUTED:
            self.mode_help_label.setText(
                "<b>Mode:</b> Precomputed RAO. Use this when the RAO has already been "
                "provided by OrcaFlex or another spectral calculation. No excitation time series is required."
            )
        elif mode == RAOMode.PAIRED:
            self.mode_help_label.setText(
                "<b>Mode:</b> Paired time-series RAO. Computes H(f) = response / excitation "
                "from two synchronized time series using Welch/CSD spectral estimates."
            )
        else:
            self.mode_help_label.setText(
                "<b>Mode:</b> Single-series simplification. This assumes unit excitation "
                "and returns a one-sided response amplitude spectrum. This is not a true RAO."
            )

        if has_precomputed:
            self.response_help_label.setText(
                f"<b>Response:</b> '{response_key}' has precomputed RAO data available. "
                "In Automatic or Precomputed mode, this will be used directly."
            )
        elif response_key in self._series_data:
            self.response_help_label.setText(
                f"<b>Response:</b> '{response_key}' is a time-domain response signal. "
                "For a true RAO, select a separate excitation time series."
            )
        else:
            self.response_help_label.setText(
                f"<b>Response:</b> No time-series or precomputed RAO data was found for '{response_key}'."
            )

        if using_paired:
            if excitation_key in self._series_data:
                self.excitation_help_label.setText(
                    f"<b>Excitation:</b> '{excitation_key}' will be used as the input signal. "
                    "This should normally be wave elevation or another known excitation. "
                    "It must have the same time vector as the response."
                )
            else:
                self.excitation_help_label.setText(
                    f"<b>Excitation:</b> No time-series data was found for '{excitation_key}'."
                )
        elif using_precomputed:
            self.excitation_help_label.setText(
                "<b>Excitation:</b> Disabled because the RAO is already available. "
                "The selected excitation is ignored in this mode."
            )
        elif using_single:
            self.excitation_help_label.setText(
                "<b>Excitation:</b> Disabled because single-series mode does not use an excitation signal. "
                "The result assumes unit excitation amplitude."
            )

        self.x_source_unit_help_label.setText(
            "<b>Precomputed RAO x-data source:</b> Select how the first precomputed RAO array "
            "should be interpreted. Use <b>Hz</b> when the source x-data is frequency. "
            "Use <b>Period [s]</b> when the source x-data is already period. "
            "This setting only affects precomputed/spectral RAO data. "
            "The plot x-axis is always Period [s]."
        )

        if using_precomputed:
            if has_precomputed:
                self.summary_label.setText(
                    f"Ready to plot precomputed OrcaFlex/spectral RAO for '{response_key}'. "
                    f"Source x-data will be interpreted as {self.x_source_unit_combo.currentText()}. "
                    "Excitation selection is ignored."
                )
            else:
                self.summary_label.setText(
                    f"No precomputed RAO data is available for '{response_key}'. "
                    "Use paired time-series mode or single-series simplification."
                )
        elif using_single:
            self.summary_label.setText(
                "Single-series simplification selected. This is a fallback approximation "
                "and should only be used when no excitation signal is available."
            )
        else:
            self.summary_label.setText(
                "Paired RAO mode selected. The response and excitation must have the same time vector. "
                "The output frequency axis from rao.py is converted to Period [s]."
            )

    def _compute(self) -> None:
        """Compute or plot the selected RAO."""

        response_key = self.response_combo.currentText()
        excitation_key = self.excitation_combo.currentText()
        mode = self._selected_mode()

        if not response_key:
            QMessageBox.warning(self, "Missing selection", "Please select a response series or RAO.")
            return

        has_precomputed = response_key in self._spectral_data

        if mode == RAOMode.AUTO:
            if has_precomputed:
                self._plot_precomputed_rao(response_key, self._spectral_data[response_key])
                return

            if not excitation_key:
                QMessageBox.warning(
                    self,
                    "Missing excitation",
                    "No precomputed RAO is available. Please select an excitation time series.",
                )
                return

            if response_key == excitation_key:
                self._compute_single_series(response_key)
                return

            self._compute_paired_series(response_key, excitation_key)
            return

        if mode == RAOMode.PRECOMPUTED:
            if not has_precomputed:
                QMessageBox.warning(
                    self,
                    "No precomputed RAO",
                    f"No precomputed OrcaFlex/spectral RAO is available for '{response_key}'.",
                )
                return

            self._plot_precomputed_rao(response_key, self._spectral_data[response_key])
            return

        if mode == RAOMode.SINGLE:
            self._compute_single_series(response_key)
            return

        if mode == RAOMode.PAIRED:
            if not excitation_key:
                QMessageBox.warning(
                    self,
                    "Missing excitation",
                    "Please select an excitation time series.",
                )
                return

            if response_key == excitation_key:
                QMessageBox.warning(
                    self,
                    "Invalid paired RAO selection",
                    "For paired RAO mode, response and excitation must be different series. "
                    "Use single-series simplification if you only have one signal.",
                )
                return

            self._compute_paired_series(response_key, excitation_key)
            return

    def _get_time_series(
        self,
        key: str,
        role: str,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Fetch and validate a time series."""

        if key not in self._series_data:
            QMessageBox.warning(
                self,
                f"Missing {role} data",
                f"No time-series data is available for {role} '{key}'.",
            )
            return None

        t, y = self._series_data[key]

        t = np.asarray(t, dtype=float)
        y = np.asarray(y, dtype=float)

        if t.ndim != 1 or y.ndim != 1:
            QMessageBox.warning(
                self,
                f"Invalid {role} data",
                f"The {role} time and value arrays must both be 1-D.",
            )
            return None

        if t.size != y.size:
            QMessageBox.warning(
                self,
                f"Invalid {role} data",
                f"The {role} time and value arrays must have the same length.",
            )
            return None

        if t.size < 8:
            QMessageBox.warning(
                self,
                "Not enough data",
                f"The {role} series needs at least 8 data points.",
            )
            return None

        if not np.all(np.isfinite(t)) or not np.all(np.isfinite(y)):
            QMessageBox.warning(
                self,
                f"Invalid {role} data",
                f"The {role} series contains NaN or infinite values.",
            )
            return None

        return t, y

    def _infer_uniform_dt(self, t: np.ndarray) -> float | None:
        """Infer and validate constant time step."""

        diffs = np.diff(t)

        if diffs.size == 0 or not np.all(np.isfinite(diffs)):
            QMessageBox.warning(
                self,
                "Invalid time base",
                "Could not infer a valid time step from the selected data.",
            )
            return None

        dt = float(np.median(diffs))

        if dt <= 0.0:
            QMessageBox.warning(
                self,
                "Invalid time base",
                "Could not infer a positive time step from the selected data.",
            )
            return None

        if not np.allclose(diffs, dt, rtol=1e-4, atol=1e-9):
            QMessageBox.warning(
                self,
                "Non-uniform time base",
                "The selected time series is not uniformly sampled. "
                "Resample before estimating RAO.",
            )
            return None

        return dt

    def _compute_paired_series(
        self,
        response_key: str,
        excitation_key: str,
    ) -> None:
        """Compute a true paired excitation/response RAO."""

        response_data = self._get_time_series(response_key, "response")
        if response_data is None:
            return

        excitation_data = self._get_time_series(excitation_key, "excitation")
        if excitation_data is None:
            return

        t_resp, y_resp = response_data
        t_exc, y_exc = excitation_data

        if t_resp.shape != t_exc.shape or not np.allclose(t_resp, t_exc, rtol=0.0, atol=1e-9):
            QMessageBox.warning(
                self,
                "Time bases differ",
                "Selected response and excitation series do not share a common time vector. "
                "Use the same source/time window or resample before generating RAO.",
            )
            return

        dt = self._infer_uniform_dt(t_resp)
        if dt is None:
            return

        try:
            freqs, amp, phase_deg, coh = compute_rao(
                excitation=y_exc,
                response=y_resp,
                dt=dt,
                nperseg=self.nperseg_spin.value(),
                remove_zero_frequency=True,
                unwrap_phase=False,
            )
        except ValueError as exc:
            QMessageBox.warning(self, "RAO error", str(exc))
            return

        self._plot_computed_rao(
            freqs=freqs,
            amp=amp,
            phase_deg=phase_deg,
            coh=coh,
            title=f"RAO: {response_key} / {excitation_key}",
            single_series=False,
        )

    def _compute_single_series(self, response_key: str) -> None:
        """Compute fallback single-series amplitude spectrum."""

        response_data = self._get_time_series(response_key, "response")
        if response_data is None:
            return

        t_resp, y_resp = response_data

        dt = self._infer_uniform_dt(t_resp)
        if dt is None:
            return

        try:
            freqs, amp, phase_deg, coh = compute_rao_from_timeseries(
                response=y_resp,
                dt=dt,
                remove_zero_frequency=True,
                unwrap_phase=False,
            )
        except ValueError as exc:
            QMessageBox.warning(self, "RAO error", str(exc))
            return

        self._plot_computed_rao(
            freqs=freqs,
            amp=amp,
            phase_deg=phase_deg,
            coh=coh,
            title=f"Single-series amplitude spectrum: {response_key}",
            single_series=True,
        )

    def _plot_computed_rao(
        self,
        freqs: np.ndarray,
        amp: np.ndarray,
        phase_deg: np.ndarray,
        coh: np.ndarray,
        title: str,
        single_series: bool,
    ) -> None:
        """Plot computed RAO or single-series spectrum versus period.

        For time-series estimates, rao.py returns frequency in Hz. This method
        always converts that to period using period = 1 / frequency_hz.
        """

        freqs = np.asarray(freqs, dtype=float)
        amp = np.asarray(amp, dtype=float)
        phase_deg = np.asarray(phase_deg, dtype=float)
        coh = np.asarray(coh, dtype=float)

        valid = np.isfinite(freqs) & np.isfinite(amp) & np.isfinite(phase_deg) & (freqs > 0.0)

        if coh.shape == freqs.shape:
            valid = valid & (np.isfinite(coh) | np.isnan(coh))

        freqs = freqs[valid]
        amp = amp[valid]
        phase_deg = phase_deg[valid]
        coh = coh[valid]

        if freqs.size == 0:
            QMessageBox.warning(
                self,
                "RAO error",
                "No positive frequencies are available in the estimate.",
            )
            return

        period = 1.0 / freqs

        order = np.argsort(period)
        period = period[order]
        freqs = freqs[order]
        amp = amp[order]
        phase_deg = phase_deg[order]
        coh = coh[order]

        use_smoothing = self._smoothing_enabled()
        smoothing_window = self._smoothing_window()

        if use_smoothing:
            amp_plot = self._rolling_mean_same(amp, smoothing_window)
            coh_plot = self._rolling_mean_same(coh, smoothing_window)
            title = f"{title} — amplitude rolling mean window {smoothing_window}"
        else:
            amp_plot = amp
            coh_plot = coh

        self.figure.clear()

        ax1 = self.figure.add_subplot(311)
        ax2 = self.figure.add_subplot(312, sharex=ax1)
        ax3 = self.figure.add_subplot(313, sharex=ax1)

        if use_smoothing:
            ax1.plot(period, amp, alpha=0.25, label="Raw amplitude")
            ax1.plot(period, amp_plot, label=f"Rolling mean ({smoothing_window})")
            ax1.legend(loc="best")
        else:
            ax1.plot(period, amp_plot)

        ax1.set_title(title)
        ax1.set_ylabel("|RAO|")
        ax1.grid(True, alpha=0.3)

        ax2.plot(period, phase_deg)
        ax2.set_ylabel("Phase [deg]")
        ax2.grid(True, alpha=0.3)

        if coh.size == 0 or np.all(np.isnan(coh)):
            ax3.text(
                0.5,
                0.5,
                "Coherence is available only for paired excitation/response RAO.",
                transform=ax3.transAxes,
                ha="center",
                va="center",
            )
            ax3.set_ylim(0.0, 1.0)
        else:
            if use_smoothing:
                ax3.plot(period, coh, alpha=0.25, label="Raw coherence")
                ax3.plot(period, coh_plot, label=f"Rolling mean ({smoothing_window})")
                ax3.legend(loc="best")
            else:
                ax3.plot(period, coh_plot)

            ax3.set_ylim(0.0, 1.0)

        ax3.set_ylabel("Coherence")
        ax3.set_xlabel("Period [s]")
        ax3.grid(True, alpha=0.3)

        self.canvas.draw_idle()

        peak_source = amp_plot if use_smoothing else amp

        if not np.any(np.isfinite(peak_source)):
            QMessageBox.warning(
                self,
                "RAO error",
                "No finite RAO amplitudes are available after smoothing.",
            )
            return

        peak_idx = int(np.nanargmax(peak_source))
        peak_label = "Smoothed peak" if use_smoothing else "Peak"

        if single_series:
            self.summary_label.setText(
                f"{peak_label} amplitude {peak_source[peak_idx]:.4g} "
                f"at period {period[peak_idx]:.4g} s "
                f"({freqs[peak_idx]:.4g} Hz, phase {phase_deg[peak_idx]:.2f}°). "
                "Single-series simplification assumes unit excitation and is not a true paired RAO."
            )
        elif coh.size == 0 or np.all(np.isnan(coh)):
            self.summary_label.setText(
                f"{peak_label} RAO amplitude {peak_source[peak_idx]:.4g} "
                f"at period {period[peak_idx]:.4g} s "
                f"({freqs[peak_idx]:.4g} Hz, phase {phase_deg[peak_idx]:.2f}°)."
            )
        else:
            self.summary_label.setText(
                f"{peak_label} RAO amplitude {peak_source[peak_idx]:.4g} "
                f"at period {period[peak_idx]:.4g} s "
                f"({freqs[peak_idx]:.4g} Hz, phase {phase_deg[peak_idx]:.2f}°, "
                f"coherence {coh_plot[peak_idx]:.2f})."
            )

    def _plot_precomputed_rao(
        self,
        response_key: str,
        spectral_resp: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Plot precomputed spectral/OrcaFlex RAO data.

        Expected input:
            spectral_resp = (x_data, rao_amplitude)

        The x-data may be:
            - frequency in Hz
            - period in seconds

        The user-selected GUI option controls the interpretation.

        The plot x-axis is always Period [s].
        """

        x_input, rao_amp = spectral_resp

        x_input = np.asarray(x_input, dtype=float)
        rao_amp = np.asarray(rao_amp, dtype=float)

        if x_input.ndim != 1 or rao_amp.ndim != 1:
            QMessageBox.warning(
                self,
                "Invalid spectral RAO",
                "Precomputed RAO x-data and amplitude arrays must be 1-D.",
            )
            return

        if x_input.size != rao_amp.size:
            QMessageBox.warning(
                self,
                "Invalid spectral RAO",
                "Precomputed RAO x-data and amplitude arrays must have the same length.",
            )
            return

        valid = np.isfinite(x_input) & np.isfinite(rao_amp) & (x_input > 0.0)
        x_input = x_input[valid]
        rao_amp = rao_amp[valid]

        if x_input.size == 0:
            QMessageBox.warning(
                self,
                "RAO error",
                "No positive x-data values are available in the precomputed RAO.",
            )
            return

        x_source_unit = self._selected_x_source_unit()

        if x_source_unit == RAOXSourceUnit.PERIOD_S:
            period = x_input
            freq_hz = 1.0 / period
        else:
            freq_hz = x_input
            period = 1.0 / freq_hz

        valid_period = (
            np.isfinite(period)
            & np.isfinite(freq_hz)
            & (period > 0.0)
            & (freq_hz > 0.0)
        )

        period = period[valid_period]
        freq_hz = freq_hz[valid_period]
        rao_amp = rao_amp[valid_period]

        if period.size == 0:
            QMessageBox.warning(
                self,
                "RAO error",
                "No positive period values are available after x-data conversion.",
            )
            return

        order = np.argsort(period)
        period = period[order]
        freq_hz = freq_hz[order]
        rao_amp = rao_amp[order]

        use_smoothing = self._smoothing_enabled()
        smoothing_window = self._smoothing_window()

        if use_smoothing:
            rao_amp_plot = self._rolling_mean_same(rao_amp, smoothing_window)
            title = (
                f"Precomputed OrcaFlex / spectral RAO: {response_key} "
                f"— amplitude rolling mean window {smoothing_window}"
            )
        else:
            rao_amp_plot = rao_amp
            title = f"Precomputed OrcaFlex / spectral RAO: {response_key}"

        self.figure.clear()

        ax = self.figure.add_subplot(111)

        if use_smoothing:
            ax.plot(period, rao_amp, alpha=0.25, label="Raw amplitude")
            ax.plot(period, rao_amp_plot, label=f"Rolling mean ({smoothing_window})")
            ax.legend(loc="best")
        else:
            ax.plot(period, rao_amp_plot)

        ax.set_xlabel("Period [s]")
        ax.set_ylabel("|RAO|")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

        self.canvas.draw_idle()

        if not np.any(np.isfinite(rao_amp_plot)):
            QMessageBox.warning(
                self,
                "RAO error",
                "No finite RAO amplitudes are available after smoothing.",
            )
            return

        peak_idx = int(np.nanargmax(rao_amp_plot))
        peak_label = "Smoothed peak" if use_smoothing else "Precomputed RAO peak"

        self.summary_label.setText(
            f"{peak_label} {rao_amp_plot[peak_idx]:.4g} "
            f"at period {period[peak_idx]:.4g} s "
            f"({freq_hz[peak_idx]:.4g} Hz). "
            f"Source x-data interpreted as {x_source_unit.value}. "
            "Excitation selection is ignored because the RAO is already available."
        )