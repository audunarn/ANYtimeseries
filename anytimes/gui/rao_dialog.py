"""Dialog for estimating RAOs from selected time series."""

from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

from ..rao import compute_rao, compute_rao_from_timeseries


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
        self.resize(850, 580)
        self._series_data = series_data
        self._spectral_data = spectral_data or {}

        layout = QVBoxLayout(self)

        form = QFormLayout()
        self.response_combo = QComboBox()
        self.response_combo.addItems(labels)
        self.excitation_combo = QComboBox()
        self.excitation_combo.addItems(labels)
        if len(labels) > 1:
            self.excitation_combo.setCurrentIndex(1)

        self.nperseg_spin = QSpinBox()
        self.nperseg_spin.setRange(8, 1_000_000)
        self.nperseg_spin.setValue(1024)

        form.addRow("Response series", self.response_combo)
        form.addRow("Excitation series", self.excitation_combo)
        form.addRow("Welch segment length", self.nperseg_spin)
        layout.addLayout(form)

        btn_row = QHBoxLayout()
        self.compute_btn = QPushButton("Compute RAO")
        btn_row.addWidget(self.compute_btn)
        btn_row.addStretch(1)
        layout.addLayout(btn_row)

        self.summary_label = QLabel("Pick response/excitation and click Compute RAO.")
        layout.addWidget(self.summary_label)

        self.figure = Figure(figsize=(8, 4), tight_layout=True)
        self.canvas = FigureCanvasQTAgg(self.figure)
        layout.addWidget(self.canvas)

        self.compute_btn.clicked.connect(self._compute)

    def _compute(self) -> None:
        response_key = self.response_combo.currentText()
        excitation_key = self.excitation_combo.currentText()

        if not response_key or not excitation_key:
            QMessageBox.warning(self, "Missing selection", "Please select both response and excitation series.")
            return
        spectral_resp = self._spectral_data.get(response_key)
        if spectral_resp is not None:
            self._plot_spectral_response(response_key, spectral_resp)
            return

        t_resp, y_resp = self._series_data[response_key]
        t_exc, y_exc = self._series_data[excitation_key]

        if t_resp.size < 8 or t_exc.size < 8:
            QMessageBox.warning(self, "Not enough data", "Each series needs at least 8 data points.")
            return

        if t_resp.shape != t_exc.shape or not np.allclose(t_resp, t_exc, rtol=0.0, atol=1e-9):
            QMessageBox.warning(
                self,
                "Time bases differ",
                "Selected series do not share a common time vector. "
                "Use the same source/time window or resample before generating RAO.",
            )
            return

        dt = float(np.median(np.diff(t_resp)))
        if dt <= 0:
            QMessageBox.warning(self, "Invalid time base", "Could not infer a positive time step from the data.")
            return

        single_series_mode = response_key == excitation_key

        try:
            if single_series_mode:
                freqs, amp, phase_deg, coh = compute_rao_from_timeseries(response=y_resp, dt=dt)
            else:
                freqs, amp, phase_deg, coh = compute_rao(
                    excitation=y_exc,
                    response=y_resp,
                    dt=dt,
                    nperseg=self.nperseg_spin.value(),
                )
        except ValueError as exc:
            QMessageBox.warning(self, "RAO error", str(exc))
            return

        valid = freqs > 0.0
        freqs = freqs[valid]
        amp = amp[valid]
        phase_deg = phase_deg[valid]
        coh = coh[valid]

        if freqs.size == 0:
            QMessageBox.warning(self, "RAO error", "No positive frequencies available in the estimate.")
            return

        self.figure.clear()
        ax1 = self.figure.add_subplot(311)
        ax2 = self.figure.add_subplot(312, sharex=ax1)
        ax3 = self.figure.add_subplot(313, sharex=ax1)

        ax1.plot(freqs, amp)
        ax1.set_ylabel("|RAO|")
        ax1.grid(True, alpha=0.3)

        ax2.plot(freqs, phase_deg)
        ax2.set_ylabel("Phase [deg]")
        ax2.grid(True, alpha=0.3)

        if np.all(np.isnan(coh)):
            ax3.text(
                0.5,
                0.5,
                "Coherence available only for paired excitation/response RAO.",
                transform=ax3.transAxes,
                ha="center",
                va="center",
            )
            ax3.set_ylabel("Coherence")
        else:
            ax3.plot(freqs, coh)
            ax3.set_ylim(0.0, 1.0)
            ax3.set_ylabel("Coherence")
        ax3.set_xlabel("Frequency [Hz]")
        ax3.grid(True, alpha=0.3)

        self.canvas.draw_idle()

        peak_idx = int(np.argmax(amp))
        if np.all(np.isnan(coh)):
            self.summary_label.setText(
                f"Peak RAO amplitude {amp[peak_idx]:.4g} at {freqs[peak_idx]:.4g} Hz "
                f"(phase {phase_deg[peak_idx]:.2f}°). Single-series mode assumes unit excitation."
            )
        else:
            self.summary_label.setText(
                f"Peak RAO amplitude {amp[peak_idx]:.4g} at {freqs[peak_idx]:.4g} Hz "
                f"(phase {phase_deg[peak_idx]:.2f}°, coherence {coh[peak_idx]:.2f})."
            )

    def _plot_spectral_response(
        self,
        response_key: str,
        spectral_resp: tuple[np.ndarray, np.ndarray],
    ) -> None:
        freq_hz, rao_amp = spectral_resp

        valid = np.isfinite(freq_hz) & np.isfinite(rao_amp) & (freq_hz > 0.0)
        freq_hz = freq_hz[valid]
        rao_amp = rao_amp[valid]
        if freq_hz.size == 0:
            QMessageBox.warning(self, "RAO error", "No positive frequencies available in spectral response.")
            return

        period = 1.0 / freq_hz

        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.plot(period, rao_amp)
        ax.set_xlabel("Period [s]")
        ax.set_ylabel("|RAO|")
        ax.set_title(f"Spectral Response RAO: {response_key}")
        ax.grid(True, alpha=0.3)
        ax.invert_xaxis()
        self.canvas.draw_idle()

        peak_idx = int(np.argmax(rao_amp))
        self.summary_label.setText(
            f"Spectral RAO peak {rao_amp[peak_idx]:.4g} at {freq_hz[peak_idx]:.4g} Hz "
            f"(period {period[peak_idx]:.4g} s)."
        )
