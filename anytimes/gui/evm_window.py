"""Extreme value analysis dialog."""
from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QGridLayout,
    QLabel,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QMessageBox,
)

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

__all__ = ['EVMWindow']

