"""Extreme value analysis dialog."""
from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
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


from anytimes.evm import calculate_extreme_value_statistics, declustering_boundaries


class EVMWindow(QDialog):
    #: Maximum number of clustered exceedances allowed when auto-iterating.
    #: Using too many points tends to bias the tail fit towards the bulk of
    #: the distribution rather than the extremes we want to model.
    _MAX_CLUSTERED_EXCEEDANCES = 120

    def __init__(self, tsdb, var_name, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Extreme Value Analysis - {var_name}")

        self.setWindowFlags(self.windowFlags() | Qt.WindowMaximizeButtonHint)

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

        self._latest_warning: str | None = None
        self._show_canvas_messages = True


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

        boundaries = declustering_boundaries(x, tail)

        while attempts < 10:
            clustered_peaks = []
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                segment = x[start:end]
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

        boundaries = declustering_boundaries(x, tail)

        clustered_peaks = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            segment = x[start:end]
            peak = np.max(segment) if tail == "upper" else np.min(segment)
            if (tail == "upper" and peak > threshold) or (
                tail == "lower" and peak < threshold
            ):
                clustered_peaks.append(peak)
        # use numpy array like helper does to ensure consistent type
        clustered_peaks_arr = np.array(clustered_peaks, dtype=float)

        return clustered_peaks_arr, boundaries


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

        self.canvas_message_checkbox = QCheckBox("Show messages on canvas")
        self.canvas_message_checkbox.setChecked(True)
        self.canvas_message_checkbox.toggled.connect(self.on_canvas_message_toggle)

        run_btn = QPushButton("Run EVM")
        run_btn.clicked.connect(self.run_evm)
        layout.addWidget(run_btn, 4, 0, 1, 2)

        iterate_btn = QPushButton("Iterate Fit")
        iterate_btn.clicked.connect(self.on_iterate_fit)
        layout.addWidget(iterate_btn, 4, 2)

        layout.addWidget(self.canvas_message_checkbox, 5, 0, 1, 3)

    def show_canvas_message(self, message: str):
        """Display *message* on the plot canvas."""

        if not self._show_canvas_messages:
            return

        self.fig.clear()
        self.fig.text(
            0.5,
            0.5,
            message,
            ha="center",
            va="center",
            wrap=True,
            fontsize=11,
        )
        self.fig_canvas.draw()

    def run_evm(self):
        tail = self.tail_combo.currentText()
        threshold = self.threshold_spin.value()

        status, data = self._fit_once(threshold, tail)

        if status == "ok":
            self._manual_threshold = threshold
            self._handle_successful_fit(data, tail)
        elif status == "insufficient":
            message = (
                "Too few clustered exceedances were found for the selected threshold. "
                "Adjust the threshold or tail selection and try again."
            )
            self._latest_warning = None
            QMessageBox.warning(
                self,
                "Too Few Points",
                f"Threshold {threshold:.3f} resulted in only {data['count']} clustered exceedances.",
            )
            self.result_text.setPlainText(message)
            self.show_canvas_message(message)
            self._evm_ran = False
        else:
            message = f"Extreme value analysis failed: {data['message']}"
            self._latest_warning = None
            self.result_text.setPlainText(message)
            self.show_canvas_message(message)
            self._evm_ran = False

    def on_ci_changed(self, value):
        if self._evm_ran:
            self.run_evm()


    def _fit_once(self, threshold: float, tail: str):
        clustered_peaks, boundaries = self._cluster_exceedances(threshold, tail)

        if len(clustered_peaks) < 10:
            return "insufficient", {"count": len(clustered_peaks), "threshold": threshold}

        try:
            evm_result = calculate_extreme_value_statistics(
                self.t,
                self.x,
                threshold,
                tail=tail,
                confidence_level=self.ci_spin.value(),
                clustered_peaks=clustered_peaks,
            )
        except Exception as exc:  # pragma: no cover - defensive GUI guard
            return "error", {"message": str(exc), "threshold": threshold}

        c = evm_result.shape

        warnings: list[str] = []
        if abs(c) > 1:
            warnings.append(
                f"Warning: large shape parameter detected (xi = {c:.4f}). Return levels may be unstable."
            )
        if c < -1e-6:
            warnings.append("Note: fitted GPD shape xi < 0 indicates a bounded tail.")

        warnings_text = "\n".join(warnings) if warnings else None

        return (
            "ok",
            {
                "evm_result": evm_result,
                "boundaries": boundaries,
                "threshold": threshold,
                "warnings": warnings_text,
            },
        )

    def _handle_successful_fit(self, data: dict, tail: str) -> None:
        evm_result = data["evm_result"]
        threshold = data["threshold"]
        boundaries = data["boundaries"]
        warnings_text = data["warnings"]

        c = evm_result.shape
        scale = evm_result.scale

        self._latest_warning = warnings_text

        units = ""
        max_val = np.max(self.x) if tail == "upper" else np.min(self.x)

        header = f"Extreme value statistics: {self.ts.name}"
        if self._latest_warning:
            header = f"{header}\n\n{self._latest_warning}"
        result = f"{header}\n\n"
        result += (
            f"The {evm_result.return_periods[-2]:.1f} hour return level is\n"
            f"{evm_result.return_levels[-2]:.5f} {units}\n\n"
        )
        result += (
            "Fitted GPD parameters:\n"
            f"Sigma: {scale:.4f}\n"
            f"Xi: {c:.4f}\n"
            f"Exceedances used: {len(evm_result.exceedances)}\n"
        )

        result += f"Total crossings/clusters found: {max(len(boundaries) - 1, 0)}\n"

        result += f"Observed maximum value: {max_val:.4f} {units}\n"
        result += f"Return level unit: {units or 'same as input'}\n\n"
        result += f"{self.ci_spin.value():.0f}% Confidence Interval:\n"
        for rp, lo, up in zip(
            evm_result.return_periods,
            evm_result.lower_bounds,
            evm_result.upper_bounds,
        ):
            result += f"{rp:.1f} hr: {lo:.3f} – {up:.3f}\n"

        self.result_text.setPlainText(result)

        self.plot_diagnostics(
            evm_result.return_periods * 3600,
            evm_result.return_levels,
            evm_result.exceedances - threshold,
            c,
            scale,
            threshold,
            evm_result.lower_bounds,
            evm_result.upper_bounds,
            warnings=self._latest_warning,
        )
        self._evm_ran = True

    def _candidate_thresholds(self, base_threshold: float, tail: str) -> list[float]:
        x = np.asarray(self.x, dtype=float)
        min_val = float(np.min(x))
        max_val = float(np.max(x))

        if not np.isfinite(min_val) or not np.isfinite(max_val):
            return [base_threshold]

        span = max_val - min_val
        if span <= 0:
            return [base_threshold]

        delta = span / 1000.0
        if delta <= 0:
            delta = max(abs(max_val), abs(min_val), 1.0) / 1000.0

        base_clamped = float(np.clip(base_threshold, min_val, max_val))

        candidates: list[float] = []
        seen: set[float] = set()

        if tail == "upper":
            current = float(max_val - delta)
            if current < min_val:
                current = max_val

            steps = 0
            while current >= min_val and steps < 2000:
                key = round(current, 6)
                if key not in seen:
                    candidates.append(current)
                    seen.add(key)
                current -= delta
                steps += 1

        else:
            current = float(min_val + delta)
            if current > max_val:
                current = min_val

            steps = 0
            while current <= max_val and steps < 2000:
                key = round(current, 6)
                if key not in seen:
                    candidates.append(current)
                    seen.add(key)
                current += delta
                steps += 1

        base_key = round(base_clamped, 6)
        if base_key not in seen:
            candidates.append(base_clamped)

        if tail == "upper":
            candidates.sort(reverse=True)
        else:
            candidates.sort()

        return candidates

    def on_iterate_fit(self) -> None:
        tail = self.tail_combo.currentText()
        base_threshold = self.threshold_spin.value()

        self.result_text.setPlainText("Iterating to find a stable fit...")
        self._evm_ran = False

        best_success: dict | None = None
        best_with_warning: dict | None = None
        failure_messages: list[str] = []

        candidate_thresholds = self._candidate_thresholds(base_threshold, tail)
        candidate_thresholds = sorted(
            candidate_thresholds,
            reverse=(tail == "upper"),
        )

        for threshold in candidate_thresholds:
            status, data = self._fit_once(threshold, tail)

            if status == "ok":
                exceedance_count = len(data["evm_result"].exceedances)

                if exceedance_count > self._MAX_CLUSTERED_EXCEEDANCES:
                    failure_messages.append(
                        (
                            "Threshold {thresh:.3f} produced {count} clustered "
                            "exceedances which is above the maximum of {max_count}."
                        ).format(
                            thresh=threshold,
                            count=exceedance_count,
                            max_count=self._MAX_CLUSTERED_EXCEEDANCES,
                        )
                    )
                    continue

                if data["warnings"]:
                    if best_with_warning is None:
                        best_with_warning = data
                    continue

                best_success = data
                break

            elif status == "insufficient":
                failure_messages.append(
                    f"Threshold {threshold:.3f} resulted in only {data['count']} clustered exceedances."
                )
            elif status == "error":
                failure_messages.append(data["message"])

        if best_success and not best_success["warnings"]:
            final_threshold = best_success["threshold"]
            self.threshold_spin.setValue(round(final_threshold, 4))
            self._manual_threshold = final_threshold
            self._handle_successful_fit(best_success, tail)
            return

        if best_with_warning:
            warning_details = best_with_warning.get("warnings") or ""
            if warning_details:
                warning_details = f"\n\n{warning_details}"
            final_threshold = best_with_warning["threshold"]
            self.threshold_spin.setValue(round(final_threshold, 4))
            self._manual_threshold = final_threshold
            self._handle_successful_fit(best_with_warning, tail)
            return

        message = "\n".join(failure_messages)
        if not message:
            message = "Iteration failed to compute a valid extreme value fit."
        self._latest_warning = None
        self.result_text.setPlainText(message)
        self.show_canvas_message(message)
        self._evm_ran = False


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
        *,
        warnings: str | None = None,
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

        if warnings:
            if self._show_canvas_messages:
                self.fig.suptitle(warnings, color="red", fontsize=10)
                self.fig.tight_layout(rect=[0, 0, 1, 0.92])
            else:
                self.fig.tight_layout()
        else:
            self.fig.tight_layout()

        self.fig_canvas.draw()

    def display_message_on_canvas(self, message: str) -> None:
        """Display a centered message in the plotting canvas."""

        if not self._show_canvas_messages:
            return

        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            message,
            ha="center",
            va="center",
            wrap=True,
        )
        self.fig_canvas.draw()

    def on_canvas_message_toggle(self, checked: bool) -> None:
        """Callback to enable/disable canvas messaging."""

        self._show_canvas_messages = checked

        if not checked:
            # Clear any existing message/warning from the canvas when disabled.
            self.fig.clear()
            self.fig_canvas.draw()

__all__ = ['EVMWindow']

