"""Extreme value analysis dialog."""
from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import Qt
from PySide6.QtGui import QPalette
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QGridLayout,
    QLineEdit,
    QLabel,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QMessageBox,
)


from anytimes.evm import (
    SUMMARY_RETURN_PERIODS_HOURS,
    calculate_extreme_value_statistics,
    decluster_peaks,
    ExtremeValueResult,
)
from .layout_utils import apply_initial_size


class EVMWindow(QDialog):
    #: Maximum number of clustered exceedances allowed when auto-iterating.
    #: Using too many points tends to bias the tail fit towards the bulk of
    #: the distribution rather than the extremes we want to model.
    _MAX_CLUSTERED_EXCEEDANCES = 120

    def __init__(self, tsdb, var_name, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Extreme Value Analysis - {var_name}")

        self.setWindowFlags(self.windowFlags() | Qt.WindowMaximizeButtonHint)

        apply_initial_size(
            self,
            desired_width=900,
            desired_height=700,
            min_width=760,
            min_height=540,
            width_ratio=0.85,
            height_ratio=0.85,
        )

        self.ts = tsdb.getm()[var_name]
        self.x = self.ts.x
        self.t = self.ts.t

        self.engine_combo = QComboBox()
        self.engine_combo.addItem("Built-in (GPD)", "builtin")
        self.engine_combo.addItem("PyExtremes (POT)", "pyextremes")
        self.engine_combo.currentIndexChanged.connect(self.on_engine_changed)

        self.distribution_label = QLabel("Distribution: Generalized Pareto (built-in)")

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

        self.declustering_spin = QDoubleSpinBox()
        self.declustering_spin.setDecimals(3)
        self.declustering_spin.setRange(0.0, 1e9)
        self.declustering_spin.setSuffix(" s")
        self.declustering_spin.setKeyboardTracking(False)
        self.declustering_spin.setValue(0.0)

        # Determine an initial reasonable threshold
        threshold = self._auto_threshold(suggested, self.tail_combo.currentText())
        self.threshold_spin.setValue(round(threshold, 4))
        self.threshold_spin.editingFinished.connect(self.on_manual_threshold)

        self._manual_threshold = threshold

        self.pyext_r_spin = QDoubleSpinBox()
        self.pyext_r_spin.setDecimals(3)
        self.pyext_r_spin.setRange(0.0, 1e9)
        self.pyext_r_spin.setSuffix(" s")
        self.pyext_r_spin.setKeyboardTracking(False)
        self.pyext_r_spin.setValue(round(self._suggest_pyextremes_window(), 3))

        self.pyext_return_size_spin = QDoubleSpinBox()
        self.pyext_return_size_spin.setDecimals(3)
        self.pyext_return_size_spin.setRange(0.001, 1e6)
        self.pyext_return_size_spin.setSuffix(" h")
        self.pyext_return_size_spin.setKeyboardTracking(False)
        self.pyext_return_size_spin.setValue(1.0)

        self.pyext_samples_spin = QSpinBox()
        self.pyext_samples_spin.setRange(50, 10000)
        self.pyext_samples_spin.setSingleStep(50)
        self.pyext_samples_spin.setValue(400)

        self.pyext_return_periods_edit = QLineEdit()
        self.pyext_return_periods_edit.setPlaceholderText("None")
        self.pyext_return_periods_edit.setText("None")

        self._plotting_positions = [
            ("Empirical CDF (m/n)", "ecdf"),
            ("Hazen ((m-0.5)/n)", "hazen"),
            ("Weibull (m/(n+1))", "weibull"),
            ("Tukey ((m-1/3)/(n+1/3))", "tukey"),
            ("Blom ((m-0.375)/(n+0.25))", "blom"),
            ("Median ((m-0.3)/(n+0.4))", "median"),
            ("Cunnane ((m-0.4)/(n+0.2))", "cunnane"),
            ("Gringorten ((m-0.44)/(n+0.12))", "gringorten"),
            ("Beard ((m-0.31)/(n+0.38))", "beard"),
        ]
        self._plotting_position_labels = {
            value: label for label, value in self._plotting_positions
        }
        self.pyext_plot_combo = QComboBox()
        for label, value in self._plotting_positions:
            self.pyext_plot_combo.addItem(label, value)
        default_index = next(
            (idx for idx, (_, value) in enumerate(self._plotting_positions) if value == "weibull"),
            0,
        )
        self.pyext_plot_combo.setCurrentIndex(default_index)

        self._pyext_widgets: list[QWidget] = []
        self._builtin_widgets: list[QWidget] = []

        #

        main_layout = QVBoxLayout(self)

        self.inputs_widget = QWidget()
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)

        self.extremes_plot_area = QWidget()
        self.extremes_plot_layout = QVBoxLayout(self.extremes_plot_area)
        self.extremes_plot_layout.setContentsMargins(0, 0, 0, 0)
        self.extremes_plot_layout.setSpacing(0)
        self.extremes_fig = Figure(figsize=(5, 4))
        self.extremes_canvas = FigureCanvasQTAgg(self.extremes_fig)
        self.extremes_plot_layout.addWidget(self.extremes_canvas)

        self.plot_area = QWidget()
        self.plot_layout = QVBoxLayout(self.plot_area)
        self.plot_layout.setContentsMargins(0, 0, 0, 0)
        self.plot_layout.setSpacing(0)
        self.fig = Figure(figsize=(6, 4))
        self._base_figure = self.fig
        self.fig_canvas = FigureCanvasQTAgg(self.fig)
        self.plot_layout.addWidget(self.fig_canvas)
        self._evm_ran = False

        self._latest_warning: str | None = None
        self._show_canvas_messages = True
        self._last_evm_result: ExtremeValueResult | None = None
        self._extremes_placeholder = (
            "Run the analysis to view the extremes over threshold plot."
        )


        self.build_inputs()


        main_layout.addWidget(self.inputs_widget)

        text_plot_splitter = QSplitter(Qt.Horizontal)
        text_plot_splitter.addWidget(self.result_text)
        text_plot_splitter.addWidget(self.extremes_plot_area)
        text_plot_splitter.setStretchFactor(0, 2)
        text_plot_splitter.setStretchFactor(1, 3)
        text_plot_splitter.setChildrenCollapsible(False)

        splitter = QSplitter(Qt.Vertical)
        splitter.addWidget(text_plot_splitter)
        splitter.addWidget(self.plot_area)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        splitter.setChildrenCollapsible(False)
        splitter.setSizes([200, 400])

        main_layout.addWidget(splitter)

        # Show initial extremes plot with the suggested threshold
        self.update_extremes_plot(threshold)

    def _auto_threshold(self, start_thresh, tail):
        threshold = start_thresh
        attempts = 0

        peaks, _ = self._declustered_peaks(tail)
        comparator = np.greater if tail == "upper" else np.less

        while attempts < 10:
            exceedances = int(np.count_nonzero(comparator(peaks, threshold)))
            if exceedances >= 10:
                break
            threshold *= 0.95 if tail == "upper" else 1.05
            attempts += 1

        return threshold

    def _suggest_pyextremes_window(self) -> float:
        """Return a representative declustering window in seconds."""

        t = np.asarray(self.t, dtype=float)
        if t.size < 2:
            return 0.0

        diffs = np.diff(np.sort(t))
        positive = diffs[diffs > 0]
        if positive.size == 0:
            return 0.0

        return float(np.median(positive))

    def _set_canvas_figure(self, figure: Figure) -> None:
        """Ensure the matplotlib canvas is showing ``figure``."""

        current_canvas = self.fig_canvas
        if current_canvas.figure is figure:
            self.fig = figure
            return

        self.plot_layout.removeWidget(current_canvas)
        current_canvas.setParent(None)
        current_canvas.deleteLater()

        self.fig_canvas = FigureCanvasQTAgg(figure)
        self.plot_layout.addWidget(self.fig_canvas)
        self.fig = figure

    def show_extremes_message(self, message: str) -> None:
        """Display *message* on the extremes plot canvas."""

        self.extremes_fig.clear()
        self.extremes_fig.text(
            0.5,
            0.5,
            message,
            ha="center",
            va="center",
            wrap=True,
            fontsize=11,
        )
        self.extremes_canvas.draw()

    def plot_pyextremes_extremes(self, evm_result: ExtremeValueResult) -> None:
        """Render the PyExtremes extremes plot on the dedicated canvas."""

        metadata = evm_result.metadata
        eva = None
        getter = None

        if metadata is not None:
            maybe_getter = getattr(metadata, "get", None)
            if callable(maybe_getter):
                getter = maybe_getter
                eva = getter("eva", None)
            else:
                try:
                    eva = metadata["eva"]  # type: ignore[index]
                except Exception:
                    eva = None

        if eva is None:
            self.show_extremes_message("PyExtremes could not provide an extremes plot.")
            return

        self.extremes_fig.clear()
        ax = self.extremes_fig.add_subplot(111)

        try:
            show_clusters = False
            method = None
            if getter is not None:
                method = getter("method", None)
            elif metadata is not None:
                try:
                    method = metadata["method"]  # type: ignore[index]
                except Exception:
                    method = None
            if isinstance(method, str) and method.upper() == "POT":
                show_clusters = True

            eva.plot_extremes(ax=ax, show_clusters=show_clusters)
            self._apply_dark_mode_to_pyextremes_axes(ax)
        except Exception:  # pragma: no cover - plotting should not abort GUI flow
            self.show_extremes_message("Failed to draw PyExtremes extremes plot.")
            return

        self.extremes_canvas.draw()

    def update_extremes_plot(
        self,
        threshold: float,
        evm_result: ExtremeValueResult | None = None,
    ) -> None:
        """Update the extremes canvas based on the active engine."""

        engine = self.engine_combo.currentData()
        if engine == "pyextremes":
            if evm_result is not None and evm_result.engine == "pyextremes":
                self.plot_pyextremes_extremes(evm_result)
            else:
                self.show_extremes_message(self._extremes_placeholder)
            return

        self.plot_timeseries_with_threshold(threshold)

    def plot_timeseries_with_threshold(self, threshold):

        self.extremes_fig.clear()
        ax = self.extremes_fig.add_subplot(111)

        ax.plot(self.t, self.x, label="Time series")
        ax.axhline(threshold, color="red", linestyle="--", label="Threshold")
        ax.set_title("Time Series")
        ax.set_xlabel("Time")
        ax.set_ylabel(self.ts.name)
        ax.grid(True)
        ax.legend()
        self.extremes_canvas.draw()

    def _declustered_peaks(self, tail: str) -> tuple[np.ndarray, np.ndarray]:
        """Return cluster peaks and their boundaries for ``tail``."""

        window_seconds = self._current_declustering_window_seconds()

        peaks, boundaries = decluster_peaks(
            np.asarray(self.x, dtype=float),
            tail,
            t=np.asarray(self.t, dtype=float),
            window_seconds=window_seconds,
        )

        return peaks, boundaries

    def _current_declustering_window_seconds(self) -> float:
        """Return the active declustering window in seconds for the current engine."""

        engine = self.engine_combo.currentData()
        if engine == "pyextremes":
            value = self.pyext_r_spin.value()
        else:
            value = self.declustering_spin.value()

        return max(0.0, float(value))

    def _is_dark_theme_active(self) -> bool:
        """Return ``True`` when the application palette corresponds to dark mode."""

        window_color = self.palette().color(QPalette.Window)
        return window_color.name().lower() in {"#31363b", "#232629"}

    def _apply_dark_mode_to_pyextremes_axes(self, ax):
        """Restyle the PyExtremes plot so it matches the dark application theme."""

        if not self._is_dark_theme_active():
            return

        figure_bg = "#31363b"
        axes_bg = "#232629"
        text_color = "#eff0f1"
        grid_color = "#6c7074"

        self.extremes_fig.patch.set_facecolor(figure_bg)
        ax.set_facecolor(axes_bg)

        ax.tick_params(axis="both", colors=text_color, which="both")
        ax.xaxis.label.set_color(text_color)
        ax.yaxis.label.set_color(text_color)
        ax.title.set_color(text_color)
        ax.xaxis.offsetText.set_color(text_color)
        ax.yaxis.offsetText.set_color(text_color)

        for spine in ax.spines.values():
            spine.set_color(text_color)

        for grid_line in ax.get_xgridlines() + ax.get_ygridlines():
            grid_line.set_color(grid_color)

        for text in ax.texts:
            text.set_color(text_color)

        legend = ax.get_legend()
        if legend is not None:
            legend.get_frame().set_facecolor(axes_bg)
            legend.get_frame().set_edgecolor(text_color)
            for text in legend.get_texts():
                text.set_color(text_color)

    def _cluster_exceedances(self, threshold, tail):
        peaks, boundaries = self._declustered_peaks(tail)

        if peaks.size == 0:
            return np.empty(0, dtype=float), boundaries

        if tail == "upper":
            mask = peaks > threshold
        else:
            mask = peaks < threshold

        return peaks[mask], boundaries


    def on_manual_threshold(self):
        self.threshold_spin.interpretText()
        threshold = self.threshold_spin.value()

        # ensure the spin box keeps the manually entered value
        self.threshold_spin.setValue(threshold)

        self._manual_threshold = threshold
        self._last_evm_result = None

        self.update_extremes_plot(threshold)
        peaks, _ = self._cluster_exceedances(threshold, self.tail_combo.currentText())
        self.result_text.setPlainText(f"Exceedances used: {len(peaks)}")
        self._evm_ran = False

    def on_calc_threshold(self):
        tail = self.tail_combo.currentText()
        suggested = 0.8 * np.max(self.x) if tail == "upper" else 0.8 * np.min(self.x)
        threshold = self._auto_threshold(suggested, tail)
        self.threshold_spin.setValue(round(threshold, 4))
        self._last_evm_result = None
        self.update_extremes_plot(threshold)

    def build_inputs(self):
        layout = QGridLayout(self.inputs_widget)

        row = 0
        layout.addWidget(QLabel("Extreme value engine:"), row, 0)
        layout.addWidget(self.engine_combo, row, 1, 1, 2)
        row += 1

        layout.addWidget(self.distribution_label, row, 0, 1, 3)
        row += 1

        layout.addWidget(QLabel("Threshold:"), row, 0)
        layout.addWidget(self.threshold_spin, row, 1)
        self.calc_threshold_btn = QPushButton("Calc Threshold")
        self.calc_threshold_btn.clicked.connect(self.on_calc_threshold)
        layout.addWidget(self.calc_threshold_btn, row, 2)
        row += 1

        self.declustering_label = QLabel("Declustering period (s):")
        layout.addWidget(self.declustering_label, row, 0)
        layout.addWidget(self.declustering_spin, row, 1)
        row += 1

        self.ci_spin = QDoubleSpinBox()
        self.ci_spin.setDecimals(1)
        self.ci_spin.setValue(95.0)
        layout.addWidget(QLabel("Extremes to analyse:"), row, 0)
        layout.addWidget(self.tail_combo, row, 1)
        row += 1

        layout.addWidget(QLabel("Confidence level (%):"), row, 0)
        layout.addWidget(self.ci_spin, row, 1)
        self.ci_spin.valueChanged.connect(self.on_ci_changed)
        row += 1

        self.pyext_r_label = QLabel("PyExtremes declustering window:")
        layout.addWidget(self.pyext_r_label, row, 0)
        layout.addWidget(self.pyext_r_spin, row, 1)
        row += 1

        self.pyext_return_label = QLabel("PyExtremes return-period base:")
        layout.addWidget(self.pyext_return_label, row, 0)
        layout.addWidget(self.pyext_return_size_spin, row, 1)
        row += 1

        self.pyext_periods_label = QLabel("PyExtremes return periods (hours):")
        layout.addWidget(self.pyext_periods_label, row, 0)
        layout.addWidget(self.pyext_return_periods_edit, row, 1, 1, 2)
        row += 1

        self.pyext_samples_label = QLabel("PyExtremes bootstrap samples:")
        layout.addWidget(self.pyext_samples_label, row, 0)
        layout.addWidget(self.pyext_samples_spin, row, 1)
        row += 1

        self.pyext_plot_label = QLabel("PyExtremes plotting position:")
        layout.addWidget(self.pyext_plot_label, row, 0)
        layout.addWidget(self.pyext_plot_combo, row, 1, 1, 2)
        row += 1

        self._pyext_widgets.extend(
            [
                self.pyext_r_label,
                self.pyext_r_spin,
                self.pyext_return_label,
                self.pyext_return_size_spin,
                self.pyext_periods_label,
                self.pyext_return_periods_edit,
                self.pyext_samples_label,
                self.pyext_samples_spin,
                self.pyext_plot_label,
                self.pyext_plot_combo,
            ]
        )

        self._builtin_widgets.extend([self.declustering_label, self.declustering_spin])

        self.canvas_message_checkbox = QCheckBox("Show messages on canvas")
        self.canvas_message_checkbox.setChecked(True)
        self.canvas_message_checkbox.toggled.connect(self.on_canvas_message_toggle)

        run_btn = QPushButton("Run EVM")
        run_btn.clicked.connect(self.run_evm)
        layout.addWidget(run_btn, row, 0, 1, 2)

        iterate_btn = QPushButton("Iterate Fit")
        iterate_btn.clicked.connect(self.on_iterate_fit)
        layout.addWidget(iterate_btn, row, 2)
        row += 1

        layout.addWidget(self.canvas_message_checkbox, row, 0, 1, 3)

        self.on_engine_changed(self.engine_combo.currentIndex())

    def show_canvas_message(self, message: str):
        """Display *message* on the plot canvas."""

        if not self._show_canvas_messages:
            return

        self._set_canvas_figure(self._base_figure)
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

    def on_engine_changed(self, index: int) -> None:
        engine = self.engine_combo.itemData(index)
        is_pyextremes = engine == "pyextremes"

        self.distribution_label.setText(
            "Distribution: PyExtremes (Generalized Pareto)"
            if is_pyextremes
            else "Distribution: Generalized Pareto (built-in)"
        )

        for widget in self._pyext_widgets:
            widget.setVisible(is_pyextremes)

        for widget in self._builtin_widgets:
            widget.setVisible(not is_pyextremes)

        matching_result = (
            self._last_evm_result
            if self._last_evm_result is not None
            and self._last_evm_result.engine == engine
            else None
        )
        self.update_extremes_plot(self.threshold_spin.value(), matching_result)

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
            self._last_evm_result = None
            QMessageBox.warning(
                self,
                "Too Few Points",
                f"Threshold {threshold:.3f} resulted in only {data['count']} clustered exceedances.",
            )
            self.result_text.setPlainText(message)
            self.show_canvas_message(message)
            self.update_extremes_plot(threshold)
            self._evm_ran = False
        else:
            message = f"Extreme value analysis failed: {data['message']}"
            self._latest_warning = None
            self._last_evm_result = None
            self.result_text.setPlainText(message)
            self.show_canvas_message(message)
            self.update_extremes_plot(threshold)
            self._evm_ran = False

    def on_ci_changed(self, value):
        if self._evm_ran:
            self.run_evm()


    def _fit_once(self, threshold: float, tail: str, *, precomputed=None):
        engine = self.engine_combo.currentData()
        declustering_window = self._current_declustering_window_seconds()

        if precomputed is None:
            clustered_peaks, boundaries = self._cluster_exceedances(threshold, tail)
        else:
            peaks, boundaries = precomputed
            if peaks.size == 0:
                clustered_peaks = np.empty(0, dtype=float)
            elif tail == "upper":
                clustered_peaks = peaks[peaks > threshold]
            else:
                clustered_peaks = peaks[peaks < threshold]

        if engine != "pyextremes" and len(clustered_peaks) < 10:
            return "insufficient", {"count": len(clustered_peaks), "threshold": threshold}

        pyext_options = None
        if engine == "pyextremes":
            r_seconds = declustering_window
            return_base = self.pyext_return_size_spin.value()

            periods_text = self.pyext_return_periods_edit.text().strip()
            default_periods = list(SUMMARY_RETURN_PERIODS_HOURS)

            def _merge_periods(user_periods: list[float] | None) -> list[float]:
                merged: list[float] = []

                def _append_unique(period: float) -> None:
                    for existing in merged:
                        if abs(existing - period) <= 1e-9:
                            return
                    merged.append(period)

                for period in default_periods:
                    _append_unique(float(period))

                if user_periods:
                    for period in user_periods:
                        _append_unique(float(period))

                return merged

            if periods_text.lower() in {"", "none"}:
                return_periods = _merge_periods(None)
            else:
                try:
                    user_periods = [
                        float(token)
                        for token in periods_text.split(",")
                        if token.strip()
                    ]
                except ValueError:
                    return "error", {
                        "message": (
                            "Invalid PyExtremes return periods. Enter a comma-separated "
                            "list of positive numbers or 'None'."
                        ),
                        "threshold": threshold,
                    }
                if any(period <= 0 for period in user_periods):
                    return "error", {
                        "message": "PyExtremes return periods must be positive.",
                        "threshold": threshold,
                    }
                return_periods = _merge_periods(user_periods)

            pyext_options = {
                "method": "POT",
                "r": r_seconds,
                "return_period_size": f"{return_base}h",
                "n_samples": self.pyext_samples_spin.value(),
                "plotting_position": self.pyext_plot_combo.currentData(),
            }

        calc_kwargs = {}
        if engine == "pyextremes":
            calc_kwargs["return_periods_hours"] = return_periods

        try:
            evm_result = calculate_extreme_value_statistics(
                self.t,
                self.x,
                threshold,
                tail=tail,
                confidence_level=self.ci_spin.value(),
                clustered_peaks=clustered_peaks if engine != "pyextremes" else None,
                engine=engine,
                pyextremes_options=pyext_options,
                **calc_kwargs,
            )
        except Exception as exc:  # pragma: no cover - defensive GUI guard
            return "error", {"message": str(exc), "threshold": threshold}

        if engine == "pyextremes" and len(evm_result.exceedances) < 10:
            return (
                "insufficient",
                {"count": len(evm_result.exceedances), "threshold": threshold},
            )

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
                "declustering_window": declustering_window,
            },
        )

    def _handle_successful_fit(self, data: dict, tail: str) -> None:
        evm_result = data["evm_result"]
        threshold = data["threshold"]
        boundaries = data["boundaries"]
        warnings_text = data["warnings"]
        declustering_window = float(data.get("declustering_window", 0.0) or 0.0)

        c = evm_result.shape
        scale = evm_result.scale

        self._latest_warning = warnings_text

        units = ""
        max_val = np.max(self.x) if tail == "upper" else np.min(self.x)

        header = f"Extreme value statistics ({self.engine_combo.currentText()}): {self.ts.name}"
        if self._latest_warning:
            header = f"{header}\n\n{self._latest_warning}"
        result = f"{header}\n\n"

        if evm_result.engine != "pyextremes":
            if declustering_window > 0.0:
                result += f"Declustering period: {declustering_window:.3f} s\n\n"
            else:
                result += "Declustering: Mean level crossings\n\n"

        if evm_result.engine == "pyextremes" and evm_result.metadata:
            meta_lines: list[str] = []
            method = evm_result.metadata.get("method")
            if isinstance(method, str):
                meta_lines.append(f"PyExtremes method: {method}")
            rp_size = evm_result.metadata.get("return_period_size")
            if rp_size is not None:
                meta_lines.append(f"Return-period base: {rp_size}")
            decluster = evm_result.metadata.get("declustering_window")
            if decluster is not None:
                meta_lines.append(f"Declustering window: {decluster}")
            samples = evm_result.metadata.get("n_samples")
            if samples is not None:
                meta_lines.append(f"Bootstrap samples: {samples}")
            distribution = evm_result.metadata.get("distribution")
            if distribution is not None:
                meta_lines.append(f"Distribution: {distribution}")
            plotting_position = evm_result.metadata.get("plotting_position")
            if plotting_position is not None:
                friendly = self._plotting_position_labels.get(
                    str(plotting_position).lower(), plotting_position
                )
                meta_lines.append(f"Plotting position: {friendly}")
            if meta_lines:
                result += "\n".join(meta_lines) + "\n\n"

        preferred_hour = 3.0
        idx_candidates = np.where(
            np.isclose(evm_result.return_periods, preferred_hour, rtol=1e-6, atol=1e-6)
        )[0]
        if idx_candidates.size:
            idx = int(idx_candidates[0])
        elif evm_result.return_periods.size:
            idx = int(
                np.argmin(np.abs(evm_result.return_periods - preferred_hour))
            )
        else:
            idx = 0

        result += (
            f"The {evm_result.return_periods[idx]:.1f} hour return level is\n"
            f"{evm_result.return_levels[idx]:.5f} {units}\n\n"
        )
        result += (
            "Fitted tail parameters:\n"
            f"Sigma: {scale:.4f}\n"
            f"Xi: {c:.4f}\n"
            f"Exceedances used: {len(evm_result.exceedances)}\n"
        )

        result += f"Total crossings/clusters found: {max(len(boundaries) - 1, 0)}\n"

        result += f"Observed maximum value: {max_val:.4f} {units}\n"
        result += f"Return level unit: {units or 'same as input'}\n\n"
        def _format_hours(value: float) -> str:
            if abs(value - round(value)) < 1e-8:
                return f"{int(round(value))}"
            return f"{value:.1f}".rstrip("0").rstrip(".")

        result += f"{self.ci_spin.value():.0f}% Confidence Interval:\n"

        available_periods = evm_result.return_periods
        for target in SUMMARY_RETURN_PERIODS_HOURS:
            matches = np.where(
                np.isclose(available_periods, target, rtol=1e-6, atol=1e-6)
            )[0]
            if matches.size:
                idx = int(matches[0])
                lo = evm_result.lower_bounds[idx]
                up = evm_result.upper_bounds[idx]
                if np.isfinite(lo) and np.isfinite(up):
                    interval_text = f"{lo:.3f} – {up:.3f}"
                else:
                    interval_text = "n/a – n/a"
            else:
                interval_text = "n/a – n/a"
            result += f"{_format_hours(target)} hr: {interval_text}\n"

        self.result_text.setPlainText(result)

        self._last_evm_result = evm_result
        self.update_extremes_plot(threshold, evm_result)

        self.plot_diagnostics(
            evm_result.return_periods * 3600,
            evm_result.return_levels,
            evm_result.exceedances,
            c,
            scale,
            threshold,
            evm_result.lower_bounds,
            evm_result.upper_bounds,
            tail=tail,
            warnings=self._latest_warning,
            diagnostic_figure=getattr(evm_result, "diagnostic_figure", None),
        )
        self._evm_ran = True

    def _candidate_thresholds(
        self, base_threshold: float, tail: str, peaks: np.ndarray
    ) -> list[float]:
        """Return iteration thresholds derived from declustered peaks."""

        if peaks.size == 0:
            return [float(base_threshold)]

        comparator = np.greater if tail == "upper" else np.less
        base_count = int(np.count_nonzero(comparator(peaks, base_threshold)))

        total_available = int(min(peaks.size, self._MAX_CLUSTERED_EXCEEDANCES))
        min_required = 10

        if total_available < min_required:
            return [float(base_threshold)]

        counts: set[int] = {min_required, total_available, max(min_required, base_count)}

        if total_available > min_required:
            n_points = min(25, total_available - min_required + 1)
            geom_counts = np.geomspace(min_required, total_available, num=n_points)
            counts.update(int(round(val)) for val in geom_counts)

        for delta in (-15, -10, -5, -2, 0, 2, 5, 10, 15):
            counts.add(base_count + delta)

        valid_counts = sorted(
            {
                int(c)
                for c in counts
                if min_required <= int(c) <= total_available
            }
        )

        if tail == "upper":
            ordered_peaks = np.sort(peaks)[::-1]
        else:
            ordered_peaks = np.sort(peaks)

        thresholds: list[float] = []
        seen: set[float] = set()

        def _add_threshold(value: float) -> None:
            key = round(value, 8)
            if np.isfinite(value) and key not in seen:
                thresholds.append(float(value))
                seen.add(key)

        _add_threshold(float(base_threshold))

        for count in valid_counts:
            idx = count - 1
            if idx < 0 or idx >= ordered_peaks.size:
                continue

            peak_value = float(ordered_peaks[idx])
            if tail == "upper":
                threshold = np.nextafter(peak_value, -np.inf)
            else:
                threshold = np.nextafter(peak_value, np.inf)

            _add_threshold(threshold)

        thresholds.sort(reverse=(tail == "upper"))
        return thresholds

    def on_iterate_fit(self) -> None:
        tail = self.tail_combo.currentText()
        base_threshold = self.threshold_spin.value()

        peaks, boundaries = self._declustered_peaks(tail)
        if peaks.size < 10:
            message = (
                "Iteration requires at least 10 clustered peaks. Lower the threshold "
                "or adjust the tail selection and try again."
            )
            self._latest_warning = None
            self.result_text.setPlainText(message)
            self.show_canvas_message(message)
            self._evm_ran = False
            return

        self.result_text.setPlainText("Iterating to find a stable fit...")
        self._evm_ran = False
        self._last_evm_result = None
        self.update_extremes_plot(self.threshold_spin.value())

        best_success: dict | None = None
        best_with_warning: dict | None = None
        failure_messages: list[str] = []

        candidate_thresholds = self._candidate_thresholds(base_threshold, tail, peaks)

        for threshold in candidate_thresholds:
            status, data = self._fit_once(
                threshold,
                tail,
                precomputed=(peaks, boundaries),
            )

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
        exceedances,
        c,
        scale,
        threshold,
        lower_bounds=None,
        upper_bounds=None,
        *,
        tail: str,
        warnings: str | None = None,
        diagnostic_figure: Figure | None = None,
    ):
        if diagnostic_figure is not None:
            self._set_canvas_figure(diagnostic_figure)
            if warnings and self._show_canvas_messages:
                diagnostic_figure.suptitle(warnings, color="red", fontsize=10)
                try:
                    diagnostic_figure.tight_layout(rect=[0, 0, 1, 0.93])
                except ValueError:
                    pass
            self.fig_canvas.draw()
            return

        from scipy.stats import genpareto

        self._set_canvas_figure(self._base_figure)

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
        ts_ax.minorticks_on()
        ts_ax.grid(True, which="both", linestyle="--", alpha=0.5)
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
        ax.minorticks_on()
        ax.grid(True, which="both", linestyle="--", alpha=0.5)
        ax.legend()

        exceedances = np.asarray(exceedances, dtype=float)
        excursions = (
            exceedances - threshold if tail == "upper" else threshold - exceedances
        )

        if np.any(excursions <= 0):
            raise ValueError("All exceedances must lie beyond the threshold")

        sorted_excursions = np.sort(excursions)
        probs = (np.arange(1, len(sorted_excursions) + 1) - 0.5) / len(sorted_excursions)
        model_excursions = genpareto.ppf(probs, c, scale=scale)

        if tail == "lower":
            sorted_excursions = sorted_excursions[::-1]
            model_excursions = model_excursions[::-1]

        if tail == "upper":
            sorted_empirical = threshold + sorted_excursions
            model_quantiles = threshold + model_excursions
        else:
            sorted_empirical = threshold - sorted_excursions
            model_quantiles = threshold - model_excursions

        qax.scatter(model_quantiles, sorted_empirical, alpha=0.6, label="Data")

        diag_min = float(np.min([model_quantiles.min(), sorted_empirical.min()]))
        diag_max = float(np.max([model_quantiles.max(), sorted_empirical.max()]))
        qax.plot([diag_min, diag_max], [diag_min, diag_max], color="red", label="1:1 line")
        qax.set_title("Quantile Plot")
        qax.set_xlabel("Theoretical Quantiles")
        qax.set_ylabel("Empirical Quantiles")
        qax.minorticks_on()
        qax.grid(True, which="both", linestyle="--", alpha=0.5)
        if tail == "lower":
            qax.invert_xaxis()
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

        self._set_canvas_figure(self._base_figure)
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
            self._set_canvas_figure(self._base_figure)
            self.fig.clear()
            self.fig_canvas.draw()

__all__ = ['EVMWindow']

