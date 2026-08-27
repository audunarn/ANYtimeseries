import numpy as np
import pandas as pd
import pytest
from PySide6.QtWidgets import QApplication, QDoubleSpinBox, QTextEdit

import anytimes.gui.evm_window as evm_window_module
from anyqats import TimeSeries
from anytimes.evm import ExtremeValueResult
from anytimes.gui.evm_window import EVMWindow


class DummyDB:
    def __init__(self, series):
        self._series = {series.name: series}

    def getm(self):
        return self._series


@pytest.fixture(scope="module")
def qt_app():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _result(*, levels=None, lower=None, upper=None, exceedance_count=30):
    levels = np.asarray([3.0, 4.0] if levels is None else levels, dtype=float)
    lower = np.asarray([2.5, 3.5] if lower is None else lower, dtype=float)
    upper = np.asarray([3.5, 4.5] if upper is None else upper, dtype=float)
    return ExtremeValueResult(
        return_periods=np.asarray([1.0, 2.0], dtype=float),
        return_levels=levels,
        lower_bounds=lower,
        upper_bounds=upper,
        shape=-0.2,
        scale=1.4,
        exceedances=np.linspace(2.0, 5.0, num=exceedance_count),
        threshold=1.0,
        exceedance_rate=1.0,
    )


def _window():
    time = np.arange(60, dtype=float) * 3600.0
    data = np.sin(np.linspace(0.0, 8.0 * np.pi, num=time.size))
    return EVMWindow(DummyDB(TimeSeries("signal", time, data)), "signal")


def test_threshold_stability_points_keep_selected_threshold(qt_app, monkeypatch):
    window = _window()
    selected_threshold = window.threshold_spin.value()
    calls = []

    monkeypatch.setattr(
        window,
        "_candidate_thresholds",
        lambda threshold, tail, peaks: [0.4, 0.8],
    )

    def fake_fit(threshold, tail, *, precomputed=None, estimate_confidence=True):
        calls.append((threshold, tail, precomputed, estimate_confidence))
        if threshold == 0.8:
            return "insufficient", {"count": 8, "threshold": threshold}
        return "ok", {"evm_result": _result(), "threshold": threshold}

    monkeypatch.setattr(window, "_fit_once", fake_fit)

    points = window._threshold_stability_points(
        "upper",
        precomputed=(np.asarray([1.0] * 12), np.asarray([0, 12])),
    )

    assert window.threshold_spin.value() == selected_threshold
    assert [point["status"] for point in points] == ["ok", "insufficient"]
    assert points[0]["return_levels"].shape == (2,)
    assert points[1]["exceedance_count"] == 8
    assert all(call[3] is False for call in calls)
    window.close()


def test_auto_threshold_targets_quality_control_count(qt_app, monkeypatch):
    window = _window()
    peaks = np.linspace(1.0, 100.0, num=100)
    monkeypatch.setattr(
        window,
        "_declustered_peaks",
        lambda tail: (peaks, np.asarray([0, peaks.size])),
    )

    threshold = window._auto_threshold(95.0, "upper")
    exceedances = int(np.count_nonzero(peaks > threshold))
    summary = window._threshold_quality_summary(threshold, "upper", peaks)

    assert exceedances == window._TARGET_CLUSTERED_EXCEEDANCES
    assert "Good starting point" in summary
    assert "Target count: 50" in summary
    window.close()


def test_iterate_fit_searches_declustering_windows(qt_app, monkeypatch):
    window = _window()
    peaks = np.linspace(1.0, 80.0, num=80)
    attempts = []
    chosen = {}

    monkeypatch.setattr(window, "_candidate_declustering_windows", lambda: [0.0, 12.0])
    monkeypatch.setattr(
        window,
        "_declustered_peaks_for_window",
        lambda tail, seconds: (peaks, np.asarray([0, peaks.size])),
    )
    monkeypatch.setattr(
        window,
        "_candidate_thresholds",
        lambda threshold, tail, values: [threshold],
    )

    def fake_fit(
        threshold,
        tail,
        *,
        precomputed=None,
        estimate_confidence=True,
        declustering_window_seconds=None,
    ):
        attempts.append((estimate_confidence, declustering_window_seconds))
        if not estimate_confidence:
            count = 18 if declustering_window_seconds == 0.0 else 50
            return "ok", {
                "evm_result": _result(exceedance_count=count),
                "threshold": threshold,
                "boundaries": precomputed[1],
                "warnings": None,
                "declustering_window": declustering_window_seconds,
            }
        return "ok", {
            "evm_result": _result(exceedance_count=50),
            "threshold": threshold,
            "boundaries": np.asarray([0, peaks.size]),
            "warnings": None,
            "declustering_window": window._current_declustering_window_seconds(),
        }

    monkeypatch.setattr(window, "_fit_once", fake_fit)
    monkeypatch.setattr(
        window,
        "_handle_successful_fit",
        lambda data, tail: chosen.update(
            data=data,
            tail=tail,
            window=window._current_declustering_window_seconds(),
        ),
    )

    window.on_iterate_fit()

    candidate_windows = {
        seconds for estimate, seconds in attempts if not estimate
    }
    assert candidate_windows == {0.0, 12.0}
    assert chosen["window"] == 12.0
    assert chosen["tail"] == "upper"
    window.close()


def test_pyextremes_fit_uses_datetime_index_and_current_settings(
    qt_app,
    monkeypatch,
):
    index = pd.date_range("2025-01-01", periods=40, freq="h")
    data = np.linspace(0.0, 4.0, num=index.size)
    window = EVMWindow(DummyDB(TimeSeries("datetime", index, data)), "datetime")
    window.engine_combo.setCurrentIndex(window.engine_combo.findData("pyextremes"))
    window.pyext_r_spin.setValue(2.0)
    window.pyext_return_periods_edit.setText("2")

    captured = {}

    def fake_calculate(*args, **kwargs):
        captured.update(kwargs)
        return _result()

    monkeypatch.setattr(evm_window_module, "calculate_extreme_value_statistics", fake_calculate)

    status, _data = window._fit_once(
        window.threshold_spin.value(),
        "upper",
        precomputed=(np.linspace(1.0, 5.0, num=12), np.asarray([0, 12])),
    )

    assert status == "ok"
    assert captured["engine"] == "pyextremes"
    assert captured["pyextremes_options"]["r"] == 2.0
    assert captured["return_periods_hours"][-1] == 2.0
    datetime_index = np.asarray(captured["pyextremes_options"]["datetime_index"])
    assert datetime_index.shape == index.shape
    assert datetime_index[0] == window.ts.dtg_time[0]
    assert window._has_datetime_time is True
    window.close()


def test_result_html_reports_return_level_between_bounds(qt_app):
    window = EVMWindow.__new__(EVMWindow)
    window.result_text = QTextEdit()
    window.ci_spin = QDoubleSpinBox()
    window.ci_spin.setValue(95.0)
    result = _result(
        levels=np.asarray([-3.0, np.nan]),
        lower=np.asarray([-3.5, np.nan]),
        upper=np.asarray([-2.5, np.nan]),
    )

    interval_html = window._confidence_interval_table_html(
        result,
        period_kind="return_period",
        units="kN",
    )
    checks_html = window._fit_checks_html(result)
    window.result_text.setHtml(interval_html + checks_html)
    rendered = window.result_text.toPlainText()

    assert "Lower" in rendered
    assert "Return level (kN)" in rendered
    assert rendered.index("-3.500") < rendered.index("-3.000") < rendered.index("-2.500")
    assert "n/a" in rendered
    assert "Fit checks" in rendered
