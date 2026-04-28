import os
import sys
import types
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

pytest.importorskip("PySide6.QtWidgets", exc_type=ImportError)
from PySide6.QtWidgets import QWidget


class _FakePage:
    def setBackgroundColor(self, *args, **kwargs):
        pass


class _FakeWebView(QWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._page = _FakePage()

    def setMinimumHeight(self, *args, **kwargs):
        pass

    def setSizePolicy(self, *args, **kwargs):
        pass

    def setStyleSheet(self, *args, **kwargs):
        pass

    def load(self, *args, **kwargs):
        pass

    def page(self):
        return self._page


# Stub optional modules used during import
stub_modules = {
    "nptdms": types.ModuleType("nptdms"),
    "PySide6.QtWebEngineWidgets": types.ModuleType("PySide6.QtWebEngineWidgets"),
}
stub_modules["nptdms"].TdmsFile = type("TdmsFile", (), {})
stub_modules["PySide6.QtWebEngineWidgets"].QWebEngineView = _FakeWebView
for name, module in stub_modules.items():
    sys.modules.setdefault(name, module)

import numpy as np
import pandas as pd
from PySide6.QtWidgets import QApplication, QMessageBox

import anytimes.gui.editor as editor_module
from anytimes.gui.editor import TimeSeriesEditorQt
from anyqats import TimeSeries


class DummyDB:
    def __init__(self, data):
        self._data = data

    def getm(self):
        return self._data

    def add(self, ts):
        self._data[ts.name] = ts


@pytest.fixture(scope="module")
def qt_app():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture
def message_spy(monkeypatch):
    calls = {"info": [], "warn": [], "crit": []}

    def _wrap(kind, retval):
        def _inner(parent, title, text, *args, **kwargs):
            calls[kind].append((title, text))
            return retval

        return _inner

    monkeypatch.setattr(QMessageBox, "information", _wrap("info", QMessageBox.Ok))
    monkeypatch.setattr(QMessageBox, "warning", _wrap("warn", QMessageBox.Ok))
    monkeypatch.setattr(QMessageBox, "critical", _wrap("crit", QMessageBox.Ok))
    return calls


def _build_editor(monkeypatch, tsdbs, paths):
    monkeypatch.setattr(TimeSeriesEditorQt, "apply_dark_palette", lambda self: None)
    monkeypatch.setattr(TimeSeriesEditorQt, "apply_light_palette", lambda self: None)
    editor = TimeSeriesEditorQt()
    editor.tsdbs = tsdbs
    editor.file_paths = paths
    editor.user_variables = set()
    editor.refresh_variable_tabs()
    return editor


def test_merge_common_single_series_creates_user_variables(qt_app, message_spy, monkeypatch):
    files = ["file1.ts", "file2.ts", "file3.ts"]
    tsdbs = []
    for idx in range(3):
        t = np.arange(5, dtype=float) + idx * 10
        x = np.arange(5, dtype=float) + idx * 100
        ts = TimeSeries("CommonVar", t, x)
        tsdbs.append(DummyDB({"CommonVar": ts}))

    editor = _build_editor(monkeypatch, tsdbs, files)
    editor.var_checkboxes["CommonVar"].setChecked(True)

    editor.merge_selected_series()
    qt_app.processEvents()

    expected = "merge(CommonVar)"
    created = set()
    for tsdb in editor.tsdbs:
        created.update(name for name in tsdb.getm() if name.startswith("merge(CommonVar)"))

    assert created == {expected}
    # Only the first database should receive the merged copy.
    assert expected in editor.tsdbs[0].getm()
    for tsdb in editor.tsdbs[1:]:
        assert expected not in tsdb.getm()
    assert not message_spy["crit"]
    assert not message_spy["warn"]


def test_merge_common_name_with_colon_not_misclassified(qt_app, message_spy, monkeypatch):
    files = ["A", "B", "C"]
    tsdbs = []
    for idx, name in enumerate(files):
        t = np.arange(5, dtype=float) + idx * 10
        x = np.arange(5, dtype=float) + idx * 100
        tsdbs.append(DummyDB({"A:Var": TimeSeries("A:Var", t, x)}))

    editor = _build_editor(monkeypatch, tsdbs, files)
    editor.var_checkboxes["A:Var"].setChecked(True)

    editor.merge_selected_series()
    qt_app.processEvents()

    expected = "merge(Var)"
    created = set()
    for tsdb in editor.tsdbs:
        created.update(name for name in tsdb.getm() if name.startswith("merge(Var)"))

    assert created == {expected}
    assert expected in editor.tsdbs[0].getm()
    for tsdb in editor.tsdbs[1:]:
        assert expected not in tsdb.getm()
    assert not message_spy["crit"]
    assert not message_spy["warn"]


def test_open_evm_user_variable_name_with_colon_uses_exact_match(qt_app, message_spy, monkeypatch):
    files = ["file1.ts", "file2.ts"]
    user_name = "sqr_sum_of_squares(file1.ts:VarA, file1.ts:VarB)"
    tsdbs = [
        DummyDB({user_name: TimeSeries(user_name, np.arange(5, dtype=float), np.arange(5, dtype=float))}),
        DummyDB({}),
    ]

    editor = _build_editor(monkeypatch, tsdbs, files)
    editor.user_variables = {user_name}
    editor.refresh_variable_tabs()
    editor.var_checkboxes[user_name].setChecked(True)

    launched = {}

    class DummyEVMWindow:
        def __init__(self, db, name, parent):
            launched["db"] = db
            launched["name"] = name
            launched["parent"] = parent

        def exec(self):
            launched["exec"] = True

    monkeypatch.setattr(editor_module, "EVMWindow", DummyEVMWindow)

    editor.open_evm_tool()
    qt_app.processEvents()

    assert launched["db"] is tsdbs[0]
    assert launched["name"] == user_name
    assert launched["parent"] is editor
    assert launched["exec"] is True
    assert not message_spy["crit"]
    assert not message_spy["warn"]




def test_calculate_series_success_popup_includes_equation(qt_app, message_spy, monkeypatch):
    files = ["file1.ts"]
    t = np.arange(5, dtype=float)
    x = np.arange(5, dtype=float) + 1.0
    tsdb = DummyDB({"VarA": TimeSeries("VarA", t, x)})

    editor = _build_editor(monkeypatch, [tsdb], files)
    editor.calc_entry.setPlainText("result = sin(radians(60)) + f1_VarA * 2")

    editor.calculate_series()
    qt_app.processEvents()

    assert "result_f1" in tsdb.getm()
    assert message_spy["info"]
    title, text = message_spy["info"][-1]
    assert title == "Success"
    assert "New variable(s): result_f1" in text
    assert "Equation used:" in text
    assert "result_f1 = sin(radians(60)) + f1_VarA * 2" in text
    assert "60" in text
    assert not message_spy["crit"]
    assert not message_spy["warn"]


def test_calculate_series_without_assignment_auto_creates_name(qt_app, message_spy, monkeypatch):
    files = ["file1.ts"]
    t = np.arange(5, dtype=float)
    x = np.arange(5, dtype=float) + 1.0
    tsdb = DummyDB({"VarA": TimeSeries("VarA", t, x)})

    editor = _build_editor(monkeypatch, [tsdb], files)
    editor.calc_entry.setPlainText("sin(radians(60)) + f1_VarA * 2")

    editor.calculate_series()
    qt_app.processEvents()

    auto_name = "calc_sin_rad_60_f1_VarA_x_2_f1"
    assert auto_name in tsdb.getm()
    assert message_spy["info"]
    title, text = message_spy["info"][-1]
    assert title == "Success"
    assert f"New variable(s): {auto_name}" in text
    assert f"{auto_name} = sin(radians(60)) + f1_VarA * 2" in text
    assert not message_spy["crit"]
    assert not message_spy["warn"]


def test_calculate_series_auto_names_distinguish_plus_and_minus(qt_app, message_spy, monkeypatch):
    files = ["file1.ts"]
    t = np.arange(5, dtype=float)
    x = np.arange(5, dtype=float) + 1.0
    tsdb = DummyDB({"VarA": TimeSeries("VarA", t, x)})

    editor = _build_editor(monkeypatch, [tsdb], files)

    editor.calc_entry.setPlainText("f1_VarA + 2")
    editor.calculate_series()
    qt_app.processEvents()

    editor.calc_entry.setPlainText("f1_VarA - 2")
    editor.calculate_series()
    qt_app.processEvents()

    assert "calc_f1_VarA_p_2_f1" in tsdb.getm()
    assert "calc_f1_VarA_m_2_f1" in tsdb.getm()
    assert not message_spy["crit"]
    assert not message_spy["warn"]


def test_calculate_series_auto_name_marks_common_variables(qt_app, message_spy, monkeypatch):
    files = ["file1.ts", "file2.ts"]
    tsdbs = []
    for idx in range(2):
        t = np.arange(5, dtype=float)
        x = np.arange(5, dtype=float) + idx
        tsdbs.append(DummyDB({"CommonVar": TimeSeries("CommonVar", t, x)}))

    editor = _build_editor(monkeypatch, tsdbs, files)
    editor.calc_entry.setPlainText("c_CommonVar + 2")

    editor.calculate_series()
    qt_app.processEvents()

    assert "calc_cc_CommonVar_p_2_f1" in tsdbs[0].getm()
    assert "calc_cc_CommonVar_p_2_f2" in tsdbs[1].getm()
    assert not message_spy["crit"]
    assert not message_spy["warn"]


def test_calculate_series_auto_name_shortens_long_equation_tokens(qt_app, message_spy, monkeypatch):
    files = ["file1.ts"]
    t = np.arange(5, dtype=float)
    x = np.arange(5, dtype=float) + 1.0
    tsdb = DummyDB({"MX_PIPE": TimeSeries("MX_PIPE", t, x), "common_f1": TimeSeries("common_f1", t, x)})

    editor = _build_editor(monkeypatch, [tsdb], files)
    editor.calc_entry.setPlainText("c_MX_PIPE * cos(radians(45)) + f1_common_f1")

    editor.calculate_series()
    qt_app.processEvents()

    created = next(name for name in tsdb.getm() if name.startswith("calc_"))
    assert "cc_MX_PIPE" in created
    assert "rad_45" in created
    assert "_x_" in created
    assert "_p_" in created
    assert "times" not in created
    assert "radians" not in created
    assert "common" not in created.removeprefix("calc_")
    assert not message_spy["crit"]
    assert not message_spy["warn"]


def test_calculate_series_uses_multiprocessing_and_updates_progress(qt_app, message_spy, monkeypatch):
    files = ["file1.ts", "file2.ts", "file3.ts"]
    tsdbs = []
    for idx in range(3):
        t = np.arange(5, dtype=float)
        x = np.arange(5, dtype=float) + idx
        tsdbs.append(DummyDB({"CommonVar": TimeSeries("CommonVar", t, x)}))

    submitted = []
    tqdm_calls = []

    class FakeFuture:
        def __init__(self, value=None, error=None):
            self._value = value
            self._error = error

        def result(self):
            if self._error is not None:
                raise self._error
            return self._value

    class FakeExecutor:
        def __init__(self, max_workers=None, initializer=None, initargs=()):
            self.max_workers = max_workers
            if initializer is not None:
                initializer(*initargs)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def submit(self, fn, *args, **kwargs):
            submitted.append(args[0])
            try:
                return FakeFuture(fn(*args, **kwargs))
            except Exception as exc:
                return FakeFuture(error=exc)

    monkeypatch.setattr(editor_module, "ProcessPoolExecutor", FakeExecutor)
    monkeypatch.setattr(editor_module, "as_completed", lambda futures: list(futures))

    class FakeTqdm:
        def __init__(self, iterable, total=None, desc=None, leave=None):
            tqdm_calls.append({"total": total, "desc": desc, "leave": leave})
            self._iterable = iterable

        def __enter__(self):
            return iter(self._iterable)

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(editor_module, "tqdm", FakeTqdm)

    editor = _build_editor(monkeypatch, tsdbs, files)
    editor.calc_entry.setPlainText("c_CommonVar + 2")

    editor.calculate_series()
    qt_app.processEvents()

    assert submitted == [0, 1, 2]
    assert tqdm_calls == [{"total": len(files), "desc": "Calculating", "leave": False}]
    assert editor.progress.maximum() == len(files)
    assert editor.progress.value() == len(files)
    assert "calc_cc_CommonVar_p_2_f1" in tsdbs[0].getm()
    assert "calc_cc_CommonVar_p_2_f2" in tsdbs[1].getm()
    assert "calc_cc_CommonVar_p_2_f3" in tsdbs[2].getm()
    assert not message_spy["crit"]
    assert not message_spy["warn"]


def test_calculate_series_preserves_per_file_lengths(qt_app, message_spy, monkeypatch):
    files = ["file1.ts", "file2.ts"]
    tsdbs = [
        DummyDB({"CommonVar": TimeSeries("CommonVar", np.arange(5, dtype=float), np.arange(5, dtype=float) + 1.0)}),
        DummyDB({"CommonVar": TimeSeries("CommonVar", np.arange(8, dtype=float), np.arange(8, dtype=float) + 10.0)}),
    ]

    editor = _build_editor(monkeypatch, tsdbs, files)
    editor.calc_entry.setPlainText("c_CommonVar + 2")

    editor.calculate_series()
    qt_app.processEvents()

    first = tsdbs[0].getm()["calc_cc_CommonVar_p_2_f1"]
    second = tsdbs[1].getm()["calc_cc_CommonVar_p_2_f2"]

    assert len(first.t) == 5
    assert len(first.x) == 5
    assert len(second.t) == 8
    assert len(second.x) == 8
    assert np.allclose(first.x, np.arange(5, dtype=float) + 3.0)
    assert np.allclose(second.x, np.arange(8, dtype=float) + 12.0)
    assert not message_spy["crit"]
    assert not message_spy["warn"]


def test_quick_transformation_uses_multiprocessing_and_updates_progress(qt_app, message_spy, monkeypatch):
    files = ["file1.ts", "file2.ts", "file3.ts"]
    tsdbs = []
    for idx in range(3):
        t = np.arange(5, dtype=float)
        x = np.arange(5, dtype=float) + idx + 1
        tsdbs.append(DummyDB({"CommonVar": TimeSeries("CommonVar", t, x)}))

    submitted = []
    tqdm_calls = []

    class FakeFuture:
        def __init__(self, value=None, error=None):
            self._value = value
            self._error = error

        def result(self):
            if self._error is not None:
                raise self._error
            return self._value

    class FakeExecutor:
        def __init__(self, max_workers=None, initializer=None, initargs=()):
            self.max_workers = max_workers
            if initializer is not None:
                initializer(*initargs)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def submit(self, fn, *args, **kwargs):
            submitted.append(args[0])
            try:
                return FakeFuture(fn(*args, **kwargs))
            except Exception as exc:
                return FakeFuture(error=exc)

    monkeypatch.setattr(editor_module, "ProcessPoolExecutor", FakeExecutor)
    monkeypatch.setattr(editor_module, "as_completed", lambda futures: list(futures))

    class FakeTqdm:
        def __init__(self, iterable, total=None, desc=None, leave=None):
            tqdm_calls.append({"total": total, "desc": desc, "leave": leave})
            self._iterable = iterable

        def __enter__(self):
            return iter(self._iterable)

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(editor_module, "tqdm", FakeTqdm)

    editor = _build_editor(monkeypatch, tsdbs, files)
    editor.var_checkboxes["CommonVar"].setChecked(True)

    editor.multiply_by_2()
    qt_app.processEvents()

    assert submitted == [0, 1, 2]
    assert tqdm_calls == [{"total": len(files), "desc": "Transforming", "leave": False}]
    assert editor.progress.maximum() == len(files)
    assert editor.progress.value() == len(files)
    assert np.allclose(tsdbs[0].getm()["CommonVar_×2_f1"].x, np.array([2, 4, 6, 8, 10], dtype=float))
    assert np.allclose(tsdbs[1].getm()["CommonVar_×2_f2"].x, np.array([4, 6, 8, 10, 12], dtype=float))
    assert np.allclose(tsdbs[2].getm()["CommonVar_×2_f3"].x, np.array([6, 8, 10, 12, 14], dtype=float))
    assert not message_spy["crit"]
    assert not message_spy["warn"]


def test_quick_transformation_preserves_per_file_lengths(qt_app, message_spy, monkeypatch):
    files = ["file1.ts", "file2.ts"]
    tsdbs = [
        DummyDB({"CommonVar": TimeSeries("CommonVar", np.arange(4, dtype=float), np.arange(4, dtype=float) + 1.0)}),
        DummyDB({"CommonVar": TimeSeries("CommonVar", np.arange(7, dtype=float), np.arange(7, dtype=float) + 10.0)}),
    ]

    editor = _build_editor(monkeypatch, tsdbs, files)
    editor.var_checkboxes["CommonVar"].setChecked(True)

    editor.multiply_by_2()
    qt_app.processEvents()

    first = tsdbs[0].getm()["CommonVar_×2_f1"]
    second = tsdbs[1].getm()["CommonVar_×2_f2"]

    assert len(first.t) == 4
    assert len(first.x) == 4
    assert len(second.t) == 7
    assert len(second.x) == 7
    assert np.allclose(first.x, np.array([2.0, 4.0, 6.0, 8.0]))
    assert np.allclose(second.x, np.array([20.0, 22.0, 24.0, 26.0, 28.0, 30.0, 32.0]))
    assert not message_spy["crit"]
    assert not message_spy["warn"]


def test_merge_preserves_irregular_time_steps(qt_app, message_spy, monkeypatch):
    files = ["file1.ts"]
    t1 = np.array([0.0, 1.0, 11.0, 21.0])
    x1 = np.arange(t1.size, dtype=float)
    ts1 = TimeSeries("VarA", t1, x1)
    ts1.dt = None

    t2 = np.array([0.0, 2.0, 5.0])
    x2 = np.arange(t2.size, dtype=float) + 100.0
    ts2 = TimeSeries("VarB", t2, x2)
    ts2.dt = None

    tsdb = DummyDB({"VarA": ts1, "VarB": ts2})

    editor = _build_editor(monkeypatch, [tsdb], files)
    editor.var_checkboxes["VarA"].setChecked(True)
    editor.var_checkboxes["VarB"].setChecked(True)

    editor.merge_selected_series()
    qt_app.processEvents()

    created = [name for name in tsdb.getm() if name.startswith("merge(")]
    assert len(created) == 1
    merged = tsdb.getm()[created[0]]

    assert merged.x.size == t1.size + t2.size
    assert np.allclose(merged.t[: t1.size], t1)

    second_segment = merged.t[t1.size :]
    assert np.allclose(second_segment - second_segment[0], t2 - t2[0])
    assert second_segment[0] > t1[-1]
    assert not message_spy["crit"]
    assert not message_spy["warn"]




def test_merge_preserves_datetime_reference(qt_app, message_spy, monkeypatch):
    files = ["file1.ts"]
    base = np.datetime64("2024-01-01T00:00:00")

    t1 = base + np.arange(3) * np.timedelta64(1, "h")
    x1 = np.array([1.0, 2.0, 3.0])
    t2 = base + np.arange(2) * np.timedelta64(30, "m")
    x2 = np.array([10.0, 11.0])

    tsdb = DummyDB({
        "VarA": TimeSeries("VarA", t1, x1),
        "VarB": TimeSeries("VarB", t2, x2),
    })

    editor = _build_editor(monkeypatch, [tsdb], files)
    editor.var_checkboxes["VarA"].setChecked(True)
    editor.var_checkboxes["VarB"].setChecked(True)

    editor.merge_selected_series()
    qt_app.processEvents()

    created = [name for name in tsdb.getm() if name.startswith("merge(")]
    assert len(created) == 1
    merged = tsdb.getm()[created[0]]

    assert merged.dtg_ref == tsdb.getm()["VarA"].dtg_ref
    assert merged.dtg_time[0] == tsdb.getm()["VarA"].dtg_time[0]
    assert merged.dtg_time[-1] > tsdb.getm()["VarA"].dtg_time[-1]
    assert not message_spy["crit"]
    assert not message_spy["warn"]


def test_time_window_filters_series_with_datetime_reference(qt_app, monkeypatch):
    dt_index = pd.date_range("2024-01-01 00:00:00", periods=6, freq="H")
    ts = TimeSeries("wind_speed", dt_index, np.arange(6, dtype=float))
    editor = _build_editor(monkeypatch, [DummyDB({"wind_speed": ts})], ["file1.ts"])

    editor.time_start.setText("2024-01-01 01:00:00")
    editor.time_end.setText("2024-01-01 03:00:00")

    mask = editor.get_time_window(ts)
    if isinstance(mask, slice):
        idx = np.arange(ts.t.size)[mask]
    else:
        idx = np.flatnonzero(mask)

    np.testing.assert_array_equal(idx, np.array([1, 2, 3]))


def test_export_selected_to_csv_uses_shared_time_column(qt_app, message_spy, monkeypatch, tmp_path):
    t = np.array([0.0, 1.0, 2.0])
    tsdb = DummyDB(
        {
            "VarA": TimeSeries("VarA", t, np.array([10.0, 11.0, 12.0])),
            "VarB": TimeSeries("VarB", t, np.array([20.0, 21.0, 22.0])),
        }
    )

    editor = _build_editor(monkeypatch, [tsdb], ["shared.ts"])
    editor.var_checkboxes["VarA"].setChecked(True)
    editor.var_checkboxes["VarB"].setChecked(True)

    export_path = tmp_path / "shared.csv"
    monkeypatch.setattr(
        "anytimes.gui.editor.QFileDialog.getSaveFileName",
        lambda *args, **kwargs: (str(export_path), "CSV files (*.csv)"),
    )

    editor.export_selected_to_csv()
    qt_app.processEvents()

    df = pd.read_csv(export_path)
    assert list(df.columns) == ["time", "VarA", "VarB"]
    assert np.allclose(df["time"].to_numpy(), t)
    assert np.allclose(df["VarA"].to_numpy(), np.array([10.0, 11.0, 12.0]))
    assert np.allclose(df["VarB"].to_numpy(), np.array([20.0, 21.0, 22.0]))
    assert not message_spy["warn"]


def test_export_selected_to_csv_keeps_per_series_time_for_different_timebases(
    qt_app, message_spy, monkeypatch, tmp_path
):
    tsdb = DummyDB(
        {
            "VarA": TimeSeries("VarA", np.array([0.0, 1.0, 2.0]), np.array([10.0, 11.0, 12.0])),
            "VarB": TimeSeries("VarB", np.array([0.0, 1.5, 3.0]), np.array([20.0, 21.0, 22.0])),
        }
    )

    editor = _build_editor(monkeypatch, [tsdb], ["mixed.ts"])
    editor.var_checkboxes["VarA"].setChecked(True)
    editor.var_checkboxes["VarB"].setChecked(True)

    export_path = tmp_path / "mixed.csv"
    monkeypatch.setattr(
        "anytimes.gui.editor.QFileDialog.getSaveFileName",
        lambda *args, **kwargs: (str(export_path), "CSV files (*.csv)"),
    )

    editor.export_selected_to_csv()
    qt_app.processEvents()

    df = pd.read_csv(export_path)
    assert list(df.columns) == ["VarA_t", "VarA", "VarB_t", "VarB"]
    assert np.allclose(df["VarA_t"].to_numpy(), np.array([0.0, 1.0, 2.0]))
    assert np.allclose(df["VarB_t"].to_numpy(), np.array([0.0, 1.5, 3.0]))
    assert not message_spy["warn"]


def test_bokeh_axis_type_detects_datetime_traces():
    traces = [{"t": pd.date_range("2024-01-01", periods=3, freq="h"), "y": [1.0, 2.0, 3.0]}]
    axis_type = TimeSeriesEditorQt._bokeh_x_axis_type_from_traces(traces)
    assert axis_type == "datetime"


def test_bokeh_axis_type_defaults_to_linear_for_numeric_traces():
    traces = [{"t": np.array([0.0, 1.0, 2.0]), "y": [1.0, 2.0, 3.0]}]
    axis_type = TimeSeriesEditorQt._bokeh_x_axis_type_from_traces(traces)
    assert axis_type == "linear"
