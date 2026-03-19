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
