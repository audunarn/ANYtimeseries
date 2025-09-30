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
from PySide6.QtWidgets import QApplication, QMessageBox

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
