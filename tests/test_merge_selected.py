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
from anyqats import TimeSeries, TsDB


class DummyDB:
    def __init__(self, data):
        self._data = data

    def getm(self):
        return self._data

    def add(self, ts, replace=False):
        self._data[ts.name] = ts


class FakeLayoutSettings:
    def __init__(self, values=None):
        self.values = dict(values or {})

    def value(self, key):
        return self.values.get(key)

    def setValue(self, key, value):
        self.values[key] = value

    def sync(self):
        pass


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


def _build_layout_editor(monkeypatch, qt_app, settings):
    monkeypatch.setattr(TimeSeriesEditorQt, "apply_dark_palette", lambda self: None)
    monkeypatch.setattr(TimeSeriesEditorQt, "apply_light_palette", lambda self: None)
    monkeypatch.setattr(TimeSeriesEditorQt, "_layout_settings", lambda self: settings)
    editor = TimeSeriesEditorQt()
    editor.resize(1900, 1040)
    editor.show()
    qt_app.processEvents()
    return editor


def test_editor_main_splitter_allows_left_pane_growth(qt_app, monkeypatch):
    editor = _build_layout_editor(monkeypatch, qt_app, FakeLayoutSettings())

    editor.main_splitter.setSizes([1200, 600])
    qt_app.processEvents()
    left, right = editor.main_splitter.sizes()

    assert editor.main_splitter.widget(1).minimumSizeHint().width() < 900
    assert left > editor._min_left_panel * 3
    assert left > right
    editor.close()


def test_editor_invalid_layout_state_falls_back_to_defaults(qt_app, monkeypatch):
    settings = FakeLayoutSettings(
        {
            "main_editor/main_splitter": b"invalid",
            "main_editor/right_splitter": b"invalid",
            "main_editor/top_row_splitter": b"invalid",
        }
    )
    editor = _build_layout_editor(monkeypatch, qt_app, settings)
    left, right = editor.main_splitter.sizes()

    assert editor._layout_state_restored is False
    assert left > 0
    assert right > 0
    editor.close()


def test_editor_splitter_state_restores_saved_sizes(qt_app, monkeypatch):
    settings = FakeLayoutSettings()
    first = _build_layout_editor(monkeypatch, qt_app, settings)
    first.main_splitter.setSizes([480, 1320])
    qt_app.processEvents()
    first._save_layout_state()
    first.close()

    second = _build_layout_editor(monkeypatch, qt_app, settings)
    left, right = second.main_splitter.sizes()

    assert second._layout_state_restored is True
    assert left < right
    second.close()


def test_editor_resize_keeps_manual_embedded_plot_height(qt_app, monkeypatch):
    editor = _build_layout_editor(monkeypatch, qt_app, FakeLayoutSettings())
    editor.right_splitter.setSizes([320, 680])
    qt_app.processEvents()

    editor.resize(1880, 980)
    qt_app.processEvents()
    controls_height, plot_height = editor.right_splitter.sizes()

    assert plot_height > controls_height
    editor.close()


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


def test_open_evm_filtered_datetime_series_preserves_reference(
    qt_app,
    message_spy,
    monkeypatch,
):
    index = pd.date_range("2025-03-01", periods=6, freq="h")
    ts = TimeSeries("Wind", index, np.arange(6, dtype=float))
    tsdb = DummyDB({"Wind": ts})
    editor = _build_editor(monkeypatch, [tsdb], ["wind.csv"])
    editor.var_checkboxes["Wind"].setChecked(True)

    launched = {}

    class DummyEVMWindow:
        def __init__(self, db, name, parent):
            launched["db"] = db
            launched["name"] = name

        def exec(self):
            launched["exec"] = True

    mask = np.asarray([False, True, True, True, False, False])
    monkeypatch.setattr(editor_module, "EVMWindow", DummyEVMWindow)
    monkeypatch.setattr(editor, "get_time_window", lambda _ts: mask)
    monkeypatch.setattr(editor, "apply_filters", lambda series: series.x)

    editor.open_evm_tool()
    qt_app.processEvents()

    filtered = launched["db"].getm()["Wind"]
    assert launched["name"] == "Wind"
    assert launched["exec"] is True
    assert filtered.dtg_ref == ts.dtg_ref
    assert filtered.dtg_time[0] == ts.dtg_time[1]
    assert filtered.dtg_time[-1] == ts.dtg_time[3]
    assert not message_spy["crit"]
    assert not message_spy["warn"]


def test_show_stats_preserves_datetime_filter_and_window_payload(
    qt_app,
    message_spy,
    monkeypatch,
):
    index = pd.date_range("2026-05-21", periods=6, freq="h")
    ts = TimeSeries("Wind", index, np.arange(6, dtype=float))
    tsdb = DummyDB({"Wind": ts})
    editor = _build_editor(monkeypatch, [tsdb], ["wind.csv"])
    editor.var_checkboxes["Wind"].setChecked(True)
    editor.filter_lowpass_rb.setChecked(True)
    editor.lowpass_cutoff.setText("0.2")

    launched = {}

    class DummyStatsDialog:
        def __init__(self, series_info, parent, preferred_plot_engine):
            launched["series_info"] = series_info
            launched["parent"] = parent
            launched["engine"] = preferred_plot_engine

        def exec(self):
            launched["exec"] = True

    mask = np.asarray([False, True, True, True, False, False])
    monkeypatch.setattr(editor_module, "StatsDialog", DummyStatsDialog)
    monkeypatch.setattr(editor, "get_time_window", lambda _ts: mask)

    editor.show_stats()
    qt_app.processEvents()

    payload = launched["series_info"][0]
    assert launched["parent"] is editor
    assert launched["exec"] is True
    assert payload["dtg_time"][0] == ts.dtg_time[1]
    assert payload["time_window"]["datetime_end"] == ts.dtg_time[3]
    assert payload["editor_filter"]["mode"] == "lowpass"
    assert "0.2" in payload["editor_filter"]["description"]
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

    # Preserve the plus token just like the adjacent plus/minus collision test;
    # otherwise distinct expressions can be assigned the same automatic name.
    auto_name = "calc_sin_rad_60_p_f1_VarA_x_2_f1"
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


def test_calculate_series_avoids_process_startup_and_updates_progress(qt_app, message_spy, monkeypatch):
    files = ["file1.ts", "file2.ts", "file3.ts"]
    t = np.arange(5, dtype=float)
    tsdbs = [DummyDB({"CommonVar": TimeSeries("CommonVar", t, t + i)}) for i in range(3)]

    def unexpected_pool(*args, **kwargs):
        pytest.fail("Vector calculator must not spawn a fresh process pool")

    monkeypatch.setattr(editor_module, "ProcessPoolExecutor", unexpected_pool)
    editor = _build_editor(monkeypatch, tsdbs, files)
    editor.calc_entry.setPlainText("c_CommonVar + 2")
    editor.calculate_series()
    qt_app.processEvents()

    assert editor.progress.maximum() == len(files)
    assert editor.progress.value() == len(files)
    for i, db in enumerate(tsdbs):
        np.testing.assert_array_equal(db.getm()[f"calc_cc_CommonVar_p_2_f{i + 1}"].x, t + i + 2)
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


def test_calculator_only_prepares_referenced_channels_and_output_files(qt_app, message_spy, monkeypatch):
    t = np.arange(6, dtype=float)
    first = DummyDB({"Unused": TimeSeries("Unused", t, t)})
    second = DummyDB({"A": TimeSeries("A", t, t), "Unused": TimeSeries("Unused", t, t)})
    editor = _build_editor(monkeypatch, [first, second], ["first.ts", "second.ts"])
    monkeypatch.setattr(editor, "refresh_variable_tabs", lambda: None)
    monkeypatch.setattr(first, "getm", lambda: pytest.fail("Unreferenced file was loaded"))
    filtered = []
    monkeypatch.setattr(editor, "apply_filters", lambda ts: filtered.append(ts.name) or ts.x * 3)
    editor.calc_entry.setPlainText("answer = f2_A + f2_A")

    editor.calculate_series()

    assert filtered == ["A"]
    np.testing.assert_array_equal(second.getm()["answer_f2"].x, t * 6)
    assert editor.progress.maximum() == editor.progress.value() == 1
    assert not message_spy["crit"]


@pytest.mark.parametrize("dated", [False, True])
def test_calculator_cross_file_formula_alignment(qt_app, message_spy, monkeypatch, dated):
    from datetime import datetime, timedelta

    ref = datetime(2026, 1, 1) if dated else None
    t1 = np.arange(6, dtype=float)
    t2 = np.array([0., 2., 4.]) if dated else np.array([1., 3., 5.])
    ref2 = ref + timedelta(seconds=1) if dated else None
    first = DummyDB({"X": TimeSeries("X", t1, 10 + t1, dtg_ref=ref)})
    second = DummyDB({
        "XPOS": TimeSeries("XPOS", t2, np.array([2., 4., 6.]), dtg_ref=ref2),
        "YAW": TimeSeries("YAW", t2, np.full(3, 30.), dtg_ref=ref2),
        "PITCH": TimeSeries("PITCH", t2, np.full(3, 5.), dtg_ref=ref2),
    })
    editor = _build_editor(monkeypatch, [first, second], ["first.ts", "second.ts"])
    editor.calc_entry.setPlainText(
        "rel = (f1_X - 14.86) - (f2_XPOS - 0.46*radians(f2_YAW) + 38.34*radians(f2_PITCH))"
    )
    editor.calculate_series()

    expected = 9. - 14.86 + 0.46*np.deg2rad(30.) - 38.34*np.deg2rad(5.)
    a, b = first.getm()["rel"], second.getm()["rel"]
    np.testing.assert_allclose(a.x, [np.nan] + [expected] * 5, equal_nan=True)
    np.testing.assert_allclose(b.x, np.full(3, expected))
    np.testing.assert_array_equal(a.t, t1)
    np.testing.assert_array_equal(b.t, t2)
    assert a.dtg_ref == ref
    assert b.dtg_ref == ref2
    assert not message_spy["crit"]


def test_calculator_common_and_user_subtraction_uses_raw_series(qt_app, message_spy, monkeypatch):
    t = np.arange(5, dtype=float)
    db = DummyDB({"A": TimeSeries("A", t, t + 5), "B": TimeSeries("B", t, t + 1)})
    editor = _build_editor(monkeypatch, [db], ["first.ts"])
    editor.user_variables = {"B"}
    monkeypatch.setattr(editor, "apply_filters", lambda ts: pytest.fail("Raw common/user input was filtered"))
    editor.calc_entry.setPlainText("answer = c_A - u_B")
    editor.calculate_series()
    np.testing.assert_array_equal(db.getm()["answer_f1"].x, np.full(5, 4.))
    assert not message_spy["crit"]


def test_calculator_repeated_runs_use_current_filters(qt_app, message_spy, monkeypatch):
    t = np.arange(5, dtype=float)
    db = DummyDB({"A": TimeSeries("A", t, t)})
    editor = _build_editor(monkeypatch, [db], ["first.ts"])
    editor.calc_entry.setPlainText("first = f1_A * 2")
    editor.calculate_series()
    monkeypatch.setattr(editor, "apply_filters", lambda ts: ts.x + 10)
    editor.calc_entry.setPlainText("second = f1_A * 2")
    editor.calculate_series()
    np.testing.assert_array_equal(db.getm()["first_f1"].x, t * 2)
    np.testing.assert_array_equal(db.getm()["second_f1"].x, (t + 10) * 2)
    np.testing.assert_array_equal(db.getm()["A"].x, t)
    assert not message_spy["crit"]


def test_calculator_reports_empty_time_window(qt_app, message_spy, monkeypatch):
    t = np.arange(5, dtype=float)
    db = DummyDB({"A": TimeSeries("A", t, t)})
    editor = _build_editor(monkeypatch, [db], ["first.ts"])
    editor.time_start.setText("100")
    editor.time_end.setText("200")
    editor.calc_entry.setPlainText("answer = f1_A * 2")
    editor.calculate_series()
    assert message_spy["crit"][0][0] == "No Time Window"
    assert "answer_f1" not in db.getm()


def test_calculator_scalar_output_and_time_window(qt_app, message_spy, monkeypatch):
    t = np.arange(6, dtype=float)
    db = DummyDB({"A": TimeSeries("A", t, t)})
    editor = _build_editor(monkeypatch, [db], ["first.ts"])
    editor.time_start.setText("1")
    editor.time_end.setText("3")
    editor.calc_entry.setPlainText("answer = 42")
    editor.calculate_series()
    np.testing.assert_array_equal(db.getm()["answer_f1"].t, [1., 2., 3.])
    np.testing.assert_array_equal(db.getm()["answer_f1"].x, [42., 42., 42.])
    assert not message_spy["crit"]


def test_no_filter_returns_independent_unchanged_data(qt_app, monkeypatch):
    values = np.array([1., np.nan, 3., np.inf])
    ts = TimeSeries("A", np.arange(4, dtype=float), values)
    editor = _build_editor(monkeypatch, [DummyDB({"A": ts})], ["first.ts"])
    monkeypatch.setattr(editor_module.np, "median", lambda *_: pytest.fail("No-filter path scanned time steps"))
    result = editor.apply_filters(ts)
    np.testing.assert_array_equal(result, values)
    assert not np.shares_memory(result, ts.x)


def test_calculator_pasted_xyz_equations_create_all_outputs(qt_app, message_spy, monkeypatch):
    t = np.arange(8, dtype=float)
    mocap = {axis: t + offset for axis, offset in zip("xyz", [15., 2., 40.])}
    motion = {name: t * scale for name, scale in zip(
        ["XPOS", "YPOS", "ZPOS", "YAW", "PITCH", "ROLL"], [.1, .2, .3, .4, .5, .6]
    )}
    first = DummyDB({f"qtm_uv_M1_{axis}pos": TimeSeries(f"qtm_uv_M1_{axis}pos", t, x)
                     for axis, x in mocap.items()})
    sixth = DummyDB({name: TimeSeries(name, t, x) for name, x in motion.items()})
    dbs = [first] + [DummyDB({"unused": TimeSeries("unused", t, t)}) for _ in range(4)] + [sixth]
    editor = _build_editor(monkeypatch, dbs, [f"file{i}.ts" for i in range(1, 7)])
    editor.calc_entry.setPlainText(
        "T5210_M1_Xrel = (f1_qtm_uv_M1_xpos - 14.86) - (f6_XPOS - 0.46*radians(f6_YAW) + 38.34*radians(f6_PITCH))\n"
        "T5210_M1_Yrel = (f1_qtm_uv_M1_ypos - 0.46) - (f6_YPOS + 14.86*radians(f6_YAW) - 38.34*radians(f6_ROLL))\n"
        "T5210_M1_Zrel = (f1_qtm_uv_M1_zpos - 38.34) - (f6_ZPOS - 14.86*radians(f6_PITCH) + 0.46*radians(f6_ROLL))"
    )
    refreshed, filtered = [], []
    monkeypatch.setattr(editor, "refresh_variable_tabs", lambda: refreshed.append(True))
    monkeypatch.setattr(editor, "apply_filters", lambda ts: filtered.append(ts.name) or ts.x.copy())
    editor.calculate_series()

    yaw, pitch, roll = [np.deg2rad(motion[key]) for key in ("YAW", "PITCH", "ROLL")]
    expected = {
        "X": mocap["x"] - 14.86 - (motion["XPOS"] - .46*yaw + 38.34*pitch),
        "Y": mocap["y"] - .46 - (motion["YPOS"] + 14.86*yaw - 38.34*roll),
        "Z": mocap["z"] - 38.34 - (motion["ZPOS"] - 14.86*pitch + .46*roll),
    }
    for axis, values in expected.items():
        name = f"T5210_M1_{axis}rel"
        for db in (first, sixth):
            np.testing.assert_allclose(db.getm()[name].x, values)
            np.testing.assert_array_equal(db.getm()[name].t, t)
        assert name in editor.user_variables
        assert name in message_spy["info"][0][1]
    assert all(set(db.getm()) == {"unused"} for db in dbs[1:5])
    assert len(filtered) == len(set(filtered)) == 9
    assert refreshed == [True]
    assert len(message_spy["info"]) == 1
    assert not message_spy["crit"]


def test_calculator_multiline_parentheses_comments_and_blank_lines(qt_app, message_spy, monkeypatch):
    t = np.arange(5, dtype=float)
    db = DummyDB({"A": TimeSeries("A", t, t)})
    editor = _build_editor(monkeypatch, [db], ["first.ts"])
    editor.calc_entry.setPlainText(
        "# Ignore this example reference: f99_Missing\n"
        "first = (\n    f1_A +\n    2\n)\n\n"
        "second = np.where(f1_A == 2, 10, 20) # equality is not an assignment\n"
    )
    editor.calculate_series()
    np.testing.assert_array_equal(db.getm()["first_f1"].x, t + 2)
    np.testing.assert_array_equal(db.getm()["second_f1"].x, [20., 20., 10., 20., 20.])
    assert not message_spy["crit"]


def test_calculator_bare_comparison_is_not_an_assignment(qt_app, message_spy, monkeypatch):
    t = np.arange(5, dtype=float)
    db = DummyDB({"A": TimeSeries("A", t, t)})
    editor = _build_editor(monkeypatch, [db], ["first.ts"])
    editor.calc_entry.setPlainText("f1_A == 2")
    editor.calculate_series()
    output = next(ts for key, ts in db.getm().items() if key != "A")
    np.testing.assert_array_equal(output.x, [0., 0., 1., 0., 0.])
    assert not message_spy["crit"]


def test_calculator_batch_keeps_each_equations_file_scope(qt_app, message_spy, monkeypatch):
    t1, t2 = np.arange(5, dtype=float), np.arange(8, dtype=float)
    first = DummyDB({"A": TimeSeries("A", t1, t1)})
    second = DummyDB({"B": TimeSeries("B", t2, t2)})
    editor = _build_editor(monkeypatch, [first, second], ["first.ts", "second.ts"])
    editor.calc_entry.setPlainText("one = f1_A * 2\ntwo = f2_B + 3")
    editor.calculate_series()
    assert set(first.getm()) == {"A", "one_f1"}
    assert set(second.getm()) == {"B", "two_f2"}
    np.testing.assert_array_equal(first.getm()["one_f1"].x, t1 * 2)
    np.testing.assert_array_equal(second.getm()["two_f2"].x, t2 + 3)
    assert not message_spy["crit"]


@pytest.mark.parametrize("reference", ["u_first_f1", "f1_first_f1"])
def test_calculator_batch_can_reference_staged_output(qt_app, message_spy, monkeypatch, reference):
    t = np.arange(5, dtype=float)
    db = DummyDB({"A": TimeSeries("A", t, t)})
    editor = _build_editor(monkeypatch, [db], ["first.ts"])
    editor.calc_entry.setPlainText(f"first = f1_A + 2\nsecond = {reference} * 3")
    editor.calculate_series()
    np.testing.assert_array_equal(db.getm()["second_f1"].x, (t + 2) * 3)
    assert not message_spy["crit"]


@pytest.mark.parametrize("last_equation", [
    "second = (f1_A +",  # syntax error
    "second = f1_Missing * 2",  # unknown channel
    "second = np.array([1., 2.])",  # result length mismatch
    "second = missing_name",  # evaluation error
])
def test_calculator_batch_failure_does_not_publish_partial_outputs(qt_app, message_spy, monkeypatch, last_equation):
    t = np.arange(5, dtype=float)
    db = DummyDB({"A": TimeSeries("A", t, t), "existing_f1": TimeSeries("existing_f1", t, t + 10)})
    editor = _build_editor(monkeypatch, [db], ["first.ts"])
    editor.calc_entry.setPlainText("first = f1_A * 2\n" + last_equation)
    editor.calculate_series()
    assert set(db.getm()) == {"A", "existing_f1"}
    np.testing.assert_array_equal(db.getm()["existing_f1"].x, t + 10)
    assert editor.user_variables == set()
    assert not message_spy["info"]
    assert len(message_spy["crit"]) == 1


def test_calculator_rerun_overwrites_result_and_time_window(qt_app, message_spy, monkeypatch):
    t = np.arange(5, dtype=float)
    db = TsDB()
    db.add(TimeSeries("A", t, t))
    editor = _build_editor(monkeypatch, [db], ["first.ts"])
    editor.calc_entry.setPlainText("answer = f1_A * 2")
    editor.calculate_series()
    original = db.get(name="answer_f1")
    keys = list(db.register_keys)

    editor.time_start.setText("1")
    editor.time_end.setText("3")
    editor.calc_entry.setPlainText("answer = f1_A * 3")
    editor.calculate_series()

    result = db.get(name="answer_f1")
    assert result is not original
    np.testing.assert_array_equal(result.t, [1., 2., 3.])
    np.testing.assert_array_equal(result.x, [3., 6., 9.])
    np.testing.assert_array_equal(original.x, t * 2)
    np.testing.assert_array_equal(db.get(name="A").x, t)
    assert db.register_keys == keys
    assert editor.user_variables == {"answer_f1"}
    assert len(message_spy["info"]) == 2
    assert not message_spy["crit"]


def test_calculator_duplicate_batch_name_uses_latest_value(qt_app, message_spy, monkeypatch):
    t = np.arange(5, dtype=float)
    db = TsDB()
    db.add(TimeSeries("A", t, t))
    editor = _build_editor(monkeypatch, [db], ["first.ts"])
    editor.calc_entry.setPlainText(
        "answer = f1_A + 1\n"
        "before = f1_answer_f1 * 2\n"
        "answer = f1_A + 5\n"
        "after = f1_answer_f1 * 2\n"
        "user_after = u_answer_f1 * 3"
    )
    editor.calculate_series()
    np.testing.assert_array_equal(db.get(name="answer_f1").x, t + 5)
    np.testing.assert_array_equal(db.get(name="before_f1").x, (t + 1) * 2)
    np.testing.assert_array_equal(db.get(name="after_f1").x, (t + 5) * 2)
    np.testing.assert_array_equal(db.get(name="user_after_f1").x, (t + 5) * 3)
    assert len(db.register_keys) == len(set(db.register_keys)) == 5
    assert not message_spy["crit"]


def test_calculator_failed_batch_preserves_existing_output(qt_app, message_spy, monkeypatch):
    t = np.arange(5, dtype=float)
    db = TsDB()
    db.add(TimeSeries("A", t, t))
    original = TimeSeries("answer_f1", t, t + 10)
    db.add(original)
    editor = _build_editor(monkeypatch, [db], ["first.ts"])
    editor.user_variables = {"answer_f1"}
    editor.calc_entry.setPlainText("answer = f1_A * 2\nnew = f1_A + 1\nbroken = unknown")
    editor.calculate_series()
    assert db.get(name="answer_f1") is original
    np.testing.assert_array_equal(original.x, t + 10)
    assert set(db.getm()) == {"A", "answer_f1"}
    assert editor.user_variables == {"answer_f1"}
    assert not message_spy["info"]
    assert len(message_spy["crit"]) == 1


def test_calculator_rerun_overwrites_common_outputs_in_each_file(qt_app, message_spy, monkeypatch):
    t = np.arange(5, dtype=float)
    dbs = [TsDB(), TsDB()]
    for i, db in enumerate(dbs):
        db.add(TimeSeries("A", t, t + i))
    editor = _build_editor(monkeypatch, dbs, ["first.ts", "second.ts"])
    editor.calc_entry.setPlainText("answer = f1_A + f2_A")
    editor.calculate_series()
    editor.calc_entry.setPlainText("answer = f1_A - f2_A")
    editor.calculate_series()
    for db in dbs:
        assert set(db.getm()) == {"A", "answer"}
        np.testing.assert_array_equal(db.get(name="answer").x, -np.ones(5))
        assert len(db.register_keys) == 2
    assert not message_spy["crit"]


def test_tsdb_add_replace_is_explicit_and_preserves_registration_order():
    t = np.arange(3, dtype=float)
    db = TsDB()
    first, second = TimeSeries("A", t, t), TimeSeries("B", t, t)
    db.add(first)
    db.add(second)
    replacement = TimeSeries("A", t, t + 10)
    with pytest.raises(KeyError):
        db.add(replacement)
    assert db.get(name="A") is first
    db.add(replacement, replace=True)
    assert db.get(name="A") is replacement
    assert db.get(ind=1) is second
    assert db.register_keys == list(db.register) == ["A", "B"]
    assert set(db.register_parent) == set(db.register_indices) == {"A", "B"}


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

    t2 = np.array([0.0, 2.0, 5.0])
    x2 = np.arange(t2.size, dtype=float) + 100.0
    ts2 = TimeSeries("VarB", t2, x2)

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
