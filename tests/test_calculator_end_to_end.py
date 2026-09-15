"""Exercise equation input, Qt button wiring, storage, plotting and CSV export."""
import math
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PySide6.QtCore import QMimeData

from test_merge_selected import _build_editor, qt_app, message_spy
from anyqats import TimeSeries
from anytimes.gui.file_loader import FileLoader


EQUATIONS = """T5210_M1_Xrel = (f1_qtm_uv_M1_xpos - 14.86) - (f6_XPOS - 0.46*radians(f6_YAW) + 38.34*radians(f6_PITCH))
T5210_M1_Yrel = (f1_qtm_uv_M1_ypos - 0.46) - (f6_YPOS + 14.86*radians(f6_YAW) - 38.34*radians(f6_ROLL))
T5210_M1_Zrel = (f1_qtm_uv_M1_zpos - 38.34) - (f6_ZPOS - 14.86*radians(f6_PITCH) + 0.46*radians(f6_ROLL))"""
OUTPUTS = [f"T5210_M1_{axis}rel" for axis in "XYZ"]


def _signals(t):
    return {
        "qtm_uv_M1_xpos": 14.86 + .01*t,
        "qtm_uv_M1_ypos": .46 + .02*t,
        "qtm_uv_M1_zpos": 38.34 - .03*t,
        "XPOS": .1 + .004*t, "YPOS": -.2 + .003*t, "ZPOS": .05 - .002*t,
        "YAW": 1. + .2*t, "PITCH": -.5 + .04*t, "ROLL": .3 - .07*t,
    }


def _expected(t):
    # Independently expanded formula with degrees-to-radians conversion.
    rad = math.pi / 180.
    return {
        OUTPUTS[0]: -.1 + .006*t + rad*(.46*(1 + .2*t) - 38.34*(-.5 + .04*t)),
        OUTPUTS[1]: .2 + .017*t + rad*(-14.86*(1 + .2*t) + 38.34*(.3 - .07*t)),
        OUTPUTS[2]: -.05 - .028*t + rad*(14.86*(-.5 + .04*t) - .46*(.3 - .07*t)),
    }


def _load_six_files(tmp_path, t1, t6):
    paths = []
    for index in range(1, 7):
        t = t6 if index == 6 else t1
        signals = _signals(t)
        names = list(signals)[:3] if index == 1 else list(signals)[3:]
        data = {name: signals[name] for name in names} if index in (1, 6) else {"unused": t}
        path = tmp_path / f"file{index}.csv"
        pd.DataFrame({"time": t, **data}).to_csv(path, index=False)
        paths.append(str(path))
    dbs, errors = FileLoader().load_files(paths)
    assert not errors
    return dbs, paths


def _paste_and_calculate(editor, text, qt_app):
    mime = QMimeData()
    mime.setText(text)
    editor.calc_entry.clear()
    editor.calc_entry.insertFromMimeData(mime)
    assert editor.calc_entry.toPlainText() == text
    editor.calc_btn.click()
    qt_app.processEvents()


@pytest.mark.parametrize("dated", [False, True])
def test_equations_interpolate_all_samples_inside_source_overlap(qt_app, message_spy, monkeypatch, tmp_path, dated):
    t1, t6 = np.arange(-1., 6., .5), np.arange(.25, 5., 1.)
    dbs, paths = _load_six_files(tmp_path, t1, t6)
    reference = datetime(2026, 1, 1)
    if dated:
        for file_index, shift in ((0, 0), (5, 10)):
            for ts in dbs[file_index].getm().values():
                dbs[file_index].add(TimeSeries(ts.name, ts.t - shift, ts.x,
                                              dtg_ref=reference + timedelta(seconds=shift)), replace=True)
    editor = _build_editor(monkeypatch, dbs, paths)
    _paste_and_calculate(editor, EQUATIONS, qt_app)
    assert not message_spy["crit"]
    for file_index, t in ((0, t1), (5, t6)):
        overlap = (t >= max(t1[0], t6[0])) & (t <= min(t1[-1], t6[-1]))
        for name, expected in _expected(t).items():
            result = dbs[file_index].get(name=name)
            expected[~overlap] = np.nan
            np.testing.assert_allclose(result.x, expected, atol=2e-14, equal_nan=True)
            np.testing.assert_array_equal(result.t, t - (10 if dated and file_index == 5 else 0))
            if dated:
                np.testing.assert_array_equal(result.dtg_time,
                                              [reference + timedelta(seconds=value) for value in t])
    editor.close()


@pytest.mark.parametrize("cropped", [False, True])
def test_equations_end_to_end_paste_overwrite_plot_export_reload(qt_app, message_spy, monkeypatch, tmp_path, cropped):
    t = np.arange(0., 5., .25)
    dbs, paths = _load_six_files(tmp_path, t, t)
    editor = _build_editor(monkeypatch, dbs, paths)
    if cropped:
        editor.time_start.setText("1")
        editor.time_end.setText("3")
        t = t[(t >= 1) & (t <= 3)]
    for offset in (0., 1.):
        equations = "\n".join(line + f" + {offset}" for line in EQUATIONS.splitlines())
        _paste_and_calculate(editor, equations, qt_app)
        assert not message_spy["crit"]
        for index in (0, 5):
            for name, expected in _expected(t).items():
                np.testing.assert_allclose(dbs[index].get(name=name).x, expected + offset, atol=2e-14)
                assert name in editor.user_tab_widget.checkboxes
                assert f"f{index + 1}_{name}" in editor.calc_variables
            assert len(dbs[index].register_keys) == len(set(dbs[index].register_keys))
        assert all(set(db.getm()) == {"unused"} for db in dbs[1:5])

    for name in OUTPUTS:
        editor.user_tab_widget.checkboxes[name].setChecked(True)
    figures = []
    editor.plot_engine_combo.setCurrentText("default")
    editor.embed_plot_cb.setChecked(False)
    editor.plot_raw_cb.setChecked(True)
    editor.plot_lowpass_cb.setChecked(False)
    editor.plot_highpass_cb.setChecked(False)
    monkeypatch.setattr(editor, "_show_mpl_figure_window", figures.append)
    editor.plot_selected()
    assert len(figures) == 1
    figure = figures[0]
    figure.canvas.draw()
    lines = [line for line in figure.axes[0].get_lines()
             if any(name in line.get_label() for name in OUTPUTS)]
    assert len(lines) == 6
    for line in lines:
        name = next(name for name in OUTPUTS if name in line.get_label())
        np.testing.assert_array_equal(line.get_xdata(), t)
        np.testing.assert_allclose(line.get_ydata(), _expected(t)[name] + 1., atol=2e-14)

    editor.export_dt_input.setText("0")
    output_path = tmp_path / "calculated.csv"
    monkeypatch.setattr("anytimes.gui.editor.QFileDialog.getSaveFileName",
                        lambda *args, **kwargs: (str(output_path), "CSV files (*.csv)"))
    editor.export_selected_to_csv()
    exported = pd.read_csv(output_path)
    expected_columns = [f"file{i}.csv::{name}" for i in (1, 6) for name in OUTPUTS]
    assert list(exported.columns) == ["time", *expected_columns]
    np.testing.assert_allclose(exported["time"], t)
    reloaded, errors = FileLoader().load_files([str(output_path)])
    assert not errors
    for column in expected_columns:
        name = column.split("::", 1)[1]
        np.testing.assert_allclose(exported[column], _expected(t)[name] + 1., atol=2e-14)
        series = reloaded[0].get(name=column)
        np.testing.assert_array_equal(series.t, t)
        np.testing.assert_allclose(series.x, _expected(t)[name] + 1., atol=2e-14)
    assert not message_spy["warn"]
    assert not message_spy["crit"]
    from matplotlib import pyplot as plt
    plt.close(figure)
    editor.close()


def test_export_keeps_common_results_on_distinct_time_axes(qt_app, message_spy, monkeypatch, tmp_path):
    t1, t6 = np.arange(-1., 6., .5), np.arange(.25, 5., 1.)
    dbs, paths = _load_six_files(tmp_path, t1, t6)
    editor = _build_editor(monkeypatch, dbs, paths)
    _paste_and_calculate(editor, EQUATIONS, qt_app)
    for name in OUTPUTS:
        editor.user_tab_widget.checkboxes[name].setChecked(True)
    # Selecting an output in two tabs must not export it twice.
    editor.var_checkboxes[f"file1.csv::{OUTPUTS[0]}"].setChecked(True)
    output_path = tmp_path / "distinct_axes.csv"
    monkeypatch.setattr("anytimes.gui.editor.QFileDialog.getSaveFileName",
                        lambda *args, **kwargs: (str(output_path), "CSV files (*.csv)"))
    editor.export_selected_to_csv()
    exported = pd.read_csv(output_path)
    assert len(exported.columns) == 12
    assert len(set(exported.columns)) == 12
    for file_idx, t in ((0, t1), (5, t6)):
        for name in OUTPUTS:
            label = f"file{file_idx + 1}.csv::{name}"
            # Explicitly selected file1 X has the same label as shared outputs.
            np.testing.assert_allclose(exported[label + "_t"][:len(t)], t)
            np.testing.assert_allclose(exported[label][:len(t)], dbs[file_idx].get(name=name).x,
                                       atol=2e-14, equal_nan=True)
    assert "Exported 6 series" in message_spy["info"][-1][1]
    editor.close()


LOCAL_PATHS = [Path(__file__).resolve().parents[1] / name for name in
               ("test5210_UVMOCAP_M1.csv", "test5210_yaw_pitch_roll.csv")]


@pytest.mark.skipif(not all(path.exists() for path in LOCAL_PATHS), reason="User CSVs are local verification inputs")
def test_local_csv_equations_against_independent_interpolation_and_export(qt_app, message_spy, monkeypatch, tmp_path):
    """Verify available terms only: supplied CSVs lack XPOS/YPOS/ZPOS."""
    paths = [str(path) for path in LOCAL_PATHS]
    dbs, errors = FileLoader().load_files(paths)
    assert not errors
    frames = [pd.read_csv(path, float_precision="round_trip") for path in paths]
    editor = _build_editor(monkeypatch, dbs, paths)
    equations = """Xcheck = f1_test5210_UVMOCAP_mat__qtm_uv_M1_xpos - 14.86 + 0.46*radians(f2_YAW) - 38.34*radians(f2_PITCH)
Ycheck = f1_test5210_UVMOCAP_mat__qtm_uv_M1_ypos - 0.46 - 14.86*radians(f2_YAW) + 38.34*radians(f2_ROLL)
Zcheck = f1_test5210_UVMOCAP_mat__qtm_uv_M1_zpos - 38.34 + 14.86*radians(f2_PITCH) - 0.46*radians(f2_ROLL)"""
    _paste_and_calculate(editor, equations, qt_app)
    assert not message_spy["crit"]
    for file_idx, frame in enumerate(frames):
        target = frame["time"].to_numpy()

        def independent_column(source_idx, column):
            source = frames[source_idx]
            return np.interp(target, source["time"], source[column], left=np.nan, right=np.nan)

        m = [independent_column(0, f"test5210_UVMOCAP.mat::qtm_uv_M1_{axis}pos") for axis in "xyz"]
        yaw, pitch, roll = [independent_column(1, name)*math.pi/180. for name in ("YAW", "PITCH", "ROLL")]
        expected = {
            "Xcheck": m[0] - 14.86 + .46*yaw - 38.34*pitch,
            "Ycheck": m[1] - .46 - 14.86*yaw + 38.34*roll,
            "Zcheck": m[2] - 38.34 + 14.86*pitch - .46*roll,
        }
        for name, values in expected.items():
            ts = dbs[file_idx].get(name=name)
            np.testing.assert_allclose(ts.x, values, atol=1e-12, rtol=1e-12, equal_nan=True)
            np.testing.assert_array_equal(ts.t, next(iter(dbs[file_idx].getm().values())).t)
            # Independent CSV parsers can differ by one float rounding unit.
            np.testing.assert_allclose(ts.t, target, atol=2e-12, rtol=0)

        # Round-trip each file's output on its own time axis through real CSV IO.
        for checkbox in editor.var_checkboxes.values():
            checkbox.setChecked(False)
        for name in expected:
            editor.var_checkboxes[f"{LOCAL_PATHS[file_idx].name}::{name}"].setChecked(True)
        output_path = tmp_path / f"local_results_f{file_idx + 1}.csv"
        monkeypatch.setattr("anytimes.gui.editor.QFileDialog.getSaveFileName",
                            lambda *args, **kwargs: (str(output_path), "CSV files (*.csv)"))
        editor.export_selected_to_csv()
        reloaded, errors = FileLoader().load_files([str(output_path)])
        assert not errors
        for name, values in expected.items():
            ts = reloaded[0].get(name=f"{LOCAL_PATHS[file_idx].name}::{name}")
            np.testing.assert_allclose(ts.t, target, atol=2e-12, rtol=0)
            np.testing.assert_allclose(ts.x, values, atol=1e-12, rtol=1e-12, equal_nan=True)
    assert not message_spy["crit"]
    assert not message_spy["warn"]
    editor.close()
