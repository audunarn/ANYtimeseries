import os
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6.QtWidgets", exc_type=ImportError)
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QApplication

from anytimes.gui.stats_dialog import StatsDialog


@pytest.fixture(scope="module")
def qt_app():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _series_info(
    *,
    t=None,
    x=None,
    dtg_time=None,
    editor_filter=None,
):
    if t is None:
        t = np.arange(len(x), dtype=float) if x is not None else np.arange(16, dtype=float)
    else:
        t = np.asarray(t, dtype=float)
    x = np.sin(t / 2.0) if x is None else np.asarray(x, dtype=float)
    editor_filter = editor_filter or {
        "mode": "none",
        "cutoff_low": "",
        "cutoff_high": "",
        "description": "Editor filter: none",
    }
    return [
        {
            "file": "case.csv",
            "uniq_file": "",
            "file_idx": 1,
            "var": "Signal",
            "t": t,
            "x": x,
            "dtg_time": dtg_time,
            "editor_filter": editor_filter,
            "time_window": {
                "start": t[0] if t.size else None,
                "end": t[-1] if t.size else None,
                "datetime_start": dtg_time[0] if dtg_time is not None and len(dtg_time) else None,
                "datetime_end": dtg_time[-1] if dtg_time is not None and len(dtg_time) else None,
                "samples": int(t.size),
            },
        }
    ]


def _header_column(dialog, name):
    headers = [
        dialog.table.horizontalHeaderItem(col).text()
        for col in range(dialog.table.columnCount())
    ]
    return headers.index(name)


def _cell(dialog, header):
    return dialog.table.item(0, _header_column(dialog, header)).text()


def test_datetime_context_and_core_column_preset(qt_app):
    dtg_time = np.asarray(
        [datetime(2026, 5, 21) + timedelta(seconds=i) for i in range(16)],
        dtype=object,
    )
    dialog = StatsDialog(_series_info(dtg_time=dtg_time))

    assert "2026-05-21" in _cell(dialog, "start")
    assert dialog.ts_dict["case.csv::Signal"].time_is_datetime is True
    assert dialog._prepare_plot_data()["time"][0]["x"][0] == dtg_time[0]

    advanced_cols = [
        col
        for col, header in enumerate(dialog._table_headers)
        if header.lower() not in dialog._CORE_STATS_HEADERS
    ]
    assert advanced_cols
    assert all(dialog.table.isColumnHidden(col) for col in advanced_cols)
    dialog.column_preset_combo.setCurrentText("All metrics")
    assert all(not dialog.table.isColumnHidden(col) for col in advanced_cols)
    dialog.close()


def test_table_and_plot_area_use_vertical_splitter(qt_app):
    dialog = StatsDialog(_series_info())

    assert dialog.results_splitter.orientation().name == "Vertical"
    assert dialog.results_splitter.widget(0) is dialog.table
    assert dialog.results_splitter.widget(1) is dialog.plot_stack
    dialog.close()


def test_qc_annotations_keep_invalid_filter_row_usable(qt_app):
    dialog = StatsDialog(_series_info(x=[1.0, np.nan, np.nan, 4.0, np.nan, 6.0] * 3))
    dialog.filter_lowpass_rb.setChecked(True)
    dialog.lowpass_cutoff.setText("bad-cutoff")
    dialog.update_data()

    assert _cell(dialog, "QC") == "Review"
    assert "invalid" in _cell(dialog, "QC messages").lower()
    assert "nan coverage" in _cell(dialog, "QC messages").lower()
    np.testing.assert_allclose(
        dialog.ts_dict["case.csv::Signal"].y,
        np.asarray([1.0, np.nan, np.nan, 4.0, np.nan, 6.0] * 3),
        equal_nan=True,
    )
    dialog.close()


@pytest.mark.parametrize(
    ("t", "x", "header", "expected"),
    [
        ([0.0, 1.0, 3.0, 6.0, 10.0], [1, 2, 3, 4, 5], "Time step QC", "Irregular"),
        ([0.0, 1.0], [1, 2], "PSD QC", "Unavailable"),
    ],
)
def test_qc_marks_irregular_and_psd_limited_rows(qt_app, t, x, header, expected):
    dialog = StatsDialog(_series_info(t=t, x=x))
    assert expected in _cell(dialog, header)
    dialog.close()


def test_metric_selection_and_tsv_copy_survive_visibility_changes(qt_app):
    dialog = StatsDialog(_series_info())
    assert dialog.metric_list.count()
    dialog.column_preset_combo.setCurrentText("Distribution metrics")
    dialog.table.selectRow(0)
    dialog.copy_all_as_tsv()
    copied = QGuiApplication.clipboard().text()

    assert "File\tUniqueness\tVariable" in copied
    assert "QC messages" in copied
    assert dialog.selected_columns
    assert dialog._prepare_plot_data()["metrics"]
    dialog.close()


def test_plotly_and_bokeh_render_prepared_statistics(qt_app):
    dialog = StatsDialog(_series_info())
    prepared = dialog._prepare_plot_data()
    pytest.importorskip("plotly", exc_type=ImportError)
    pytest.importorskip("bokeh", exc_type=ImportError)

    assert "plotly" in dialog._render_plotly_html(prepared).lower()
    assert "bokeh" in dialog._render_bokeh_html(prepared).lower()
    dialog.close()


def test_web_plot_data_downsamples_lines_but_bins_histogram(qt_app):
    t = np.arange(50000, dtype=float)
    dialog = StatsDialog(_series_info(t=t, x=np.sin(t / 15.0)))
    prepared = dialog._prepare_plot_data()
    web_prepared = dialog._web_prepared_plot_data(prepared)
    counts, edges = dialog._histogram_bins(prepared["hist"][0]["values"])

    assert len(web_prepared["time"][0]["x"]) == dialog._WEB_TIME_MAX_POINTS
    assert len(web_prepared["psd"][0]["x"]) <= dialog._WEB_PSD_MAX_POINTS
    assert counts.size == dialog._HISTOGRAM_BINS
    assert edges.size == dialog._HISTOGRAM_BINS + 1
    assert int(np.sum(counts)) == t.size
    dialog.close()


def test_web_plot_html_is_loaded_from_temp_file(qt_app):
    dialog = StatsDialog(_series_info())

    class FakeWebView:
        loaded = None

        def load(self, url):
            self.loaded = url

    fake_view = FakeWebView()
    dialog.web_plot_view = fake_view
    dialog._load_web_plot_html("<html><body>Stats</body></html>")

    assert dialog._temp_plot_file is not None
    assert Path(fake_view.loaded.toLocalFile()) == Path(dialog._temp_plot_file)
    dialog.close()


def test_plot_renderer_failure_falls_back_to_matplotlib(qt_app, monkeypatch):
    dialog = StatsDialog(_series_info(), preferred_plot_engine="plotly")
    dialog.web_plot_view = dialog.mpl_plot_widget

    def fail(_prepared):
        raise RuntimeError("renderer burst")

    monkeypatch.setattr(dialog, "_render_plotly_html", fail)
    dialog.update_plots()

    assert dialog.plot_stack.currentWidget() is dialog.mpl_plot_widget
    assert not dialog.render_warning_label.isHidden()
    assert "renderer burst" in dialog.render_warning_label.text()
    dialog.close()
