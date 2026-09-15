"""Opt-in source/frozen application smoke test: --smoke-test report.json."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile
import traceback
from unittest.mock import patch


def run_smoke_test(report_path: Path) -> int:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    report = {"frozen": bool(getattr(sys, "frozen", False)), "python": sys.version,
              "status": "failed"}
    try:
        import numpy as np
        import pandas as pd
        from PySide6.QtCore import QSettings
        from PySide6.QtWidgets import QApplication, QMessageBox
        from anytimes import __version__
        from anytimes.gui.editor import TimeSeriesEditorQt
        from anytimes.gui.file_loader import FileLoader
        from plotly.graph_objects import Figure, Scatter
        from bokeh.plotting import figure
        from bokeh.embed import file_html
        from bokeh.resources import INLINE

        report["version"] = __version__
        app = QApplication.instance() or QApplication([])
        with tempfile.TemporaryDirectory(prefix="anytimes-smoke-") as temporary:
            folder = Path(temporary)
            t = np.arange(32, dtype=float) / 10
            inputs = []
            for index in (1, 2):
                path = folder / f"input{index}.csv"
                pd.DataFrame({"time": t, "A": t + index}).to_csv(path, index=False)
                inputs.append(str(path))
            databases, errors = FileLoader().load_files(inputs)
            assert not errors, errors
            settings = QSettings(str(folder / "settings.ini"), QSettings.IniFormat)
            with patch.object(TimeSeriesEditorQt, "_layout_settings", lambda self: settings), \
                 patch.object(QMessageBox, "information"), \
                 patch.object(QMessageBox, "critical") as critical, \
                 patch.object(QMessageBox, "warning") as warning:
                window = TimeSeriesEditorQt()
                window.tsdbs, window.file_paths = databases, inputs
                window.user_variables = set()
                window.refresh_variable_tabs()
                window.show()
                app.processEvents()
                for factor in (2, 3):
                    window.calc_entry.setPlainText(
                        f"sum_result = (f1_A + f2_A) * {factor}\n"
                        "difference = f2_A - f1_A"
                    )
                    window.calc_btn.click()
                    app.processEvents()
                    assert not critical.called, critical.call_args
                    for db in databases:
                        np.testing.assert_allclose(db.get(name="sum_result").x, (2*t + 3)*factor)
                        np.testing.assert_allclose(db.get(name="difference").x, np.ones_like(t))
                        assert len(db.register_keys) == 3
                for name in ("sum_result", "difference"):
                    window.user_tab_widget.checkboxes[name].setChecked(True)
                window.plot_engine_combo.setCurrentText("default")
                window.embed_plot_cb.setChecked(False)
                window.plot_raw_cb.setChecked(True)
                with patch.object(window, "_show_mpl_figure_window") as plotted:
                    window.plot_selected()
                    assert plotted.call_count == 1
                    chart = plotted.call_args.args[0]
                    chart.canvas.draw()
                target = folder / "export.csv"
                with patch("anytimes.gui.editor.QFileDialog.getSaveFileName", return_value=(str(target), "CSV")):
                    window.export_selected_to_csv()
                exported = pd.read_csv(target)
                assert len(exported.columns) == 5
                np.testing.assert_allclose(exported["input1.csv::sum_result"], (2*t + 3)*3)
                np.testing.assert_allclose(exported["input2.csv::difference"], np.ones_like(t))
                assert not warning.called, warning.call_args
                assert not critical.called, critical.call_args
                window.close()
                app.processEvents()
            # Verify dynamic plotting imports and their bundled JavaScript data.
            assert "plotly" in Figure(Scatter(x=t, y=t)).to_html(include_plotlyjs=True).lower()
            bokeh_chart = figure()
            bokeh_chart.line(t, t)
            assert "bokeh" in file_html(bokeh_chart, INLINE, "Smoke").lower()
        report["checks"] = ["GUI launch", "CSV loading", "batch equations", "overwrite",
                            "Matplotlib rendering", "CSV export", "Plotly HTML", "Bokeh HTML"]
        report["status"] = "passed"
    except Exception:
        report["error"] = traceback.format_exc()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0 if report["status"] == "passed" else 1
