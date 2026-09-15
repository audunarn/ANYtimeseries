"""Bounded calculator benchmark; excludes GUI construction, tab refresh and dialogs.

Run from the repository root: python tools/benchmark_calculator.py --profile
"""
from __future__ import annotations

import argparse
import ast
import cProfile
from datetime import datetime
import json
import os
from pathlib import Path
import pstats
import statistics
import subprocess
import sys
from time import perf_counter
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from anyqats import TimeSeries, TsDB
from anytimes.gui.editor import TimeSeriesEditorQt
import anytimes.gui.editor as editor_module
from anytimes.gui.file_loader import FileLoader


class CalculatorHarness:
    calculate_series = TimeSeriesEditorQt.calculate_series
    _calculate_series_equation = TimeSeriesEditorQt._calculate_series_equation
    get_time_window = TimeSeriesEditorQt.get_time_window
    _format_calculator_equation = TimeSeriesEditorQt._format_calculator_equation

    def __init__(self, files, channels, samples, dated, expression):
        self.tsdbs = []
        t = np.arange(samples, dtype=float) * 0.1
        for file_idx in range(files):
            db = TsDB()
            for channel_idx in range(channels):
                db.add(TimeSeries(
                    f"v{channel_idx}", t, np.sin(t) + file_idx + channel_idx,
                    dtg_ref=datetime(2026, 1, 1) if dated else None,
                ))
            self.tsdbs.append(db)
        self.file_paths = [f"file{i}.ts" for i in range(files)]
        self.calc_entry = SimpleNamespace(toPlainText=lambda: expression)
        self.time_start = self.time_end = SimpleNamespace(text=lambda: "")
        unchecked = SimpleNamespace(isChecked=lambda: False)
        for name in ("lowpass", "highpass", "bandpass", "bandblock"):
            setattr(self, f"filter_{name}_rb", unchecked)
        self.progress = SimpleNamespace(setFormat=lambda *_: None, reset=lambda: None)
        self.common_lookup = {}
        self.user_variables = set()
        self.filter_calls = 0

    def apply_filters(self, ts):
        self.filter_calls += 1
        return TimeSeriesEditorQt.apply_filters(self, ts)

    def update_progressbar(self, *_):
        pass

    def refresh_variable_tabs(self):
        pass

    def _filter_tag(self):
        return ""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--files", type=int, default=3)
    parser.add_argument("--channels", type=int, default=12)
    parser.add_argument("--samples", type=int, default=20000)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--numeric", action="store_true")
    parser.add_argument("--expression", default="result = f1_v0 * 2")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--csv", nargs="+", help="Local CSV inputs in calculator file order")
    parser.add_argument("--compare-head", action="store_true", help="Compare with committed calculator")
    args = parser.parse_args()
    if args.csv:
        benchmark_csv(args)
        return
    timings, counts = [], []
    profiler = cProfile.Profile()
    for repeat in range(args.repeats):
        harness = CalculatorHarness(args.files, args.channels, args.samples,
                                    not args.numeric, args.expression)
        with patch("anytimes.gui.editor.QMessageBox.information"), patch(
            "anytimes.gui.editor.QMessageBox.critical"
        ) as errors:
            if args.profile and repeat == 0:
                profiler.enable()
            start = perf_counter()
            harness.calculate_series()
            timings.append(perf_counter() - start)
            profiler.disable()
            if errors.called:
                raise RuntimeError(str(errors.call_args))
        counts.append(harness.filter_calls)
        # The default formula has an independent numerical oracle.
        if args.expression == "result = f1_v0 * 2":
            db = harness.tsdbs[0]
            np.testing.assert_array_equal(db.get(name="result_f1").x, db.get(name="v0").x * 2)
    print(json.dumps({"configuration": vars(args), "seconds": timings,
                      "median_seconds": statistics.median(timings),
                      "filter_calls": counts}, indent=2))
    if args.profile:
        pstats.Stats(profiler).strip_dirs().sort_stats("cumulative").print_stats(20)


def benchmark_csv(args):
    """Load with the production loader; compare every output sample and timestamp."""
    dbs, errors = FileLoader().load_files(args.csv)
    if errors:
        raise RuntimeError(errors)
    source = [db.getm() for db in dbs]
    print(json.dumps({"inputs": [
        {"path": path, "channels": list(series),
         "samples": len(next(iter(series.values())).t),
         "time_range": [float(next(iter(series.values())).t[0]),
                        float(next(iter(series.values())).t[-1])]}
        for path, series in zip(args.csv, source)
    ], "expression": args.expression}, indent=2))
    implementations = {"current": CalculatorHarness}
    if args.compare_head:
        root = Path(__file__).resolve().parents[1]
        original = subprocess.check_output(
            ["git", "-c", f"safe.directory={root.as_posix()}", "show", "HEAD:anytimes/gui/editor.py"],
            cwd=root, text=True, encoding="utf-8",
        )
        tree = ast.parse(original)
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                   and node.name == "TimeSeriesEditorQt")
        methods = [node for node in cls.body if isinstance(node, ast.FunctionDef)
                   and node.name in {"calculate_series", "apply_filters"}]
        namespace = dict(vars(editor_module))
        exec(compile(ast.Module(body=methods, type_ignores=[]), "committed_editor.py", "exec"), namespace)

        class OriginalHarness(CalculatorHarness):
            calculate_series = namespace["calculate_series"]

            def apply_filters(self, ts):
                self.filter_calls += 1
                return namespace["apply_filters"](self, ts)

        implementations = {"committed": OriginalHarness, **implementations}
    reference = None
    for label, harness_type in implementations.items():
        timings, counts = [], []
        for _ in range(args.repeats):
            harness = harness_type(0, 0, 0, False, args.expression)
            harness.file_paths = args.csv
            harness.tsdbs = []
            for series in source:
                db = TsDB()
                for ts in series.values():
                    db.add(TimeSeries(ts.name, ts.t.copy(), ts.x.copy(), dtg_ref=ts.dtg_ref))
                harness.tsdbs.append(db)
            with patch("anytimes.gui.editor.QMessageBox.information"), patch(
                "anytimes.gui.editor.QMessageBox.critical"
            ) as errors:
                start = perf_counter()
                harness.calculate_series()
                timings.append(perf_counter() - start)
                if errors.called:
                    raise RuntimeError(str(errors.call_args))
            counts.append(harness.filter_calls)
            outputs = {(i, key): (ts.t.copy(), ts.x.copy())
                       for i, db in enumerate(harness.tsdbs)
                       for key, ts in db.getm().items() if key not in source[i]}
            if not outputs:
                raise AssertionError("Calculator produced no outputs")
            if reference is None:
                reference = outputs
            assert outputs.keys() == reference.keys()
            for key, (t, x) in outputs.items():
                np.testing.assert_array_equal(t, reference[key][0])
                np.testing.assert_array_equal(x, reference[key][1])
        print(json.dumps({"implementation": label, "seconds": timings,
                          "median_seconds": statistics.median(timings),
                          "filter_calls": counts,
                          "outputs": {str(key): {"samples": len(x),
                                                  "finite_samples": int(np.isfinite(x).sum())}
                                      for key, (_, x) in outputs.items()},
                          "all_outputs_equal": True}, indent=2))


if __name__ == "__main__":
    main()
