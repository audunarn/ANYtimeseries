# Calculator end-to-end verification — 2026-09-15

## Scope and outcome

**Full suite: 193 tests passed in 94.75 seconds**, with 12 existing
calendar/deprecation warnings. This includes all six new end-to-end cases,
including the local user CSV test (not skipped). `git diff --check` passed.

The verification exercises the production CSV loader, the Qt calculator input's
paste handler, the Calculate button signal, equation parsing and execution,
TimeSeries storage, replacement of existing results, refreshed variable tabs
and autocomplete, raw Matplotlib plot generation, CSV export and CSV reload.

It uses real Qt widgets offscreen and real database objects. Message boxes and
the save-file chooser are intercepted to avoid interaction. The optional web
view is stubbed; Matplotlib renders the actual figure. This verifies the local
source application, not an installed executable or the Plotly/Bokeh renderers.

## Two issues reproduced and fixed

1. **Interpolation lost valid edge samples.** The calculator cropped source
   samples to the target window before interpolation, removing the samples
   needed to interpolate the first or last target point. The new test exposed
   two unexpected NaNs on an offset time axis. Alignment now retains the full
   source for interpolation and still returns NaN outside its time range.
2. **Export could discard results sharing a name across files.** Selecting a
   calculated user variable present in two files yielded two plot curves, but
   the CSV dictionary used the same column key for both. Export now qualifies
   repeated names with filenames, keeps each time axis, and avoids exporting
   the same file/channel twice when checked in multiple tabs. The reproduced
   three-equation example now exports six distinct result columns rather than
   collapsing them to three.

These fixes are covered by tests that failed before the changes. The
interpolation fix intentionally changes the earlier calculator's edge NaNs;
the historical exact-baseline comparisons in the performance report describe
the state before this correction.

## Full requested equations

The user's three X/Y/Z equations are pasted verbatim into the calculator with
six controlled CSV input files, including all XPOS/YPOS/ZPOS and rotation
channels. Expected results use independently expanded arithmetic and a
degrees-to-radians conversion, rather than the calculator expression evaluator.

Checks include:

- All three output names, signs, constants and degree conversions.
- Storage only in files 1 and 6 and visibility in user-variable controls.
- Rerunning with changed results, with no duplicate outputs or stale values.
- Equal and offset time axes, non-overlap NaNs, datetime reference offsets,
  and cropped time windows.
- Every raw plot sample and timestamp on all six curves.
- Unique CSV columns and each exported value/time axis.
- Reloading exports that share one time axis through the production CSV loader.

The numerical comparison uses an absolute tolerance of 2e-14 with NumPy's
default relative tolerance for these controlled fixtures.

## Local user CSVs

Inputs are unchanged:

| File | Samples | Channels |
| --- | ---: | --- |
| `test5210_UVMOCAP_M1.csv` | 276,429 | mocap X/Y/Z |
| `test5210_yaw_pitch_roll.csv` | 199,693 | PITCH/ROLL/YAW |

**These files do not contain XPOS/YPOS/ZPOS.** The real-data check therefore
omits those three unavailable translation terms and explicitly names its
outputs Xcheck/Ycheck/Zcheck. File references are mapped to f1/f2 and the
exported mocap headers are used exactly as provided by the loader.

All six resulting series (1,428,366 output samples) are compared against an
independent CSV read and direct NumPy interpolation/arithmetic. Each file's
three outputs are then exported on their own time axis and reloaded through
the production loader. Values agree with both absolute and relative tolerances
of 1e-12, including NaN locations. Calculation preserves the loaded timestamps
exactly; separate CSV parsers and export/reload agree within 2e-12 seconds.

This verifies the available measurements and the complete equation workflow.
It does not validate the missing translation measurements, physical coordinate
conventions, or additional filtering/resampling selected after calculation.

## Reproduction

From `C:\Github\ANYtimeseries`, use a fresh writable pytest temporary directory:

```powershell
$verificationTemp = Join-Path (Get-Location) ('.pytest_tmp_calculator_e2e_' + [guid]::NewGuid().ToString('N'))
python -m pytest tests/test_calculator_end_to_end.py -q -p no:cacheprovider --basetemp $verificationTemp
```

The local-data test is skipped when the two user CSVs are unavailable; the
controlled end-to-end tests remain runnable without those files.
