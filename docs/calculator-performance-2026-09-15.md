# Calculator performance verification — 2026-09-15

## Result

The calculator performed unnecessary work before evaluating a formula:

- It filtered every channel in every loaded file, even when unreferenced.
- It aligned unreferenced files' channels to each output file's time axis.
- It evaluated output files whose results were then discarded.
- Every multi-file calculation started and stopped a new process pool,
  including Python and GUI-module imports in Windows worker processes.
- With filters disabled, `apply_filters` still copied/indexed arrays and
  computed a median time step.

The local fix prepares only referenced channels and destination files, caches
database mappings and time coordinates for one calculation, evaluates NumPy
expressions in the current process, and returns a copy immediately when no
filter is selected. Caches are discarded between calculations.

## User-provided CSV verification

Loaded using the application's `FileLoader`:

| File | Channels | Samples per channel | Time range (seconds) |
| --- | ---: | ---: | --- |
| `test5210_UVMOCAP_M1.csv` | 3 | 276,429 | -1.910942 to 4981.465698 |
| `test5210_yaw_pitch_roll.csv` | 3 | 199,693 | 648.999230 to 4248.997956 |

The first file contains mocap X/Y/Z. The second contains PITCH/ROLL/YAW.
**Neither contains XPOS.** The exact requested equation therefore cannot yet
be verified with these two files alone. File IDs were mapped to f1/f2 and the
exported mocap header was converted to the calculator's identifier syntax.
The measured expression explicitly omits the missing XPOS term:

```python
T5210_M1_Xrel_partial = (f1_test5210_UVMOCAP_mat__qtm_uv_M1_xpos - 14.86) - (-0.46*radians(f2_YAW) + 38.34*radians(f2_PITCH))
```

Windows, Python 3.14; three runs per version; comparison baseline:
`c25b69dba825f34aafc2abc7075a5b056d230857`.

| Measurement | Original | Fixed |
| --- | ---: | ---: |
| Run 1 | 1.888953 s | 0.017224 s |
| Run 2 | 1.952544 s | 0.017178 s |
| Run 3 | 1.903281 s | 0.017365 s |
| Median | **1.903281 s** | **0.017224 s** |
| Filter calls per calculation | 6 | 3 |

This is approximately **110× faster for calculator preparation and execution**.
Timing includes constructing output TimeSeries objects. It excludes file
loading, application/widget construction, variable-tab rebuilding and dialogs;
it is not an end-to-end desktop latency measurement.

All output timestamps and values match the original exactly, using
`numpy.testing.assert_array_equal`, including NaNs. Output lengths are 276,429
and 199,693. Finite value counts are 199,693 and 199,692 respectively. The
existing alignment behavior is preserved: it crops source samples to the
target time window before interpolation, so a boundary sample may be NaN
even when the complete source spans that time. No extrapolation was added.

Reproduce from the repository root while HEAD still identifies the baseline:

```powershell
python tools/benchmark_calculator.py --csv test5210_UVMOCAP_M1.csv test5210_yaw_pitch_roll.csv --expression 'T5210_M1_Xrel_partial = (f1_test5210_UVMOCAP_mat__qtm_uv_M1_xpos - 14.86) - (-0.46*radians(f2_YAW) + 38.34*radians(f2_PITCH))' --compare-head
```

The command loads CSVs without modifying them. It compares every generated
sample and timestamp against the committed implementation on every run.

## Synthetic scaling checks

Three files, twelve channels per file, 20,000 samples per channel, with datetime
references; three runs per expression:

| Expression | Original median | Fixed median | Filter calls, before → after |
| --- | ---: | ---: | ---: |
| `result = f1_v0 * 2` | 3.803904 s | 0.029047 s | 36 → 1 |
| `result = c_v0 + 2` | 4.202963 s | 0.082073 s | 36 → 0 |

Common/user references retain their existing raw-data semantics; explicit
`fN_` references retain the selected frequency filtering.

## Regression coverage

Full repository suite: **172 passed**, with 12 calendar/deprecation warnings,
in 91.10 seconds. Pytest used a fresh temporary directory inside the repository
because the default user temporary directory is inaccessible to the sandbox.
`git diff --check` passed.

The calculator regression tests cover output naming, progress, preservation of
per-file lengths, skipping unrelated files/channels, applying filters once per
referenced channel, numeric and absolute-datetime interpolation, the full
requested equation structure against an independent synthetic numerical
reference, common/user subtraction, changed filters on repeated runs, empty
time windows, scalar broadcasting, and independent no-filter output arrays.

Two adjacent correctness issues found during verification were also corrected:
common/user token matching no longer consumes subtraction and spaces, and an
empty time-window slice now produces the intended error instead of indexing an
empty target axis. User file references numbered zero are rejected.

The calculator still evaluates synchronously on the GUI thread. Very expensive
custom expressions can still block the interface; those were not benchmarked.

## Follow-up: multiple pasted equations

After this change, the full repository suite passed: **184 tests**, 12 existing
calendar/deprecation warnings, 89.77 seconds.

The calculator now splits the input into Python assignment statements, saves
each named output, and recognizes parenthesized continuations, blank lines and
comments. A batch shares the per-file time windows, time coordinates and
filtered source data, and refreshes the variable tabs once after all equations
succeed. Output file selection remains specific to each equation.

The exact three X/Y/Z equations supplied by the user are covered by a numerical
regression test with six synthetic input files, including XPOS/YPOS/ZPOS.
The nine referenced channels are filtered only once each across the batch.
Regression tests also cover failed batches without partial output creation,
duplicate/existing names, independent file scopes, wrapped expressions and
references to earlier staged outputs.

An additional three-equation run on the two local CSVs used all six available
mocap/rotation channels, omitting the unavailable XPOS/YPOS/ZPOS terms. It
created Xcheck/Ycheck/Zcheck on both time axes in 0.055342, 0.056209 and
0.057078 seconds (median **0.056209 seconds**), with six filter calls per batch.
Repeated runs produced identical values and timestamps. The same timing
exclusions listed above apply; this is not a full-equation data validation or
an end-to-end GUI measurement.

## Follow-up: overwrite repeated output names

Full repository verification: **187 tests passed** in 88.71 seconds, with 12
existing calendar/deprecation warnings. Whitespace checks passed.

Rerunning a named equation now replaces its previous output in each destination
file. Repeating a name within a pasted batch keeps the last result. Intermediate
equations can read the value available at that point; cached filters and time
coordinates are invalidated when that staged value changes.

Replacement updates the stored TimeSeries, including its time window, without
adding duplicate database keys or moving its position. All results remain
staged until evaluation succeeds, so a later error leaves previous outputs
untouched. This supersedes the initial batch implementation's rejection of
duplicate or existing output names.
