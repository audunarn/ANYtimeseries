# ANYtimeSeries Documentation

ANYtimeSeries is a Qt-based application for exploring and editing time-series data. This document provides an overview of the application's capabilities and how to get started.

## Installation

Install the package from PyPI:

```bash
pip install anytimes
```

The core dependencies (``numpy``, ``pandas``, ``scipy``, ``PySide6``, ``matplotlib`` and ``anyqats``) will be installed automatically.

## Launching the GUI

After installation the GUI can be started from Python or from the command line.

```python
from anytimes import anytimes_gui
anytimes_gui.main()
```

or simply:

```bash
anytimes
```

## Basic Workflow

1. **Load data** – open one or more time-series files. The application automatically groups files with common variables for an efficient workflow.
2. **Inspect variables** – select variables and preview the data in table form.
3. **Manipulate series** – apply predefined operations or build custom expressions to create new series.
4. **Visualise** – choose from several plot types and engines (Plotly, Bokeh or Matplotlib) to explore the results.
5. **Export** – save modified series for later use.

## Advanced Features

- Frequency filtering for noise reduction.
- Equation-based transformations for complex processing.
- Extreme value statistics and a statistics table summarising key metrics.
- Support for OrcaFlex `.sim` files.
- Configurable light and dark themes for comfortable viewing.

## Supported File Types

The loader accepts a broad range of common formats:

- Text and binary time-series: `csv`, `mat`, `dat`, `ts`, `h5`, `tda`, `asc`, `tdms`, `pkl`, `pickle`, `bin`.
- Tabular data: `xlsx`, `json`, `feather`, `parquet`.
- OrcaFlex simulation files: `.sim` with a dialog for selecting objects and variables.

## Button Reference

### Variable navigation
- **Go to Common / Go to User Variables** – scrolls to predefined sections in the variable list.
- **Unselect All** – clears all check boxes.
- **Select all by list pos.** – selects by row number across all tabs.

### File controls
- **Load time series file** – open one or more data files.
- **Save Files** – write all loaded series back to disk.
- **Clear All** – remove every loaded file and variable.
- **Save Values… / Load Values…** – store or restore current offsets and scaling factors.
- **Export Selected to CSV** – export ticked variables; optional resampling via the adjacent `dt` field.
- **Clear OrcaFlex Selection / Re-select OrcaFlex Variables** – manage previous `.sim` selections.
- **Dark Theme** – toggle light/dark appearance.
- **Embed Plot** – draw plots inside the main window instead of a popup.

### Quick transformations
- Multiply or divide by common factors (`1000`, `10`, `2`) or `-1`.
- Convert between **Radians** and **Degrees**.
- **Shift Mean → 0 / Shift Min to Zero** – translate series; optionally ignore the lowest 1 % of values.
- **Sqrt(sum of squares)**, **Mean**, **Absolute**, **Rolling Avg** – derive a new series from the selection.
- **Shift Min → 0 : if repeted minima as per input** and **Common Shift Min → 0…** – advanced zero-shifting using tolerance and minimum count inputs.

### Variable input operations
- **Apply Values** – apply expressions typed in each variable’s field (e.g. `*2`, `/2`, `+1`).
- **Create user variable instead of overwriting?** – stores results as new user-defined series.

### File list and time window
- **Remove File** – unload the highlighted file.
- **Time Window Start/End** – restrict operations and plots to a time interval; **Reset** clears the limits.

### Frequency filter
Radio buttons select **None**, **Low-pass**, **High-pass**, **Band-pass** or **Band-block** filters with associated cutoff inputs. The chosen filter applies to transformations and statistics.

### Tools
- **Open in AnyQATS** – launch the external viewer.
- **Open Extreme Value Statistics Tool** – open the dedicated extreme value analysis window.

### Plot controls
- **Plot Selected (one graph / side-by-side)** – display selected variables on a single axis or in a grid; **Same axes** forces a common scale.
- **Plot Mean / Rolling Mean** – combine selected signals.
- **Animate XYZ scatter (all points)** – show a 3-D animation from three variables.
- **Raw / Low-pass / High-pass** – choose which filtered components to plot.
- **Engine** – pick the plotting backend (`plotly`, `bokeh` or the built-in matplotlib).
- **Show components (used in mean)** – include raw variables when plotting the mean.
- Label trimming, Y-axis label and rolling-window controls refine the appearance of plots.
- **Mark max/min** – annotate extrema on the time plot.

### Calculator
- **Calculate** – evaluate Python-style expressions to create new variables.
- **?** – open a short help message about calculator syntax.

## Statistics Window

Selecting **Show statistic for selected variables** opens a table with descriptive statistics. The window provides:

- Load order toggle between *Files → Variables* and *Variables → Files*.
- Frequency filters identical to the main window.
- Histogram line input with optional bar-value text.
- Sortable table of statistics where columns can be toggled via header right-click.
- Linked plots: line graph of the time series and histograms by row and column.
- **Copy as TSV** button for clipboard export.

## Extreme Value Statistics Window

Launched from **Open Extreme Value Statistics Tool**, this dialog estimates return levels using a Generalized Pareto distribution. Features include:

- Tail selection (**upper** or **lower**).
- Threshold spin box with **Calc Threshold** helper.
- Confidence level input for bootstrap intervals.
- **Run EVM** performs the analysis and updates the result text.
- Plots of the raw time series with threshold, return level curve and a quantile comparison.

## Command Line Helpers

For quick access you can create a small batch script that launches the GUI with a specific Python interpreter:

```batch
@echo off
C:\Python\Python313\python.exe C:\Github\ANYtimeseries\anytimes\anytimes_gui.py
pause
```

Update the paths to match your environment.

## Screenshots

![Dark mode](../dark_mode.png)
![Light mode](../light_mode.png)
![Statistics table](../statistics_table.png)

## License

Released under the MIT License. See [LICENSE](../LICENSE) for details.

