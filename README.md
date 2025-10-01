<p align="center">
  <img src="ANYtimes_logo.png" alt="AnytimeSeries logo" width="200"/>
</p>

# ANYtimeSeries

ANYtimeSeries provides a Qt-based interface for exploring, editing and analysing time-series data. The goal is efficient processing and a clear understanding of the loaded signals, whether they originate from laboratory tests, numerical simulations or OrcaFlex studies.

The application integrates with the bundled **anyqats** package and supports a broad range of formats for loading and visualising time-series information. For a comprehensive walkthrough—covering the full workflow, control reference and screenshots—see the [documentation](docs/README.md).

## Key features

### Data import and management
- Load multiple files at once; the editor automatically identifies common variables for quick comparison across tests.
- Native support for text, binary and tabular formats (`csv`, `xlsx`, `mat`, `h5`, `tdms`, `parquet`, …) alongside OrcaFlex `.sim` studies.
- Preload OrcaFlex simulations, reuse previous selections and broadcast object/variable choices across matching simulations to keep large projects in sync.
- Extract surface pressure time series by pairing `.sim` files with diffraction (`.owr`) models and a set of user-specified panel coordinates.
- Cache OrcaFlex selections and diffraction models so subsequent loads are instant.

### Editing and transformation
- Quickly manipulate series via predefined operations (multiply/divide, negate, convert degrees↔radians, rolling averages, absolute value, square-root sum of squares, mean, zero-shifting utilities and more).
- Apply custom expressions per variable (e.g. `*2`, `+offset`) or evaluate full calculator expressions to build user-defined channels instead of overwriting the originals.
- Limit operations to a time window and optionally resample data when exporting selections to CSV.

### Filtering, statistics and extreme values
- Apply low-, high-, band-pass or band-stop filters to the active selection and feed the filtered data into plots or statistics.
- Open the Statistics window for a sortable table, histograms, descriptive metrics and clipboard export (TSV).
- Launch the Extreme Value Statistics tool to perform Generalized Pareto fits, return-level calculations and diagnostic plotting, including guidance for stabilising challenging fits.

### Visualisation
- Plot selected variables in a single figure or side-by-side grid using Plotly, Bokeh or Matplotlib backends.
- Embed plots inside the main window or open them in a separate browser window, toggle shared axes, annotate maxima/minima and trim labels.
- Generate rolling means, mean overlays and XYZ scatter animations to inspect vector responses.

### OrcaFlex integration
- Search objects and variables with live filtering, strip redundant substrings from labels and specify arc-length/extras directly in the selector.
- Reuse selections for future `.sim` files, apply them to batches of simulations and automatically align similarly named objects via configurable stripping rules.
- Load AnyQATS directly from the GUI for side-by-side inspection of the same datasets.

### Convenience utilities
- Switch between light and dark themes, embed plots, open external viewers and manage offsets/scaling presets via save/load dialogs.
- Track file loading progress with an integrated progress bar and optional preloading callbacks.


<p align="center">
  <img src="dark_mode.png" alt="AnytimeSeries dark mode" width="700"/>
</p>


<p align="center">
  <img src="light_mode.png" alt="AnytimeSeries light mode" width="700"/>
</p>


<p align="center">
  <img src="statistics_table.png" alt="AnytimeSeries statistics" width="700"/>
</p>

## Installation

```bash
pip install anytimes
```

For running on Windows without a Python environment installed, download the `.exe` file from [releases][2].

## Requirements

- numpy
- pandas
- scipy
- PySide6
- matplotlib

## Optional Requirements
- plotly
- bokeh
- OrcFxAPI and Orcaflex (licenced or [Demo][1])

## Usage

After installation, import the GUI module in your Python project:

```python
from anytimes import anytimes_gui
```

The module exposes Qt widgets for building custom time-series exploration tools. 

You can also launch the GUI from the command line using the `anytimes` entry point or via Python's module launcher:
```cmd
C:\Python\Python313\Scripts\anytimes
python -m anytimes
```

You can start the GUI programmatically by calling:

```python
anytimes_gui.main()
```

Another approach is to make <b>some_file.bat</b> and put it on your desktop. The contents should look something like this:

```batch
@echo off
REM Run script with specific Python interpreter

C:\Python\Python313\python.exe C:\Github\ANYtimeseries\anytimes\anytimes_gui.py
pause
```

Update it with the correct location of your Python environment and the `.py` file.


## License

Released under the MIT License. See [LICENSE](LICENSE) for details.

[1]: <https://www.orcina.com/orcaflex/demo/> "Demo version of Orcaflex"
[2]: <https://github.com/audunarn/ANYtimeseries/releases> "ANYtimes releases"

