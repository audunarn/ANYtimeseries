<p align="center">
  <img src="ANYtimes_logo.png" alt="AnytimeSeries logo" width="200"/>
</p>

# AnytimeSeries

ANYtimeSeries provides a PySide6-based interface for exploring and editing time-series data. 

The application integrates with the bundled anyqats package and supports various file formats for loading and visualising time-series information.

Some of the features of ANYtimes
- loading of multiple files
- detection of commom variables files for effiecent work flow
- quick manipulation of time series using predefined operations
- complex manipulation of time series using equation input
- frequency filtering
- time sereis statistics
- many plotting options
- embdedded and in browser plotting
- Orcaflex .sim files compability
- extreme value statstics
- selection of plotting engine (plotly, bokeh or matplotlib)


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

## Requirements

- numpy
- pandas
- scipy
- anyqats
- PySide6
- matplotlib

## Usage

After installation, import the GUI module in your Python project:

```python
from anytimes import anytimes_gui
```

The module exposes Qt widgets for building custom time-series exploration tools. You can also launch the GUI from the command line using the `anytimes` entry point.

## License

Released under the MIT License. See [LICENSE](LICENSE) for details.

