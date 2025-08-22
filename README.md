<p align="center">
  <img src="ANYtimes_logo.png" alt="AnytimeSeries logo" width="200"/>
</p>

# AnytimeSeries

AnytimeSeries provides a PySide6-based interface for exploring and editing time-series data. The application integrates with the [qats](https://pypi.org/project/qats/) package and supports various file formats for loading and visualising time-series information.

## Installation

```bash
pip install anytimes
```

## Requirements

- numpy
- pandas
- scipy
- qats
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

