<p align="center">
  <img src="ANYtimes_logo.png" alt="AnytimeSeries logo" width="200"/>
</p>

# ANYtimeSeries

ANYtimeSeries provides a QT-based interface for exploring and editing time-series data.

The intension is efficent processing and understanding of the loaded data.

The application integrates with the bundled anyqats package and supports various file formats for loading and visualising time-series information.

For a more comprehensive guide, including workflow examples and screenshots, see the [documentation](docs/README.md).

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

For running in windows without Python environment installed, download the .exe file from [releases][2]

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

You can also launch the GUI from the command line using the `anytimes` entry point:
```cmd
C:\Python\Python313\Scripts>anytimes
```

You can start the GUI by typing:

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

Update it with the correct location of you Python environment and location of .py file.


## License

Released under the MIT License. See [LICENSE](LICENSE) for details.

[1]: <https://www.orcina.com/orcaflex/demo/> "Demo version of Orcaflex"
[2]: <https://github.com/audunarn/ANYtimeseries/releases> "ANYtimes releases"

