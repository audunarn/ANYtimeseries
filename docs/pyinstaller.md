# Building the Windows executable

The repository's maintained PyInstaller configuration is `anytimes.spec`.
Build from a clean virtual environment so the executable contains only the
intended runtime dependencies.

```powershell
python -m venv build/windows-venv
build/windows-venv/Scripts/python -m pip install -r tools/requirements-pyinstaller.txt
build/windows-venv/Scripts/python -m pip install --no-deps .
build/windows-venv/Scripts/python -m PyInstaller --noconfirm anytimes.spec
```

The windowed executable is written to `dist/ANYtimeSeries.exe`. PyInstaller is
a packaging tool and is not a runtime dependency of the Python package.

If resources or hidden imports are added, update `anytimes.spec` and verify the
result on a Windows machine without the development environment. The release
artifact should be smoke-tested by launching the GUI and loading a small CSV
before distribution.

The pinned configuration uses Python 3.13 x64 and bundles Plotly/Bokeh resources,
QtWebEngine and the file-reader dependencies. Run the built-in smoke test from
an unrelated working directory to verify the executable without source imports:

```powershell
C:/path/to/ANYtimeSeries.exe --smoke-test C:/path/to/smoke-report.json
```

It opens the Qt application offscreen, loads CSV data, calculates and overwrites
multiple equations, renders Matplotlib and Plotly/Bokeh output, and verifies CSV
export. The JSON report records the version, frozen status and checks performed.

Release assets include the Windows executable, this specification, the Python
wheel and source distribution, and `SHA256SUMS`. The release ledger binds all
four artifacts; only the wheel and source distribution are submitted to PyPI.
