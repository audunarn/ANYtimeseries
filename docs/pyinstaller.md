# Building the Windows executable

The repository's maintained PyInstaller configuration is `anytimes.spec`.
Build from a clean virtual environment so the executable contains only the
intended runtime dependencies.

```powershell
python -m pip install -e .
python -m pip install pyinstaller
pyinstaller anytimes.spec
```

The windowed executable is written to `dist/ANYtimeSeries.exe`. PyInstaller is
a packaging tool and is not a runtime dependency of the Python package.

If resources or hidden imports are added, update `anytimes.spec` and verify the
result on a Windows machine without the development environment. The release
artifact should be smoke-tested by launching the GUI and loading a small CSV
before distribution.
