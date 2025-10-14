# Building a standalone executable with PyInstaller

This project can be bundled into a single-file Windows executable with
[PyInstaller](https://pyinstaller.org/) so that end users do not need a
separate Python installation. The instructions below assume PyInstaller is
installed in the active Python environment (e.g. `pip install pyinstaller`).

## Basic single-file build

From the repository root run:

```bash
pyinstaller \
  --onefile \
  --windowed \
  --name ANYtimeSeries \
  --collect-data anyqats.app \
  anytimes/__main__.py
```

Key options:

- `--onefile` creates a single self-extracting `ANYtimeSeries.exe`.
- `--windowed` hides the console window when the GUI is launched.
- `--collect-data anyqats.app` ensures the bundled AnyQATS resources (such as
  the application icon) are copied into the executable.
- `anytimes/__main__.py` is the entry point that starts the GUI.

After the command completes, the executable is available at
`dist/ANYtimeSeries.exe`.

## Including additional resources

If you have added local plugins, themes or other data files that are loaded at
runtime, list them with extra `--collect-data` options, e.g.

```bash
pyinstaller --onefile --windowed \
  --collect-data anyqats.app \
  --collect-data mypackage.resources \
  anytimes/__main__.py
```

Alternatively, create a `anytimes.spec` file (generated automatically the first
run) and edit the `datas` section to point at extra files. Subsequent builds can
then be triggered with:

```bash
pyinstaller anytimes.spec
```

## Running the build on Windows

1. Install the project and optional dependencies into a virtual environment.
2. Install PyInstaller (`pip install pyinstaller`).
3. Run the command above from a Developer Command Prompt or PowerShell.
4. Distribute the resulting `dist/ANYtimeSeries.exe`.

Running the executable may require the Microsoft Visual C++ Redistributable if
it is not already installed on the target machine.
