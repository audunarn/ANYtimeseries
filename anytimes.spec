# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller 6 configuration for the standalone Windows application."""
from pathlib import Path
from PyInstaller.utils.hooks import collect_all, collect_submodules, copy_metadata

root = Path(SPECPATH)
datas = [(str(root / "ANYtimes_logo.png"), ".")]
binaries = []
hiddenimports = collect_submodules("anyqats.io")
for package in ("plotly", "bokeh", "pyextremes"):
    package_data, package_binaries, package_imports = collect_all(package)
    datas += package_data
    binaries += package_binaries
    hiddenimports += package_imports
datas += copy_metadata("anytimes")

a = Analysis(
    [str(root / "anytimes" / "__main__.py")],
    pathex=[str(root)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hooksconfig={"matplotlib": {"backends": ["QtAgg", "Agg"]}},
    excludes=["tkinter", "PyQt5", "PyQt6", "PySide2", "IPython", "pytest"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz, a.scripts, a.binaries, a.datas, [],
    name="ANYtimeSeries",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
)
