"""Tests for OrcaFlex selection reuse logic in FileLoader."""

from pathlib import Path
import importlib.util
import sys
import types


def _install_qt_stubs(monkeypatch):
    class _AutoClassModule(types.ModuleType):
        def __getattr__(self, name):
            value = type(name, (), {})
            setattr(self, name, value)
            return value

    qt_core = _AutoClassModule("PySide6.QtCore")

    class _Signal:
        def __init__(self, *_, **__):
            pass

        def connect(self, *_):
            pass

        def emit(self, *_):
            pass

    qt_core.Signal = lambda *args, **kwargs: _Signal()
    qt_core.Slot = lambda *args, **kwargs: (lambda func: func)
    qt_core.Qt = type("Qt", (), {})

    qt_gui = _AutoClassModule("PySide6.QtGui")
    qt_widgets = _AutoClassModule("PySide6.QtWidgets")

    monkeypatch.setitem(sys.modules, "PySide6", types.ModuleType("PySide6"))
    sys.modules["PySide6"].QtCore = qt_core
    sys.modules["PySide6"].QtGui = qt_gui
    sys.modules["PySide6"].QtWidgets = qt_widgets
    monkeypatch.setitem(sys.modules, "PySide6.QtCore", qt_core)
    monkeypatch.setitem(sys.modules, "PySide6.QtGui", qt_gui)
    monkeypatch.setitem(sys.modules, "PySide6.QtWidgets", qt_widgets)


def _install_optional_stubs(monkeypatch):
    for name in ["h5py", "pymatreader", "nptdms"]:
        if name in sys.modules:
            continue
        module = types.ModuleType(name)
        if name == "pymatreader":
            module.read_mat = lambda *args, **kwargs: {}
        if name == "nptdms":
            module.TdmsFile = type("TdmsFile", (), {})
        monkeypatch.setitem(sys.modules, name, module)


def _install_anytimes_stub(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    if "anytimes" not in sys.modules:
        pkg = types.ModuleType("anytimes")
        pkg.__path__ = [str(root / "anytimes")]
        monkeypatch.setitem(sys.modules, "anytimes", pkg)
    if "anytimes.gui" not in sys.modules:
        gui_pkg = types.ModuleType("anytimes.gui")
        gui_pkg.__path__ = [str(root / "anytimes" / "gui")]
        monkeypatch.setitem(sys.modules, "anytimes.gui", gui_pkg)


def _load_file_loader(monkeypatch):
    try:  # Try the normal import path first.
        from anytimes.gui import file_loader  # type: ignore
        return file_loader
    except Exception:
        _install_qt_stubs(monkeypatch)
        _install_optional_stubs(monkeypatch)
        _install_anytimes_stub(monkeypatch)

        spec = importlib.util.spec_from_file_location(
            "anytimes.gui.file_loader", Path(__file__).resolve().parents[1] / "anytimes/gui/file_loader.py"
        )
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, "anytimes.gui.file_loader", module)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module


class _FakeOrcaObject:
    def __init__(self, name):
        self.Name = name
        self.typeName = "Line"
        self.NodeArclengths = [0.0, 1.0]


class _FakeModel:
    def __init__(self, obj):
        self._obj = obj

    def __getitem__(self, key):
        if key == self._obj.Name:
            return self._obj
        raise KeyError(key)


def test_reuseable_selection_only_keeps_names(monkeypatch):
    file_loader = _load_file_loader(monkeypatch)

    obj = _FakeOrcaObject("LineA")
    model = _FakeModel(obj)

    fake_module = types.SimpleNamespace(Model=lambda _: model)
    monkeypatch.setitem(sys.modules, "OrcFxAPI", fake_module)
    monkeypatch.setattr(file_loader, "OrcFxAPI", fake_module)

    def fake_selector(*_):
        return ([("LineA", "Axial", "")], [], True)

    monkeypatch.setattr(
        file_loader.OrcaflexVariableSelector,
        "get_selection",
        staticmethod(fake_selector),
    )

    loaded_tsdb = object()
    monkeypatch.setattr(
        file_loader.FileLoader,
        "_load_orcaflex_data_from_specs",
        lambda self, mdl, specs: loaded_tsdb,
    )

    loader = file_loader.FileLoader()
    result = loader._load_orcaflex_file("fake.sim")

    assert result is loaded_tsdb
    assert loader._last_orcaflex_selection == [("LineA", "Axial", None, None)]
    assert isinstance(loader._last_orcaflex_selection[0][0], str)


def test_add_unique_timeseries_suffixes_duplicates(monkeypatch):
    file_loader = _load_file_loader(monkeypatch)
    loader = file_loader.FileLoader()
    
    class _StubTsDB:
        def __init__(self):
            self.names = []

        def add(self, ts):
            if ts.name in self.names:
                raise KeyError(ts.name)
            self.names.append(ts.name)

    tsdb = _StubTsDB()

    time = [0.0, 1.0]
    data = [1.0, 2.0]

    first_name = loader._add_unique_timeseries(tsdb, "LineA:Axial", time, data)
    second_name = loader._add_unique_timeseries(tsdb, "LineA:Axial", time, data)

    assert first_name == "LineA:Axial"
    assert second_name == "LineA:Axial (2)"
    assert tsdb.names == ["LineA:Axial", "LineA:Axial (2)"]
