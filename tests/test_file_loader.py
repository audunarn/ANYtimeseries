"""Tests for OrcaFlex selection reuse logic in FileLoader."""

import importlib.util
from pathlib import Path
import sys
import types

import numpy as np
import pandas as pd
import pytest
import xarray as xr


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
    captured = {}

    def _fake_load(self, mdl, specs):
        captured["specs"] = specs
        return loaded_tsdb

    monkeypatch.setattr(
        file_loader.FileLoader,
        "_load_orcaflex_data_from_specs",
        _fake_load,
    )

    loader = file_loader.FileLoader()
    loader.cache_orcaflex_buffers = False
    result = loader._load_orcaflex_file("fake.sim")

    assert result is loaded_tsdb
    assert captured["specs"] == [("LineA", "Axial", None, None)]
    assert isinstance(captured["specs"][0][0], str)


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


def test_add_unique_timeseries_attaches_metadata(monkeypatch):
    file_loader = _load_file_loader(monkeypatch)
    loader = file_loader.FileLoader()

    class _StubTsDB:
        def __init__(self):
            self.register = {}

        def add(self, ts):
            if ts.name in self.register:
                raise KeyError(ts.name)
            self.register[ts.name] = ts

    tsdb = _StubTsDB()
    metadata = {"freq_hz": np.array([0.1, 0.2]), "rao_amp": np.array([1.0, 2.0])}
    loader._add_unique_timeseries(tsdb, "LineA:Axial", [0.0, 1.0], [1.0, 2.0], metadata=metadata)

    ts = tsdb.register["LineA:Axial"]
    np.testing.assert_array_equal(ts.freq_hz, np.array([0.1, 0.2]))
    np.testing.assert_array_equal(ts.rao_amp, np.array([1.0, 2.0]))


def test_is_frequency_domain_model_detects_general_string(monkeypatch):
    file_loader = _load_file_loader(monkeypatch)
    loader = file_loader.FileLoader()

    class _General:
        DynamicsSolutionMethod = "Frequency domain"

    class _Model:
        def __getitem__(self, key):
            assert key == "General"
            return _General()

    assert loader._is_frequency_domain_model(_Model()) is True


def test_is_frequency_domain_model_rejects_time_domain_method(monkeypatch):
    file_loader = _load_file_loader(monkeypatch)
    loader = file_loader.FileLoader()

    class _General:
        DynamicsSolutionMethod = "Implicit time domain"

    class _Model:
        def __getitem__(self, key):
            assert key == "General"
            return _General()

    assert loader._is_frequency_domain_model(_Model()) is False




def test_extract_model_time_uses_sample_times_for_frequency_domain(monkeypatch):
    file_loader = _load_file_loader(monkeypatch)
    loader = file_loader.FileLoader()

    called = {}

    class _Model:
        def SampleTimes(self, spec):
            called["spec"] = spec
            return [0.0, 2.0, 4.0]

    monkeypatch.setattr(loader, "_is_frequency_domain_model", lambda _: True)

    time = loader._extract_model_time(_Model(), "spec-token")

    np.testing.assert_array_equal(time, np.array([0.0, 2.0, 4.0]))
    assert called["spec"] == "spec-token"


def test_extract_model_time_falls_back_to_general_timehistory(monkeypatch):
    file_loader = _load_file_loader(monkeypatch)
    loader = file_loader.FileLoader()

    class _General:
        def TimeHistory(self, var, spec):
            assert var == "Time"
            assert spec == "time-window"
            return [10.0, 20.0]

    class _Model:
        def __getitem__(self, key):
            assert key == "General"
            return _General()

    monkeypatch.setattr(loader, "_is_frequency_domain_model", lambda _: False)

    time = loader._extract_model_time(_Model(), "time-window")

    assert time == [10.0, 20.0]


def test_extract_model_time_falls_back_when_sample_times_decode_fails(monkeypatch):
    file_loader = _load_file_loader(monkeypatch)
    loader = file_loader.FileLoader()

    class _General:
        def TimeHistory(self, var, spec):
            assert var == "Time"
            assert spec == "time-window"
            return [1.0, 2.0, 3.0]

    class _Model:
        def SampleTimes(self, _spec):
            raise UnicodeDecodeError("charmap", b"\x8d", 0, 1, "character maps to <undefined>")

        def __getitem__(self, key):
            assert key == "General"
            return _General()

    monkeypatch.setattr(loader, "_is_frequency_domain_model", lambda _: True)

    time = loader._extract_model_time(_Model(), "time-window")

    assert time == [1.0, 2.0, 3.0]


def test_resolve_time_array_uses_fallback_index_for_length_mismatch(monkeypatch):
    file_loader = _load_file_loader(monkeypatch)
    loader = file_loader.FileLoader()

    out = loader._resolve_time_array(np.array([0.0, 1.0]), np.array([10.0, 20.0, 30.0]))

    np.testing.assert_array_equal(out, np.array([0.0, 1.0, 2.0]))


def test_load_orcaflex_time_histories_individually_uses_resolved_time(monkeypatch):
    file_loader = _load_file_loader(monkeypatch)
    loader = file_loader.FileLoader()

    captured = []

    def _fake_add_unique(_tsdb, name, time, data, metadata=None):
        captured.append((name, np.asarray(time), np.asarray(data), metadata))

    monkeypatch.setattr(loader, "_add_unique_timeseries", _fake_add_unique)

    class _Obj:
        def TimeHistory(self, *_args, **_kwargs):
            return [5.0, 6.0, 7.0]

    model = {"ObjA": _Obj()}

    error = loader._load_orcaflex_time_histories_individually(
        model=model,
        tsdb=object(),
        fallback_specs=[("ObjA", "A", None)],
        names=["obj:A"],
        time_spec="window",
        time=np.array([0.0, 1.0]),
        spectral_lookup={},
    )

    assert error is None
    np.testing.assert_array_equal(captured[0][1], np.array([0.0, 1.0, 2.0]))


def test_load_orcaflex_time_histories_individually_handles_object_extra(monkeypatch):
    file_loader = _load_file_loader(monkeypatch)
    loader = file_loader.FileLoader()

    captured = []

    def _fake_add_unique(tsdb, name, time, data, metadata=None):
        captured.append((name, np.asarray(time), np.asarray(data), metadata))

    monkeypatch.setattr(loader, "_add_unique_timeseries", _fake_add_unique)

    class _Obj:
        def TimeHistory(self, var, spec, *args):
            if args:
                return [3.0, 4.0]
            return [1.0, 2.0]

    model = {"ObjA": _Obj(), "ObjB": _Obj()}
    fallback_specs = [("ObjA", "A", None), ("ObjB", "B", "extra")]
    names = ["obj:A", "obj:B"]
    spectral_lookup = {"obj:B": {"freq_hz": np.array([0.1]), "rao_amp": np.array([1.1])}}

    error = loader._load_orcaflex_time_histories_individually(
        model=model,
        tsdb=object(),
        fallback_specs=fallback_specs,
        names=names,
        time_spec="window",
        time=np.array([0.0, 1.0]),
        spectral_lookup=spectral_lookup,
    )

    assert error is None
    assert [item[0] for item in captured] == ["obj:A", "obj:B"]
    np.testing.assert_array_equal(captured[0][2], np.array([1.0, 2.0]))
    np.testing.assert_array_equal(captured[1][2], np.array([3.0, 4.0]))
    assert captured[0][3] is None
    assert captured[1][3] == spectral_lookup["obj:B"]


def test_load_orcaflex_time_histories_individually_returns_last_error(monkeypatch):
    file_loader = _load_file_loader(monkeypatch)
    loader = file_loader.FileLoader()

    monkeypatch.setattr(loader, "_add_unique_timeseries", lambda *args, **kwargs: None)

    class _Obj:
        def TimeHistory(self, *_args, **_kwargs):
            raise RuntimeError("not available")

    model = {"ObjA": _Obj()}
    fallback_specs = [("ObjA", "A", None)]

    error = loader._load_orcaflex_time_histories_individually(
        model=model,
        tsdb=object(),
        fallback_specs=fallback_specs,
        names=["obj:A"],
        time_spec="window",
        time=np.array([0.0, 1.0]),
        spectral_lookup={},
    )

    assert isinstance(error, RuntimeError)
    assert str(error) == "not available"


def test_open_orcaflex_picker_includes_preload_error_details(monkeypatch):
    file_loader = _load_file_loader(monkeypatch)
    loader = file_loader.FileLoader()

    def _fail_load(_self, filepath):
        raise RuntimeError(
            f"Problem reading time history file 'missing.txt' while opening '{filepath}'"
        )

    monkeypatch.setattr(file_loader.FileLoader, "_load_sim_model", _fail_load)

    with pytest.raises(RuntimeError) as excinfo:
        loader.open_orcaflex_picker(["/tmp/example.sim"])

    message = str(excinfo.value)
    assert message.startswith("Models not preloaded.\n")
    assert "example.sim: Problem reading time history file 'missing.txt'" in message

def test_load_era5_netcdf_single_point(monkeypatch, tmp_path):
    file_loader = _load_file_loader(monkeypatch)

    times = pd.date_range("2020-01-01", periods=4, freq="H")
    ds = xr.Dataset(
        {
            "u10": (
                ("time", "latitude", "longitude"),
                np.arange(times.size, dtype=float).reshape((times.size, 1, 1)),
            ),
            "mwd": ("time", np.linspace(0.0, 30.0, times.size)),
        },
        coords={"time": times, "latitude": [60.0], "longitude": [5.0]},
    )
    nc_path = tmp_path / "era5_sample.nc"
    ds.to_netcdf(nc_path)

    loader = file_loader.FileLoader()
    tsdb = loader._load_generic_file(str(nc_path))

    data = tsdb.getm()
    assert set(data) == {"mwd", "u10"}
    assert data["u10"].t.shape[0] == times.size
    assert data["u10"].dtg_ref == times[0].to_pydatetime()
    np.testing.assert_array_equal(data["mwd"].x, np.linspace(0.0, 30.0, times.size))


def test_load_netcdf_flexible_time_detection(monkeypatch, tmp_path):
    file_loader = _load_file_loader(monkeypatch)

    times = pd.date_range("2021-06-01", periods=3, freq="D")
    ds = xr.Dataset(
        {"temp": ("forecast_time", np.linspace(5.0, 7.0, times.size))},
        coords={"forecast_time": times},
    )
    nc_path = tmp_path / "forecast_sample.nc"
    ds.to_netcdf(nc_path)

    loader = file_loader.FileLoader()
    tsdb = loader._load_generic_file(str(nc_path))

    data = tsdb.getm()
    assert set(data) == {"temp"}
    np.testing.assert_array_equal(data["temp"].t, np.array([0.0, 86400.0, 172800.0]))
    np.testing.assert_array_equal(data["temp"].x, np.linspace(5.0, 7.0, times.size))
    assert data["temp"].dtg_ref == times[0].to_pydatetime()
    np.testing.assert_array_equal(data["temp"].dtg_time, times.to_pydatetime())


def test_load_netcdf_ignores_unused_time_coord(monkeypatch, tmp_path):
    file_loader = _load_file_loader(monkeypatch)

    times = pd.date_range("2022-04-01", periods=2, freq="D")
    ds = xr.Dataset(
        {"temperature": ("valid_time", [12.3, 12.5])},
        coords={
            "valid_time": times,
            # An unused time-like coordinate that should not be picked as the time axis.
            "time": pd.date_range("2000-01-01", periods=2, freq="H"),
        },
    )

    nc_path = tmp_path / "mixed_time_coords.nc"
    ds.to_netcdf(nc_path)

    loader = file_loader.FileLoader()
    tsdb = loader._load_generic_file(str(nc_path))

    data = tsdb.getm()
    assert set(data) == {"temperature"}
    np.testing.assert_array_equal(data["temperature"].x, np.array([12.3, 12.5]))




def test_load_netcdf_multidimensional_depth_series(monkeypatch):
    file_loader = _load_file_loader(monkeypatch)

    loader = file_loader.FileLoader()
    nc_path = Path(__file__).resolve().parent / "subset.nc"
    tsdb = loader._load_generic_file(str(nc_path))

    data = tsdb.getm()
    expected = {
        "temperature [depth=0.0]",
        "temperature [depth=300.0]",
        "salinity [depth=0.0]",
        "u_eastward [depth=0.0]",
        "v_northward [depth=0.0]",
    }
    assert expected.issubset(set(data.keys()))

    for name in [
        "temperature [depth=0.0]",
        "salinity [depth=0.0]",
        "u_eastward [depth=0.0]",
        "v_northward [depth=0.0]",
    ]:
        assert data[name].t.shape[0] == 39

def test_load_netcdf_uses_cftime_decoder(monkeypatch, tmp_path):
    file_loader = _load_file_loader(monkeypatch)

    times = xr.cftime_range("2001-02-27", periods=3, freq="D", calendar="noleap")
    ds = xr.Dataset({"height": ("time", [1.0, 2.0, 3.0])}, coords={"time": times})

    nc_path = tmp_path / "cftime_calendar.nc"
    ds.to_netcdf(nc_path)

    loader = file_loader.FileLoader()
    tsdb = loader._load_generic_file(str(nc_path))

    data = tsdb.getm()["height"]
    np.testing.assert_array_equal(data.x, np.array([1.0, 2.0, 3.0]))
    np.testing.assert_array_equal(data.t, np.array([0.0, 86400.0, 172800.0]))
    np.testing.assert_array_equal(
        data.dtg_time,
        pd.to_datetime(times.to_datetimeindex()).to_pydatetime(),
    )
