"""Tests for CSV import behavior in :class:`anyqats.tsdb.TsDB`."""

import sys
import types

import numpy as np



def test_tsdb_csv_loader_skips_string_columns(tmp_path, monkeypatch):
    for name in ["h5py", "pymatreader", "nptdms"]:
        if name not in sys.modules:
            module = types.ModuleType(name)
            if name == "pymatreader":
                module.read_mat = lambda *args, **kwargs: {}
            if name == "nptdms":
                module.TdmsFile = type("TdmsFile", (), {})
            monkeypatch.setitem(sys.modules, name, module)

    from anyqats.tsdb import TsDB

    csv_path = tmp_path / "mixed_columns.csv"
    csv_path.write_text(
        "time,hs,tp,source_file\n"
        "16/03/2023 18:00:00,0.027045375,1.3719087,mywavewam800_nordnorge.an.2023031618.nc\n"
        "16/03/2023 19:00:00,0.02169195,1.2471896,mywavewam800_nordnorge.an.2023031618.nc\n",
        encoding="utf-8",
    )

    tsdb = TsDB()
    tsdb.load(str(csv_path), read=False)

    names = tsdb.list(relative=True, display=False)
    assert names == ["hs", "tp"]

    data = tsdb.getm()
    assert set(data) == {"hs", "tp"}
    np.testing.assert_allclose(data["hs"].x, np.array([0.027045375, 0.02169195]))
    np.testing.assert_allclose(data["tp"].x, np.array([1.3719087, 1.2471896]))


def test_tsdb_csv_loader_supports_long_met_format(tmp_path, monkeypatch):
    for name in ["h5py", "pymatreader", "nptdms"]:
        if name not in sys.modules:
            module = types.ModuleType(name)
            if name == "pymatreader":
                module.read_mat = lambda *args, **kwargs: {}
            if name == "nptdms":
                module.TdmsFile = type("TdmsFile", (), {})
            monkeypatch.setitem(sys.modules, name, module)

    from anyqats.tsdb import TsDB

    csv_path = tmp_path / "met_long.csv"
    csv_path.write_text(
        "elementId\tvalue\tunit\tlevel\ttimeOffset\ttimeResolution\ttimeSeriesId\tperformanceCategory\texposureCategory\tqualityCode\treferenceTime\tsourceId\n"
        "wind_from_direction\t330\tdegrees\t{'levelType': 'height_above_ground', 'unit': 'm', 'value': 10}\tPT0H\tPT6H\t0\tC\t2\t2\t1976-04-20T00:00:00.000Z\tSN75550:0\n"
        "wind_speed\t11.3\tm/s\t{'levelType': 'height_above_ground', 'unit': 'm', 'value': 10}\tPT0H\tPT6H\t0\tC\t2\t2\t1976-04-20T00:00:00.000Z\tSN75550:0\n"
        "wind_from_direction\t320\tdegrees\t{'levelType': 'height_above_ground', 'unit': 'm', 'value': 10}\tPT0H\tPT6H\t0\tC\t2\t2\t1976-04-20T06:00:00.000Z\tSN75550:0\n"
        "wind_speed\t10.3\tm/s\t{'levelType': 'height_above_ground', 'unit': 'm', 'value': 10}\tPT0H\tPT6H\t0\tC\t2\t2\t1976-04-20T06:00:00.000Z\tSN75550:0\n",
        encoding="utf-8",
    )

    tsdb = TsDB()
    tsdb.load(str(csv_path), read=False)

    names = tsdb.list(relative=True, display=False)
    assert names == ["wind_from_direction", "wind_speed"]

    data = tsdb.getm()
    np.testing.assert_allclose(data["wind_from_direction"].x, np.array([330.0, 320.0]))
    np.testing.assert_allclose(data["wind_speed"].x, np.array([11.3, 10.3]))
    assert data["wind_speed"].dtg_time[0].isoformat().startswith("1976-04-20T00:00:00")


def test_tsdb_csv_loader_supports_tab_delimited_with_bom(tmp_path, monkeypatch):
    for name in ["h5py", "pymatreader", "nptdms"]:
        if name not in sys.modules:
            module = types.ModuleType(name)
            if name == "pymatreader":
                module.read_mat = lambda *args, **kwargs: {}
            if name == "nptdms":
                module.TdmsFile = type("TdmsFile", (), {})
            monkeypatch.setitem(sys.modules, name, module)

    from anyqats.tsdb import TsDB

    csv_path = tmp_path / "met_tab_bom.csv"
    csv_path.write_text(
        "\ufeffelementId\tvalue\tunit\treferenceTime\n"
        "wind_from_direction\t36\tdegrees\t2003-01-01T00:00:00.000Z\n"
        "wind_speed\t8.5\tm/s\t2003-01-01T00:00:00.000Z\n"
        "wind_from_direction\t43\tdegrees\t2003-01-01T01:00:00.000Z\n"
        "wind_speed\t9.0\tm/s\t2003-01-01T01:00:00.000Z\n",
        encoding="utf-8",
    )

    tsdb = TsDB()
    tsdb.load(str(csv_path), read=False)

    names = tsdb.list(relative=True, display=False)
    assert names == ["wind_from_direction", "wind_speed"]
    data = tsdb.getm()
    np.testing.assert_allclose(data["wind_from_direction"].x, np.array([36.0, 43.0]))
    np.testing.assert_allclose(data["wind_speed"].x, np.array([8.5, 9.0]))
