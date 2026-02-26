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
