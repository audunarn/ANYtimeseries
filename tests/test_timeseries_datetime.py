"""Tests for :class:`anyqats.ts.TimeSeries` datetime compatibility."""

from datetime import datetime

import numpy as np
import pandas as pd

from anyqats import TimeSeries


def _hours(n):
    """Return ``n`` hours as seconds."""

    return n * 3600.0


def test_timeseries_accepts_numpy_datetime64_array():
    base = np.datetime64("2024-01-01T00:00:00")
    t = base + np.arange(4) * np.timedelta64(1, "h")
    x = np.arange(4, dtype=float)

    ts = TimeSeries("np_datetime", t, x)

    assert isinstance(ts.dtg_ref, datetime)
    np.testing.assert_allclose(ts.t, np.array([_hours(i) for i in range(4)]))
    np.testing.assert_allclose(ts.x, x)
    dtg_time = ts.dtg_time
    assert isinstance(dtg_time[0], datetime)
    assert dtg_time[0] == datetime(2024, 1, 1, 0, 0)
    assert dtg_time[-1] == datetime(2024, 1, 1, 3, 0)


def test_timeseries_accepts_pandas_datetime_index_and_series():
    index = pd.date_range("2024-05-01", periods=5, freq="30min")
    data = pd.Series(np.linspace(0, 1, num=5), index=index)

    ts = TimeSeries("pd_series", data.index, data)

    assert ts.dtg_ref == index[0].to_pydatetime()
    np.testing.assert_allclose(
        ts.t,
        np.arange(5, dtype=float) * 1800.0,
    )
    np.testing.assert_allclose(ts.x, data.to_numpy(dtype=float))
    dtg_time = ts.dtg_time
    assert dtg_time.shape == index.shape
    assert all(isinstance(val, datetime) for val in dtg_time)
    assert dtg_time[-1] == index[-1].to_pydatetime()


def test_timeseries_rejects_empty_datetime_input():
    t = np.array([], dtype="datetime64[ns]")
    x = np.array([], dtype=float)

    try:
        TimeSeries("empty", t, x)
    except ValueError as err:
        assert "contain at least one value" in str(err)
    else:  # pragma: no cover - defensive
        raise AssertionError("Expected ValueError for empty time series")
