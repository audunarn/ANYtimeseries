import numpy as np
import xarray as xr

from env_prediction.norkyst800_data import _pick_coords


def test_pick_coords_supports_lat_lon_aliases_from_dataset_variables():
    ds = xr.Dataset(
        data_vars={
            "wind_speed_10m": (("y", "x"), np.ones((2, 3))),
            "lat": (("y", "x"), np.array([[64.9, 64.9, 64.9], [65.0, 65.0, 65.0]])),
            "lon": (("y", "x"), np.array([[11.0, 11.1, 11.2], [11.0, 11.1, 11.2]])),
        }
    )

    lon, lat = _pick_coords(ds, "wind_speed_10m")

    assert lon.shape == (2, 3)
    assert lat.shape == (2, 3)
    np.testing.assert_allclose(lon[0], [11.0, 11.1, 11.2])


def test_pick_coords_supports_latitude_longitude_coords():
    ds = xr.Dataset(
        data_vars={"mwd": (("time", "latitude", "longitude"), np.ones((2, 1, 1)))},
        coords={"time": [0, 1], "latitude": [60.0], "longitude": [5.0]},
    )

    lon, lat = _pick_coords(ds, "mwd")

    np.testing.assert_allclose(lon, [5.0])
    np.testing.assert_allclose(lat, [60.0])
