import numpy as np

from anyqats.io.other import read_dat_data, read_dat_names


def test_read_openfoam_dat_names_and_data(tmp_path):
    path = tmp_path / "forces.dat"
    path.write_text(
        "# Force\n"
        "# CofR : (0 0 0)\n"
        "#\n"
        "# Time total_x total_y total_z pressure_x pressure_y pressure_z viscous_x viscous_y viscous_z\n"
        "0.1 1 2 3 4 5 6 7 8 9\n"
        "0.2 10 20 30 40 50 60 70 80 90\n",
        encoding="utf-8",
    )

    names = read_dat_names(str(path))
    assert names == [
        "total_x",
        "total_y",
        "total_z",
        "pressure_x",
        "pressure_y",
        "pressure_z",
        "viscous_x",
        "viscous_y",
        "viscous_z",
    ]

    data = read_dat_data(str(path), ind=[0, 1, 9])
    assert data.shape == (3, 2)
    assert np.allclose(data[0], [0.1, 0.2])
    assert np.allclose(data[1], [1.0, 10.0])
    assert np.allclose(data[2], [9.0, 90.0])
