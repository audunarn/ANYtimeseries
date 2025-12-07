"""Unit tests for SWAN helper utilities."""

from pathlib import Path

from swan_helpers import ManualWaveConfig, NONSTAT_DT, write_manual_run_configuration


def test_nonstat_dt_defined():
    """The NONSTAT_DT constant should be available for GUI callers."""

    assert isinstance(NONSTAT_DT, int)
    assert NONSTAT_DT > 0


def test_manual_wave_off_quad_label():
    """Zero or missing wind values should yield the OFF QUAD directive."""

    config = ManualWaveConfig(
        wave_height=2.5,
        peak_period=10.0,
        direction=270.0,
        wind_speed=0.0,
        physics="GEN3",
    )

    assert config.wind_label == "OFF QUAD"


def test_write_manual_run_configuration(tmp_path: Path):
    """Manual mode should emit a readable configuration summary."""

    config = ManualWaveConfig(
        wave_height=1.2,
        peak_period=8.5,
        direction=180.0,
        wind_speed=5.0,
        physics="WESTHUYSEN",
        flat_bottom_depth=12.0,
    )

    output_file = tmp_path / "manual" / "manual_run.txt"
    write_manual_run_configuration(config, output_file)

    contents = output_file.read_text()
    assert "Significant wave height (Hm0): 1.20 m" in contents
    assert "Peak period (Tp): 8.50 s" in contents
    assert "Wind: 5.00 m/s aligned with waves" in contents
    assert "Physics: WESTHUYSEN" in contents
    assert "Flat bottom: 12.00 m" in contents
