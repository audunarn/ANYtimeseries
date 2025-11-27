import pytest

from anyqats.era5_to_swan_utils import (
    MisalignedBoundaryError,
    calculate_misalignment,
    validate_boundary_alignment,
)


def test_calculate_misalignment_wraps_angles():
    wind = [350, 10]
    wave = [10, 350]
    diff = calculate_misalignment(wind, wave)
    assert diff.tolist() == [20, 20]


def test_validate_boundary_alignment_passes_for_aligned_inputs():
    boundary_wind = {"west": [270, 280, 290], "east": [90, 95, 100]}
    boundary_wave = {"west": [265, 285, 295], "east": [92, 93, 101]}

    # Should not raise
    validate_boundary_alignment(boundary_wind, boundary_wave, tolerance=15)


def test_validate_boundary_alignment_raises_for_misaligned_inputs():
    boundary_wind = {"north": [0, 45, 90]}
    boundary_wave = {"north": [180, 225, 270]}

    with pytest.raises(MisalignedBoundaryError) as exc:
        validate_boundary_alignment(boundary_wind, boundary_wave, tolerance=60)

    assert "Tolerance=60.0" in str(exc.value)
    assert "north" in str(exc.value)


def test_validate_boundary_alignment_rejects_mismatched_keys():
    boundary_wind = {"north": [0], "south": [180]}
    boundary_wave = {"north": [0], "east": [90]}

    with pytest.raises(ValueError):
        validate_boundary_alignment(boundary_wind, boundary_wave)


def test_validate_boundary_alignment_reports_success(tmp_path):
    boundary_wind = {"west": [270, 275]}
    boundary_wave = {"west": [268, 280]}
    report_file = tmp_path / "alignment_report.txt"

    validate_boundary_alignment(
        boundary_wind, boundary_wave, tolerance=15, report_path=report_file
    )

    report = report_file.read_text()
    assert "passed" in report.lower()
    assert "west" in report


def test_validate_boundary_alignment_reports_failure(tmp_path):
    boundary_wind = {"east": [90]}
    boundary_wave = {"east": [210]}
    report_file = tmp_path / "alignment_report.txt"

    with pytest.raises(MisalignedBoundaryError):
        validate_boundary_alignment(
            boundary_wind, boundary_wave, tolerance=30, report_path=report_file
        )

    report = report_file.read_text()
    assert "failed" in report.lower()
    assert "east[0]" in report


def test_validate_boundary_alignment_creates_report_directory(tmp_path):
    boundary_wind = {"south": [180, 182]}
    boundary_wave = {"south": [175, 185]}
    nested_report = tmp_path / "nested" / "reports" / "alignment.txt"

    validate_boundary_alignment(
        boundary_wind,
        boundary_wave,
        tolerance=10,
        report_path=nested_report,
    )

    assert nested_report.exists()
    assert "passed" in nested_report.read_text().lower()
