"""Integration tests for the fatigue damage helper used by the GUI."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from anyqats.fatigue.sn import SNCurve

from anytimes.fatigue import FatigueSeries, compute_fatigue_damage, summarize_damage

DATA_DIR = Path(__file__).resolve().parent


def _load_series(path: Path, label: str) -> FatigueSeries:
    data = np.loadtxt(path, delimiter=",", skiprows=2)
    duration = float(data[-1, 0])
    values = data[:, 1]
    return FatigueSeries(label=label, values=values, duration=duration, source_file=path.name)


def test_tension_fatigue_matches_orcaflex_reference() -> None:
    """The helper reproduces the provided OrcaFlex damage summary."""

    series = [
        _load_series(DATA_DIR / "ts_fatigue1.csv", "Load case 1"),
        _load_series(DATA_DIR / "ts_fatigue2.csv", "Load case 2"),
    ]

    # Each load case represents 230746.468 exposure hours (~26.35 years).
    exposure_hours = [230_746.468_010_831_36, 230_746.468_010_831_36]

    curve = SNCurve("T-N curve1", m1=3.0, a1=1000.0)
    results, logs = compute_fatigue_damage(
        series,
        exposure_hours,
        curve,
        scf=1.0,
        thickness=None,
        load_basis="tension",
        curve_type="tn",
        unit_factor=1e-3,  # Convert kN to MN
        reference_strength=8.0,  # 8000 kN expressed in MN
    )

    assert len(results) == 2
    assert len(logs) == 2

    total_damage = sum(result.damage for result in results)
    assert total_damage == pytest.approx(0.19182317093977239, rel=1e-6)
    assert all(result.total_cycles > 0 for result in results)


def test_damage_summary_matches_gui_expectations() -> None:
    """Summarized damage matches the reference image values."""

    series = [
        _load_series(DATA_DIR / "ts_fatigue1.csv", "Load case 1"),
        _load_series(DATA_DIR / "ts_fatigue2.csv", "Load case 2"),
    ]

    exposure_hours = [230_746.468_010_831_36, 230_746.468_010_831_36]

    curve = SNCurve("T-N curve1", m1=3.0, a1=1000.0)
    results, _ = compute_fatigue_damage(
        series,
        exposure_hours,
        curve,
        scf=1.0,
        thickness=None,
        load_basis="tension",
        curve_type="tn",
        unit_factor=1e-3,
        reference_strength=8.0,
    )

    summary = summarize_damage(results, exposure_hours)

    assert summary.total_damage == pytest.approx(0.19182317093977239, rel=1e-6)
    assert summary.total_exposure_hours == pytest.approx(sum(exposure_hours), rel=1e-12)
    assert summary.estimated_life_years == pytest.approx(
        (sum(exposure_hours) / 8760.0) / summary.total_damage, rel=1e-12
    )
