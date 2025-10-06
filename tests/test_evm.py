from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from anytimes import evm
from anytimes.evm import (
    calculate_extreme_value_statistics,
    cluster_exceedances,
    declustering_boundaries,
)

try:
    import pyextremes  # noqa: F401
except ImportError:  # pragma: no cover - optional dependency guard
    pyextremes = None


def _synthetic_series():
    rng = np.random.default_rng(42)
    size = 720
    base = rng.normal(0, 0.2, size)
    indices = rng.choice(size, size // 20, replace=False)
    excess = rng.standard_gamma(2.5, size=indices.size)
    base[indices] += 1.5 + excess
    return np.arange(size, dtype=float), base


def test_cluster_exceedances_matches_expected_profile():
    t, x = _synthetic_series()
    threshold = 1.2

    clusters = cluster_exceedances(x, threshold, "upper")


    assert clusters.size == 33

    expected_first = np.array(
        [
            3.255648,
            2.516285,
            4.036607,
            5.324857,
            5.090921,
            5.634556,
            4.319638,
            4.548014,

            4.114666,
            3.789485,

        ]
    )
    np.testing.assert_allclose(clusters[: expected_first.size], expected_first, rtol=0, atol=1e-6)


def test_cluster_exceedances_accepts_high_low_aliases():
    _, x = _synthetic_series()

    threshold_upper = 1.2
    peaks_upper = cluster_exceedances(x, threshold_upper, "upper")
    peaks_high = cluster_exceedances(x, threshold_upper, "high")
    np.testing.assert_allclose(peaks_high, peaks_upper)

    threshold_lower = -0.3
    peaks_lower = cluster_exceedances(x, threshold_lower, "lower")
    peaks_low = cluster_exceedances(x, threshold_lower, "low")
    np.testing.assert_allclose(peaks_low, peaks_lower)


def test_calculate_extreme_value_statistics_matches_known_values():
    t, x = _synthetic_series()
    threshold = 1.2

    res = calculate_extreme_value_statistics(
        t,
        x,
        threshold,
        tail="upper",
        confidence_level=90.0,
        n_bootstrap=10_000,
        rng=np.random.default_rng(2024),
    )


    assert res.exceedances.size == 33
    assert np.isclose(res.shape, -0.7942124974671382)
    assert np.isclose(res.scale, 4.8572359369876255)

    expected_levels = np.array([6.65656619, 7.1321767, 7.20990746, 7.27154184, 7.28629789])
    expected_lower = np.array([5.93958204, 6.82033183, 7.01422591, 7.17422058, 7.20959305])
    expected_upper = np.array([6.97593568, 7.35651581, 7.45919467, 7.57530795, 7.62314262])


    np.testing.assert_allclose(res.return_levels, expected_levels, rtol=0, atol=1e-6)
    np.testing.assert_allclose(res.lower_bounds, expected_lower, rtol=0, atol=1e-6)
    np.testing.assert_allclose(res.upper_bounds, expected_upper, rtol=0, atol=1e-6)
    assert res.engine == "builtin"



def test_declustering_boundaries_include_extremes():
    x = np.array([0.2, -0.1, 0.3, -0.4, 0.5])
    bounds = declustering_boundaries(x, "upper")
    assert bounds[0] == 0
    assert bounds[-1] == x.size


def test_return_levels_zero_shape_limit_matches_log_expression():
    threshold = 3.0
    scale = 1.5
    shape = 0.0
    exceed_rate = 4.0
    durations = np.array([0.25, 1.0, 2.0])

    expected = threshold + scale * np.log(exceed_rate * durations)
    calculated = evm._return_levels(
        threshold=threshold,
        scale=scale,
        shape=shape,
        exceedance_rate=exceed_rate,
        return_durations=durations,
        tail="upper",
    )

    np.testing.assert_allclose(calculated, expected, rtol=0, atol=1e-12)


def test_return_levels_lower_tail_reflects_negative_extremes():
    threshold = -1.0
    scale = 0.8
    shape = -0.2
    exceed_rate = 6.0
    durations = np.array([0.5, 1.5])

    expected = threshold - (scale / shape) * (
        np.power(exceed_rate * durations, shape) - 1.0
    )
    calculated = evm._return_levels(
        threshold=threshold,
        scale=scale,
        shape=shape,
        exceedance_rate=exceed_rate,
        return_durations=durations,
        tail="lower",
    )

    np.testing.assert_allclose(calculated, expected, rtol=0, atol=1e-12)


@pytest.mark.parametrize(
    (
        "column",
        "tail",
        "threshold",
        "expected",
    ),
    [
        (
            "In-frame connection moment",
            "upper",
            41_000.0,
            {
                "exceedances": 48,
                "shape": -0.11058,
                "scale": 5628.31,
                "return_level": 58_724.77085556258,
                "lower_bound": 54_840.982400358895,
                "upper_bound": 74_527.62784681482,
                "lower_tolerance": 1_200.0,
                "upper_tolerance": 15_000.0,
                "shape_se": 0.18128,
                "scale_se": 1301.11,
            },
        ),
        (
            "In-frame connection GY moment",
            "upper",
            38_630.0,
            {
                "exceedances": 21,
                "shape": -0.0161,
                "scale": 4400.84,
                "return_level": 51_705.40634088432,
                "lower_bound": 47_471.09092177025,
                "upper_bound": 82_159.58909065329,
                "lower_tolerance": 3_500.0,
                "upper_tolerance": 25_000.0,
                "shape_se": 0.32253,
                "scale_se": 1713.67,
            },
        ),
        (
            "In-frame connection GY moment",
            "lower",
            -41_630.0,
            {
                "exceedances": 27,
                "shape": -0.01297,
                "scale": 5133.53,
                "return_level": -58_192.663151184766,
                "lower_bound": -89_306.37327445572,
                "upper_bound": -53_249.29580064948,
                "lower_tolerance": 20_000.0,
                "upper_tolerance": 4_000.0,
                "shape_se": 0.29053,
                "scale_se": 1788.95,
            },
        ),
    ],
)
def test_extreme_value_statistics_matches_orcaflex_reference(column, tail, threshold, expected):
    df = pd.read_excel(Path(__file__).with_name("ts_test.xlsx"))
    t = df["Time"].to_numpy()
    x = df[column].to_numpy()

    res = calculate_extreme_value_statistics(
        t,
        x,
        threshold=threshold,
        tail=tail,
        return_periods_hours=(3,),
        confidence_level=95.0,
        n_bootstrap=100_000,
        rng=np.random.default_rng(321),
        sample_exceedance_rate=True,
    )

    assert res.exceedances.size == expected["exceedances"]
    assert res.shape == pytest.approx(expected["shape"], abs=5e-5)
    assert res.scale == pytest.approx(expected["scale"], abs=0.2)

    assert res.return_levels[0] == pytest.approx(expected["return_level"], abs=50.0)
    assert abs(res.lower_bounds[0] - expected["lower_bound"]) <= expected["lower_tolerance"]
    assert abs(res.upper_bounds[0] - expected["upper_bound"]) <= expected["upper_tolerance"]

    if tail == "upper":
        excesses = res.exceedances - res.threshold
    else:
        excesses = res.threshold - res.exceedances

    covariance = evm._gpd_parameter_covariance(
        shape=res.shape,
        scale=res.scale,
        excesses=excesses,
    )
    assert covariance is not None
    se_shape = float(np.sqrt(covariance[0, 0]))
    se_scale = float(np.sqrt(covariance[1, 1]))

    assert se_shape == pytest.approx(expected["shape_se"], abs=5e-4)
    assert se_scale == pytest.approx(expected["scale_se"], abs=5.0)


@pytest.mark.skipif(pyextremes is None, reason="pyextremes is not installed")
def test_pyextremes_engine_returns_consistent_structure():
    t, x = _synthetic_series()
    threshold = 1.2

    res = calculate_extreme_value_statistics(
        t,
        x,
        threshold,
        tail="upper",
        return_periods_hours=(0.5, 1.0, 2.0),
        confidence_level=90.0,
        engine="pyextremes",
        pyextremes_options={
            "r": 1.0,
            "return_period_size": "1h",
            "n_samples": 200,
        },
        rng=np.random.default_rng(1234),
    )

    assert res.engine == "pyextremes"
    assert res.exceedances.size >= 10
    assert res.return_levels.shape == (3,)
    assert np.all(np.isfinite(res.return_levels))
    assert np.all(np.isfinite(res.lower_bounds))
    assert np.all(np.isfinite(res.upper_bounds))
    assert np.isfinite(res.shape)
    assert np.isfinite(res.scale)
    assert res.metadata is not None
    assert "distribution" in res.metadata
    assert res.metadata.get("plotting_position") == "weibull"
    assert res.metadata.get("diagnostic_return_periods") == (0.5, 1.0, 2.0)


@pytest.mark.skipif(pyextremes is None, reason="pyextremes is not installed")
def test_pyextremes_engine_accepts_low_tail_alias():
    t, x = _synthetic_series()
    threshold = -0.3

    base_kwargs = dict(
        t=t,
        x=x,
        threshold=threshold,
        return_periods_hours=(0.5, 1.5),
        confidence_level=85.0,
        engine="pyextremes",
        pyextremes_options={
            "r": 1.0,
            "return_period_size": "1h",
            "n_samples": 150,
        },
    )

    res_lower = calculate_extreme_value_statistics(
        tail="lower",
        rng=np.random.default_rng(999),
        **base_kwargs,
    )
    res_low = calculate_extreme_value_statistics(
        tail="low",
        rng=np.random.default_rng(999),
        **base_kwargs,
    )

    np.testing.assert_allclose(res_low.return_levels, res_lower.return_levels)
    np.testing.assert_allclose(res_low.lower_bounds, res_lower.lower_bounds)
    np.testing.assert_allclose(res_low.upper_bounds, res_lower.upper_bounds)
    assert res_low.metadata is not None
    assert res_low.metadata.get("extremes_type") == "low"


@pytest.mark.skipif(pyextremes is None, reason="pyextremes is not installed")
def test_pyextremes_engine_accepts_plotting_position_selection():
    t, x = _synthetic_series()
    threshold = 1.2

    res = calculate_extreme_value_statistics(
        t,
        x,
        threshold,
        tail="upper",
        return_periods_hours=(0.5, 1.0, 2.0),
        confidence_level=90.0,
        engine="pyextremes",
        pyextremes_options={
            "r": 1.0,
            "return_period_size": "1h",
            "n_samples": 200,
            "plotting_position": "median",
        },
        rng=np.random.default_rng(4321),
    )

    assert res.metadata is not None
    assert res.metadata.get("plotting_position") == "median"


@pytest.mark.skipif(pyextremes is None, reason="pyextremes is not installed")
def test_pyextremes_engine_allows_default_diagnostic_return_periods():
    t, x = _synthetic_series()
    threshold = 1.2

    res = calculate_extreme_value_statistics(
        t,
        x,
        threshold,
        tail="upper",
        return_periods_hours=(0.5, 1.0, 2.0),
        confidence_level=90.0,
        engine="pyextremes",
        pyextremes_options={
            "r": 1.0,
            "return_period_size": "1h",
            "n_samples": 200,
            "diagnostic_return_periods": None,
        },
        rng=np.random.default_rng(5678),
    )

    assert res.metadata is not None
    assert res.metadata.get("diagnostic_return_periods") is None


@pytest.mark.skipif(pyextremes is None, reason="pyextremes is not installed")
def test_pyextremes_engine_rejects_invalid_plotting_position():
    t, x = _synthetic_series()

    with pytest.raises(ValueError):
        calculate_extreme_value_statistics(
            t,
            x,
            threshold=1.2,
            tail="upper",
            engine="pyextremes",
            pyextremes_options={"plotting_position": "invalid"},
        )

