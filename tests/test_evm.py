import numpy as np

from anytimes import evm
from anytimes.evm import (
    calculate_extreme_value_statistics,
    cluster_exceedances,
    declustering_boundaries,
)


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


def test_calculate_extreme_value_statistics_matches_known_values():
    t, x = _synthetic_series()
    threshold = 1.2

    res = calculate_extreme_value_statistics(
        t,
        x,
        threshold,
        tail="upper",
        confidence_level=90.0,
        n_bootstrap=200,
        rng=np.random.default_rng(2024),
    )

    assert res.exceedances.size == 33
    assert np.isclose(res.shape, -0.7942124974671382)
    assert np.isclose(res.scale, 4.8572359369876255)

    expected_levels = np.array([6.65656619, 7.1321767, 7.20990746, 7.27154184, 7.28629789])
    expected_lower = np.array([5.85069148, 6.00889926, 6.01775253, 6.03363989, 6.04274598])
    expected_upper = np.array([7.09042912, 7.22463645, 7.26022048, 7.35106889, 7.38627693])

    np.testing.assert_allclose(res.return_levels, expected_levels, rtol=0, atol=1e-6)
    np.testing.assert_allclose(res.lower_bounds, expected_lower, rtol=0, atol=1e-6)
    np.testing.assert_allclose(res.upper_bounds, expected_upper, rtol=0, atol=1e-6)


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
