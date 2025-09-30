import numpy as np

from anytimes.evm import calculate_extreme_value_statistics, cluster_exceedances


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

    assert clusters.size == 35
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
            3.063981,
            4.114666,
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

    assert res.exceedances.size == 35
    assert np.isclose(res.shape, -0.7473962018962973)
    assert np.isclose(res.scale, 4.591897801701656)

    expected_levels = np.array([6.6211765, 7.12681975, 7.21457406, 7.28698103, 7.30503227])
    expected_lower = np.array([6.01020245, 6.18871786, 6.19542769, 6.19878085, 6.1993328])
    expected_upper = np.array([7.08325542, 7.22327264, 7.28170387, 7.40015782, 7.43073501])

    np.testing.assert_allclose(res.return_levels, expected_levels, rtol=0, atol=1e-6)
    np.testing.assert_allclose(res.lower_bounds, expected_lower, rtol=0, atol=1e-6)
    np.testing.assert_allclose(res.upper_bounds, expected_upper, rtol=0, atol=1e-6)
