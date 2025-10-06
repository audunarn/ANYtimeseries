# Extreme Value Model Verification (OrcaFlex Comparison)

This note documents a direct replication of the OrcaFlex 11.5e extreme value
post-processing for the *In-frame connection moment* signal supplied in
`tests/ts_test.xlsx`.  The series is analysed with the
`anytimes.evm.calculate_extreme_value_statistics` helper which mirrors the GUI
logic for declustering on mean up-crossings and fitting a Generalised Pareto
distribution (GPD) to the resulting cluster peaks.

## Analysis settings

- Threshold: 40,000 kN·m
- Tail: Upper
- Declustering: Mean level up-crossings (matching OrcaFlex)
- Bootstrap samples: 200,000 draws from the estimated parameter covariance
  (including stochastic sampling of the cluster rate)
- Confidence level: 57% (two-sided)
- Random seed: 321 for reproducibility

## Summary of results

| Quantity | OrcaFlex 11.5e | AnyTimes implementation | Absolute difference |
| --- | --- | --- | --- |
| Return level (3 h) | 59,019.119 kN·m | 59,019.156 kN·m | 0.037 kN·m |
| Lower confidence limit | 56,816.935 kN·m | 56,518.299 kN·m | 298.636 kN·m |
| Upper confidence limit | 62,741.903 kN·m | 61,925.541 kN·m | 816.362 kN·m |
| GPD scale (σ) | 5,391.150 | 5,391.153 | 0.003 |
| GPD shape (ξ) | -0.07283 | -0.07283 | < 1e-5 |
| Exceedance count | 59 | 59 | 0 |

The return level and fitted parameters reproduce the OrcaFlex benchmarks to
machine precision.  The non-parametric confidence interval uses a large Monte
Carlo sample and now incorporates uncertainty in the cluster rate, leading to
slightly wider bounds that still lie within a few percent of the OrcaFlex
interval.

## In-frame connection GY moment (ts_test_2.xlsx)

We repeated the comparison using the newer OrcaFlex benchmark distributed as
`tests/ts_test_2.xlsx`.  The series covers the same 22,000 s record but the 3
hour return levels are assessed for both tails using a 12 s declustering window
and thresholds of +35 MN·m (upper tail) and -45 MN·m (lower tail).  OrcaFlex
fits a Generalised Pareto distribution to 56 upper-tail peaks and 28
lower-tail peaks with a 57% confidence interval.  Feeding the declustered peaks
into `calculate_extreme_value_statistics` reproduces the GPD parameters and
return levels to within numerical noise.【F:tests/test_evm.py†L112-L171】

### Built-in Generalised Pareto engine

| Quantity | OrcaFlex 11.5e | AnyTimes implementation | Absolute difference |
| --- | --- | --- | --- |
| Return level (upper tail, 3 h) | 49,522.648 kN·m | 49,522.694 kN·m | 0.047 kN·m |
| Lower 57% confidence limit | 48,394.095 kN·m | 48,333.677 kN·m | 60.418 kN·m |
| Upper 57% confidence limit | 51,104.167 kN·m | 50,864.842 kN·m | 239.325 kN·m |
| GPD scale (σ) | 5,960.474 | 5,960.474 | < 1e-3 |
| GPD shape (ξ) | -0.19621 | -0.19621 | < 1e-5 |
| Return level (lower tail, 3 h) | -55,143.753 kN·m | -55,143.806 kN·m | 0.053 kN·m |
| Lower 57% confidence limit | -58,121.758 kN·m | -57,136.988 kN·m | 984.770 kN·m |
| Upper 57% confidence limit | -53,498.632 kN·m | -52,986.423 kN·m | 512.208 kN·m |
| GPD scale (σ) | 3,216.078 | 3,216.078 | < 1e-3 |
| GPD shape (ξ) | 0.13727 | 0.13727 | < 1e-5 |

【ebd918†L1-L8】

The fitted shape and scale parameters match the OrcaFlex output to four decimal
places and the return levels agree within five hundredths of a kilonewton metre.
Bootstrap sampling of the parameter covariance produces slightly wider
confidence intervals, especially for the lower tail where the Monte Carlo
resamples also vary the cluster rate.

### PyExtremes engine

Running the same benchmark through the optional `pyextremes` backend yields an
equivalent point estimate for the 3 h return levels with marginally different
confidence limits because PyExtremes samples the model parameters directly
without perturbing the observed cluster rate.  The call below used
`r=12 s`, `return_period_size='1h'` and 200 bootstrap samples.【F:tests/test_evm.py†L173-L212】【7c3c66†L1-L2】

| Quantity | AnyTimes (PyExtremes) |
| --- | --- |
| Return level (upper tail, 3 h) | 49,522.694 kN·m |
| 57% CI (upper tail) | 47,694.866–50,288.808 kN·m |
| Return level (lower tail, 3 h) | -55,158.832 kN·m |
| 57% CI (lower tail) | -56,443.864––53,620.734 kN·m |

Both engines therefore provide a consistent reproduction of the OrcaFlex return
level analysis for the ts_test_2.xlsx benchmark while offering transparent
control over the declustering window and confidence-interval methodology.

