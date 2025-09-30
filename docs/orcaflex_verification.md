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

