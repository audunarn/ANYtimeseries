# Extreme-value verification against OrcaFlex

The maintained benchmark is implemented in
[`tests/test_evm.py`](../tests/test_evm.py) using the
`In-frame connection GY moment` signal in `tests/ts_test_2.xlsx`. The tests fit
a Generalized Pareto distribution to declustered peaks and compare the 3-hour
return level, fitted parameters, confidence bounds, and parameter uncertainty
with OrcaFlex 11.5e reference values.

## Reference cases

| Confidence | Tail | Threshold (kN m) | Window (s) | Peaks | Shape | Scale | 3 h return level (kN m) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 57% | Upper | 35,000 | 12.00 | 56 | -0.19621 | 5,960.47 | 49,522.65 |
| 57% | Lower | -45,000 | 12.00 | 28 | 0.13727 | 3,216.08 | -55,143.75 |
| 95% | Upper | 38,500 | 15.75 | 30 | -0.21794 | 5,426.69 | 49,544.51 |
| 95% | Lower | -40,000 | 15.00 | 58 | -0.22085 | 6,394.12 | -55,133.34 |

The 15.75-second upper-tail window is intentional: maxima in the sampled
record occur slightly after the threshold up-crossings, and this setting
reproduces the OrcaFlex peak count.

The tests require the fitted shape and scale and the point return level to
match tightly. Bootstrap confidence limits use explicit, wider tolerances
because ANYtimeSeries samples both parameter covariance and cluster-rate
uncertainty; this is not the same interval construction used internally by
OrcaFlex. The optional pyextremes test checks structural and point-estimate
consistency without claiming identical confidence limits.

## Run the evidence

From the repository root:

```powershell
python -m pytest tests/test_evm.py -q
```

The test file is the source of truth for exact reference values, random seeds,
bootstrap counts, and tolerances. Do not copy stochastic implementation output
into this document as a fixed benchmark.
