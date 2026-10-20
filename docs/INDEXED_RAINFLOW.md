# Indexed rainflow — living task record

## Scope and prospective acceptance, 2026-10-20

User authorized changes to ANYtimeseries and ANYfatigue. ANYtimeseries owns the
sole counting core; ANYfatigue owns the explicitly versioned shear association
and fatigue interpretation. Preserve legacy numerical output on finite valid
records. Add indexed individual cycles and a chunkable state with one explicit
finalization per independent finite record. No silent drop of invalid samples.
No periodic counting claim, package publication, version bump or main merge.

Base: 7eea1ca053c7d2b77e23d9c85d2ca56a63cae6bf. Temporary read-only snapshot
workflow will be removed. The workstation is offline; its global instructions
were unavailable and its files are untouched. Repository guides and the
accessible ANYopenSoft philosophy apply.

Acceptance: frozen pre-change scalar fixtures, published/hand-worked cases,
plateaus, nested and equal ranges, full/initial-half/residual-half identification,
all chunk partitions of short records, random chunk parity, invalid-input atomic
rejection, installed-wheel tests and owner/consumer CI. No results yet.

Principal uncertainty: a turning-point pair is not a full chronological loop.
The owner will retain the closing reversal as well as the pair. The consumer
will find first return to the starting normal level on the original sampled
history; it must retain between-reversal shear extrema and fractional closure.

## Implemented public contract

```python
from anyqats.fatigue.rainflow import IndexedRainflowCounter, count_cycles_indexed

records = count_cycles_indexed([0, 4, 2, 3, 1, 5, 0], endpoints=True)
for cycle in records:
    print(cycle.range, cycle.mean, cycle.count,
          cycle.start_index, cycle.end_index, cycle.closing_index, cycle.kind)

counter = IndexedRainflowCounter(endpoints=True)
a = counter.process([0, 4, 2])
b = counter.process([3, 1, 5, 0])
c = counter.finalize()
assert a + b + c == records
```

`INDEXED_CYCLES_VERSION == 1`. Records are immutable dataclasses in detection
order. Their indices refer to the original concatenated samples, not a reduced
reversal array or range-sorted table. The turning point is `end_index`; the
closing reversal is `closing_index`. The latter is not necessarily the instant
where the original stress first returns to the starting level. Full cycles have
count 1 and kind `full`; boundary halves have count 0.5 and kind `initial_half`
or `residual_half`. Final residuals have no closing reversal (None). Zero-range
records are retained, but are not fatigue cycles.

The counter maintains pending extreme/plateau state across chunks. Internal
plateaus use the first occurrence; the final endpoint is the actual final sample.
Chunk boundaries are never endpoints. A second finalize or processing after
finalization fails. Start a new counter for an independent record. Empty chunks
are harmless; empty/single-sample records have no cycles. No periodic convention
is implemented or inferred by appending an endpoint.

The legacy `cycles`/`count_cycles` functions now consume this same core, retaining
full-before-half grouping and the old range/mean sorting. There is no second
production counter. Frozen pre-change fixtures preserve successful ordinary-scale
behaviour. Two old edge cases are intentionally corrected: construction of empty
cycle tables now returns shape (0,3), and sign comparisons replace multiplication
of slopes, avoiding numerical underflow at very small scales. Invalid/non-finite
or complex histories are rejected, not silently cleaned. No noise threshold is
introduced. Finite samples with an unrepresentable float64 range are rejected.

Counting is amortized O(N); streaming state grows with unresolved reversals and
the current chunk/output, not all raw samples. The convenience batch function
validates/holds its full input and optionally splits it into computational chunks.

## Evidence

Local Linux/Python 3.13.5: **73 passed** for the new 71-case indexed suite plus the
existing two fatigue-tool tests (including the retained OrcaFlex reference).
A bounded external-to-the-new-core comparison against the unmodified original
file passed **6000** exact numerical comparisons and corresponding chunk checks
on deterministic random integer/normal records. The shipped frozen fixtures name
source commit and blob identity. Tests also include hand-worked nested indices,
every partition of short plateau/nested records, variation invariants, invalid
chunk rollback, overflow and lifecycle guards.

The broader local application run could not establish a full-suite result: Qt
was absent, and NetCDF tests lacked `cftime`; that diagnostic was stopped after
the first missing-dependency failure. No application tests or tolerances were
weakened. The added CI runs the affected numerical tests over its stated matrix
and a separate complete application suite with project dependencies. Exact CI
and installed-wheel results belong to the PR evidence, not this pre-CI note.

Independent design reference: pyLife's documented detector/recorder split and
original-sample indexing (including first-plateau convention),
https://pylife.readthedocs.io/en/stable/stress/rainflow.html . No pyLife source or
production dependency was copied. The numerical oracle also includes the
ASTM example already documented in the owner's legacy rainflow module.

Implementation evidence is not experimental fatigue-life qualification. No
version bump, release/tag, publication or main merge is performed in this task.
