"""Indexed owner-contract tests; frozen pre-change evidence plus analytical cases."""
import itertools
import json
from pathlib import Path

import numpy as np
import pytest

from anyqats.fatigue.rainflow import (
    INDEXED_CYCLES_VERSION, IndexedRainflowCounter, count_cycles, count_cycles_indexed,
)

FIXTURES = json.loads((Path(__file__).parent/'data/indexed_legacy_reference.json').read_text())


@pytest.mark.parametrize('case', FIXTURES['cases'])
def test_frozen_legacy_numerical_output(case):
    expected = np.asarray(case['table']).reshape(-1, 3)
    np.testing.assert_array_equal(count_cycles(case['samples'], case['endpoints']), expected)
    records = count_cycles_indexed(case['samples'], case['endpoints'])
    rows = sorted([(c.range, c.mean, c.count) for c in records])
    assert rows == sorted(map(tuple, expected))


def test_nested_hand_worked_indices_and_kinds():
    records = count_cycles_indexed([0, 4, 2, 3, 1, 5, 0], endpoints=True)
    assert [(c.range,c.mean,c.count,c.start_index,c.end_index,c.closing_index,c.kind) for c in records] == [
        (1, 2.5, 1, 2, 3, 4, 'full'),
        (3, 2.5, 1, 1, 4, 5, 'full'),
        (5, 2.5, .5, 0, 5, 6, 'initial_half'),
        (5, 2.5, .5, 5, 6, None, 'residual_half'),
    ]
    assert INDEXED_CYCLES_VERSION == 1


@pytest.mark.parametrize('endpoints', [False, True])
@pytest.mark.parametrize('values', [
    [0,4,2,3,1,5,0], [0,0,3,3,1,1,2,2,0,0], [1]*8, [1,-1,1,-1,1], [0,1,2,3],
])
def test_every_chunk_partition_has_identical_events(values, endpoints):
    expected = count_cycles_indexed(values, endpoints)
    for bits in itertools.product((False, True), repeat=len(values)-1):
        counter = IndexedRainflowCounter(endpoints=endpoints)
        rows, first = [], 0
        for i, boundary in enumerate(bits, 1):
            if boundary:
                rows.extend(counter.process(values[first:i])); first=i
                assert counter.process([]) == ()
        rows.extend(counter.process(values[first:])); rows.extend(counter.finalize())
        assert tuple(rows) == expected
        assert counter.finalized and counter.sample_count == len(values)


def test_internal_plateaus_and_final_endpoint_use_documented_indices():
    records = count_cycles_indexed([0,4,4,2,2,3,3,1,1,5,5,0,0], True)
    assert [(c.start_index,c.end_index,c.closing_index) for c in records] == [
        (3,5,7), (1,7,9), (0,9,12), (9,12,None),
    ]


def test_random_record_invariants_and_chunked_indices():
    rng = np.random.default_rng(56120)
    for _ in range(120):
        x = rng.integers(-50,51,size=250).astype(float)
        records = count_cycles_indexed(x, True)
        assert records == count_cycles_indexed(x, True, chunk_size=13)
        assert sum(2*c.count*c.range for c in records) == pytest.approx(np.abs(np.diff(x)).sum(), rel=1e-14)
        for c in records:
            assert c.range == abs(x[c.start_index]-x[c.end_index])
            assert c.mean == (x[c.start_index]+x[c.end_index])/2
            if c.closing_index is not None:
                assert c.start_index < c.end_index < c.closing_index


@pytest.mark.parametrize('values', [[], [1], [1,1], [1,1,1]])
def test_short_and_constant_records(values):
    assert all(c.range == 0 for c in count_cycles_indexed(values, True))
    assert count_cycles(values, False).shape == (0,3)


@pytest.mark.parametrize('bad', [[1,np.nan,2], [np.inf], [[1,2]], [1+2j], ['1','2'], [True], 2])
def test_bad_chunk_is_rejected_without_changing_state(bad):
    c = IndexedRainflowCounter(endpoints=True)
    before = c.process([0,4,2])
    state = (c.sample_count, c.residual_indices)
    with pytest.raises(ValueError):
        c.process(bad)
    assert (c.sample_count,c.residual_indices) == state
    after = c.process([3,1,5,0])+c.finalize()
    assert before+after == count_cycles_indexed([0,4,2,3,1,5,0], True)


def test_overflow_across_chunks_is_atomic():
    c = IndexedRainflowCounter(endpoints=True)
    c.process([1e308])
    with pytest.raises(ValueError, match='overflow'):
        c.process([-1e308])
    assert c.sample_count == 1
    assert c.process([1e308]) == ()
    assert c.finalize()[0].mean == 1e308


def test_sign_detection_does_not_multiply_tiny_slopes():
    records = count_cycles_indexed([0,1e-200,0,-1e-200,0], True)
    assert sum(2*c.count*c.range for c in records) == pytest.approx(4e-200, rel=1e-14, abs=0)


def test_finalization_does_not_invent_a_chunk_endpoint():
    c = IndexedRainflowCounter(endpoints=True)
    first = c.process([0,4,2])
    assert first == ()
    assert c.residual_indices == (0,1)
    rest = c.process([3,1,5,0])+c.finalize()
    assert first+rest == count_cycles_indexed([0,4,2,3,1,5,0], True)
    with pytest.raises(ValueError, match='finalized'): c.process([0])
    with pytest.raises(ValueError, match='finalized'): c.finalize()


@pytest.mark.parametrize('chunk', [0,-1,1.5,True,'3'])
def test_invalid_chunk_size(chunk):
    with pytest.raises(ValueError, match='chunk_size'): count_cycles_indexed([0,1,0],chunk_size=chunk)


def test_generator_input_and_read_only_endpoint_policy():
    assert count_cycles_indexed(iter([0,1,0]),True) == count_cycles_indexed([0,1,0],True)
    c = IndexedRainflowCounter(endpoints=True)
    with pytest.raises(AttributeError): c.endpoints = False
