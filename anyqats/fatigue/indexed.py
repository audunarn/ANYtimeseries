"""Indexed, chunkable form of the owner's ASTM three-point rainflow counter.

The closing reversal is *not* the physical return instant. Downstream consumers
can locate that instant on the original samples. See docs/INDEXED_RAINFLOW.md.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from numbers import Integral
from typing import Optional, Tuple

import numpy as np

INDEXED_CYCLES_VERSION = 1
__all__ = ["INDEXED_CYCLES_VERSION", "IndexedCycle", "IndexedRainflowCounter", "count_cycles_indexed"]


@dataclass(frozen=True)
class IndexedCycle:
    """One loop in detection order, not aggregated or sorted.

    Indices are zero-based in the concatenated input. ``end_index`` is the
    turning reversal, not full-loop closure. ``closing_index`` is the reversal
    that established closure (None for a final residual). Internal plateaus use
    their first sample; the final endpoint is the actual last input sample.
    Zero-range cycles are retained for compatibility, with no fatigue meaning.
    """
    range: float
    mean: float
    count: float
    start_index: int
    end_index: int
    closing_index: Optional[int]
    kind: str  # full, initial_half, residual_half


@dataclass(frozen=True)
class _Reversal:
    index: int
    value: float


def _samples(series):
    try:
        raw = np.asarray(series if isinstance(series, (np.ndarray, list, tuple)) else list(series))
        if raw.ndim != 1 or np.iscomplexobj(raw) or raw.dtype.kind in "bSU":
            raise ValueError
        values = np.asarray(raw, dtype=float)
    except (TypeError, ValueError, OverflowError):
        raise ValueError("rainflow samples must be a one-dimensional real numeric sequence") from None
    if not np.all(np.isfinite(values)):
        raise ValueError("rainflow samples must be finite; missing samples must not be dropped")
    return values


class IndexedRainflowCounter:
    """State for one independent finite record, fed in arbitrary-sized chunks.

    ``process`` returns newly detected cycles. Call ``finalize`` exactly once
    to include the last endpoint (when requested) and emit residual half cycles.
    Chunk boundaries never act as endpoints. Storage is O(unresolved reversals),
    plus the validated input chunk and emitted cycles, not the full raw record.
    Invalid chunks are rejected before mutating state. Empty chunks are allowed.
    """
    def __init__(self, *, endpoints=False):
        if not isinstance(endpoints, (bool, np.bool_)):
            raise ValueError("endpoints must be boolean")
        self._endpoints = bool(endpoints)
        self._points = deque()
        self._n = 0
        self._last = None
        self._extreme_index = 0
        self._slope = 0
        self._minimum = math.inf
        self._maximum = -math.inf
        self._finalized = False

    @property
    def endpoints(self):
        return self._endpoints

    @property
    def sample_count(self):
        return self._n

    @property
    def finalized(self):
        return self._finalized

    @property
    def residual_indices(self):
        """Confirmed unresolved reversals; the pending last extreme is separate."""
        return tuple(p.index for p in self._points)

    def _cycle(self, a, b, count, closing, kind):
        total = a.value + b.value
        mean = 0.5 * total if math.isfinite(total) else 0.5*a.value + 0.5*b.value
        return IndexedCycle(abs(b.value-a.value), mean, count, a.index, b.index, closing, kind)

    def _push(self, point, out):
        self._points.append(point)
        while len(self._points) >= 3:
            a, b, c = self._points[-3], self._points[-2], self._points[-1]
            if abs(b.value-c.value) < abs(a.value-b.value):
                break
            if len(self._points) == 3:
                out.append(self._cycle(a, b, 0.5, c.index, "initial_half"))
                self._points.popleft()
            else:
                out.append(self._cycle(a, b, 1.0, c.index, "full"))
                last = self._points.pop()
                self._points.pop()
                self._points.pop()
                self._points.append(last)

    def process(self, series) -> Tuple[IndexedCycle, ...]:
        if self._finalized:
            raise ValueError("rainflow record is finalized; create a new counter for a new record")
        values = _samples(series)
        if not len(values):
            return ()
        low = min(self._minimum, float(values.min()))
        high = max(self._maximum, float(values.max()))
        if not math.isfinite(high-low):
            raise ValueError("rainflow range overflows float64; rescale the input")
        out = []
        for offset, value in enumerate(values):
            value = float(value)
            index = self._n + offset
            if self._last is None:
                if self.endpoints:
                    self._points.append(_Reversal(index, value))
                self._extreme_index = index
            elif value != self._last:
                slope = 1 if value > self._last else -1
                if self._slope and slope != self._slope:
                    self._push(_Reversal(self._extreme_index, self._last), out)
                self._slope = slope
                self._extreme_index = index
            self._last = value
        self._n += len(values)
        self._minimum, self._maximum = low, high
        return tuple(out)

    def finalize(self) -> Tuple[IndexedCycle, ...]:
        if self._finalized:
            raise ValueError("rainflow record is already finalized")
        out = []
        if self.endpoints and self._n >= 2:
            self._push(_Reversal(self._n-1, self._last), out)
        # Preserve the legacy residual order (latest pair first).
        while len(self._points) > 1:
            out.append(self._cycle(self._points[-2], self._points[-1], 0.5, None, "residual_half"))
            self._points.pop()
        self._points.clear()
        self._finalized = True
        return tuple(out)


def count_cycles_indexed(series, endpoints=False, *, chunk_size=None) -> Tuple[IndexedCycle, ...]:
    """Count one finite record. ``chunk_size`` changes computation, not semantics.

    Empty/single-sample records have no cycles. Input is validated as a whole;
    use IndexedRainflowCounter directly to avoid retaining the full input.
    """
    values = _samples(series)
    if chunk_size is not None and (
        isinstance(chunk_size, (bool, np.bool_)) or not isinstance(chunk_size, Integral) or chunk_size <= 0
    ):
        raise ValueError("chunk_size must be a positive integer or None")
    counter = IndexedRainflowCounter(endpoints=endpoints)
    size = (len(values) or 1) if chunk_size is None else int(chunk_size)
    out = []
    for first in range(0, len(values), size):
        out.extend(counter.process(values[first:first+size]))
    out.extend(counter.finalize())
    return tuple(out)
