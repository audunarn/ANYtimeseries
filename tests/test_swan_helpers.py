"""Unit tests for SWAN helper utilities."""

from swan_helpers import NONSTAT_DT


def test_nonstat_dt_defined():
    """The NONSTAT_DT constant should be available for GUI callers."""

    assert isinstance(NONSTAT_DT, int)
    assert NONSTAT_DT > 0
