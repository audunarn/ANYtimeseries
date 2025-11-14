"""Utility helpers for parsing metadata encoded in filenames."""

from __future__ import annotations

import os
import re
from typing import Dict

_KEY_VALUE_RE = re.compile(r"([A-Za-z]+)([-+]?(?:\d+(?:[._]\d+)*))")


def parse_embedded_values(name: str) -> Dict[str, float]:
    """Extract ``{"Name": value}`` pairs embedded within ``name``.

    The parser scans the filename (minus its extension) looking for repeating
    ``<letters><number>`` patterns.  Decimal separators may be either ``.`` or
    ``_`` since the latter is frequently used when filenames cannot contain a
    period.  ``name`` may optionally include a path.
    """

    if not name:
        return {}

    base = os.path.splitext(os.path.basename(name))[0]
    parsed: Dict[str, float] = {}
    for key, val in _KEY_VALUE_RE.findall(base):
        if not val:
            continue
        try:
            parsed[key] = float(val.replace("_", "."))
        except ValueError:
            continue
    return parsed
__all__ = ["parse_embedded_values"]
