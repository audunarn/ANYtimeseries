"""Utility helpers for parsing metadata encoded in filenames."""

from __future__ import annotations

import os
import re
from typing import Dict, Optional

_PROBABILITY_KEYS = {"probability", "prob", "probaility", "p"}
_EXPOSURE_KEYS = {"exposure", "exposure_time", "ex_t", "ex_time"}
_EXPOSURE_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])"
    r"(?P<key>probability|prob|probaility|p|exposure|exposure_time|ex_t|ex_time)"
    r"(?P<value>-?[0-9]+(?:_[0-9]+)*)",
    re.IGNORECASE,
)


def parse_general_filename(filename: str) -> dict:
    """Parse key/value pairs embedded in ``filename``.

    Examples of supported names::

        FSRU_Dir0_Hs0_3_Tp6_Uw2_5_Uc0_1_prob0_0047_fatigue.sim
        FSRU_Dir202.5_Hs2_3_Tp7_5_Uw12_Uc0_15_prob0_0118_fatigue.sim

    Any token of the form ``<letters><number or number.with.decimal>`` is
    parsed, and if the next token is pure digits it is taken as extra decimal
    places.  For instance ``Hs0_3`` becomes ``{"Hs": 0.3}``.
    """

    base = os.path.basename(filename)
    tokens = base.split("_")

    result = {}
    i = 0
    while i < len(tokens):
        token = tokens[i]

        # Match: letters + number (possibly with a dot), e.g. Dir202.5, Hs0, Tp6
        m = re.match(r"([A-Za-z]+)([0-9.]+)$", token)
        if not m:
            i += 1
            continue

        key, num_part = m.group(1), m.group(2)

        # If next token is pure digits, treat it as extra decimal places
        frac_part = None
        if i + 1 < len(tokens) and tokens[i + 1].isdigit():
            frac_part = tokens[i + 1]
            i += 1  # consume fractional token

        if frac_part:
            num_str = f"{num_part}.{frac_part}"
        else:
            num_str = num_part

        try:
            value = float(num_str)
        except ValueError:
            i += 1
            continue

        result[key] = value
        i += 1

    return result


def choose_parse_target(*candidates: Optional[str]) -> str:
    """Return the first non-empty candidate name for filename parsing."""

    for candidate in candidates:
        if candidate:
            return candidate
    return ""


def parse_embedded_values(name: str) -> Dict[str, float]:
    """Backward-compatible wrapper for the general filename parser."""

    if not name:
        return {}

    return parse_general_filename(name)


def exposure_hours_from_name(name: str, design_life_years: float) -> Optional[float]:
    """Return exposure hours encoded in ``name`` if possible.

    The parser looks for tokens that match one of the supported probability or
    exposure keys (e.g. ``prob``, ``probability``, ``exposure``).  If a
    probability is found the result is calculated as ``design_life_years * 365 *
    24 * probability``.  If an exposure token is found the numeric value is
    interpreted directly as hours.
    """

    if not name:
        return None

    design_life_years = max(float(design_life_years), 0.0)

    def _normalized_value(token: str) -> Optional[float]:
        token = token.strip()
        if not token:
            return None
        if "_" in token:
            integer, _, fractional = token.partition("_")
            fractional = fractional.replace("_", "")
            if fractional:
                token = f"{integer}.{fractional}"
            else:
                token = integer
        token = token.replace(",", "")
        try:
            return float(token)
        except ValueError:
            return None

    for match in _EXPOSURE_PATTERN.finditer(name):
        key = match.group("key").lower()
        value = _normalized_value(match.group("value"))
        if value is None:
            continue
        if key in _PROBABILITY_KEYS:
            return design_life_years * 365.0 * 24.0 * value
        if key in _EXPOSURE_KEYS:
            return value

    try:
        values = parse_general_filename(name)
    except Exception:  # pragma: no cover - defensive
        return None

    if not values:
        return None

    for key, value in values.items():
        lowered = key.lower()
        if lowered in _PROBABILITY_KEYS:
            return design_life_years * 365.0 * 24.0 * float(value)
        if lowered in _EXPOSURE_KEYS:
            return float(value)

    return None


__all__ = [
    "choose_parse_target",
    "exposure_hours_from_name",
    "parse_embedded_values",
    "parse_general_filename",
]
