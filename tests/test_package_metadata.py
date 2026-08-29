"""Public package metadata must identify the canonical distribution source."""

from __future__ import annotations

from pathlib import Path
import re

import anytimes


ROOT = Path(__file__).resolve().parents[1]


def _project_value(name: str) -> str:
    source = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(
        rf'^{re.escape(name)}\s*=\s*"([^"]+)"\s*$',
        source,
        flags=re.MULTILINE,
    )
    assert match is not None, name
    return match.group(1)


def test_public_runtime_version_matches_distribution_metadata() -> None:
    assert anytimes.__version__ == _project_value("version") == "1.0.1"
    assert "__version__" in anytimes.__all__


def test_project_homepage_is_the_canonical_repository() -> None:
    assert _project_value("Homepage") == (
        "https://github.com/audunarn/ANYtimeseries"
    )
