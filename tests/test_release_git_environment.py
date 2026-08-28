from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
VERIFIER = ROOT / "tools" / "verify_release_authority.py"


def _load_verifier():
    spec = importlib.util.spec_from_file_location(
        "_release_git_environment", VERIFIER
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_release_verifier_scrubs_inherited_git_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_verifier()
    for key in (
        "GIT_CONFIG",
        "GIT_CONFIG_COUNT",
        "GIT_CONFIG_PARAMETERS",
        "GIT_CONFIG_KEY_0",
        "GIT_CONFIG_VALUE_0",
    ):
        monkeypatch.setenv(key, "attacker-controlled")
    environment = module._git_environment()
    assert not {
        "GIT_CONFIG",
        "GIT_CONFIG_COUNT",
        "GIT_CONFIG_PARAMETERS",
        "GIT_CONFIG_KEY_0",
        "GIT_CONFIG_VALUE_0",
    } & set(environment)
    assert environment["GIT_CONFIG_NOSYSTEM"] == "1"
    assert environment["GIT_CONFIG_GLOBAL"]
    assert environment["GIT_ATTR_NOSYSTEM"] == "1"
    assert environment["GIT_NO_REPLACE_OBJECTS"] == "1"


def test_release_verifier_closes_external_git_authority() -> None:
    source = VERIFIER.read_text(encoding="utf-8")
    for token in (
        '"core.attributesFile="',
        '"--no-ext-diff"',
        '"refs/replace"',
        '"info/grafts"',
        '"info/attributes"',
        "Git replacement objects are forbidden",
        "Git grafts are forbidden",
        "Git info attributes are forbidden",
    ):
        assert token in source

