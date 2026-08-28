from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import zipfile

import pytest


ROOT = Path(__file__).resolve().parents[1]
VERIFIER = ROOT / "tools" / "verify_release_authority.py"
DISTRIBUTION = "anytimes"
NORMALIZED = "anytimes"
VERSION = "1.0.1"
TAG = f"v{VERSION}"
EXPECTED_TERMINAL = "ACCEPTED_ANYTIMES_1_0_1_RELEASE"
WRONG_TAG = "v1.0.0"
WHEEL = f"{NORMALIZED}-{VERSION}-py3-none-any.whl"
SDIST = f"{NORMALIZED}-{VERSION}.tar.gz"
LEDGER = Path("docs/release") / f"{NORMALIZED}-{VERSION}-ledger.json"
CHECKOUT_ACTION = "actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1"
SETUP_ACTION = "actions/setup-python@5fda3b95a4ea91299a34e894583c3862153e4b97"
PUBLISH_ACTION = (
    "pypa/gh-action-pypi-publish@"
    "dc37677b2e1c63e2034f94d8a5b11f265b73ba33"
)


def _git(repository: Path, *arguments: str) -> str:
    completed = subprocess.run(
        [
            "git",
            "-c",
            "user.name=Release Authority Test",
            "-c",
            "user.email=release-authority@example.invalid",
            "-c",
            "commit.gpgsign=false",
            *arguments,
        ],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _metadata(distribution: str = DISTRIBUTION) -> bytes:
    return (
        "Metadata-Version: 2.1\n"
        f"Name: {distribution}\n"
        f"Version: {VERSION}\n\n"
    ).encode("utf-8")


def _write_wheel(
    path: Path,
    payload: bytes,
    *,
    distribution: str = DISTRIBUTION,
) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(f"{NORMALIZED}/__init__.py", payload)
        archive.writestr(
            f"{NORMALIZED}-{VERSION}.dist-info/METADATA",
            _metadata(distribution),
        )


def _write_sdist(path: Path) -> None:
    info = tarfile.TarInfo(f"{NORMALIZED}-{VERSION}/PKG-INFO")
    metadata = _metadata()
    info.size = len(metadata)
    with tarfile.open(path, "w:gz") as archive:
        archive.addfile(info, io.BytesIO(metadata))


def _write_checksums(assets: Path) -> None:
    text = "".join(
        f"{hashlib.sha256((assets / name).read_bytes()).hexdigest()}  {name}\n"
        for name in sorted((WHEEL, SDIST))
    )
    (assets / "SHA256SUMS").write_text(text, encoding="ascii", newline="\n")


def _run_verifier(tmp_path: Path, mutation: str = "") -> subprocess.CompletedProcess[str]:
    repository = tmp_path / "repository"
    remote = tmp_path / "origin.git"
    assets = tmp_path / "release-assets"
    repository.mkdir(parents=True)
    remote.mkdir()
    assets.mkdir()
    _git(repository, "init", "--quiet")
    _git(remote, "init", "--bare", "--quiet")
    (repository / "source.txt").write_text("frozen artifact source\n", encoding="utf-8")
    _git(repository, "add", "source.txt")
    _git(repository, "commit", "--quiet", "-m", "freeze artifact source")
    source_commit = _git(repository, "rev-parse", "HEAD")
    source_tree = _git(repository, "rev-parse", "HEAD^{tree}")
    _git(repository, "branch", "-M", "main")
    _git(repository, "remote", "add", "origin", str(remote))
    _git(repository, "push", "--quiet", "-u", "origin", "main")

    _write_wheel(assets / WHEEL, b"accepted build\n")
    if mutation == "wrong-metadata":
        _write_wheel(
            assets / WHEEL,
            b"accepted build\n",
            distribution="DifferentDistribution",
        )
    _write_sdist(assets / SDIST)
    artifact_rows = []
    for name in sorted((WHEEL, SDIST)):
        raw = (assets / name).read_bytes()
        artifact_rows.append(
            {
                "bytes": len(raw),
                "filename": name,
                "sha256": hashlib.sha256(raw).hexdigest().upper(),
            }
        )
    ledger = {
        "artifact_source": {"commit": source_commit, "tree": source_tree},
        "artifacts": artifact_rows,
        "distribution": DISTRIBUTION,
        "publication_authorized": True,
        "qualification": {
            "accepted_terminal": EXPECTED_TERMINAL,
            "evidence_sha256": "A" * 64,
            "independent_review_sha256": "B" * 64,
        },
        "schema": "anyecosystem.release-ledger-v1",
        "tag": TAG,
        "version": VERSION,
    }
    if mutation == "wrong-byte-count":
        ledger["artifacts"][0]["bytes"] += 1
    elif mutation == "wrong-terminal":
        ledger["qualification"]["accepted_terminal"] = "REJECTED_RELEASE"
    elif mutation == "evidence-hash":
        ledger["qualification"]["evidence_sha256"] = "0" * 64
    elif mutation == "review-hash":
        ledger["qualification"]["independent_review_sha256"] = "A" * 64
    if mutation == "wrong-source":
        ledger["artifact_source"]["tree"] = "0" * 40

    target = repository / LEDGER
    target.parent.mkdir(parents=True)
    if mutation == "noncanonical":
        target.write_text(json.dumps(ledger), encoding="utf-8")
    else:
        target.write_text(
            json.dumps(ledger, allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    _git(repository, "add", LEDGER.as_posix())
    if mutation == "extra-child-path":
        (repository / "unexpected.txt").write_text("not ledger-only\n", encoding="utf-8")
        _git(repository, "add", "unexpected.txt")
    _git(repository, "commit", "--quiet", "-m", "docs: authorize release artifacts")
    _git(repository, "tag", TAG)
    if mutation != "unmerged-tag-child":
        _git(repository, "push", "--quiet", "origin", "HEAD:main")

    _write_checksums(assets)
    invoked_tag = TAG
    if mutation == "paired-replacement":
        _write_wheel(assets / WHEEL, b"replacement build\n")
        _write_checksums(assets)
    elif mutation == "checksum":
        (assets / "SHA256SUMS").write_text(
            "0" * 64 + f"  {WHEEL}\n"
            + hashlib.sha256((assets / SDIST).read_bytes()).hexdigest()
            + f"  {SDIST}\n",
            encoding="ascii",
            newline="\n",
        )
    elif mutation == "extra-asset":
        (assets / "unregistered.txt").write_text("extra\n", encoding="utf-8")
    elif mutation == "tag":
        invoked_tag = WRONG_TAG

    return subprocess.run(
        [
            sys.executable,
            str(VERIFIER),
            "--repository-root",
            str(repository),
            "--ledger",
            LEDGER.as_posix(),
            "--assets",
            str(assets),
            "--output",
            str(tmp_path / "dist"),
            "--tag",
            invoked_tag,
            "--protected-ref",
            "refs/remotes/origin/main",
            "--expected-terminal",
            EXPECTED_TERMINAL,
            "--distribution",
            DISTRIBUTION,
            "--version",
            VERSION,
            "--artifact",
            WHEEL,
            "--artifact",
            SDIST,
        ],
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
    )


def test_production_workflow_uses_immutable_ledger_authority() -> None:
    production = (ROOT / ".github/workflows/publish-release-assets.yml").read_text(encoding="utf-8")
    assert "types: [published]" in production
    assert "github.event.release.prerelease == false" in production
    assert f"ref: ${{{{ github.event.release.tag_name }}}}" in production
    assert "fetch-depth: 0" in production
    assert "--protected-ref refs/remotes/origin/main" in production
    assert "--expected-terminal " + EXPECTED_TERMINAL in production
    assert CHECKOUT_ACTION in production
    assert SETUP_ACTION in production
    assert PUBLISH_ACTION in production
    assert "@release/v1" not in production
    assert "gh release download \"$RELEASE_TAG\"" in production
    assert "--pattern" not in production
    assert "tools/verify_release_authority.py" in production
    assert LEDGER.as_posix() in production
    assert "--artifact " + WHEEL in production
    assert "--artifact " + SDIST in production
    assert "python -m build" not in production
    assert "id-token: write" in production


def test_manual_workflow_builds_but_cannot_publish() -> None:
    workflow = (ROOT / ".github/workflows/publish.yml").read_text(encoding="utf-8")
    assert "workflow_dispatch:" in workflow
    assert "release:" not in workflow
    assert "sha256sum *.whl *.tar.gz > SHA256SUMS" in workflow
    assert "gh-action-pypi-publish" not in workflow


def test_release_authority_accepts_exact_ledger_bound_artifacts(tmp_path: Path) -> None:
    completed = _run_verifier(tmp_path)
    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize(
    "mutation",
    [
        "paired-replacement",
        "checksum",
        "extra-asset",
        "tag",
        "wrong-source",
        "unmerged-tag-child",
        "wrong-terminal",
        "evidence-hash",
        "review-hash",
        "wrong-byte-count",
        "wrong-metadata",
        "extra-child-path",
        "noncanonical",
    ],
)
def test_release_authority_rejects_mutation(tmp_path: Path, mutation: str) -> None:
    completed = _run_verifier(tmp_path / mutation, mutation)
    assert completed.returncode != 0, mutation


def test_paired_asset_and_checksum_replacement_is_not_authority(tmp_path: Path) -> None:
    completed = _run_verifier(tmp_path, "paired-replacement")
    assert completed.returncode != 0
    assert "committed authority" in completed.stderr
