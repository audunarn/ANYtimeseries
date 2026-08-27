import hashlib
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def _verifier() -> str:
    workflow = (ROOT / ".github/workflows/publish-release-assets.yml").read_text(encoding="utf-8")
    return workflow.split("          python - <<'PY'\n", 1)[1].split("\n          PY", 1)[0].replace("          ", "", 1).replace("\n          ", "\n")


def _run(tmp_path: Path, *, tag: str = "v1.0.1", wheel: str = "anytimes-1.0.1-py3-none-any.whl", digest_ok: bool = True) -> subprocess.CompletedProcess[str]:
    release = tmp_path / "release"
    release.mkdir(parents=True)
    files = {wheel: b"wheel", "anytimes-1.0.1.tar.gz": b"sdist"}
    for name, data in files.items():
        (release / name).write_bytes(data)
    rows = [f"{hashlib.sha256(data).hexdigest()}  {name}" for name, data in files.items()]
    if not digest_ok:
        rows[0] = "0" * 64 + "  " + wheel
    (release / "SHA256SUMS").write_text("\n".join(rows) + "\n", encoding="ascii")
    environment = os.environ.copy()
    environment["RELEASE_TAG"] = tag
    return subprocess.run([sys.executable, "-c", _verifier()], cwd=tmp_path, env=environment, capture_output=True, text=True, check=False)


def test_production_release_publishes_only_verified_prebuilt_assets() -> None:
    workflow = (ROOT / ".github/workflows/publish-release-assets.yml").read_text(encoding="utf-8")
    assert "types: [published]" in workflow
    assert "gh release download" in workflow
    assert "SHA256SUMS does not bind the exact artifact set" in workflow
    assert "hashlib.sha256(path.read_bytes()).hexdigest()" in workflow
    assert "python -m build" not in workflow
    assert "id-token: write" in workflow
    assert "timeout-minutes: 20" not in workflow
    assert "anytimes-{version}-py3-none-any.whl" in workflow
    assert "path.is_symlink()" in workflow


def test_manual_workflow_does_not_publish() -> None:
    workflow = (ROOT / ".github/workflows/publish.yml").read_text(encoding="utf-8")
    assert "workflow_dispatch:" in workflow
    assert "release:" not in workflow
    assert "sha256sum *.whl *.tar.gz > SHA256SUMS" in workflow
    assert "gh-action-pypi-publish" not in workflow


def test_release_verifier_rejects_wrong_tag_name_and_hash(tmp_path: Path) -> None:
    assert _run(tmp_path / "valid").returncode == 0
    assert _run(tmp_path / "tag", tag="v1.0.0").returncode != 0
    assert _run(tmp_path / "name", wheel="other-1.0.1-py3-none-any.whl").returncode != 0
    assert _run(tmp_path / "hash", digest_ok=False).returncode != 0
