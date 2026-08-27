from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_production_release_publishes_only_verified_prebuilt_assets() -> None:
    workflow = (ROOT / ".github/workflows/publish-release-assets.yml").read_text(encoding="utf-8")
    assert "types: [published]" in workflow
    assert "gh release download" in workflow
    assert "SHA256SUMS does not bind the exact artifact set" in workflow
    assert "hashlib.sha256(path.read_bytes()).hexdigest()" in workflow
    assert "python -m build" not in workflow
    assert "id-token: write" in workflow
    assert "timeout-minutes: 20" not in workflow


def test_manual_workflow_does_not_publish() -> None:
    workflow = (ROOT / ".github/workflows/publish.yml").read_text(encoding="utf-8")
    assert "workflow_dispatch:" in workflow
    assert "release:" not in workflow
    assert "sha256sum *.whl *.tar.gz > SHA256SUMS" in workflow
    assert "if: startsWith(github.ref, 'refs/tags/')" in workflow
