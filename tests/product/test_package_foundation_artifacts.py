from pathlib import Path


def _repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("repository root not found")


ROOT = _repo_root()


def test_installer_artifacts_exist_for_supported_targets():
    for relative in (
        "artifacts/installers/install-slmcortex-macos.sh",
        "artifacts/installers/install-slmcortex-linux.sh",
        "artifacts/installers/install-slmcortex-windows.ps1",
    ):
        assert ROOT.joinpath(relative).exists()


def test_packaged_install_doc_lists_workspace_contract_and_smokes():
    text = ROOT.joinpath("docs/user-guide/packaged-install.md").read_text()

    assert "App Workspace Contract" in text
    assert "run_package_product_smoke.py" in text
    assert "run_packaged_install_smoke.py" in text
    assert "install-slmcortex-macos.sh" in text
    assert "install-slmcortex-linux.sh" in text
    assert "install-slmcortex-windows.ps1" in text