from pathlib import Path

from slmcortex.shared.product import (
    PRODUCT_MODES,
    ensure_app_workspace,
    environment_diagnostics,
    resolve_app_workspace,
    runtime_name_for_folder,
)


def test_workspace_layout_is_externalizable(tmp_path):
    workspace = ensure_app_workspace(tmp_path / "app-workspace")

    assert workspace.root == (tmp_path / "app-workspace").resolve()
    assert workspace.packages_dir == workspace.root / "packages"
    assert workspace.runtimes_dir == workspace.root / "runtimes"
    assert workspace.exports_dir == workspace.root / "exports"
    assert workspace.logs_dir == workspace.root / "logs"
    assert workspace.diagnostics_dir == workspace.root / "diagnostics"
    assert workspace.packages_dir.exists()
    assert workspace.runtimes_dir.exists()


def test_runtime_name_for_folder_normalizes_user_input(tmp_path):
    folder = tmp_path / "My Fancy Repo"
    folder.mkdir()

    assert runtime_name_for_folder(folder) == "my-fancy-repo"
    assert runtime_name_for_folder(folder, " Runtime 1 ") == "runtime-1"


def test_environment_diagnostics_reports_machine_and_workspace(tmp_path):
    diagnostics = environment_diagnostics(
        workspace_root=tmp_path / "workspace",
        product_mode="composer",
    )

    assert diagnostics["status"] == "complete"
    assert diagnostics["product_mode"] in PRODUCT_MODES
    assert diagnostics["workspace"]["root"] == str((tmp_path / "workspace").resolve())
    assert diagnostics["default_runtime_backend"] in {"mlx", "gguf"}
    assert isinstance(diagnostics["backends"], list)
    assert diagnostics["summary_lines"]
    assert diagnostics["optional_backend_provisioning"]
    assert diagnostics["recovery_guidance"]


def test_resolve_app_workspace_is_stable_for_given_root(tmp_path):
    root = tmp_path / "workspace"

    first = resolve_app_workspace(root)
    second = resolve_app_workspace(Path(str(root)))

    assert first.as_dict() == second.as_dict()