import json

from slmcortex.cli import main
from slmcortex.composer_app import run_composer_app
from slmcortex.packaging.artifacts import package_checksums


def _package_fastapi_contract(workspace_root, tmp_path):
    eval_summary = tmp_path / "eval-summary.json"
    eval_summary.write_text(
        json.dumps(
            {
                "hypothesis": None,
                "modes": {"single-slm": {"count": 1, "fuzzy_score": 1.0}},
                "tasks": {"python_generation": {"single-slm": {"count": 1}}},
            }
        )
        + "\n"
    )
    package = workspace_root / "packages" / "fastapi_contract"
    assert (
        main(
            [
                "package-slm",
                "--slm-id",
                "fastapi_contract",
                "--name",
                "FastAPI Contract Slm",
                "--adapter-dir",
                "artifacts/adapters/python_slm",
                "--output",
                str(package),
                "--train-dataset",
                "data/train.jsonl",
                "--eval-dataset",
                "data/eval.jsonl",
                "--eval-summary",
                str(eval_summary),
                "--description",
                "FastAPI endpoints with Pydantic validation.",
                "--allowed-task-types",
                "python_generation",
                "--activation-scope",
                "task",
            ]
        )
        == 0
    )
    (package / "routing_card.json").write_text(
        json.dumps(
            {
                "positive_examples": [
                    "Create a FastAPI endpoint with Pydantic validation",
                ],
                "negative_examples": ["Fix a React hydration bug"],
            }
        )
        + "\n"
    )
    metadata = json.loads((package / "metadata.json").read_text())
    metadata["checksums"] = package_checksums(package)
    (package / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")


def test_composer_app_persists_onboarding_and_project_state(tmp_path, capsys):
    workspace = tmp_path / "workspace"
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "app.py").write_text("from fastapi import FastAPI\nfrom pydantic import BaseModel\n")
    _package_fastapi_contract(workspace, tmp_path)
    capsys.readouterr()

    assert main(["composer-app", "--workspace", str(workspace), "--folder", str(repo), "--outcome", "export_bundle"]) == 0
    first = json.loads(capsys.readouterr().out)

    assert first["status"] == "complete"
    assert first["onboarding"]["first_run"] is True
    assert first["onboarding"]["completed"] is True
    assert first["project"]["scan_summary"]["framework_signals"] == ["fastapi", "pydantic"]
    assert first["composition"]["runtime"]["validation_status"] == "passed"
    assert first["outcome"]["requested"] == "export_bundle"
    assert first["outcome"]["status"] == "written"

    assert main(["composer-app", "--workspace", str(workspace), "--folder", str(repo), "--outcome", "export_bundle"]) == 0
    second = json.loads(capsys.readouterr().out)

    assert second["onboarding"]["first_run"] is False
    assert second["project"]["reopened"] is True
    state_path = workspace / "state" / "composer-app-state.json"
    state = json.loads(state_path.read_text())
    assert state["onboarding_completed"] is True
    assert str(repo.resolve()) in state["projects"]


def test_composer_app_local_run_reports_dry_run_server_when_requested(tmp_path, capsys):
    workspace = tmp_path / "workspace"
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "app.py").write_text("from fastapi import FastAPI\nfrom pydantic import BaseModel\n")
    _package_fastapi_contract(workspace, tmp_path)
    capsys.readouterr()

    assert (
        main(
            [
                "composer-app",
                "--workspace",
                str(workspace),
                "--folder",
                str(repo),
                "--task",
                "Create a FastAPI endpoint with Pydantic validation",
                "--outcome",
                "local_run",
                "--run-target",
                "compatibility_server",
                "--dry-run",
                "--export-logs",
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)

    assert result["status"] == "complete"
    assert result["outcome"]["requested"] == "local_run"
    assert result["outcome"]["run_target"] == "compatibility_server"
    assert result["outcome"]["server"]["status"] == "dry-run"
    assert result["support"]["support_bundle"] is not None


def test_composer_app_local_run_supports_agent_flow(tmp_path, capsys):
    workspace = tmp_path / "workspace"
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "app.py").write_text("from fastapi import FastAPI\nfrom pydantic import BaseModel\n")
    _package_fastapi_contract(workspace, tmp_path)
    capsys.readouterr()

    assert (
        main(
            [
                "composer-app",
                "--workspace",
                str(workspace),
                "--folder",
                str(repo),
                "--task",
                "Create a FastAPI endpoint with Pydantic validation",
                "--outcome",
                "local_run",
                "--run-target",
                "agent_flow",
                "--dry-run",
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)

    assert result["status"] == "complete"
    assert result["outcome"]["run_target"] == "agent_flow"
    assert result["outcome"]["agent"]["status"] == "dry-run"


def test_composer_app_reports_dry_run_only_when_no_runtime_backend_is_available(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    repo = tmp_path / "repo"
    repo.mkdir()
    runtime_path = workspace / "runtimes" / "repo"
    runtime_path.mkdir(parents=True, exist_ok=True)

    import slmcortex.composer_app as composer_app_module

    monkeypatch.setattr(
        composer_app_module,
        "environment_diagnostics",
        lambda **_: {
            "status": "complete",
            "product_mode": "composer",
            "workspace": composer_app_module.ensure_app_workspace(workspace).as_dict(),
            "available_runtime_backends": [],
            "backends": [],
            "warnings": ["no runtime backend dependencies detected"],
        },
    )
    monkeypatch.setattr(
        composer_app_module,
        "compose_from_folder",
        lambda **_: {
            "status": "complete",
            "folder": str(repo.resolve()),
            "task": "Create a FastAPI endpoint",
            "runtime": {
                "path": str(runtime_path.resolve()),
                "composition_strategy": "routed",
                "composition_status": "written",
                "validation_status": "passed",
            },
            "routing_decision": {"routing_mode": "capability", "fallback": "base"},
            "selected_slms": [],
            "export_bundle": None,
            "warnings": [],
            "errors": [],
        },
    )

    observed = {}

    def fake_serve_runtime(**kwargs):
        observed.update(kwargs)
        return {"status": "dry-run", "runtime": "repo", "host": kwargs["host"], "port": kwargs["port"]}

    monkeypatch.setattr(composer_app_module, "serve_runtime", fake_serve_runtime)

    result = run_composer_app(folder=repo, workspace_root=workspace, outcome="local_run")

    assert result["status"] == "complete"
    assert result["onboarding"]["capabilities"]["dry_run_only"] is True
    assert result["onboarding"]["capabilities"]["local_run"]["available"] is False
    assert result["onboarding"]["capabilities"]["local_inference"]["available"] is False
    assert result["outcome"]["status"] == "dry-run"
    assert observed["dry_run"] is True


def test_composer_app_maps_validation_failure_to_product_error(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    repo = tmp_path / "repo"
    repo.mkdir()

    import slmcortex.composer_app as composer_app_module

    monkeypatch.setattr(
        composer_app_module,
        "compose_from_folder",
        lambda **_: {
            "status": "failed",
            "runtime": {"path": str((workspace / "runtimes" / "repo").resolve())},
            "warnings": [],
            "errors": ["runtime validation failed: invalid"],
        },
    )

    result = run_composer_app(folder=repo, workspace_root=workspace)

    assert result["status"] == "failed"
    assert result["product_error"]["code"] == "validation_failed"


def test_composer_app_maps_backend_and_incompatibility_errors(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    repo = tmp_path / "repo"
    repo.mkdir()

    import slmcortex.composer_app as composer_app_module

    errors = iter(
        [
            "MLX backend requires macOS arm64",
            "slm fastapi_contract is incompatible with selected slm debugging_slm",
        ]
    )

    def fake_compose(**_):
        return {
            "status": "failed",
            "runtime": {"path": str((workspace / "runtimes" / "repo").resolve())},
            "warnings": [],
            "errors": [next(errors)],
        }

    monkeypatch.setattr(composer_app_module, "compose_from_folder", fake_compose)

    backend_result = run_composer_app(folder=repo, workspace_root=workspace)
    incompatible_result = run_composer_app(folder=repo, workspace_root=workspace)

    assert backend_result["product_error"]["code"] == "unsupported_backend_choice"
    assert incompatible_result["product_error"]["code"] == "incompatible_selection"


def test_composer_app_translates_missing_packages_into_repair_guidance(tmp_path, capsys):
    workspace = tmp_path / "workspace"
    repo = tmp_path / "repo"
    repo.mkdir()

    exit_code = main(["composer-app", "--workspace", str(workspace), "--folder", str(repo), "--export-logs"])
    result = json.loads(capsys.readouterr().out)

    assert exit_code == 2
    assert result["status"] == "failed"
    assert result["product_error"]["code"] in {"missing_package_metadata", "incompatible_selection"}
    assert result["product_error"]["recommended_next_action"]
    assert result["support"]["support_bundle"] is not None


def test_composer_app_migrates_v1_state_and_tracks_runtime_provenance(tmp_path, capsys):
    workspace = tmp_path / "workspace"
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "app.py").write_text("from fastapi import FastAPI\nfrom pydantic import BaseModel\n")
    _package_fastapi_contract(workspace, tmp_path)
    state_path = workspace / "state" / "composer-app-state.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(
        json.dumps(
            {
                "onboarding_completed": True,
                "projects": {
                    str(repo.resolve()): {
                        "runtime_name": "repo",
                        "task": "Create a FastAPI endpoint",
                    }
                },
            }
        )
        + "\n"
    )
    capsys.readouterr()

    assert main(["composer-app", "--workspace", str(workspace), "--folder", str(repo), "--outcome", "export_bundle"]) == 0
    result = json.loads(capsys.readouterr().out)
    state = json.loads(state_path.read_text())

    assert result["status"] == "complete"
    assert state["schema_version"] == "2"
    assert state["migrations"]
    assert state["migrations"][0]["backup_path"]
    project = state["projects"][str(repo.resolve())]
    assert project["selected_packages"]
    assert project["selected_packages"][0]["fingerprint"]
    assert project["runtime_bundle"]["checksums_sha256"]


def test_composer_app_rejects_future_state_schema_with_recovery_guidance(tmp_path):
    workspace = tmp_path / "workspace"
    repo = tmp_path / "repo"
    repo.mkdir()
    state_path = workspace / "state" / "composer-app-state.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps({"schema_version": "99", "projects": {}}) + "\n")

    result = run_composer_app(folder=repo, workspace_root=workspace, export_logs=True)

    assert result["status"] == "failed"
    assert result["product_error"]["code"] == "state_schema_incompatible"
    assert "clean workspace root" in result["product_error"]["recommended_next_action"]
    assert result["support"]["support_bundle"] is not None