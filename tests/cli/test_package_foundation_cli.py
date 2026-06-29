import json

from slmcortex.cli import main
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


def test_doctor_reports_external_workspace_contract(tmp_path, capsys):
    workspace = tmp_path / "workspace"

    assert main(["doctor", "--workspace", str(workspace)]) == 0

    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "complete"
    assert result["product_mode"] == "composer"
    assert result["workspace"]["root"] == str(workspace.resolve())
    assert result["workspace"]["packages_dir"].endswith("/packages")
    assert result["summary_lines"]


def test_compose_folder_uses_workspace_layout_and_writes_export_descriptor(tmp_path, capsys):
    workspace = tmp_path / "workspace"
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "app.py").write_text("from fastapi import FastAPI\nfrom pydantic import BaseModel\n")
    _package_fastapi_contract(workspace, tmp_path)
    capsys.readouterr()

    export_descriptor = workspace / "exports" / "repo.json"
    assert (
        main(
            [
                "compose-folder",
                "--workspace",
                str(workspace),
                "--folder",
                str(repo),
                "--task",
                "Create a FastAPI endpoint with Pydantic validation",
                "--export-descriptor",
                str(export_descriptor),
            ]
        )
        == 0
    )

    result = json.loads(capsys.readouterr().out)
    runtime_path = workspace / "runtimes" / "repo"
    assert result["status"] == "complete"
    assert result["slms_dir"] == str((workspace / "packages").resolve())
    assert result["runtime"]["path"] == str(runtime_path.resolve())
    assert result["runtime"]["validation_status"] == "passed"
    assert result["export_bundle"]["descriptor_path"] == str(export_descriptor.resolve())
    assert runtime_path.joinpath("composition.yaml").exists()
    assert export_descriptor.exists()
    assert workspace.joinpath("logs", "compose-repo.json").exists()


def test_compose_folder_returns_structured_failure_for_missing_packages(tmp_path, capsys):
    workspace = tmp_path / "workspace"
    repo = tmp_path / "repo"
    repo.mkdir()

    exit_code = main(
        [
            "compose-folder",
            "--workspace",
            str(workspace),
            "--folder",
            str(repo),
            "--task",
            "Create a FastAPI endpoint",
        ]
    )

    result = json.loads(capsys.readouterr().out)
    assert exit_code == 2
    assert result["status"] == "failed"
    assert result["error_code"] in {"not_found", "invalid_request"}
    assert result["runtime"]["composition_status"] == "failed"
    assert result["errors"]