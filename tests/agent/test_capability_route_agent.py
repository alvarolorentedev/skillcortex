import io
import json
from pathlib import Path

import yaml

from slmcortex.cli import main
from slmcortex.packaging.artifacts import package_checksums


def write_fastapi_slm(slms_dir):
    package = slms_dir / "fastapi_contract"
    package.mkdir(parents=True)
    (package / "slm.yaml").write_text(
        yaml.safe_dump(
            {
                "slm_id": "fastapi_contract",
                "name": "FastAPI Contract Slm",
                "description": "FastAPI endpoints with Pydantic validation.",
                "capabilities": ["fastapi", "pydantic"],
                "activation_cues": ["FastAPI", "Pydantic"],
            },
            sort_keys=False,
        )
    )


def package_fastapi_slm(tmp_path):
    slms_dir = tmp_path / "slms"
    package = slms_dir / "fastapi_contract"
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
                ]
            }
        )
        + "\n"
    )
    metadata = json.loads((package / "metadata.json").read_text())
    metadata["checksums"] = package_checksums(package)
    (package / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return slms_dir


def test_agent_run_slms_dir_dry_run_executes_dynamic_agent_without_writes(tmp_path, capsys):
    slms_dir = package_fastapi_slm(tmp_path)
    capsys.readouterr()
    repo = tmp_path / "repo"
    repo.mkdir()
    app = repo / "app.py"
    app.write_text("from fastapi import FastAPI\n")

    assert (
        main(
            [
                "agent",
                "run",
                "--slms-dir",
                str(slms_dir),
                "--repo",
                str(repo),
                "--task",
                "Create a FastAPI endpoint with Pydantic validation",
                "--dry-run",
                "--compose-runtime-out",
                str(tmp_path / "runtime"),
            ]
        )
        == 0
    )

    result = json.loads(capsys.readouterr().out)
    assert result["mode"] == "dynamic_agent"
    assert result["agent_execution_status"] == "dry_run_completed"
    assert result["write_mode"] == "dry_run"
    assert result["selected_slms"] == [str(slms_dir / "fastapi_contract")]
    assert result["agent_result"]["status"] == "dry-run"
    assert app.read_text() == "from fastapi import FastAPI\n"


def test_agent_run_slms_dir_confirm_uses_review_path_without_silent_writes(tmp_path, monkeypatch, capsys):
    slms_dir = package_fastapi_slm(tmp_path)
    capsys.readouterr()
    repo = tmp_path / "repo"
    repo.mkdir()
    app = repo / "app.py"
    app.write_text("from fastapi import FastAPI\n")

    def fake_run_agent(**kwargs):
        assert kwargs["writes"] == "confirm"
        assert kwargs["dry_run"] is False
        assert kwargs["runtime_path"] == tmp_path / "runtime"
        return {
            "status": "review_required",
            "review_artifact_path": str(tmp_path / "review.patch"),
            "runtime": str(kwargs["runtime_path"].resolve()),
        }

    monkeypatch.setattr("slmcortex.cli.handlers.run_agent", fake_run_agent)

    assert (
        main(
            [
                "agent",
                "run",
                "--slms-dir",
                str(slms_dir),
                "--repo",
                str(repo),
                "--task",
                "Create a FastAPI endpoint with Pydantic validation",
                "--write-mode",
                "confirm",
                "--compose-runtime-out",
                str(tmp_path / "runtime"),
            ]
        )
        == 0
    )

    result = json.loads(capsys.readouterr().out)
    assert result["mode"] == "dynamic_agent"
    assert result["agent_execution_status"] == "review_required"
    assert result["write_mode"] == "confirm"
    assert result["agent_result"]["review_artifact_path"]
    assert app.read_text() == "from fastapi import FastAPI\n"


def test_agent_run_slms_dir_reads_tasks_from_stdin(tmp_path, monkeypatch, capsys):
    slms_dir = package_fastapi_slm(tmp_path)
    capsys.readouterr()
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "app.py").write_text("from fastapi import FastAPI\n")
    tasks = [
        "Create a FastAPI endpoint with Pydantic validation",
        "Add a user endpoint",
    ]

    def fake_run_agent(**kwargs):
        assert kwargs["task"] == [tasks[0]]
        assert kwargs["task_provider"]() == tasks[1]
        assert kwargs["task_provider"]() is None
        assert kwargs["writes"] == "on"
        assert kwargs["dry_run"] is False
        return {
            "status": "applied",
            "runtime": str(kwargs["runtime_path"].resolve()),
        }

    monkeypatch.setattr("slmcortex.cli.handlers.run_agent", fake_run_agent)
    monkeypatch.setattr("sys.stdin", io.StringIO("\n".join(tasks) + "\n"))

    assert (
        main(
            [
                "agent",
                "run",
                "--slms-dir",
                str(slms_dir),
                "--repo",
                str(repo),
                "--compose-runtime-out",
                str(tmp_path / "runtime"),
            ]
        )
        == 0
    )

    result = json.loads(capsys.readouterr().out)
    assert result["mode"] == "dynamic_agent"
    assert result["agent_execution_status"] == "completed"
    assert result["write_mode"] == "on"


def test_agent_run_slms_dir_uses_available_slms_when_route_selects_none(tmp_path, monkeypatch, capsys):
    slms_dir = package_fastapi_slm(tmp_path)
    capsys.readouterr()
    repo = tmp_path / "repo"
    repo.mkdir()
    task = "Write CSS for a landing page"

    def fake_run_agent(**kwargs):
        assert kwargs["writes"] == "on"
        assert kwargs["dry_run"] is False
        assert kwargs["runtime_path"].joinpath("composition.yaml").exists()
        return {
            "status": "applied",
            "runtime": str(kwargs["runtime_path"].resolve()),
        }

    monkeypatch.setattr("slmcortex.cli.handlers.run_agent", fake_run_agent)

    assert (
        main(
            [
                "agent",
                "run",
                "--slms-dir",
                str(slms_dir),
                "--repo",
                str(repo),
                "--task",
                task,
                "--compose-runtime-out",
                str(tmp_path / "runtime"),
            ]
        )
        == 0
    )

    result = json.loads(capsys.readouterr().out)
    assert result["routing_decision"]["selected_slms"] == []
    assert result["selected_slms"] == [str(slms_dir / "fastapi_contract")]
    assert result["agent_execution_status"] == "completed"
    assert any("base fallback" in warning for warning in result["warnings"])


def test_agent_run_slms_dir_repairs_stale_package_checksums(tmp_path, monkeypatch, capsys):
    slms_dir = package_fastapi_slm(tmp_path)
    capsys.readouterr()
    package = slms_dir / "fastapi_contract"
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "app.py").write_text("from fastapi import FastAPI\n")
    (package / "README.md").write_text("local install note\n")

    def fake_run_agent(**kwargs):
        assert kwargs["dry_run"] is False
        return {
            "status": "review_required",
            "review_artifact_path": str(tmp_path / "review.patch"),
            "runtime": str(kwargs["runtime_path"].resolve()),
        }

    monkeypatch.setattr("slmcortex.cli.handlers.run_agent", fake_run_agent)

    assert (
        main(
            [
                "agent",
                "run",
                "--slms-dir",
                str(slms_dir),
                "--repo",
                str(repo),
                "--task",
                "Create a FastAPI endpoint with Pydantic validation",
                "--compose-runtime-out",
                str(tmp_path / "runtime"),
            ]
        )
        == 0
    )

    result = json.loads(capsys.readouterr().out)
    metadata = json.loads((package / "metadata.json").read_text())
    assert result["agent_execution_status"] == "review_required"
    assert metadata["checksums"]["README.md"] == package_checksums(package)["README.md"]


def test_agent_run_slms_dir_repairs_stale_protected_inputs(tmp_path, monkeypatch, capsys):
    slms_dir = package_fastapi_slm(tmp_path)
    capsys.readouterr()
    package = slms_dir / "fastapi_contract"
    metadata_path = package / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    protected_file = next(iter(metadata["protected_inputs"]["files"]))
    metadata["protected_inputs"]["files"][protected_file]["after_sha256"] = "stale"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "app.py").write_text("from fastapi import FastAPI\n")

    def fake_run_agent(**kwargs):
        return {
            "status": "review_required",
            "review_artifact_path": str(tmp_path / "review.patch"),
            "runtime": str(kwargs["runtime_path"].resolve()),
        }

    monkeypatch.setattr("slmcortex.cli.handlers.run_agent", fake_run_agent)

    assert (
        main(
            [
                "agent",
                "run",
                "--slms-dir",
                str(slms_dir),
                "--repo",
                str(repo),
                "--task",
                "Create a FastAPI endpoint with Pydantic validation",
                "--compose-runtime-out",
                str(tmp_path / "runtime"),
            ]
        )
        == 0
    )

    repaired = json.loads(metadata_path.read_text())
    assert repaired["protected_inputs"] == {"all_unchanged": True, "files": {}}


def test_agent_run_slms_dir_write_mode_on_is_supported(tmp_path, monkeypatch, capsys):
    slms_dir = package_fastapi_slm(tmp_path)
    capsys.readouterr()
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "app.py").write_text("from fastapi import FastAPI\n")

    def fake_run_agent(**kwargs):
        assert kwargs["writes"] == "on"
        return {
            "status": "applied",
            "runtime": str(kwargs["runtime_path"].resolve()),
        }

    monkeypatch.setattr("slmcortex.cli.handlers.run_agent", fake_run_agent)

    assert (
        main(
            [
                "agent",
                "run",
                "--slms-dir",
                str(slms_dir),
                "--repo",
                str(repo),
                "--task",
                "Create a FastAPI endpoint with Pydantic validation",
                "--write-mode",
                "on",
                "--compose-runtime-out",
                str(tmp_path / "runtime"),
            ]
        )
        == 0
    )

    result = json.loads(capsys.readouterr().out)
    assert result["agent_execution_status"] == "completed"
    assert result["write_mode"] == "on"


def test_agent_run_slms_dir_uses_dynamic_base_when_no_slm_packages_available(tmp_path, monkeypatch, capsys):
    slms_dir = tmp_path / "slms"
    repo = tmp_path / "repo"
    slms_dir.mkdir()
    repo.mkdir()

    def fake_run_agent(**kwargs):
        assert kwargs["runtime_path"] is None
        assert kwargs["runtime"].validate()["runtime"] == "dynamic"
        assert kwargs["writes"] == "on"
        assert kwargs["dry_run"] is False
        return {
            "status": "applied",
            "runtime": "dynamic",
        }

    monkeypatch.setattr("slmcortex.cli.handlers.run_agent", fake_run_agent)

    assert (
        main(
            [
                "agent",
                "run",
                "--slms-dir",
                str(slms_dir),
                "--repo",
                str(repo),
                "--task",
                "Create a FastAPI endpoint",
            ]
        )
        == 0
    )

    result = json.loads(capsys.readouterr().out)
    assert result["runtime_out"] is None
    assert result["composition_strategy"] == "dynamic"
    assert result["agent_execution_status"] == "completed"
    assert any("base fallback" in warning for warning in result["warnings"])


def test_agent_run_slms_dir_composition_failure_prevents_agent_execution(tmp_path, monkeypatch, capsys):
    slms_dir = tmp_path / "slms"
    repo = tmp_path / "repo"
    repo.mkdir()
    write_fastapi_slm(slms_dir)

    def fail_run_agent(**kwargs):
        raise AssertionError("run_agent should not be called")

    monkeypatch.setattr("slmcortex.cli.handlers.run_agent", fail_run_agent)

    assert (
        main(
            [
                "agent",
                "run",
                "--slms-dir",
                str(slms_dir),
                "--repo",
                str(repo),
                "--task",
                "Create a FastAPI endpoint with Pydantic validation",
                "--dry-run",
            ]
        )
        == 2
    )

    assert "not composable" in capsys.readouterr().err


def test_agent_run_slms_dir_validation_failure_prevents_agent_execution(tmp_path, monkeypatch, capsys):
    slms_dir = package_fastapi_slm(tmp_path)
    capsys.readouterr()
    repo = tmp_path / "repo"
    runtime = tmp_path / "runtime"
    repo.mkdir()
    (repo / "app.py").write_text("from fastapi import FastAPI\n")

    def fake_validation(path):
        return {"status": "invalid"}

    def fail_run_agent(**kwargs):
        raise AssertionError("run_agent should not be called")

    monkeypatch.setattr("slmcortex.catalog.validate_runtime_bundle", fake_validation)
    monkeypatch.setattr("slmcortex.cli.handlers.run_agent", fail_run_agent)

    assert (
        main(
            [
                "agent",
                "run",
                "--slms-dir",
                str(slms_dir),
                "--repo",
                str(repo),
                "--task",
                "Create a FastAPI endpoint with Pydantic validation",
                "--dry-run",
                "--compose-runtime-out",
                str(runtime),
            ]
        )
        == 2
    )

    assert "validation failed" in capsys.readouterr().err


def test_agent_run_slms_dir_trace_out_writes_dynamic_wrapper(tmp_path, capsys):
    slms_dir = package_fastapi_slm(tmp_path)
    capsys.readouterr()
    repo = tmp_path / "repo"
    trace = tmp_path / "dynamic-trace.json"
    repo.mkdir()
    (repo / "app.py").write_text("from fastapi import FastAPI\n")

    assert (
        main(
            [
                "agent",
                "run",
                "--slms-dir",
                str(slms_dir),
                "--repo",
                str(repo),
                "--task",
                "Create a FastAPI endpoint with Pydantic validation",
                "--dry-run",
                "--trace-out",
                str(trace),
            ]
        )
        == 0
    )

    result = json.loads(capsys.readouterr().out)
    payload = json.loads(trace.read_text())
    assert result["trace_out"] == str(trace.resolve())
    assert payload["mode"] == "dynamic_agent"
    assert payload["routing_decision"]
    assert payload["composition_status"] == "written"
    assert payload["validation_status"] == "passed"
    assert payload["agent_result"]


def test_agent_run_slms_dir_runtime_overwrite_behavior(tmp_path, capsys):
    slms_dir = package_fastapi_slm(tmp_path)
    capsys.readouterr()
    repo = tmp_path / "repo"
    runtime = tmp_path / "runtime"
    repo.mkdir()
    (repo / "app.py").write_text("from fastapi import FastAPI\n")
    runtime.mkdir()
    (runtime / "old.txt").write_text("old\n")
    args = [
        "agent",
        "run",
        "--slms-dir",
        str(slms_dir),
        "--repo",
        str(repo),
        "--task",
        "Create a FastAPI endpoint with Pydantic validation",
        "--dry-run",
        "--compose-runtime-out",
        str(runtime),
    ]

    assert main(args) == 2
    assert "exists" in capsys.readouterr().err

    assert main([*args, "--overwrite"]) == 0
    assert not (runtime / "old.txt").exists()


def test_agent_run_slms_dir_default_runtime_path_is_deterministic(tmp_path, capsys):
    slms_dir = package_fastapi_slm(tmp_path)
    capsys.readouterr()
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "app.py").write_text("from fastapi import FastAPI\n")

    assert (
        main(
            [
                "agent",
                "run",
                "--slms-dir",
                str(slms_dir),
                "--repo",
                str(repo),
                "--task",
                "Create a FastAPI endpoint with Pydantic validation",
                "--dry-run",
            ]
        )
        == 0
    )

    result = json.loads(capsys.readouterr().out)
    runtime_path = Path(result["runtime_out"])
    assert runtime_path.parent == repo / ".slmcortex" / "runtimes"
    assert (runtime_path / "composition.yaml").exists()
