import json
from pathlib import Path

import yaml

from skillcortex.cli import main
from skillcortex.packaging.artifacts import package_checksums


def write_fastapi_skill(skills_dir):
    package = skills_dir / "fastapi_contract"
    package.mkdir(parents=True)
    (package / "skill.yaml").write_text(
        yaml.safe_dump(
            {
                "skill_id": "fastapi_contract",
                "name": "FastAPI Contract Skill",
                "description": "FastAPI endpoints with Pydantic validation.",
                "capabilities": ["fastapi", "pydantic"],
                "activation_cues": ["FastAPI", "Pydantic"],
            },
            sort_keys=False,
        )
    )


def package_fastapi_skill(tmp_path):
    skills_dir = tmp_path / "skills"
    package = skills_dir / "fastapi_contract"
    eval_summary = tmp_path / "eval-summary.json"
    eval_summary.write_text(
        json.dumps(
            {
                "hypothesis": None,
                "modes": {"single-skill": {"count": 1, "fuzzy_score": 1.0}},
                "tasks": {"python_generation": {"single-skill": {"count": 1}}},
            }
        )
        + "\n"
    )
    assert (
        main(
            [
                "package-skill",
                "--skill-id",
                "fastapi_contract",
                "--name",
                "FastAPI Contract Skill",
                "--adapter-dir",
                "artifacts/adapters/python_skill",
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
    return skills_dir


def test_agent_run_skills_dir_dry_run_executes_dynamic_agent_without_writes(tmp_path, capsys):
    skills_dir = package_fastapi_skill(tmp_path)
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
                "--skills-dir",
                str(skills_dir),
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
    assert result["selected_skills"] == [str(skills_dir / "fastapi_contract")]
    assert result["agent_result"]["status"] == "dry-run"
    assert app.read_text() == "from fastapi import FastAPI\n"


def test_agent_run_skills_dir_confirm_uses_review_path_without_silent_writes(tmp_path, monkeypatch, capsys):
    skills_dir = package_fastapi_skill(tmp_path)
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

    monkeypatch.setattr("skillcortex.cli.handlers.run_agent", fake_run_agent)

    assert (
        main(
            [
                "agent",
                "run",
                "--skills-dir",
                str(skills_dir),
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


def test_agent_run_skills_dir_write_mode_on_fails_clearly(tmp_path, capsys):
    skills_dir = tmp_path / "skills"
    repo = tmp_path / "repo"
    repo.mkdir()
    write_fastapi_skill(skills_dir)

    assert (
        main(
            [
                "agent",
                "run",
                "--skills-dir",
                str(skills_dir),
                "--repo",
                str(repo),
                "--task",
                "Create a FastAPI endpoint",
                "--write-mode",
                "on",
            ]
        )
        == 2
    )

    assert "only supports --dry-run or --write-mode confirm" in capsys.readouterr().err


def test_agent_run_skills_dir_fails_when_no_skill_selected(tmp_path, monkeypatch, capsys):
    skills_dir = tmp_path / "skills"
    repo = tmp_path / "repo"
    skills_dir.mkdir()
    repo.mkdir()

    def fail_run_agent(**kwargs):
        raise AssertionError("run_agent should not be called")

    monkeypatch.setattr("skillcortex.cli.handlers.run_agent", fail_run_agent)

    assert (
        main(
            [
                "agent",
                "run",
                "--skills-dir",
                str(skills_dir),
                "--repo",
                str(repo),
                "--task",
                "Create a FastAPI endpoint",
                "--dry-run",
            ]
        )
        == 2
    )

    assert "no skill selected" in capsys.readouterr().err


def test_agent_run_skills_dir_composition_failure_prevents_agent_execution(tmp_path, monkeypatch, capsys):
    skills_dir = tmp_path / "skills"
    repo = tmp_path / "repo"
    repo.mkdir()
    write_fastapi_skill(skills_dir)

    def fail_run_agent(**kwargs):
        raise AssertionError("run_agent should not be called")

    monkeypatch.setattr("skillcortex.cli.handlers.run_agent", fail_run_agent)

    assert (
        main(
            [
                "agent",
                "run",
                "--skills-dir",
                str(skills_dir),
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


def test_agent_run_skills_dir_validation_failure_prevents_agent_execution(tmp_path, monkeypatch, capsys):
    skills_dir = package_fastapi_skill(tmp_path)
    capsys.readouterr()
    repo = tmp_path / "repo"
    runtime = tmp_path / "runtime"
    repo.mkdir()
    (repo / "app.py").write_text("from fastapi import FastAPI\n")

    def fake_validation(path):
        return {"status": "invalid"}

    def fail_run_agent(**kwargs):
        raise AssertionError("run_agent should not be called")

    monkeypatch.setattr("skillcortex.catalog.validate_runtime_bundle", fake_validation)
    monkeypatch.setattr("skillcortex.cli.handlers.run_agent", fail_run_agent)

    assert (
        main(
            [
                "agent",
                "run",
                "--skills-dir",
                str(skills_dir),
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


def test_agent_run_skills_dir_trace_out_writes_dynamic_wrapper(tmp_path, capsys):
    skills_dir = package_fastapi_skill(tmp_path)
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
                "--skills-dir",
                str(skills_dir),
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


def test_agent_run_skills_dir_runtime_overwrite_behavior(tmp_path, capsys):
    skills_dir = package_fastapi_skill(tmp_path)
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
        "--skills-dir",
        str(skills_dir),
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


def test_agent_run_skills_dir_default_runtime_path_is_deterministic(tmp_path, capsys):
    skills_dir = package_fastapi_skill(tmp_path)
    capsys.readouterr()
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "app.py").write_text("from fastapi import FastAPI\n")

    assert (
        main(
            [
                "agent",
                "run",
                "--skills-dir",
                str(skills_dir),
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
    assert runtime_path.parent == repo / ".skillcortex" / "runtimes"
    assert (runtime_path / "composition.yaml").exists()
