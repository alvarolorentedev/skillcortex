import json

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
                ],
                "negative_examples": ["Fix a React hydration bug"],
            }
        )
        + "\n"
    )
    metadata = json.loads((package / "metadata.json").read_text())
    metadata["checksums"] = package_checksums(package)
    (package / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return skills_dir


def test_route_command_emits_stable_json_contract(tmp_path, capsys):
    skills_dir = tmp_path / "skills"
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "app.py").write_text("from fastapi import FastAPI\nfrom pydantic import BaseModel\n")
    write_fastapi_skill(skills_dir)

    assert (
        main(
            [
                "route",
                "--skills-dir",
                str(skills_dir),
                "--repo",
                str(repo),
                "--task",
                "Create a FastAPI endpoint with Pydantic validation",
                "--explain",
            ]
        )
        == 0
    )

    result = json.loads(capsys.readouterr().out)
    assert list(result) == [
        "routing_mode",
        "skills_dir",
        "repo",
        "task",
        "repo_context",
        "selected_skills",
        "candidates",
        "fallback",
        "errors",
        "warnings",
    ]
    assert result["routing_mode"] == "capability"
    assert result["selected_skills"][0]["skill_id"] == "fastapi_contract"
    candidate = result["candidates"][0]
    assert list(candidate) == [
        "skill_id",
        "score",
        "selected",
        "compatible",
        "matched_signals",
        "negative_signals",
        "score_breakdown",
        "reason",
    ]


def test_compose_from_route_writes_and_validates_runtime(tmp_path, capsys):
    skills_dir = package_fastapi_skill(tmp_path)
    capsys.readouterr()
    repo = tmp_path / "repo"
    runtime = tmp_path / "runtime"
    repo.mkdir()
    (repo / "app.py").write_text("from fastapi import FastAPI\nfrom pydantic import BaseModel\n")

    assert (
        main(
            [
                "compose-from-route",
                "--skills-dir",
                str(skills_dir),
                "--repo",
                str(repo),
                "--task",
                "Create a FastAPI endpoint with Pydantic validation",
                "--runtime-out",
                str(runtime),
                "--explain",
            ]
        )
        == 0
    )

    result = json.loads(capsys.readouterr().out)
    assert (runtime / "composition.yaml").exists()
    assert result["routing_decision"]["routing_mode"] == "capability"
    assert result["selected_skills"] == [str(skills_dir / "fastapi_contract")]
    assert result["runtime_out"] == str(runtime.resolve())
    assert result["composition_strategy"] == "routed"
    assert result["composition_status"] == "written"
    assert result["validation_status"] == "passed"
    assert result["warnings"] == []
    assert result["errors"] == []


def test_compose_from_route_requires_selected_skill_without_allow_base(tmp_path, capsys):
    skills_dir = tmp_path / "skills"
    repo = tmp_path / "repo"
    skills_dir.mkdir()
    repo.mkdir()

    assert (
        main(
            [
                "compose-from-route",
                "--skills-dir",
                str(skills_dir),
                "--repo",
                str(repo),
                "--task",
                "Create a FastAPI endpoint",
                "--runtime-out",
                str(tmp_path / "runtime"),
            ]
        )
        == 2
    )

    assert "no skill selected" in capsys.readouterr().err


def test_compose_from_route_allow_base_skips_runtime_write(tmp_path, capsys):
    skills_dir = tmp_path / "skills"
    repo = tmp_path / "repo"
    runtime = tmp_path / "runtime"
    skills_dir.mkdir()
    repo.mkdir()

    assert (
        main(
            [
                "compose-from-route",
                "--skills-dir",
                str(skills_dir),
                "--repo",
                str(repo),
                "--task",
                "Create a FastAPI endpoint",
                "--runtime-out",
                str(runtime),
                "--allow-base",
            ]
        )
        == 0
    )

    result = json.loads(capsys.readouterr().out)
    assert result["composition_status"] == "skipped"
    assert result["validation_status"] == "not_run"
    assert result["runtime_out"] is None
    assert result["warnings"]
    assert not runtime.exists()


def test_compose_from_route_overwrite_controls_existing_runtime(tmp_path, capsys):
    skills_dir = package_fastapi_skill(tmp_path)
    capsys.readouterr()
    repo = tmp_path / "repo"
    runtime = tmp_path / "runtime"
    repo.mkdir()
    (repo / "app.py").write_text("from fastapi import FastAPI\n")
    runtime.mkdir()
    (runtime / "old.txt").write_text("old\n")
    args = [
        "compose-from-route",
        "--skills-dir",
        str(skills_dir),
        "--repo",
        str(repo),
        "--task",
        "Create a FastAPI endpoint with Pydantic validation",
        "--runtime-out",
        str(runtime),
    ]

    assert main(args) == 2
    assert "exists" in capsys.readouterr().err

    assert main([*args, "--overwrite"]) == 0
    assert json.loads(capsys.readouterr().out)["composition_status"] == "written"
    assert not (runtime / "old.txt").exists()
