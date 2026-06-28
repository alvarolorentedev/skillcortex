import json

import pytest
import yaml

from slmcortex.catalog import SkillCatalog, route_task


def write_skill(root, name, payload, *, routing_card=None, optional_bad=False):
    package = root / name
    package.mkdir(parents=True)
    (package / "skill.yaml").write_text(yaml.safe_dump(payload, sort_keys=False))
    if routing_card is not None:
        text = "{bad json" if optional_bad else json.dumps(routing_card)
        (package / "routing_card.json").write_text(text + "\n")
    return package


def fastapi_skill():
    return {
        "skill_id": "fastapi_contract",
        "name": "FastAPI Contract Skill",
        "description": "Improves FastAPI endpoints with Pydantic validation.",
        "capabilities": ["fastapi", "pydantic", "api endpoint creation", "request validation"],
        "activation_cues": ["FastAPI", "Pydantic", "APIRouter", "response model"],
        "avoid_when": ["frontend-only task"],
        "base_model": "demo-base",
        "adapter_path": "adapter",
    }


def test_discovers_valid_skill_with_missing_optional_files(tmp_path):
    skills_dir = tmp_path / "skills"
    write_skill(skills_dir, "fastapi_contract", fastapi_skill())

    result = SkillCatalog.discover(skills_dir)

    assert [skill.skill_id for skill in result.skills] == ["fastapi_contract"]
    assert result.errors == []
    assert result.warnings == []
    assert result.skills[0].adapter_path.name == "adapter"


def test_invalid_required_metadata_skips_skill(tmp_path):
    skills_dir = tmp_path / "skills"
    write_skill(skills_dir, "bad", {"name": "Missing id"})

    result = SkillCatalog.discover(skills_dir)

    assert result.skills == []
    assert result.errors
    assert "skill_id" in result.errors[0]


def test_invalid_optional_metadata_warns_without_blocking(tmp_path):
    skills_dir = tmp_path / "skills"
    write_skill(skills_dir, "fastapi_contract", fastapi_skill(), routing_card={}, optional_bad=True)

    result = SkillCatalog.discover(skills_dir)

    assert [skill.skill_id for skill in result.skills] == ["fastapi_contract"]
    assert result.errors == []
    assert result.warnings
    assert "routing_card.json" in result.warnings[0]


def test_old_task_type_maps_to_hint(tmp_path):
    skills_dir = tmp_path / "skills"
    payload = {"skill_id": "legacy", "name": "Legacy", "task_type": "api_generation"}
    write_skill(skills_dir, "legacy", payload)

    result = SkillCatalog.discover(skills_dir)

    assert result.skills[0].task_type_hint == "api_generation"


def test_routes_fastapi_task_and_rejects_unrelated_skill(tmp_path):
    skills_dir = tmp_path / "skills"
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "pyproject.toml").write_text('[project]\ndependencies = ["fastapi", "pydantic"]\n')
    write_skill(
        skills_dir,
        "fastapi_contract",
        fastapi_skill(),
        routing_card={
            "positive_examples": ["Create a FastAPI endpoint with Pydantic validation"],
            "negative_examples": ["Fix a React hydration bug"],
        },
    )
    write_skill(
        skills_dir,
        "react_ui",
        {
            "skill_id": "react_ui",
            "name": "React UI Skill",
            "description": "Fixes React hydration and frontend components.",
            "capabilities": ["react", "frontend", "hydration"],
            "activation_cues": ["React", "component"],
            "avoid_when": ["backend api task"],
        },
    )

    decision = route_task(
        skills_dir=skills_dir,
        repo=repo,
        task="Create a FastAPI endpoint for creating a user with Pydantic validation",
        explain=True,
        current_base_model="demo-base",
    )

    assert decision["selected_skills"][0]["skill_id"] == "fastapi_contract"
    candidates = {item["skill_id"]: item for item in decision["candidates"]}
    assert candidates["fastapi_contract"]["selected"] is True
    assert candidates["react_ui"]["selected"] is False
    assert candidates["fastapi_contract"]["score_breakdown"]


def test_known_incompatible_base_model_prevents_selection(tmp_path):
    skills_dir = tmp_path / "skills"
    repo = tmp_path / "repo"
    repo.mkdir()
    write_skill(skills_dir, "fastapi_contract", fastapi_skill())

    decision = route_task(
        skills_dir=skills_dir,
        repo=repo,
        task="Create a FastAPI endpoint with Pydantic validation",
        current_base_model="other-base",
    )

    candidate = decision["candidates"][0]
    assert candidate["compatible"] is False
    assert candidate["selected"] is False
    assert decision["selected_skills"] == []


def test_unknown_base_model_does_not_penalize_declared_base(tmp_path):
    skills_dir = tmp_path / "skills"
    repo = tmp_path / "repo"
    repo.mkdir()
    write_skill(skills_dir, "fastapi_contract", fastapi_skill())

    decision = route_task(
        skills_dir=skills_dir,
        repo=repo,
        task="Create a FastAPI endpoint with Pydantic validation",
    )

    assert decision["candidates"][0]["compatible"] is True
    assert decision["selected_skills"][0]["skill_id"] == "fastapi_contract"


def test_repo_scan_is_bounded_and_skips_binary_and_ignored_dirs(tmp_path, monkeypatch):
    skills_dir = tmp_path / "skills"
    repo = tmp_path / "repo"
    (repo / "node_modules").mkdir(parents=True)
    (repo / "node_modules" / "fastapi.txt").write_text("fastapi pydantic")
    (repo / "app.py").write_text("from fastapi import FastAPI\n")
    (repo / "blob.bin").write_bytes(b"\x00\x01\x02fastapi")
    write_skill(skills_dir, "fastapi_contract", fastapi_skill())

    decision = route_task(
        skills_dir=skills_dir,
        repo=repo,
        task="Create an endpoint",
    )

    context = decision["repo_context"]
    assert "fastapi" in context["framework_signals"]
    assert context["files_scanned"] == 1
    assert context["skipped_binary_files"] == 1
    assert "node_modules/fastapi.txt" not in context["scanned_files"]
