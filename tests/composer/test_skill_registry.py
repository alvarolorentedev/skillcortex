import copy
import json
from pathlib import Path

import pytest

from scripts.validate_skill_registry import build_report, validate_registry


def _repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("repository root not found")


ROOT = _repo_root()
REGISTRY = ROOT / "configs/skill_registry.json"
ROUTER_REPORT = ROOT / "artifacts/governance-fixtures/skillcortex-router-v1/summary.json"


def load_registry():
    return json.loads(REGISTRY.read_text())


def test_registry_tracks_known_skills_and_capacity():
    registry = load_registry()
    validate_registry(registry, json.loads(ROUTER_REPORT.read_text()))
    skills = {skill["skill_name"]: skill for skill in registry["skills"]}

    assert set(skills) == {
        "python_skill",
        "debugging_skill",
        "test_generation_skill",
        "alternating_skill",
    }
    assert all(skills[name]["origin"] == "seed_skill" for name in skills if name != "alternating_skill")
    alternating = skills["alternating_skill"]
    assert alternating["status"] == "promoted"
    assert alternating["origin"] == "failure_born"
    assert alternating["activation_scope"] == "strict_gate"
    assert alternating["historical_quarantine"]["quarantined"] is True
    assert alternating["rollback_supported"] is True
    assert registry["capacity_budget"]["current_total_adapter_parameters"] == 1_245_184


def test_report_answers_governance_questions():
    registry = load_registry()
    report = build_report(registry, json.loads(ROUTER_REPORT.read_text()))

    assert report["core_seed_skills"] == [
        "python_skill",
        "debugging_skill",
        "test_generation_skill",
    ]
    assert report["failure_born_skills"] == ["alternating_skill"]
    assert report["promoted_skills"] == ["alternating_skill"]
    assert report["quarantined_skills"] == []
    assert report["within_capacity_budget"] is True
    assert report["alternating_skill_rollback_supported"] is True
    assert report["ready_for_another_failure_born_skill_experiment"] is True
    assert report["active_parameter_behavior"]["fixed_benchmark_average"] == pytest.approx(
        419_211.94666666666
    )


def test_validation_rejects_missing_promotion_evidence():
    registry = copy.deepcopy(load_registry())
    registry["skills"][-1]["promotion_reason"] = ""

    with pytest.raises(ValueError, match="promotion evidence"):
        validate_registry(registry, json.loads(ROUTER_REPORT.read_text()))


def test_validation_rejects_exceeded_capacity():
    registry = copy.deepcopy(load_registry())
    registry["capacity_budget"]["max_total_adapter_parameters"] = 1_000_000

    with pytest.raises(ValueError, match="capacity budget exceeded"):
        validate_registry(registry, json.loads(ROUTER_REPORT.read_text()))
