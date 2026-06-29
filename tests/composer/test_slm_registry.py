import copy
import json
from pathlib import Path

import pytest

from scripts.validate_slm_registry import build_report, validate_registry


def _repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("repository root not found")


ROOT = _repo_root()
REGISTRY = ROOT / "src/slmcortex_resources/configs/slm_registry.json"
ROUTER_REPORT = ROOT / "artifacts/governance-fixtures/slmcortex-router-v1/summary.json"


def load_registry():
    return json.loads(REGISTRY.read_text())


def test_registry_tracks_known_slms_and_capacity():
    registry = load_registry()
    validate_registry(registry, json.loads(ROUTER_REPORT.read_text()))
    slms = {slm["slm_name"]: slm for slm in registry["slms"]}

    assert set(slms) == {
        "python_slm",
        "debugging_slm",
        "test_generation_slm",
        "alternating_slm",
    }
    assert all(slms[name]["origin"] == "seed_slm" for name in slms if name != "alternating_slm")
    alternating = slms["alternating_slm"]
    assert alternating["status"] == "promoted"
    assert alternating["origin"] == "failure_born"
    assert alternating["activation_scope"] == "strict_gate"
    assert alternating["historical_quarantine"]["quarantined"] is True
    assert alternating["rollback_supported"] is True
    assert registry["capacity_budget"]["current_total_adapter_parameters"] == 1_245_184


def test_report_answers_governance_questions():
    registry = load_registry()
    report = build_report(registry, json.loads(ROUTER_REPORT.read_text()))

    assert report["core_seed_slms"] == [
        "python_slm",
        "debugging_slm",
        "test_generation_slm",
    ]
    assert report["failure_born_slms"] == ["alternating_slm"]
    assert report["promoted_slms"] == ["alternating_slm"]
    assert report["quarantined_slms"] == []
    assert report["within_capacity_budget"] is True
    assert report["alternating_slm_rollback_supported"] is True
    assert report["ready_for_another_failure_born_slm_experiment"] is True
    assert report["active_parameter_behavior"]["fixed_benchmark_average"] == pytest.approx(
        419_211.94666666666
    )


def test_validation_rejects_missing_promotion_evidence():
    registry = copy.deepcopy(load_registry())
    registry["slms"][-1]["promotion_reason"] = ""

    with pytest.raises(ValueError, match="promotion evidence"):
        validate_registry(registry, json.loads(ROUTER_REPORT.read_text()))


def test_validation_rejects_exceeded_capacity():
    registry = copy.deepcopy(load_registry())
    registry["capacity_budget"]["max_total_adapter_parameters"] = 1_000_000

    with pytest.raises(ValueError, match="capacity budget exceeded"):
        validate_registry(registry, json.loads(ROUTER_REPORT.read_text()))
