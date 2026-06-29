import hashlib
import json
from pathlib import Path

import yaml

from scripts.build_slmcortex_router_v1_report import main
from slmcortex.contracts import PROMOTED_SLMS, QUARANTINED_SLMS


def _repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("repository root not found")


ROOT = _repo_root()
SOURCE = ROOT / "artifacts/governance-fixtures/alternating_slm/summary.json"


def test_alternating_slm_is_promoted_and_historical_quarantine_is_preserved(tmp_path):
    source_before = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    benchmark = ROOT / "data/eval.jsonl"
    benchmark_before = hashlib.sha256(benchmark.read_bytes()).hexdigest()

    assert "alternating_slm" in PROMOTED_SLMS
    assert "alternating_slm" not in QUARANTINED_SLMS
    assert "alternating_slm" in yaml.safe_load(
        (ROOT / "src/slmcortex_resources/configs/slms.yaml").read_text()
    )["slms"]

    assert main(["--source", str(SOURCE), "--output", str(tmp_path)]) == 0

    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == source_before
    assert hashlib.sha256(benchmark.read_bytes()).hexdigest() == benchmark_before
    historical = json.loads(SOURCE.read_text())
    assert historical["quarantine"]["active_by_default"] is False
    assert historical["quarantine"]["auto_promote"] is False
    assert historical["promotion_decision"]["status"] == "recommend_promotion"


def test_report_reuses_fixed_and_holdout_results_without_training_or_inference(tmp_path):
    assert main(["--source", str(SOURCE), "--output", str(tmp_path)]) == 0
    summary = json.loads((tmp_path / "summary.json").read_text())

    assert summary["validation"] == {
        "uses_existing_artifacts": True,
        "new_training": False,
        "new_inference": False,
        "integration_validation_only": True,
    }
    assert set(summary["fixed_benchmark"]["routers"]) == {
        "protected_slm_router_without_failure_born",
        "slmcortex_router_v1",
    }
    assert set(summary["independent_alternating_holdout"]["routers"]) == {
        "protected_slm_router_without_failure_born",
        "slmcortex_router_v1",
    }
    assert summary["fixed_benchmark"]["pass_fail_vs_previous_protected_router"] == {
        "fail_to_pass": 5,
        "pass_to_fail": 0,
    }
    assert summary["independent_alternating_holdout"][
        "pass_fail_vs_previous_protected_router"
    ] == {"fail_to_pass": 38, "pass_to_fail": 0}
    markdown = (tmp_path / "summary.md").read_text()
    assert "Fixed benchmark" in markdown
    assert "Independent alternating holdout" in markdown
