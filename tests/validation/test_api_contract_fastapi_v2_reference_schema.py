import importlib.util
import json
import sys
from pathlib import Path


def _repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("repository root not found")


ROOT = _repo_root()
MODULE_PATH = ROOT / "artifacts/validation/api_contract_fastapi_v2/validate_v2_reference_schema.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("fastapi_v2_reference_schema", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_schema_validator_accepts_representative_rows_and_rejects_bad_rows(tmp_path, monkeypatch):
    module = _load_module()
    result_path = tmp_path / "schema_validation_results.json"
    monkeypatch.setattr(module, "RESULT_JSON", result_path)

    assert module.main() == 0

    payload = json.loads(result_path.read_text())
    assert payload["required_fields"] == sorted(module.REQUIRED_FIELDS)
    assert payload["size_limits"] == module.SIZE_LIMITS
    assert len(payload["accepted_representatives"]) == 4
    assert all(row["decision"] == "accept" for row in payload["accepted_representatives"])
    assert len(payload["rejected_representatives"]) == 2
    assert all(row["decision"] == "reject" for row in payload["rejected_representatives"])


def test_schema_validator_enforces_hard_gates_and_isolation_signals():
    module = _load_module()
    accepted = module.validate_row(dict(module.REPRESENTATIVE_ROWS[0]))
    rejected_domain_overlap = module.validate_row(dict(module.REJECTED_ROWS[0]))
    rejected_shape_mixing = module.validate_row(dict(module.REJECTED_ROWS[1]))

    assert accepted["decision"] == "accept"
    assert accepted["size_guard"]["estimated_target_tokens"] <= 192
    assert accepted["size_guard"]["character_count"] <= 768
    assert accepted["size_guard"]["non_empty_line_count"] <= 32
    assert rejected_domain_overlap["decision"] == "reject"
    assert any(reason.startswith("leakage_overlap_") for reason in rejected_domain_overlap["reasons"])
    assert rejected_shape_mixing["decision"] == "reject"
    assert "shape_gate_no_app_test_mixing" in rejected_shape_mixing["reasons"]


def test_schema_validator_writes_only_report_file(tmp_path, monkeypatch):
    module = _load_module()
    result_path = tmp_path / "schema_validation_results.json"
    monkeypatch.setattr(module, "RESULT_JSON", result_path)

    assert module.main() == 0
    created = {path.name for path in tmp_path.iterdir()}
    assert created == {"schema_validation_results.json"}
    assert "train.jsonl" not in created
    assert "holdout.jsonl" not in created