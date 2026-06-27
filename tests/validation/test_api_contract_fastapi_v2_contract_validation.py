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
MODULE_PATH = ROOT / "artifacts/validation/api_contract_fastapi_v2/validate_v2_contract.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("fastapi_v2_contract_validation", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_representative_examples_fit_budget_and_pass_fixtures(tmp_path, monkeypatch):
    module = _load_module()
    result_path = tmp_path / "validation_results.json"
    monkeypatch.setattr(module, "RESULT_JSON", result_path)

    assert module.main() == 0

    payload = json.loads(result_path.read_text())
    assert payload["active_budget"]["active_generation_max_tokens"] == 256
    assert payload["active_budget"]["maximum_allowed_estimated_target_tokens"] == 192
    assert payload["final_recommendation"] == "proceed_to_v2_data_design"
    assert all(row["fits_under_active_budget"] for row in payload["representative_results"])
    assert all(row["has_required_headroom"] for row in payload["representative_results"])
    assert all(row["fixture_passed"] for row in payload["representative_results"])
    assert not any(row["mixes_app_and_test_code"] for row in payload["representative_results"])
    assert all(value == "safe_for_single_file_v2" for value in payload["shape_risk_classification"].values())


def test_anchor_and_shape_guards_detect_invalid_outputs():
    module = _load_module()

    mixed = "from fastapi import FastAPI\napp = FastAPI()\nfrom fastapi.testclient import TestClient\n"
    assert module.mixes_app_and_test_code(mixed) is True
    assert module.single_artifact("# file: a.py\nprint('x')") is False
    assert module.parse_safely("def broken(:\n") is False

    app_checks = module.app_anchor_checks("from fastapi import FastAPI\napp = FastAPI()\n@app.get('/')\ndef ok():\n    return {}\n")
    assert app_checks["valid_python_syntax"] is True
    assert app_checks["app_fastapi_present"] is True
    assert app_checks["route_decorator_present"] is True

    test_checks = module.test_anchor_checks("from fastapi.testclient import TestClient\nfrom solution import app\nclient = TestClient(app)\n\ndef test_ok():\n    assert client.get('/').status_code == 200\n")
    assert test_checks["test_framework_imports"] is True
    assert test_checks["imports_app_under_test"] is True
    assert test_checks["test_functions_exist"] is True


def test_validation_does_not_create_train_or_holdout_files(tmp_path, monkeypatch):
    module = _load_module()
    result_path = tmp_path / "validation_results.json"
    monkeypatch.setattr(module, "RESULT_JSON", result_path)

    assert module.main() == 0
    created = {path.name for path in tmp_path.iterdir()}
    assert created == {"validation_results.json"}
    assert "train.jsonl" not in created
    assert "holdout.jsonl" not in created