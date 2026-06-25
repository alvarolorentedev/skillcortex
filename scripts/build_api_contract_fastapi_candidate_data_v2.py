#!/usr/bin/env python3
"""Build and validate quarantined v2 FastAPI candidate reference data deterministically."""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path


SEED = 240624
CANDIDATE = "api_contract_fastapi_skill"
FAMILY = "fastapi_contract"
VERSION = "v2"
SCHEMA_VERSION = 2
GENERATOR_VERSION = 2
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "data/failure_born/api_contract_fastapi_skill/v2"
VALIDATOR_PATH = ROOT / "artifacts/experiments/api_contract_fastapi_skill/v2-contract-validation/validate_v2_reference_schema.py"
FASTAPI_BENCHMARK = ROOT / "data/benchmarks/fastapi_contract/v1/benchmark.jsonl"
EXISTING_BENCHMARK = ROOT / "data/eval.jsonl"
V1_TRAIN = ROOT / "data/failure_born/api_contract_fastapi_skill/v1/train.jsonl"
V1_HOLDOUT = ROOT / "data/failure_born/api_contract_fastapi_skill/v1/holdout.jsonl"
V1_MANIFEST = ROOT / "data/failure_born/api_contract_fastapi_skill/v1/manifest.json"
ROUTER = ROOT / "src/skill_lattice_coder/router.py"
REGISTRY = ROOT / "configs/skill_registry.json"

TASKS = (
    "fastapi_contract_generation",
    "fastapi_contract_debugging",
    "fastapi_contract_test_generation",
    "fastapi_contract_refactor",
)
GROUPS = (
    "request_body_validation",
    "path_parameter_validation",
    "response_model_correctness",
    "status_code_correctness",
    "not_found_error_behavior",
    "invalid_state_error_behavior",
    "list_envelope_shape",
    "enum_literal_validation",
    "optional_field_handling",
    "contract_preserving_refactor",
    "contract_drift_detection",
    "bounded_route_service_separation",
)
DOMAINS = {
    "train": (
        "ledgers",
        "rosters",
        "journals",
        "permits",
        "badges",
        "bundles",
        "digests",
        "kiosks",
        "charters",
        "folios",
        "capsules",
        "playlists",
    ),
    "holdout": (
        "vouchers",
        "lockers",
        "forums",
        "clinics",
        "memos",
        "audits",
        "parcels",
        "recipes",
        "galleries",
        "notebooks",
        "briefings",
        "terminals",
    ),
}
FIXED_DOMAINS = {
    "payments", "reports", "devices", "comments", "teams", "schedules",
    "reviews", "catalogs", "messages", "profiles", "events", "quotes",
    "orders", "tickets", "invoices", "products", "customers", "tasks",
    "projects", "assets", "notifications", "files", "memberships", "approvals",
    "bookings", "shipments", "subscriptions", "alerts", "documents", "workspaces",
    "incidents", "surveys", "contracts", "warehouses", "calendars", "expenses",
}
PRIMARY_MUTANT_CLASSES = {
    "request_body_validation": "permissive request model",
    "path_parameter_validation": "wrong path parameter type",
    "response_model_correctness": "missing required response field",
    "status_code_correctness": "wrong success status",
    "not_found_error_behavior": "wrong error status",
    "invalid_state_error_behavior": "wrong error status",
    "list_envelope_shape": "wrong list envelope",
    "enum_literal_validation": "wrong enum handling",
    "optional_field_handling": "wrong optional field behavior",
    "contract_preserving_refactor": "changed public contract",
    "contract_drift_detection": "drifted response contract",
    "bounded_route_service_separation": "missing service helper",
}


def _load_validator():
    spec = importlib.util.spec_from_file_location("fastapi_v2_reference_schema", VALIDATOR_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


VALIDATOR = _load_validator()
REQUIRED_FIELDS = VALIDATOR.REQUIRED_FIELDS
SIZE_LIMITS = VALIDATOR.SIZE_LIMITS


def _sha_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _jsonl(rows: list[dict[str, object]]) -> str:
    return "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows)


def _pytest_command() -> list[str]:
    if importlib.util.find_spec("pytest") is not None:
        return [sys.executable, "-m", "pytest"]
    if shutil.which("uv"):
        return [
            shutil.which("uv"),
            "run",
            "--project",
            str(ROOT),
            "--extra",
            "test",
            "python",
            "-m",
            "pytest",
        ]
    raise RuntimeError("pytest is unavailable for fixture validation")


def _title(domain: str) -> str:
    return "".join(part.title() for part in domain.split("_"))


def _model_name(prefix: str, domain: str) -> str:
    return f"{prefix}{_title(domain)}"


def _route_name(prefix: str, domain: str) -> str:
    return f"{prefix}_{domain}"


def _helper_name(domain: str) -> str:
    return f"_build_{domain}"


def _path(domain: str) -> str:
    return f"/{domain}"


def _verifier(primary: str, secondary: str) -> str:
    command = _pytest_command() + ["-q", "test_generated.py"]
    return (
        "import os\nimport pathlib\nimport shutil\nimport subprocess\n\n"
        "root = pathlib.Path(__file__).parent\n"
        "env = {**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'}\n"
        f"command = {command!r}\n"
        "def run():\n"
        "    shutil.rmtree(root / '__pycache__', ignore_errors=True)\n"
        "    return subprocess.run(command, cwd=root, env=env).returncode\n"
        "correct = run()\n"
        f"(root / 'solution.py').write_text({primary!r})\n"
        "primary = run()\n"
        f"(root / 'solution.py').write_text({secondary!r})\n"
        "secondary = run()\n"
        "raise SystemExit(0 if correct == 0 and primary != 0 and secondary != 0 else 1)\n"
    )


def _request_app(domain: str) -> str:
    model = _model_name("Create", domain)
    func = _route_name("create", domain)
    return (
        "from fastapi import FastAPI\n"
        "from pydantic import BaseModel, Field\n\n"
        "app = FastAPI()\n\n"
        f"class {model}(BaseModel):\n"
        "    name: str = Field(min_length=1)\n\n"
        f"@app.post('{_path(domain)}', status_code=201)\n"
        f"def {func}(payload: {model}):\n"
        "    return {'id': 1, 'name': payload.name}\n"
    )


def _path_app(domain: str) -> str:
    func = _route_name("get", domain)
    return (
        "from fastapi import FastAPI, Path\n\n"
        "app = FastAPI()\n\n"
        f"@app.get('{_path(domain)}/{{item_id}}')\n"
        f"def {func}(item_id: int = Path(ge=1)):\n"
        "    return {'id': item_id}\n"
    )


def _response_app(domain: str) -> str:
    model = _model_name("Create", domain)
    func = _route_name("create", domain)
    return (
        "from typing import Literal\n\n"
        "from fastapi import FastAPI\n"
        "from pydantic import BaseModel, Field\n\n"
        "app = FastAPI()\n\n"
        f"class {model}(BaseModel):\n"
        "    name: str = Field(min_length=1)\n\n"
        f"@app.post('{_path(domain)}', status_code=201)\n"
        f"def {func}(payload: {model}):\n"
        "    return {'id': 1, 'name': payload.name, 'status': 'active', 'note': None}\n"
    )


def _not_found_app(domain: str) -> str:
    func = _route_name("get", domain)
    detail = f"{domain} missing record"
    return (
        "from fastapi import FastAPI, HTTPException, Path\n\n"
        "app = FastAPI()\n"
        "items = {1: {'id': 1, 'name': 'stored'}}\n\n"
        f"@app.get('{_path(domain)}/{{item_id}}')\n"
        f"def {func}(item_id: int = Path(ge=1)):\n"
        "    if item_id not in items:\n"
        f"        raise HTTPException(status_code=404, detail='{detail}')\n"
        "    return items[item_id]\n"
    )


def _invalid_state_app(domain: str) -> str:
    model = _model_name("Patch", domain)
    func = _route_name("patch", domain)
    detail = f"{domain} state conflict"
    return (
        "from typing import Literal\n\n"
        "from fastapi import FastAPI, HTTPException, Path\n"
        "from pydantic import BaseModel\n\n"
        "app = FastAPI()\n"
        "items = {1: {'id': 1, 'status': 'closed'}}\n\n"
        f"class {model}(BaseModel):\n"
        "    status: Literal['active', 'closed']\n\n"
        f"@app.patch('{_path(domain)}/{{item_id}}')\n"
        f"def {func}(payload: {model}, item_id: int = Path(ge=1)):\n"
        "    if items[item_id]['status'] == 'closed':\n"
        f"        raise HTTPException(status_code=409, detail='{detail}')\n"
        "    items[item_id]['status'] = payload.status\n"
        "    return items[item_id]\n"
    )


def _list_app(domain: str) -> str:
    model = _model_name("Create", domain)
    create_func = _route_name("create", domain)
    list_func = _route_name("list", domain)
    return (
        "from fastapi import FastAPI\n"
        "from pydantic import BaseModel, Field\n\n"
        "app = FastAPI()\n"
        "items = []\n\n"
        f"class {model}(BaseModel):\n"
        "    name: str = Field(min_length=1)\n\n"
        f"@app.post('{_path(domain)}', status_code=201)\n"
        f"def {create_func}(payload: {model}):\n"
        "    item = {'id': len(items) + 1, 'name': payload.name, 'status': 'active', 'note': None}\n"
        "    items.append(item)\n"
        "    return item\n\n"
        f"@app.get('{_path(domain)}')\n"
        f"def {list_func}():\n"
        "    return {'items': items, 'total': len(items)}\n"
    )


def _enum_app(domain: str) -> str:
    model = _model_name("Create", domain)
    func = _route_name("create", domain)
    return (
        "from typing import Literal\n\n"
        "from fastapi import FastAPI\n"
        "from pydantic import BaseModel, Field\n\n"
        "app = FastAPI()\n\n"
        f"class {model}(BaseModel):\n"
        "    name: str = Field(min_length=1)\n"
        "    status: Literal['active', 'pending'] = 'active'\n\n"
        f"@app.post('{_path(domain)}', status_code=201)\n"
        f"def {func}(payload: {model}):\n"
        "    return {'id': 1, 'name': payload.name, 'status': payload.status}\n"
    )


def _optional_app(domain: str) -> str:
    model = _model_name("Create", domain)
    func = _route_name("create", domain)
    return (
        "from fastapi import FastAPI\n"
        "from pydantic import BaseModel, Field\n\n"
        "app = FastAPI()\n\n"
        f"class {model}(BaseModel):\n"
        "    name: str = Field(min_length=1)\n"
        "    note: str | None = None\n\n"
        f"@app.post('{_path(domain)}', status_code=201)\n"
        f"def {func}(payload: {model}):\n"
        "    return {'id': 1, 'name': payload.name, 'note': payload.note}\n"
    )


def _refactor_contract_app(domain: str) -> str:
    model = _model_name("Create", domain)
    helper = _helper_name(domain)
    create_func = _route_name("create", domain)
    get_func = _route_name("get", domain)
    return (
        "from fastapi import FastAPI, Path\n"
        "from pydantic import BaseModel, Field\n\n"
        "app = FastAPI()\n"
        "items = {}\n\n"
        f"class {model}(BaseModel):\n"
        "    name: str = Field(min_length=1)\n\n"
        f"def {helper}(payload: {model}) -> dict:\n"
        "    item = {'id': len(items) + 1, 'name': payload.name}\n"
        "    items[item['id']] = item\n"
        "    return item\n\n"
        f"@app.post('{_path(domain)}', status_code=201)\n"
        f"def {create_func}(payload: {model}):\n"
        f"    return {helper}(payload)\n\n"
        f"@app.get('{_path(domain)}/{{item_id}}')\n"
        f"def {get_func}(item_id: int = Path(ge=1)):\n"
        "    return items[item_id]\n"
    )


def _drift_app(domain: str) -> str:
    model = _model_name("Create", domain)
    func = _route_name("create", domain)
    return (
        "from typing import Literal\n\n"
        "from fastapi import FastAPI\n"
        "from pydantic import BaseModel, Field\n\n"
        "app = FastAPI()\n\n"
        f"class {model}(BaseModel):\n"
        "    name: str = Field(min_length=1)\n"
        "    status: Literal['active', 'pending'] = 'active'\n\n"
        f"@app.post('{_path(domain)}', status_code=201)\n"
        f"def {func}(payload: {model}):\n"
        "    return {'id': 1, 'name': payload.name, 'status': payload.status}\n"
    )


def _service_app(domain: str) -> str:
    model = _model_name("Create", domain)
    helper = _helper_name(domain)
    func = _route_name("create", domain)
    return (
        "from fastapi import FastAPI\n"
        "from pydantic import BaseModel, Field\n\n"
        "app = FastAPI()\n\n"
        f"class {model}(BaseModel):\n"
        "    name: str = Field(min_length=1)\n\n"
        f"def {helper}(payload: {model}) -> dict:\n"
        "    return {'id': 1, 'name': payload.name}\n\n"
        f"@app.post('{_path(domain)}', status_code=201)\n"
        f"def {func}(payload: {model}):\n"
        f"    return {helper}(payload)\n"
    )


APP_BUILDERS = {
    "request_body_validation": _request_app,
    "path_parameter_validation": _path_app,
    "response_model_correctness": _response_app,
    "status_code_correctness": _request_app,
    "not_found_error_behavior": _not_found_app,
    "invalid_state_error_behavior": _invalid_state_app,
    "list_envelope_shape": _list_app,
    "enum_literal_validation": _enum_app,
    "optional_field_handling": _optional_app,
    "contract_preserving_refactor": _refactor_contract_app,
    "contract_drift_detection": _drift_app,
    "bounded_route_service_separation": _service_app,
}


def _primary_mutant(group: str, app: str, domain: str) -> str:
    replacements = {
        "request_body_validation": ("min_length=1", "min_length=0"),
        "path_parameter_validation": ("Path(ge=1)", "Path(ge=0)"),
        "response_model_correctness": ("'note': None",),
        "status_code_correctness": ("status_code=201", "status_code=200"),
        "not_found_error_behavior": ("status_code=404", "status_code=400"),
        "invalid_state_error_behavior": ("status_code=409", "status_code=400"),
        "list_envelope_shape": ("'total': len(items)",),
        "enum_literal_validation": ("Literal['active', 'pending']", "str"),
        "optional_field_handling": ("payload.note", "''"),
        "contract_preserving_refactor": ("len(items) + 1", "99"),
        "contract_drift_detection": ("payload.status", "'active'"),
        "bounded_route_service_separation": (_helper_name(domain), f"missing_{domain}"),
    }
    if group == "response_model_correctness":
        return app.replace(", 'note': None", "", 1)
    if group == "list_envelope_shape":
        return app.replace("{'items': items, 'total': len(items)}", "items", 1)
    old, new = replacements[group]
    return app.replace(old, new, 1)


def _secondary_mutant(app: str, domain: str) -> str:
    return app.replace(f"'{_path(domain)}", f"'{_path(domain)}-changed", 1)


def _tangled_app(group: str, app: str, domain: str) -> str:
    helper = _helper_name(domain)
    if helper not in app:
        return app
    lines = app.splitlines()
    helper_body = []
    capture = False
    for line in lines:
        if line.startswith(f"def {helper}("):
            capture = True
            continue
        if capture and line.startswith("@app"):
            break
        if capture:
            helper_body.append(line)
    body = "\n".join(line for line in helper_body if line.strip())
    tangled = app.replace("\n\n" + "\n".join([lines[i] for i, line in enumerate(lines) if line.startswith(f"def {helper}(") or (i > 0 and lines[i - 1].startswith(f"def {helper}(") )]), "")
    if body:
        return app.replace(f"    return {helper}(payload)", body.replace("    ", "", 1), 1)
    return app


def _test_file(group: str, domain: str) -> str:
    path = _path(domain)
    helper = _helper_name(domain)
    tests = {
        "request_body_validation": (
            f"assert client.post('{path}', json={{'name': ''}}).status_code == 422\n"
            f"    assert client.post('{path}', json={{'name': 'sample-{domain}'}}).status_code == 201"
        ),
        "path_parameter_validation": (
            f"assert client.get('{path}/0').status_code == 422\n"
            f"    assert client.get('{path}/1').json() == {{'id': 1}}"
        ),
        "response_model_correctness": (
            f"response = client.post('{path}', json={{'name': 'sample-{domain}'}})\n"
            "    assert set(response.json()) == {'id', 'name', 'status', 'note'}"
        ),
        "status_code_correctness": (
            f"assert client.post('{path}', json={{'name': 'sample-{domain}'}}).status_code == 201"
        ),
        "not_found_error_behavior": (
            f"response = client.get('{path}/999')\n"
            "    assert response.status_code == 404\n"
            f"    assert response.json()['detail'] == '{domain} missing record'"
        ),
        "invalid_state_error_behavior": (
            f"assert client.patch('{path}/1', json={{'status': 'active'}}).status_code == 409"
        ),
        "list_envelope_shape": (
            f"client.post('{path}', json={{'name': 'sample-{domain}'}})\n"
            f"    assert client.get('{path}').json() == {{'items': [{{'id': 1, 'name': 'sample-{domain}', 'status': 'active', 'note': None}}], 'total': 1}}"
        ),
        "enum_literal_validation": (
            f"assert client.post('{path}', json={{'name': 'sample-{domain}', 'status': 'unknown'}}).status_code == 422"
        ),
        "optional_field_handling": (
            f"missing = client.post('{path}', json={{'name': 'sample-{domain}'}}).json()\n"
            f"    explicit = client.post('{path}', json={{'name': 'sample-{domain}-b', 'note': None}}).json()\n"
            f"    supplied = client.post('{path}', json={{'name': 'sample-{domain}-c', 'note': 'note-{domain}'}}).json()\n"
            "    assert [missing['note'], explicit['note'], supplied['note']] == [None, None, 'note-" + domain + "']"
        ),
        "contract_preserving_refactor": (
            f"created = client.post('{path}', json={{'name': 'sample-{domain}'}})\n"
            "    assert created.status_code == 201\n"
            f"    assert client.get('{path}/1').json() == created.json()"
        ),
        "contract_drift_detection": (
            f"response = client.post('{path}', json={{'name': 'sample-{domain}', 'status': 'pending'}})\n"
            "    assert response.status_code == 201\n"
            "    assert response.json()['status'] == 'pending'"
        ),
        "bounded_route_service_separation": (
            f"from solution import {helper}\n"
            f"    assert callable({helper})\n"
            f"    assert client.post('{path}', json={{'name': 'sample-{domain}'}}).status_code == 201"
        ),
    }
    return (
        "from fastapi.testclient import TestClient\n"
        "from solution import app\n\n"
        "client = TestClient(app)\n\n"
        f"def test_{group}_{domain}():\n"
        f"    {tests[group]}\n"
    )


def _base_row(split: str, task: str, group: str, domain: str, index: int) -> dict[str, object]:
    app = APP_BUILDERS[group](domain)
    tests = _test_file(group, domain)
    primary = _primary_mutant(group, app, domain)
    secondary = _secondary_mutant(app, domain)
    kind = task.removeprefix("fastapi_contract_")
    prefix = "Training exercise" if split == "train" else "Independent holdout challenge"
    prompt = f"{prefix}: {kind.replace('_', ' ')} the bounded `{_path(domain)}` FastAPI contract for {group.replace('_', ' ')}. Return code only."
    if kind == "debugging":
        prompt += "\n\nBroken app:\n" + primary
    elif kind == "refactor":
        prompt += "\n\nWorking tangled app:\n" + _tangled_app(group, app, domain)
    elif kind == "test_generation":
        prompt += "\n\nCorrect app:\n" + app

    if kind == "test_generation":
        artifact_type = "test_file"
        target = tests
        execution = {
            "files": {
                "solution.py": app,
                "verify_tests.py": _verifier(primary, secondary),
            },
            "command": ["python", "verify_tests.py"],
            "timeout_seconds": 20,
        }
        mutants_meta = [
            {"class": PRIMARY_MUTANT_CLASSES[group], "sha256": _sha_text(primary)},
            {"class": "changed route path or method", "sha256": _sha_text(secondary)},
        ]
    else:
        artifact_type = "app_file"
        target = app
        execution = {
            "files": {"test_contract.py": tests},
            "command": ["python", "-m", "pytest", "-q"],
            "timeout_seconds": 20,
        }
        mutants_meta = []

    return {
        "id": f"api-contract-candidate-v2-{split}-{kind}-{index:03d}",
        "version": VERSION,
        "split": split,
        "candidate_skill": CANDIDATE,
        "benchmark_family": FAMILY,
        "schema_version": SCHEMA_VERSION,
        "task_type": task,
        "behavior_group": group,
        "domain": domain,
        "artifact_type": artifact_type,
        "prompt": prompt,
        "target": target,
        "execution": execution,
        "metadata": {
            "source": "deterministic_v2_generator",
            "evaluation_only": False,
            "active_by_default": False,
            "generation_source": "build_api_contract_fastapi_candidate_data_v2",
            "generation_version": GENERATOR_VERSION,
            "deterministic_seed": SEED,
            "mutants": mutants_meta,
        },
        "size_guard": {},
        "shape_guard": {},
        "anchor_guard": {},
        "fixture_guard": {},
        "leakage_guard": {
            "normalized_case_key": f"{split}:{task}:{group}:{domain}:001",
        },
    }


def _field_sets(rows: list[dict[str, object]]) -> dict[str, set[str]]:
    surfaces = {
        "domains": set(),
        "paths": set(),
        "models": set(),
        "functions": set(),
        "payload_literals": set(),
        "error_messages": set(),
        "prompts": set(),
        "targets": set(),
        "fixtures": set(),
        "mutant_sources": set(),
        "ast_hashes": set(),
        "case_keys": set(),
    }
    for row in rows:
        fields = VALIDATOR.extract_surface_fields(row)
        for key in surfaces:
            surfaces[key].update(value for value in fields[key] if value)
    return surfaces


def _assert_disjoint(left: dict[str, set[str]], right: dict[str, set[str]], labels: tuple[str, ...]) -> None:
    for label in labels:
        overlap = left[label] & right[label]
        if overlap:
            raise ValueError(f"{label} overlap: {sorted(overlap)[:3]}")


def _protected_hashes() -> dict[Path, str]:
    return {path: _hash_file(path) for path in (FASTAPI_BENCHMARK, EXISTING_BENCHMARK, V1_TRAIN, V1_HOLDOUT, V1_MANIFEST, ROUTER, REGISTRY)}


def validate_row(row: dict[str, object], execute_fixtures: bool = True) -> dict[str, object]:
    reasons: list[str] = []
    if set(row) != REQUIRED_FIELDS:
        reasons.append("schema_fields_mismatch")
    if row["candidate_skill"] != CANDIDATE:
        reasons.append("candidate_skill_mismatch")
    if row["benchmark_family"] != FAMILY:
        reasons.append("benchmark_family_mismatch")
    size_ok, size_reasons = VALIDATOR.validate_size_guard(row)
    shape_ok, shape_reasons = VALIDATOR.validate_shape_guard(row)
    task_ok, task_reasons = VALIDATOR.validate_task_artifact_rules(row)
    anchor_ok, anchor_reasons = VALIDATOR.validate_anchor_guard(row)
    reasons.extend(size_reasons + shape_reasons + task_reasons + anchor_reasons)

    if execute_fixtures:
        fixture_ok, fixture_reasons = VALIDATOR.validate_fixture_guard(row)
        reasons.extend(fixture_reasons)
    else:
        fixture_ok = True
        row["fixture_guard"].update({"fixture_validation_skipped": True})

    leakage_ok, leakage_reasons = VALIDATOR.validate_leakage_guard(row)
    reasons.extend(leakage_reasons)
    decision = "accept" if all((size_ok, shape_ok, task_ok, anchor_ok, fixture_ok, leakage_ok)) and not reasons else "reject"
    row["metadata"].update({
        "artifact_type": row["artifact_type"],
        "anchor_validation_result": anchor_ok,
        "fixture_validation_result": fixture_ok,
        "leakage_validation_result": leakage_ok,
        "normalized_ast_sha256": row["leakage_guard"].get("normalized_ast_sha256", VALIDATOR.normalized_ast_sha(row["target"])),
        "normalized_case_key": row["leakage_guard"]["normalized_case_key"],
        "curation_decision": decision,
        "rejection_reason": reasons[0] if reasons else None,
        "all_rejection_reasons": reasons,
    })
    if decision != "accept":
        raise ValueError(f"row rejected: {row['id']} {reasons[:3]}")
    return row


def build_candidate_data_v2(execute_fixtures: bool = True) -> dict[str, object]:
    rows_by_split: dict[str, list[dict[str, object]]] = {"train": [], "holdout": []}
    for split in ("train", "holdout"):
        index = 1
        for task in TASKS:
            for group, domain in zip(GROUPS, DOMAINS[split]):
                row = _base_row(split, task, group, domain, index)
                rows_by_split[split].append(validate_row(row, execute_fixtures=execute_fixtures))
                index += 1

    train = rows_by_split["train"]
    holdout = rows_by_split["holdout"]
    manifest = _manifest(train, holdout, execute_fixtures=execute_fixtures)
    result = {"train": train, "holdout": holdout, "manifest": manifest}
    validate_candidate_data_v2(result, execute_fixtures=execute_fixtures)
    return result


def validate_candidate_data_v2(result: dict[str, object], execute_fixtures: bool = True) -> bool:
    before = _protected_hashes()
    train = result["train"]
    holdout = result["holdout"]
    rows = train + holdout

    if len(train) != 48 or len(holdout) != 48:
        raise ValueError("expected 48 train and 48 holdout rows")
    if len(rows) != 96:
        raise ValueError("expected 96 total rows")
    if set(DOMAINS["train"]) & set(DOMAINS["holdout"]):
        raise ValueError("train and holdout domains overlap")
    if (set(DOMAINS["train"]) | set(DOMAINS["holdout"])) & FIXED_DOMAINS:
        raise ValueError("v2 domains overlap excluded domains")

    for split, split_rows in (("train", train), ("holdout", holdout)):
        if any(set(row) != REQUIRED_FIELDS for row in split_rows):
            raise ValueError("row schema mismatch")
        if any(row["split"] != split for row in split_rows):
            raise ValueError("split metadata mismatch")
        if any(row["metadata"]["curation_decision"] != "accept" for row in split_rows):
            raise ValueError("rejected row present in accepted candidate set")
        if set(Counter(row["task_type"] for row in split_rows).values()) != {12}:
            raise ValueError("unexpected task counts")
        if set(Counter(row["behavior_group"] for row in split_rows).values()) != {4}:
            raise ValueError("unexpected behavior counts")
        if any(row["size_guard"]["estimated_target_tokens"] > SIZE_LIMITS["estimated_target_tokens_max"] for row in split_rows):
            raise ValueError("size gate failure")
        if any(row["size_guard"]["character_count"] > SIZE_LIMITS["character_count_max"] for row in split_rows):
            raise ValueError("character gate failure")
        if any(row["size_guard"]["non_empty_line_count"] > SIZE_LIMITS["non_empty_lines_max"] for row in split_rows):
            raise ValueError("line gate failure")
        if execute_fixtures and any(not row["metadata"]["fixture_validation_result"] for row in split_rows):
            raise ValueError("fixture gate failure")

    train_fields = _field_sets(train)
    holdout_fields = _field_sets(holdout)
    _assert_disjoint(
        train_fields,
        holdout_fields,
        (
            "domains",
            "paths",
            "models",
            "functions",
            "payload_literals",
            "error_messages",
            "prompts",
            "targets",
            "fixtures",
            "mutant_sources",
            "ast_hashes",
            "case_keys",
        ),
    )

    manifest = result["manifest"]
    if manifest["row_counts_by_split"] != {"train": 48, "holdout": 48}:
        raise ValueError("manifest split counts mismatch")
    if manifest["validation_passed"] is not True:
        raise ValueError("manifest validation flag mismatch")

    after = _protected_hashes()
    if before != after:
        raise RuntimeError("protected repo surfaces changed during v2 validation")
    return True


def _manifest(train: list[dict[str, object]], holdout: list[dict[str, object]], execute_fixtures: bool) -> dict[str, object]:
    train_payload = _jsonl(train)
    holdout_payload = _jsonl(holdout)
    combined_payload = holdout_payload + train_payload
    all_rows = holdout + train
    size_gates = {
        "estimated_target_tokens_max": SIZE_LIMITS["estimated_target_tokens_max"],
        "character_count_max": SIZE_LIMITS["character_count_max"],
        "non_empty_lines_max": SIZE_LIMITS["non_empty_lines_max"],
        "all_rows_within_limits": all(
            row["size_guard"]["estimated_target_tokens"] <= SIZE_LIMITS["estimated_target_tokens_max"]
            and row["size_guard"]["character_count"] <= SIZE_LIMITS["character_count_max"]
            and row["size_guard"]["non_empty_line_count"] <= SIZE_LIMITS["non_empty_lines_max"]
            for row in all_rows
        ),
    }
    shape_gates = {
        "required_field_set_valid": all(set(row) == REQUIRED_FIELDS for row in all_rows),
        "row_counts_by_split_valid": len(train) == 48 and len(holdout) == 48,
        "row_counts_by_task_type_split_valid": all(
            set(Counter(row["task_type"] for row in rows).values()) == {12}
            for rows in (train, holdout)
        ),
        "row_counts_by_behavior_group_split_valid": all(
            set(Counter(row["behavior_group"] for row in rows).values()) == {4}
            for rows in (train, holdout)
        ),
    }
    anchor_gates = {
        "all_rows_anchor_valid": all(row["metadata"]["anchor_validation_result"] for row in all_rows),
    }
    fixture_gates = {
        "fixtures_executed": execute_fixtures,
        "all_rows_fixture_valid": all(row["metadata"]["fixture_validation_result"] for row in all_rows),
    }
    leakage_gates = {
        "train_holdout_disjoint_domains": not (set(DOMAINS["train"]) & set(DOMAINS["holdout"])),
        "excluded_domains_isolated": not ((set(DOMAINS["train"]) | set(DOMAINS["holdout"])) & FIXED_DOMAINS),
        "all_rows_leakage_valid": all(row["metadata"]["leakage_validation_result"] for row in all_rows),
    }
    manifest = {
        "candidate_skill": CANDIDATE,
        "benchmark_family": FAMILY,
        "schema_version": SCHEMA_VERSION,
        "generator_version": GENERATOR_VERSION,
        "version": VERSION,
        "deterministic_seed": SEED,
        "row_counts_by_split": {"train": len(train), "holdout": len(holdout)},
        "row_counts_by_task_type_split": {
            split: dict(sorted(Counter(row["task_type"] for row in rows).items()))
            for split, rows in (("train", train), ("holdout", holdout))
        },
        "row_counts_by_behavior_group_split": {
            split: dict(sorted(Counter(row["behavior_group"] for row in rows).items()))
            for split, rows in (("train", train), ("holdout", holdout))
        },
        "domain_partitions": {"train": list(DOMAINS["train"]), "holdout": list(DOMAINS["holdout"])},
        "size_gates": size_gates,
        "shape_gates": shape_gates,
        "anchor_gates": anchor_gates,
        "fixture_gates": fixture_gates,
        "leakage_gates": leakage_gates,
        "training_performed": False,
        "adapter_created": False,
        "candidate_activated": False,
        "active_by_default": False,
        "checksums": {
            "train_jsonl_sha256": _sha_text(train_payload),
            "holdout_jsonl_sha256": _sha_text(holdout_payload),
            "combined_jsonl_sha256": _sha_text(combined_payload),
        },
        "validation_passed": True,
        "fixtures_executed": execute_fixtures,
        "write_performed": False,
        "next_recommended_phase": "v2_candidate_training_execution",
    }
    manifest_payload = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    manifest["checksums"]["manifest_sha256"] = _sha_text(manifest_payload)
    return manifest


def summary_from_result(result: dict[str, object]) -> dict[str, object]:
    manifest = result["manifest"]
    return {
        "candidate_skill": CANDIDATE,
        "schema_version": SCHEMA_VERSION,
        "generator_version": GENERATOR_VERSION,
        "version": VERSION,
        "deterministic_seed": SEED,
        "row_counts": {
            "train": len(result["train"]),
            "holdout": len(result["holdout"]),
            "total": len(result["train"]) + len(result["holdout"]),
        },
        "task_counts_by_split": manifest["row_counts_by_task_type_split"],
        "behavior_group_counts_by_split": manifest["row_counts_by_behavior_group_split"],
        "checksums": manifest["checksums"],
        "validation_passed": manifest["validation_passed"],
        "fixtures_executed": manifest["fixtures_executed"],
        "write_performed": manifest["write_performed"],
        "next_recommended_phase": manifest["next_recommended_phase"],
    }


def _output_allowed(output: Path) -> bool:
    resolved = output.resolve()
    default = DEFAULT_OUTPUT.resolve()
    temp_root = Path(tempfile.gettempdir()).resolve()
    return resolved == default or temp_root in resolved.parents


def write_candidate_data_v2(result: dict[str, object], output: Path) -> Path:
    output = Path(output)
    if not _output_allowed(output):
        raise ValueError("output path is not allowed for v2 candidate write mode")
    validate_candidate_data_v2(result, execute_fixtures=result["manifest"]["fixtures_executed"])
    output.mkdir(parents=True, exist_ok=True)
    result["manifest"]["write_performed"] = True
    manifest_payload = json.dumps(result["manifest"], indent=2, sort_keys=True) + "\n"
    result["manifest"]["checksums"]["manifest_sha256"] = _sha_text(manifest_payload)
    manifest_payload = json.dumps(result["manifest"], indent=2, sort_keys=True) + "\n"
    payloads = {
        "train.jsonl": _jsonl(result["train"]),
        "holdout.jsonl": _jsonl(result["holdout"]),
        "manifest.json": manifest_payload,
    }
    for name, content in payloads.items():
        tmp = output / f".{name}.tmp"
        tmp.write_text(content)
        os.replace(tmp, output / name)
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="build and validate rows in memory")
    parser.add_argument("--write", action="store_true", help="explicitly write validated rows to an allowed output path")
    parser.add_argument("--output", type=Path, help="output directory for explicit write mode")
    parser.add_argument("--skip-fixtures", action="store_true", help="skip fixture execution during validation")
    args = parser.parse_args(argv)

    if args.write and not args.output:
        parser.error("--write requires --output")
    if not args.check and not args.write:
        args.check = True

    try:
        result = build_candidate_data_v2(execute_fixtures=not args.skip_fixtures)
        if args.write:
            write_candidate_data_v2(result, args.output)
        print(json.dumps(summary_from_result(result), indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        failure = {
            "candidate_skill": CANDIDATE,
            "schema_version": SCHEMA_VERSION,
            "generator_version": GENERATOR_VERSION,
            "version": VERSION,
            "deterministic_seed": SEED,
            "validation_passed": False,
            "failure": str(exc),
            "next_recommended_phase": "revise_v2_data_design",
        }
        print(json.dumps(failure, indent=2, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())