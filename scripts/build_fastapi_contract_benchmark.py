#!/usr/bin/env python3
"""Build the deterministic FastAPI contract benchmark."""

import hashlib
import json
import sys
from pathlib import Path


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
DOMAINS = (
    "payments",
    "reports",
    "devices",
    "comments",
    "teams",
    "schedules",
    "reviews",
    "catalogs",
    "messages",
    "profiles",
    "events",
    "quotes",
)
BENCHMARK_SHA256 = "0ec79d983ba1a9ee2363789288242843e46c78fc0ed997b5a934c2978b89bcc6"

APP = '''from typing import Literal

from fastapi import FastAPI, HTTPException, Path
from pydantic import BaseModel, ConfigDict, Field

app = FastAPI()
records = {}


class CreateRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1)
    status: Literal["active", "pending"] = "active"
    note: str | None = None


class PatchRecord(BaseModel):
    status: Literal["active", "pending", "archived"]


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: int
    name: str
    status: Literal["active", "pending", "archived"]
    note: str | None = None


class RecordList(BaseModel):
    items: list[Record]
    total: int


def _create_record(payload: CreateRecord) -> dict:
    item = {"id": len(records) + 1, **payload.model_dump()}
    records[item["id"]] = item
    return item


@app.post("/__DOMAIN__", response_model=Record, status_code=201)
def create_record(payload: CreateRecord):
    return _create_record(payload)


@app.get("/__DOMAIN__/{item_id}", response_model=Record)
def get_record(item_id: int = Path(ge=1)):
    if item_id not in records:
        raise HTTPException(status_code=404, detail="not found")
    return records[item_id]


@app.get("/__DOMAIN__", response_model=RecordList)
def list_records():
    return {"items": list(records.values()), "total": len(records)}


@app.patch("/__DOMAIN__/{item_id}", response_model=Record)
def patch_record(payload: PatchRecord, item_id: int = Path(ge=1)):
    if item_id not in records:
        raise HTTPException(status_code=404, detail="not found")
    if records[item_id]["status"] == "archived":
        raise HTTPException(status_code=409, detail="invalid state")
    records[item_id]["status"] = payload.status
    return records[item_id]
'''

TESTS = {
    "request_body_validation": (
        "assert client.post(PATH, json={'name': ''}).status_code == 422"
    ),
    "path_parameter_validation": (
        "assert client.get(PATH + '/0').status_code == 422"
    ),
    "response_model_correctness": (
        "response = client.post(PATH, json={'name': 'alpha'})\n"
        "    assert set(response.json()) == {'id', 'name', 'status', 'note'}"
    ),
    "status_code_correctness": (
        "assert client.post(PATH, json={'name': 'alpha'}).status_code == 201"
    ),
    "not_found_error_behavior": (
        "response = client.get(PATH + '/999')\n"
        "    assert response.status_code == 404\n"
        "    assert response.json()['detail'] == 'not found'"
    ),
    "invalid_state_error_behavior": (
        "created = client.post(PATH, json={'name': 'alpha'}).json()\n"
        "    client.patch(PATH + f\"/{created['id']}\", json={'status': 'archived'})\n"
        "    assert client.patch(PATH + f\"/{created['id']}\", json={'status': 'active'}).status_code == 409"
    ),
    "list_envelope_shape": (
        "client.post(PATH, json={'name': 'alpha'})\n"
        "    assert client.get(PATH).json() == {'items': [{'id': 1, 'name': 'alpha', 'status': 'active', 'note': None}], 'total': 1}"
    ),
    "enum_literal_validation": (
        "assert client.post(PATH, json={'name': 'alpha', 'status': 'unknown'}).status_code == 422"
    ),
    "optional_field_handling": (
        "missing = client.post(PATH, json={'name': 'alpha'}).json()\n"
        "    explicit = client.post(PATH, json={'name': 'beta', 'note': None}).json()\n"
        "    supplied = client.post(PATH, json={'name': 'gamma', 'note': 'ok'}).json()\n"
        "    assert [missing['note'], explicit['note'], supplied['note']] == [None, None, 'ok']"
    ),
    "contract_preserving_refactor": (
        "created = client.post(PATH, json={'name': 'alpha'})\n"
        "    assert created.status_code == 201\n"
        "    assert client.get(PATH + '/1').json() == created.json()"
    ),
    "contract_drift_detection": (
        "response = client.post(PATH, json={'name': 'alpha', 'status': 'pending'})\n"
        "    assert response.status_code == 201\n"
        "    assert response.json()['status'] == 'pending'"
    ),
    "bounded_route_service_separation": (
        "from solution import _create_record\n"
        "    assert callable(_create_record)\n"
        "    assert client.post(PATH, json={'name': 'alpha'}).status_code == 201"
    ),
}


def _app(domain):
    return APP.replace("__DOMAIN__", domain)


def _test(domain, group):
    return (
        "from fastapi.testclient import TestClient\n"
        "from solution import app, records\n\n"
        f"PATH = '/{domain}'\n"
        "client = TestClient(app)\n\n"
        "def setup_function():\n"
        "    records.clear()\n\n"
        f"def test_{group}():\n"
        f"    {TESTS[group]}\n"
    )


def _primary_mutant(app, group):
    replacements = {
        "request_body_validation": ("Field(min_length=1)", "Field(min_length=0)"),
        "path_parameter_validation": ("Path(ge=1)", "Path(ge=0)"),
        "response_model_correctness": (
            'item = {"id": len(records) + 1, **payload.model_dump()}',
            'item = {"name": payload.name, "status": payload.status, "note": payload.note}',
        ),
        "status_code_correctness": ("status_code=201", "status_code=200"),
        "not_found_error_behavior": ("status_code=404", "status_code=400"),
        "invalid_state_error_behavior": ("status_code=409", "status_code=400"),
        "list_envelope_shape": (
            'return {"items": list(records.values()), "total": len(records)}',
            'return {"items": list(records.values())}',
        ),
        "enum_literal_validation": (
            'status: Literal["active", "pending"] = "active"',
            'status: str = "active"',
        ),
        "optional_field_handling": (
            "note: str | None = None",
            'note: str | None = ""',
        ),
        "contract_preserving_refactor": (
            "return _create_record(payload)",
            'return {"id": 99, **payload.model_dump()}',
        ),
        "contract_drift_detection": (
            "return _create_record(payload)",
            'return {**_create_record(payload), "status": "active"}',
        ),
        "bounded_route_service_separation": (
            "def _create_record(payload: CreateRecord) -> dict:",
            "def create_internal(payload: CreateRecord) -> dict:",
        ),
    }
    old, new = replacements[group]
    return app.replace(old, new, 1)


def _secondary_mutant(app, domain):
    return app.replace(f'"/{domain}', f'"/{domain}-changed')


def _verifier(primary, secondary):
    return (
        "import os\nimport pathlib\nimport shutil\nimport subprocess\nimport sys\n\n"
        "root = pathlib.Path(__file__).parent\n"
        "env = {**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'}\n"
        "def run():\n"
        "    shutil.rmtree(root / '__pycache__', ignore_errors=True)\n"
        "    return subprocess.run([sys.executable, '-m', 'pytest', '-q', "
        "'test_generated.py'], cwd=root, env=env).returncode\n"
        "correct = run()\n"
        f"(root / 'solution.py').write_text({primary!r})\n"
        "primary = run()\n"
        f"(root / 'solution.py').write_text({secondary!r})\n"
        "secondary = run()\n"
        "raise SystemExit(0 if correct == 0 and primary != 0 and secondary != 0 else 1)\n"
    )


def _row(task, group, domain, index):
    app = _app(domain)
    tests = _test(domain, group)
    kind = task.removeprefix("fastapi_contract_")
    if kind == "test_generation":
        target = tests
        files = {
            "solution.py": app,
            "verify_tests.py": _verifier(
                _primary_mutant(app, group), _secondary_mutant(app, domain)
            ),
        }
        command = ["python", "verify_tests.py"]
    else:
        target = app
        files = {"test_contract.py": tests}
        command = ["python", "-m", "pytest", "-q"]
    prompt = (
        f"{kind.replace('_', ' ').title()} the bounded FastAPI contract for "
        f"`/{domain}`. Focus on {group.replace('_', ' ')}. Return code only."
    )
    if kind == "debugging":
        prompt += "\n\nBroken implementation:\n" + _primary_mutant(app, group)
    elif kind == "refactor":
        prompt += (
            "\n\nPreserve all observable behavior and keep service logic "
            "separate from the route."
        )
    return {
        "id": f"fastapi-contract-v1-{kind}-{index:03d}",
        "benchmark_family": "fastapi_contract",
        "schema_version": 1,
        "task_type": task,
        "behavior_group": group,
        "domain": domain,
        "prompt": prompt,
        "target": target,
        "execution": {
            "files": files,
            "command": command,
            "timeout_seconds": 20,
        },
        "metadata": {
            "evaluation_only": True,
            "candidate_skill": "api_contract_fastapi_skill",
            "requires_candidate_activation": False,
        },
    }


def build_benchmark(output="data/benchmarks/fastapi_contract/v1"):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = [
        _row(task, group, domain, task_index * len(GROUPS) + group_index + 1)
        for task_index, task in enumerate(TASKS)
        for group_index, (group, domain) in enumerate(zip(GROUPS, DOMAINS))
    ]
    payload = "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
        for row in rows
    )
    benchmark = output / "benchmark.jsonl"
    benchmark.write_text(payload)
    manifest = {
        "benchmark_family": "fastapi_contract",
        "schema_version": 1,
        "builder_version": 1,
        "seed": 0,
        "case_count": len(rows),
        "task_counts": {task: len(GROUPS) for task in TASKS},
        "behavior_group_counts": {group: len(TASKS) for group in GROUPS},
        "domain_allocation": dict(zip(GROUPS, DOMAINS)),
        "dependency_versions": {
            "fastapi": "0.138.0",
            "httpx": "0.28.1",
            "pydantic": "2.13.4",
            "pytest": ">=8,<10",
        },
        "benchmark_sha256": hashlib.sha256(payload.encode()).hexdigest(),
        "existing_benchmark_sha256": BENCHMARK_SHA256,
        "evaluation_only": True,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return benchmark, output / "manifest.json"


if __name__ == "__main__":
    benchmark, manifest = build_benchmark(
        sys.argv[1] if len(sys.argv) > 1 else "data/benchmarks/fastapi_contract/v1"
    )
    print(benchmark)
    print(manifest)
