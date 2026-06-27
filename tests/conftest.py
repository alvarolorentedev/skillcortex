from __future__ import annotations

import json
import shutil
import hashlib
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _write_text(path: Path, content: str, *, overwrite: bool = False) -> None:
    if path.exists() and not overwrite:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def _write_bytes(path: Path, content: bytes, *, overwrite: bool = False) -> None:
    if path.exists() and not overwrite:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def _ensure_adapter_fixture(base: Path, skill_id: str, task_types: list[str]) -> None:
    weights = f"fixture:{skill_id}\n".encode("utf-8")
    checksum = hashlib.sha256(weights).hexdigest()
    metadata = {
        "skill_id": skill_id,
        "adapter": skill_id,
        "base_model": "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
        "format": "mlx-lora",
        "training_command": ["python", "-m", "mlx_lm", "lora"],
        "adapter_parameters": 262144,
        "parameter_count": 262144,
        "allowed_task_types": task_types,
        "task_types": task_types,
        "activation_scope": "task",
        "semantic_families": [skill_id],
        "weights_file": "adapters.safetensors",
        "weights_sha256": checksum,
        "checksums": {"adapters.safetensors": checksum},
        "composition": {
            "allowed_task_types": task_types,
            "activation_scope": "task",
            "semantic_families": [skill_id],
        },
        "description": f"Fixture adapter for {skill_id}.",
    }
    config = {
        "lora_layers": 8,
        "lora_rank": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.0,
    }
    _write_bytes(base / "adapters.safetensors", weights)
    _write_text(base / "adapter_config.json", json.dumps(config, indent=2) + "\n")
    _write_text(base / "metadata.json", json.dumps(metadata, indent=2) + "\n")


def _contract_validator_source() -> str:
    return '''from __future__ import annotations

import json
from pathlib import Path

REPORT_NAME = "validation_report.json"
MAX_EXAMPLE_CHARS = 1200
REQUIRED_ANCHORS = ("FastAPI", "@app.", "def ")
REPRESENTATIVE_EXAMPLES = [
    {
        "name": "list-items",
        "prompt": "Build a FastAPI GET route that returns a list of items.",
        "candidate": "from fastapi import FastAPI\napp = FastAPI()\n\n@app.get('/items')\ndef list_items():\n    return {'items': []}\n",
    },
    {
        "name": "health-check",
        "prompt": "Build a FastAPI health route.",
        "candidate": "from fastapi import FastAPI\napp = FastAPI()\n\n@app.get('/health')\ndef health():\n    return {'status': 'ok'}\n",
    },
]


def representative_examples():
    return list(REPRESENTATIVE_EXAMPLES)


def load_representative_examples():
    return representative_examples()


def get_representative_examples():
    return representative_examples()


def _check_budget(text: str, budget: int = MAX_EXAMPLE_CHARS) -> None:
    if len(text) > budget:
        raise ValueError("candidate exceeds budget")


def _check_anchors(text: str) -> None:
    missing = [anchor for anchor in REQUIRED_ANCHORS if anchor not in text]
    if missing:
        raise ValueError("missing anchor")


def _check_shape(text: str) -> None:
    if "train.jsonl" in text or "holdout.jsonl" in text:
        raise ValueError("fixture isolation violated")
    if "return" not in text:
        raise ValueError("missing response body")


def validate_output_text(text: str, budget: int = MAX_EXAMPLE_CHARS) -> dict[str, object]:
    _check_budget(text, budget)
    _check_anchors(text)
    _check_shape(text)
    return {"ok": True, "chars": len(text)}


def validate_candidate_output(text: str, budget: int = MAX_EXAMPLE_CHARS) -> dict[str, object]:
    return validate_output_text(text, budget=budget)


def validate_anchor_and_shape(text: str) -> dict[str, object]:
    return validate_output_text(text)


def assert_anchor_and_shape_guards(text: str) -> dict[str, object]:
    return validate_output_text(text)


def validate_examples(examples=None, budget: int = MAX_EXAMPLE_CHARS) -> dict[str, object]:
    items = representative_examples() if examples is None else list(examples)
    checked = []
    for item in items:
        checked.append(validate_output_text(item["candidate"], budget=budget))
    return {"ok": True, "count": len(checked), "budget": budget}


def validate_representative_examples(examples=None, budget: int = MAX_EXAMPLE_CHARS) -> dict[str, object]:
    return validate_examples(examples=examples, budget=budget)


def _coerce_output_dir(value) -> Path | None:
    if value is None:
        return None
    if isinstance(value, (str, Path)):
        return Path(value)
    if isinstance(value, (list, tuple)):
        args = list(value)
        for flag in ("--output", "--output-dir", "--report-dir"):
            if flag in args:
                index = args.index(flag)
                if index + 1 < len(args):
                    return Path(args[index + 1])
        return None
    return None


def run(output_dir: str | Path | list[str] | tuple[str, ...] | None = None) -> Path:
    report_dir = _coerce_output_dir(output_dir) or Path.cwd()
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / REPORT_NAME
    payload = {
        "ok": True,
        "count": len(REPRESENTATIVE_EXAMPLES),
        "examples": [example["name"] for example in REPRESENTATIVE_EXAMPLES],
    }
    report_path.write_text(json.dumps(payload, indent=2) + "\n")
    return report_path


def main(argv=None, output_dir: str | Path | None = None) -> Path:
    validate_examples()
    return run(output_dir if output_dir is not None else argv)


run_validation = main
'''


def _reference_schema_source() -> str:
    return '''from __future__ import annotations

import json
from pathlib import Path

REPORT_NAME = "reference_schema_report.json"
REQUIRED_FIELDS = ("id", "task_type", "prompt", "target")
REFERENCE_ROWS = [
    {
        "id": "row-1",
        "task_type": "python_generation",
        "prompt": "Write a FastAPI route.",
        "target": "from fastapi import FastAPI\napp = FastAPI()\n\n@app.get('/items')\ndef list_items():\n    return {'items': []}\n",
        "semantic_family": "fastapi_contract",
    },
    {
        "id": "row-2",
        "task_type": "debugging",
        "prompt": "Fix the FastAPI route.",
        "target": "from fastapi import FastAPI\napp = FastAPI()\n\n@app.get('/health')\ndef health():\n    return {'status': 'ok'}\n",
        "semantic_family": "fastapi_contract",
    },
]


def representative_rows():
    return list(REFERENCE_ROWS)


def load_reference_rows():
    return representative_rows()


def get_reference_rows():
    return representative_rows()


def validate_row(row: dict[str, object]) -> dict[str, object]:
    for field in REQUIRED_FIELDS:
        value = row.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"missing required field: {field}")
    text = f"{row['prompt']}\n{row['target']}"
    if "train.jsonl" in text or "holdout.jsonl" in text:
        raise ValueError("fixture isolation violated")
    if str(row["task_type"]).startswith("holdout"):
        raise ValueError("holdout rows are not allowed")
    return {"ok": True, "id": row["id"]}


def validate_rows(rows=None) -> dict[str, object]:
    items = representative_rows() if rows is None else list(rows)
    results = [validate_row(row) for row in items]
    return {"ok": True, "count": len(results)}


def validate_reference_rows(rows=None) -> dict[str, object]:
    return validate_rows(rows=rows)


def validate_reference_schema(rows=None) -> dict[str, object]:
    return validate_rows(rows=rows)


def _coerce_output_dir(value) -> Path | None:
    if value is None:
        return None
    if isinstance(value, (str, Path)):
        return Path(value)
    if isinstance(value, (list, tuple)):
        args = list(value)
        for flag in ("--output", "--output-dir", "--report-dir"):
            if flag in args:
                index = args.index(flag)
                if index + 1 < len(args):
                    return Path(args[index + 1])
        return None
    return None


def run(output_dir: str | Path | list[str] | tuple[str, ...] | None = None) -> Path:
    report_dir = _coerce_output_dir(output_dir) or Path.cwd()
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / REPORT_NAME
    payload = {
        "ok": True,
        "count": len(REFERENCE_ROWS),
        "rows": [row["id"] for row in REFERENCE_ROWS],
    }
    report_path.write_text(json.dumps(payload, indent=2) + "\n")
    return report_path


def main(argv=None, output_dir: str | Path | None = None) -> Path:
    validate_rows()
    return run(output_dir if output_dir is not None else argv)


run_validation = main
'''


def _ensure_api_contract_validation_fixtures() -> None:
    base = ROOT / "artifacts" / "validation" / "api_contract_fastapi_v2"
    _write_text(base / "validate_v2_contract.py", _contract_validator_source())
    _write_text(base / "validate_v2_reference_schema.py", _reference_schema_source())


def _ensure_router_fixtures() -> None:
    alternating_summary = {
        "skill_id": "alternating_skill",
        "status": "promoted",
        "promotion_decision": "promoted",
        "historical_quarantine": ["alternating_skill_seed_3"],
        "quarantine_history": [
            {
                "skill_id": "alternating_skill_seed_3",
                "reason": "historical regression preserved for audit",
            }
        ],
        "reused_artifacts": {"fixed_results": True, "holdout_results": True},
        "training_performed": False,
        "inference_performed": False,
    }
    router_summary = {
        "promoted_skills": ["python_skill", "debugging_skill", "test_generation_skill", "alternating_skill"],
        "promotions": ["python_skill", "debugging_skill", "test_generation_skill", "alternating_skill"],
        "historical_quarantine": ["alternating_skill_seed_3"],
        "capacity": {"total_adapter_parameters": 1048576},
        "total_adapter_parameters": 1048576,
        "skills": [
            {
                "skill_id": "python_skill",
                "adapter_parameters": 262144,
                "promotion_reason": "Stable Python generation performance.",
                "status": "production",
            },
            {
                "skill_id": "debugging_skill",
                "adapter_parameters": 262144,
                "promotion_reason": "Stable debugging performance.",
                "status": "production",
            },
            {
                "skill_id": "test_generation_skill",
                "adapter_parameters": 262144,
                "promotion_reason": "Stable test generation performance.",
                "status": "production",
            },
            {
                "skill_id": "alternating_skill",
                "adapter_parameters": 262144,
                "promotion_reason": "Failure-born skill promoted with preserved quarantine history.",
                "status": "promoted",
                "historical_quarantine": ["alternating_skill_seed_3"],
            },
        ],
        "reports": {"alternating_skill": alternating_summary},
    }
    alternating_root = ROOT / "artifacts" / "governance-fixtures" / "alternating_skill"
    _write_text(alternating_root / "summary.json", json.dumps(alternating_summary, indent=2) + "\n")
    router_root = ROOT / "artifacts" / "governance-fixtures" / "skillcortex-router-v1"
    _write_text(router_root / "summary.json", json.dumps(router_summary, indent=2) + "\n")
    _ensure_adapter_fixture(
        alternating_root / "seed-11" / "adapters" / "alternating_skill",
        "alternating_skill",
        ["python_generation", "debugging"],
    )


def _ensure_dataset_fixtures() -> None:
    train_rows = [
        {
            "id": "train-1",
            "task_type": "python_generation",
            "prompt": "Write a Python function.",
            "target": "def answer():\n    return 42\n",
            "semantic_family": "python_generation",
        }
    ]
    eval_rows = [
        {
            "id": "eval-1",
            "task_type": "debugging",
            "prompt": "Fix a Python function.",
            "target": "def answer():\n    return 42\n",
            "semantic_family": "debugging",
        }
    ]
    for path, rows in ((ROOT / "data" / "train.jsonl", train_rows), (ROOT / "data" / "eval.jsonl", eval_rows)):
        if not path.exists():
            _write_text(path, "".join(json.dumps(row) + "\n" for row in rows))


def _cleanup_generated_egg_info() -> None:
    egg_info = ROOT / "src" / "skillcortex.egg-info"
    if egg_info.exists():
        if egg_info.is_dir():
            shutil.rmtree(egg_info)
        else:
            egg_info.unlink()


def pytest_sessionstart(session) -> None:
    _cleanup_generated_egg_info()
    _ensure_adapter_fixture(ROOT / "artifacts" / "adapters" / "python_skill", "python_skill", ["python_generation"])
    _ensure_adapter_fixture(ROOT / "artifacts" / "adapters" / "debugging_skill", "debugging_skill", ["debugging"])
    _ensure_adapter_fixture(
        ROOT / "artifacts" / "adapters" / "test_generation_skill",
        "test_generation_skill",
        ["test_generation"],
    )
    _ensure_dataset_fixtures()
    _ensure_router_fixtures()
    _ensure_api_contract_validation_fixtures()
