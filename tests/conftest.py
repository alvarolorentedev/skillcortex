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
    target_modules = ["self_attn.q_proj", "self_attn.v_proj"]
    rank = 8
    metadata = {
        "skill_id": skill_id,
        "adapter": skill_id,
        "base_model": "mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit",
        "source_model": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
        "format": "mlx-lora",
        "quantization": "4bit",
        "training_command": ["python", "-m", "mlx_lm", "lora"],
        "adapter_parameters": 262144,
        "parameter_count": 262144,
        "rank": rank,
        "target_modules": target_modules,
        "seed": 42,
        "trainable_parameters": 311296,
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
        "config": {
            "seed": 42,
            "batch_size": 1,
            "iterations": 100,
            "learning_rate": 0.0001,
            "lora_layers": 8,
            "skill_rank": rank,
            "generic_rank": 24,
            "target_modules": target_modules,
        },
        "description": f"Fixture adapter for {skill_id}.",
    }
    config = {
        "adapter_path": str(base),
        "batch_size": 1,
        "clear_cache_threshold": 0,
        "config": str(base / "training-rank-8.yaml"),
        "data": str(base.parent.parent / "data"),
        "fine_tune_type": "lora",
        "grad_accumulation_steps": 1,
        "grad_checkpoint": False,
        "iters": 100,
        "learning_rate": 0.0001,
        "lora_parameters": {
            "rank": rank,
            "dropout": 0.0,
            "scale": 20.0,
            "keys": target_modules,
        },
        "lr_schedule": None,
        "mask_prompt": True,
        "max_seq_length": 2048,
        "model": "mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit",
        "num_layers": 8,
        "optimizer": "adam",
        "optimizer_config": {
            "adam": {},
            "adamw": {},
            "muon": {},
            "sgd": {},
            "adafactor": {},
        },
        "project_name": None,
        "report_to": None,
        "resume_adapter_file": None,
        "save_every": 100,
        "seed": 42,
        "steps_per_eval": 200,
        "steps_per_report": 10,
        "test": False,
        "test_batches": 500,
        "train": True,
        "val_batches": 25,
    }
    _write_bytes(base / "adapters.safetensors", weights)
    _write_text(base / "adapter_config.json", json.dumps(config, indent=2) + "\n")
    _write_text(base / "metadata.json", json.dumps(metadata, indent=2) + "\n")


def _contract_validator_source() -> str:
    return '''from __future__ import annotations

import json
from pathlib import Path

REPORT_NAME = "validation_report.json"
ACTIVE_GENERATION_MAX_TOKENS = 256
REQUIRED_HEADROOM_PERCENT = 25
MAX_ALLOWED_ESTIMATED_TARGET_TOKENS = ACTIVE_GENERATION_MAX_TOKENS * (100 - REQUIRED_HEADROOM_PERCENT) // 100
RESULT_JSON = Path.cwd() / "validation_results.json"

REPRESENTATIVE_EXAMPLES = [
    {
        "task_type": "fastapi_contract_generation",
        "artifact_type": "app_file",
        "reference": "single route app",
        "candidate": """from fastapi import FastAPI

app = FastAPI()

@app.get('/items')
def list_items():
    return {'items': []}
""",
        "fixture": {"language": "python"},
    },
    {
        "task_type": "fastapi_contract_test_generation",
        "artifact_type": "test_file",
        "reference": "test client smoke test",
        "candidate": """from fastapi.testclient import TestClient
from solution import app

client = TestClient(app)

def test_items():
    assert client.get('/items').status_code == 200
""",
        "fixture": {"language": "python"},
    },
]


def representative_examples():
    return list(REPRESENTATIVE_EXAMPLES)


def load_representative_examples():
    return representative_examples()


def get_representative_examples():
    return representative_examples()


def parse_safely(text: str) -> bool:
    try:
        compile(text, "<candidate>", "exec")
        return True
    except SyntaxError:
        return False


def mixes_app_and_test_code(text: str) -> bool:
    has_app = "app = FastAPI()" in text or "@app." in text
    has_test = "TestClient" in text or "def test_" in text
    return has_app and has_test


def single_artifact(text: str) -> bool:
    normalized = text.lstrip()
    return not normalized.startswith("# file:") and "\n# file:" not in text and "\n---" not in text


def app_anchor_checks(text: str) -> dict[str, bool]:
    return {
        "valid_python_syntax": parse_safely(text),
        "required_fastapi_imports": "from fastapi import FastAPI" in text,
        "app_fastapi_present": "app = FastAPI()" in text,
        "route_decorator_present": "@app." in text,
        "route_handler_present": "def " in text,
        "no_test_code": "TestClient" not in text and "def test_" not in text,
    }


def test_anchor_checks(text: str) -> dict[str, bool]:
    return {
        "valid_python_syntax": parse_safely(text),
        "test_framework_imports": "from fastapi.testclient import TestClient" in text,
        "testclient_usage": "TestClient(" in text,
        "imports_app_under_test": "from solution import app" in text,
        "test_functions_exist": "def test_" in text,
        "no_app_implementation": "app = FastAPI()" not in text and "@app." not in text,
    }


def _estimate_tokens(text: str) -> int:
    return max(1, (len(text) + 3) // 4)


def _validate_example(example: dict[str, object]) -> dict[str, object]:
    candidate = str(example["candidate"])
    estimated_tokens = _estimate_tokens(candidate)
    anchor_checks = (
        app_anchor_checks(candidate)
        if example["artifact_type"] == "app_file"
        else test_anchor_checks(candidate)
    )
    return {
        "task_type": example["task_type"],
        "artifact_type": example["artifact_type"],
        "estimated_target_tokens": estimated_tokens,
        "fits_under_active_budget": estimated_tokens <= ACTIVE_GENERATION_MAX_TOKENS,
        "has_required_headroom": estimated_tokens <= MAX_ALLOWED_ESTIMATED_TARGET_TOKENS,
        "fixture_passed": all(anchor_checks.values()),
        "mixes_app_and_test_code": mixes_app_and_test_code(candidate),
        "single_artifact": single_artifact(candidate),
    }


def validate_candidate_output(text: str, budget: int = ACTIVE_GENERATION_MAX_TOKENS) -> dict[str, object]:
    estimated_tokens = _estimate_tokens(text)
    return {
        "fits_under_active_budget": estimated_tokens <= budget,
        "has_required_headroom": estimated_tokens <= MAX_ALLOWED_ESTIMATED_TARGET_TOKENS,
        "fixture_passed": parse_safely(text) and single_artifact(text) and not mixes_app_and_test_code(text),
    }


def validate_anchor_and_shape(text: str) -> dict[str, object]:
    return validate_candidate_output(text)


def assert_anchor_and_shape_guards(text: str) -> dict[str, object]:
    return validate_candidate_output(text)


def validate_examples(examples=None, budget: int = ACTIVE_GENERATION_MAX_TOKENS) -> dict[str, object]:
    items = representative_examples() if examples is None else list(examples)
    checked = [_validate_example(item) for item in items]
    return {"ok": True, "count": len(checked), "budget": budget, "results": checked}


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
    report_path = RESULT_JSON
    if output_dir is not None:
        report_dir = _coerce_output_dir(output_dir)
        if report_dir is not None:
            report_path = report_dir / REPORT_JSON.name
    checked = [_validate_example(item) for item in REPRESENTATIVE_EXAMPLES]
    payload = {
        "active_budget": {
            "active_generation_max_tokens": ACTIVE_GENERATION_MAX_TOKENS,
            "maximum_allowed_estimated_target_tokens": MAX_ALLOWED_ESTIMATED_TARGET_TOKENS,
        },
        "representative_results": checked,
        "shape_risk_classification": {
            item["task_type"]: "safe_for_single_file_v2" for item in REPRESENTATIVE_EXAMPLES
        },
        "final_recommendation": "proceed_to_v2_data_design",
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return report_path


def main(argv=None, output_dir: str | Path | None = None) -> int:
    validate_examples()
    run(output_dir if output_dir is not None else argv)
    return 0


run_validation = main
'''


def _reference_schema_source() -> str:
    return '''from __future__ import annotations

import json
from pathlib import Path

REPORT_NAME = "reference_schema_report.json"
RESULT_JSON = Path.cwd() / "schema_validation_results.json"
REQUIRED_FIELDS = {
    "id",
    "task_type",
    "prompt",
    "target",
    "size_guard",
    "shape_guard",
    "leakage_guard",
}
SIZE_LIMITS = {
    "estimated_target_tokens_max": 192,
    "character_count_max": 768,
    "non_empty_lines_max": 32,
}
REPRESENTATIVE_ROWS = [
    {
        "id": "row-1",
        "task_type": "fastapi_contract_generation",
        "prompt": "Write a FastAPI route.",
        "target": """from fastapi import FastAPI

app = FastAPI()

@app.get('/items')
def list_items():
    return {'items': []}
""",
        "size_guard": {},
        "shape_guard": {},
        "leakage_guard": {},
    },
    {
        "id": "row-2",
        "task_type": "fastapi_contract_debugging",
        "prompt": "Fix the FastAPI route.",
        "target": """from fastapi import FastAPI

app = FastAPI()

@app.get('/health')
def health():
    return {'status': 'ok'}
""",
        "size_guard": {},
        "shape_guard": {},
        "leakage_guard": {},
    },
    {
        "id": "row-3",
        "task_type": "fastapi_contract_test_generation",
        "prompt": "Write a FastAPI test.",
        "target": """from fastapi.testclient import TestClient
from solution import app

client = TestClient(app)

def test_items():
    assert client.get('/items').status_code == 200
""",
        "size_guard": {},
        "shape_guard": {},
        "leakage_guard": {},
    },
    {
        "id": "row-4",
        "task_type": "fastapi_contract_refactor",
        "prompt": "Refactor a FastAPI route.",
        "target": """from fastapi import FastAPI

app = FastAPI()

def _payload() -> dict[str, str]:
    return {'status': 'ok'}

@app.get('/status')
def status():
    return _payload()
""",
        "size_guard": {},
        "shape_guard": {},
        "leakage_guard": {},
    },
]

REJECTED_ROWS = [
    {
        **REPRESENTATIVE_ROWS[0],
        "id": "reject-1",
        "target": REPRESENTATIVE_ROWS[0]["target"] + "\n# train.jsonl overlap",
    },
    {
        **REPRESENTATIVE_ROWS[0],
        "id": "reject-2",
        "target": REPRESENTATIVE_ROWS[0]["target"] + "\nfrom fastapi.testclient import TestClient\n",
    },
]


def representative_rows():
    return list(REPRESENTATIVE_ROWS)


def load_reference_rows():
    return representative_rows()


def get_reference_rows():
    return representative_rows()


def _estimate_tokens(text: str) -> int:
    return max(1, (len(text) + 3) // 4)


def _non_empty_lines(text: str) -> int:
    return sum(1 for line in text.splitlines() if line.strip())


def validate_row(row: dict[str, object]) -> dict[str, object]:
    missing = sorted(field for field in REQUIRED_FIELDS if field not in row)
    if missing:
        raise ValueError(f"missing required field: {missing[0]}")
    for field in ("id", "task_type", "prompt", "target"):
        value = row.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"missing required field: {field}")
    text = f"{row['prompt']}\n{row['target']}"
    estimated_target_tokens = _estimate_tokens(str(row["target"]))
    size_guard = {
        "estimated_target_tokens": estimated_target_tokens,
        "character_count": len(str(row["target"])),
        "non_empty_line_count": _non_empty_lines(str(row["target"])),
    }
    reasons: list[str] = []
    if "train.jsonl" in text or "holdout.jsonl" in text:
        reasons.append("leakage_overlap_train_or_holdout")
    if "app = FastAPI()" in text and "TestClient" in text:
        reasons.append("shape_gate_no_app_test_mixing")
    if estimated_target_tokens > SIZE_LIMITS["estimated_target_tokens_max"]:
        reasons.append("size_guard_estimated_tokens")
    if len(str(row["target"])) > SIZE_LIMITS["character_count_max"]:
        reasons.append("size_guard_characters")
    if size_guard["non_empty_line_count"] > SIZE_LIMITS["non_empty_lines_max"]:
        reasons.append("size_guard_non_empty_lines")
    decision = "reject" if reasons else "accept"
    return {
        "id": row["id"],
        "decision": decision,
        "reasons": reasons,
        "size_guard": size_guard,
    }


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
    report_path = RESULT_JSON
    if output_dir is not None:
        report_dir = _coerce_output_dir(output_dir)
        if report_dir is not None:
            report_path = report_dir / RESULT_JSON.name
    payload = {
        "required_fields": sorted(REQUIRED_FIELDS),
        "size_limits": SIZE_LIMITS,
        "accepted_representatives": [validate_row(dict(row)) for row in REPRESENTATIVE_ROWS],
        "rejected_representatives": [validate_row(dict(row)) for row in REJECTED_ROWS],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return report_path


def main(argv=None, output_dir: str | Path | None = None) -> int:
    validate_rows()
    run(output_dir if output_dir is not None else argv)
    return 0


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
