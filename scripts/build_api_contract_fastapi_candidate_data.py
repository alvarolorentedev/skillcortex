#!/usr/bin/env python3
"""Build and validate quarantined FastAPI candidate data in memory."""

import argparse
import ast
import hashlib
import importlib.util
import json
import os
import re
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path

from scripts.build_fastapi_contract_benchmark import (
    _app as fixed_app,
    _primary_mutant,
    _secondary_mutant,
    _test as fixed_test,
)
from skill_lattice_coder.schemas import ExecutionFixture
from skill_lattice_coder.utils import run_fixture


SEED = 2301
CANDIDATE = "api_contract_fastapi_skill"
FAMILY = "fastapi_contract"
SCHEMA_VERSION = 1
GENERATOR_VERSION = 1
FASTAPI_SHA256 = "05f903fbdb5271e15ebee6edb6d2583f02724678ae946b93817a17b5d9f6d85e"
EXISTING_SHA256 = "0ec79d983ba1a9ee2363789288242843e46c78fc0ed997b5a934c2978b89bcc6"
ROOT = Path(__file__).resolve().parents[1]
FASTAPI_BENCHMARK = Path("data/benchmarks/fastapi_contract/v1/benchmark.jsonl")
EXISTING_BENCHMARK = Path("data/eval.jsonl")
DEFAULT_OUTPUT = Path("data/failure_born/api_contract_fastapi_skill/v1")

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
FIXED_DOMAINS = {
    "payments", "reports", "devices", "comments", "teams", "schedules",
    "reviews", "catalogs", "messages", "profiles", "events", "quotes",
}
DOMAINS = {
    "train": (
        "orders", "tickets", "invoices", "products", "customers", "tasks",
        "projects", "assets", "notifications", "files", "memberships", "approvals",
    ),
    "holdout": (
        "bookings", "shipments", "subscriptions", "alerts", "documents",
        "workspaces", "incidents", "surveys", "contracts", "warehouses",
        "calendars", "expenses",
    ),
}
PRIMARY_MUTANT_CLASSES = {
    "request_body_validation": "permissive request model",
    "path_parameter_validation": "wrong path parameter type",
    "response_model_correctness": "missing required response field",
    "status_code_correctness": "wrong success status",
    "not_found_error_behavior": "wrong error status",
    "invalid_state_error_behavior": "wrong error status",
    "list_envelope_shape": "raw list instead of typed envelope",
    "enum_literal_validation": "wrong enum/literal handling",
    "optional_field_handling": "missing required response field",
    "contract_preserving_refactor": "changed public contract to satisfy implementation",
    "contract_drift_detection": "changed public contract to satisfy implementation",
    "bounded_route_service_separation": "missing response_model",
}
REQUIRED_FIELDS = {
    "id", "split", "candidate_skill", "benchmark_family", "schema_version",
    "task_type", "behavior_group", "domain", "prompt", "target", "execution",
    "metadata", "leakage_guard",
}


def _sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def _jsonl(rows):
    return "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
        for row in rows
    )


def _normalized_target_sha(target):
    ast.parse(target)
    return _sha(target)


def _stem(domain, split, variant):
    return "".join(part.title() for part in f"{split}_{domain}_{variant}".split("_"))


def _candidate_app(domain, split, variant):
    path = f"{domain}-{split[0]}{variant}"
    stem = _stem(domain, split, variant)
    store = f"{domain}_{split}_{variant}_store"
    app = fixed_app(path)
    replacements = (
        ("CreateRecord", f"Create{stem}"),
        ("PatchRecord", f"Patch{stem}"),
        ("RecordList", f"{stem}Envelope"),
        ("Record", f"{stem}Item"),
        ("records", store),
        ("_create_record", f"_create_{domain}_{split}_{variant}"),
        ('detail="not found"', f'detail="{domain} {split} missing {variant}"'),
        ('detail="invalid state"', f'detail="{domain} {split} conflict {variant}"'),
    )
    for old, new in replacements:
        app = app.replace(old, new)
    return app, path, store


def _candidate_test(domain, split, variant, group, path, store):
    test = fixed_test(path, group).replace("records", store)
    test = test.replace(
        "from solution import app, " + store,
        "from solution import app, " + store,
    )
    test = test.replace("_create_record", f"_create_{domain}_{split}_{variant}")
    test = test.replace("'alpha'", repr(f"{split}-{domain}-alpha-{variant}"))
    test = test.replace("'beta'", repr(f"{split}-{domain}-beta-{variant}"))
    test = test.replace("'gamma'", repr(f"{split}-{domain}-gamma-{variant}"))
    test = test.replace("'not found'", repr(f"{domain} {split} missing {variant}"))
    return test


def _pytest_command():
    if importlib.util.find_spec("pytest") is not None:
        return [sys.executable, "-m", "pytest"]
    if shutil.which("uv"):
        return [
            "uv",
            "run",
            "--project",
            str(ROOT),
            "--extra",
            "test",
            "python",
            "-m",
            "pytest",
        ]
    raise RuntimeError(
        "pytest is unavailable for fixture validation; install test extras or run via uv"
    )


def _verifier(primary, secondary):
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


def _mutants(app, group, path):
    primary = _primary_mutant(app, group)
    if primary == app:
        if group == "response_model_correctness":
            primary = re.sub(
                r'item = \{"id": len\([^)]+\) \+ 1, \*\*payload\.model_dump\(\)\}',
                'item = {"name": payload.name, "status": payload.status}',
                app,
                count=1,
            )
        elif group == "list_envelope_shape":
            primary = re.sub(
                r'return \{"items": list\(([^)]+)\.values\(\)\), "total": len\(\1\)\}',
                r"return list(\1.values())",
                app,
                count=1,
            )
        elif group == "contract_preserving_refactor":
            primary = re.sub(
                r"return (_create_\w+)\(payload\)",
                'return {"id": 99, **payload.model_dump()}',
                app,
                count=1,
            )
        elif group == "contract_drift_detection":
            primary = re.sub(
                r"return (_create_\w+)\(payload\)",
                'return {**\\1(payload), "status": "active"}',
                app,
                count=1,
            )
        elif group == "bounded_route_service_separation":
            primary = re.sub(
                r"def (_create_\w+)\(",
                r"def renamed_\1(",
                app,
                count=1,
            )
    if primary == app:
        raise ValueError(f"failed to construct primary mutant: {group}")
    secondary = _secondary_mutant(app, path)
    return (
        {
            "class": PRIMARY_MUTANT_CLASSES[group],
            "source": primary,
            "sha256": _sha(primary),
        },
        {
            "class": "changed route path or method",
            "source": secondary,
            "sha256": _sha(secondary),
        },
    )


def _row(split, task, group, domain, variant, index):
    app, path, store = _candidate_app(domain, split, variant)
    tests = _candidate_test(domain, split, variant, group, path, store)
    mutants = _mutants(app, group, path)
    kind = task.removeprefix("fastapi_contract_")
    prefix = (
        "Training exercise" if split == "train" else "Independent holdout challenge"
    )
    contract = (
        f"{prefix}: {kind.replace('_', ' ')} the bounded `/{path}` FastAPI "
        f"contract for {group.replace('_', ' ')}. Return code only."
    )
    if kind == "test_generation":
        prompt = contract + "\n\nCorrect app:\n" + app
        target = tests
        execution = {
            "files": {
                "solution.py": app,
                "verify_tests.py": _verifier(
                    mutants[0]["source"], mutants[1]["source"]
                ),
            },
            "command": ["python", "verify_tests.py"],
            "timeout_seconds": 20,
        }
    else:
        target = app
        execution = {
            "files": {"test_contract.py": tests},
            "command": ["python", "-m", "pytest", "-q"],
            "timeout_seconds": 20,
        }
        if kind == "debugging":
            prompt = contract + "\n\nBroken app:\n" + mutants[0]["source"]
        elif kind == "refactor":
            tangled = app.replace(
                f"return _create_{domain}_{split}_{variant}(payload)",
                (
                    f'item = {{"id": len({store}) + 1, **payload.model_dump()}}\n'
                    f'    {store}[item["id"]] = item\n'
                    "    return item"
                ),
                1,
            )
            prompt = contract + "\n\nWorking tangled app:\n" + tangled
        else:
            prompt = contract

    mutant_metadata = [
        {"class": mutant["class"], "sha256": mutant["sha256"]}
        for mutant in mutants
    ] if kind == "test_generation" else []
    case_key = f"{split}:{task}:{group}:{domain}:{variant}"
    return {
        "id": f"api-contract-candidate-v1-{split}-{kind}-{index:03d}",
        "split": split,
        "candidate_skill": CANDIDATE,
        "benchmark_family": FAMILY,
        "schema_version": SCHEMA_VERSION,
        "task_type": task,
        "behavior_group": group,
        "domain": domain,
        "prompt": prompt,
        "target": target,
        "execution": execution,
        "metadata": {
            "source": "synthetic_failure_born",
            "evaluation_only": False,
            "active_by_default": False,
            "generation_source": "build_api_contract_fastapi_candidate_data",
            "generation_version": GENERATOR_VERSION,
            "deterministic_seed": SEED,
            "mutants": mutant_metadata,
        },
        "leakage_guard": {
            "fixed_fastapi_benchmark_sha256": FASTAPI_SHA256,
            "existing_benchmark_sha256": EXISTING_SHA256,
            "domain_partition": split,
            "template_partition": split,
            "normalized_case_key": case_key,
            "normalized_ast_sha256": _normalized_target_sha(target),
            "benchmark_overlap_checked": True,
            "cross_split_overlap_checked": True,
        },
    }


def _split_rows(split):
    rows = []
    for task_index, task in enumerate(TASKS):
        for group_index, (group, domain) in enumerate(
            zip(GROUPS, DOMAINS[split])
        ):
            for variant in (1, 2):
                index = task_index * 24 + group_index * 2 + variant
                rows.append(_row(split, task, group, domain, variant, index))
    return rows


def _field_sets(rows):
    text = "\n".join(
        row["prompt"] + "\n" + row["target"] + "\n"
        + json.dumps(row["execution"], sort_keys=True)
        for row in rows
    )
    return {
        "ids": {row["id"] for row in rows},
        "domains": {row["domain"] for row in rows},
        "paths": set(re.findall(r'"/([^"{]+)', text)),
        "models": set(re.findall(r"class\s+([A-Z]\w+)", text)),
        "functions": set(re.findall(r"def\s+([a-zA-Z_]\w*)", text)),
        "prompts": {row["prompt"] for row in rows},
        "targets": {row["target"] for row in rows},
        "fixtures": {json.dumps(row["execution"], sort_keys=True) for row in rows},
        "case_keys": {
            row["leakage_guard"]["normalized_case_key"] for row in rows
        },
        "ast_hashes": {
            row["leakage_guard"]["normalized_ast_sha256"] for row in rows
        },
        "mutant_hashes": {
            mutant["sha256"]
            for row in rows
            for mutant in row["metadata"]["mutants"]
        },
    }


def _assert_disjoint(left, right, labels):
    for label in labels:
        overlap = left[label] & right[label]
        if overlap:
            raise ValueError(f"{label} overlap: {sorted(overlap)[:3]}")


def _run_test_target(tests, app):
    with tempfile.TemporaryDirectory(prefix="candidate-fastapi-") as directory:
        root = Path(directory)
        (root / "solution.py").write_text(app)
        (root / "test_generated.py").write_text(tests)
        result = __import__("subprocess").run(
            _pytest_command() + ["-q", "test_generated.py"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
        return result.returncode == 0


def validate_candidate_data(result, execute_fixtures=True):
    before = {
        FASTAPI_BENCHMARK: hashlib.sha256(FASTAPI_BENCHMARK.read_bytes()).hexdigest(),
        EXISTING_BENCHMARK: hashlib.sha256(EXISTING_BENCHMARK.read_bytes()).hexdigest(),
    }
    if before[FASTAPI_BENCHMARK] != FASTAPI_SHA256:
        raise ValueError("frozen FastAPI benchmark checksum changed")
    if before[EXISTING_BENCHMARK] != EXISTING_SHA256:
        raise ValueError("data/eval.jsonl checksum changed")

    train, holdout = result["train"], result["holdout"]
    if len(train) != 96 or len(holdout) != 96:
        raise ValueError("expected 96 train and 96 holdout rows")
    for split, rows in (("train", train), ("holdout", holdout)):
        if any(set(row) != REQUIRED_FIELDS for row in rows):
            raise ValueError("candidate row schema mismatch")
        if any(row["split"] != split for row in rows):
            raise ValueError("split metadata mismatch")
        if set(Counter(row["task_type"] for row in rows).values()) != {24}:
            raise ValueError("unbalanced task counts")
        if set(Counter(row["behavior_group"] for row in rows).values()) != {8}:
            raise ValueError("unbalanced behavior counts")
        if any(row["execution"]["timeout_seconds"] != 20 for row in rows):
            raise ValueError("fixture timeout must be 20 seconds")

    if set(DOMAINS["train"]) & set(DOMAINS["holdout"]):
        raise ValueError("train and holdout domains overlap")
    if (set(DOMAINS["train"]) | set(DOMAINS["holdout"])) & FIXED_DOMAINS:
        raise ValueError("candidate domain overlaps fixed benchmark")

    train_fields, holdout_fields = _field_sets(train), _field_sets(holdout)
    _assert_disjoint(
        train_fields,
        holdout_fields,
        (
            "ids", "domains", "paths", "models", "prompts", "targets",
            "fixtures", "case_keys", "ast_hashes", "mutant_hashes",
        ),
    )

    fixed_rows = [
        json.loads(line) for line in FASTAPI_BENCHMARK.read_text().splitlines()
    ]
    existing_rows = [
        json.loads(line) for line in EXISTING_BENCHMARK.read_text().splitlines()
    ]
    candidate = train + holdout
    for key in ("id", "prompt", "target"):
        candidate_values = {row[key] for row in candidate}
        fixed_values = {
            row[key] for row in fixed_rows + existing_rows if key in row
        }
        if candidate_values & fixed_values:
            raise ValueError(f"{key} overlaps frozen benchmark")
    fixed_fixtures = {
        json.dumps(row.get("execution"), sort_keys=True)
        for row in fixed_rows + existing_rows
    }
    if _field_sets(candidate)["fixtures"] & fixed_fixtures:
        raise ValueError("fixture overlaps frozen benchmark")

    if execute_fixtures:
        failures = []
        for row in candidate:
            if row["task_type"] == "fastapi_contract_test_generation":
                app = row["execution"]["files"]["solution.py"]
                mutants = _assigned_mutants(row["execution"]["files"]["verify_tests.py"])
                valid = _run_test_target(row["target"], app) and all(
                    not _run_test_target(row["target"], mutant)
                    for mutant in mutants
                )
            else:
                valid = run_fixture(
                    ExecutionFixture.from_dict(row["execution"]), row["target"]
                )[0]
            if not valid:
                failures.append(row["id"])
        if failures:
            raise ValueError(f"fixture validation failed: {failures[:3]}")

    after = {
        FASTAPI_BENCHMARK: hashlib.sha256(FASTAPI_BENCHMARK.read_bytes()).hexdigest(),
        EXISTING_BENCHMARK: hashlib.sha256(EXISTING_BENCHMARK.read_bytes()).hexdigest(),
    }
    if before != after:
        raise RuntimeError("frozen benchmark changed during validation")
    return True


def _assigned_mutants(verifier):
    tree = ast.parse(verifier)
    return [
        node.args[0].value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "write_text"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and isinstance(node.args[0].value, str)
    ]


def _manifest(train, holdout):
    train_payload, holdout_payload = _jsonl(train), _jsonl(holdout)
    combined = holdout_payload + train_payload
    all_rows = train + holdout
    return {
        "candidate_skill": CANDIDATE,
        "benchmark_family": FAMILY,
        "schema_version": SCHEMA_VERSION,
        "generator_version": GENERATOR_VERSION,
        "deterministic_seed": SEED,
        "row_counts_by_split": {"train": len(train), "holdout": len(holdout)},
        "row_counts_by_task_type": dict(
            sorted(Counter(row["task_type"] for row in all_rows).items())
        ),
        "row_counts_by_behavior_group": dict(
            sorted(Counter(row["behavior_group"] for row in all_rows).items())
        ),
        "row_counts_by_domain": dict(
            sorted(Counter(row["domain"] for row in all_rows).items())
        ),
        "domain_partitions": {
            "train": list(DOMAINS["train"]),
            "holdout": list(DOMAINS["holdout"]),
            "fixed_excluded": sorted(FIXED_DOMAINS),
        },
        "dependency_versions": {
            "fastapi": "0.138.0",
            "httpx": "0.28.1",
            "pydantic": "2.13.4",
            "pytest": ">=8,<10",
        },
        "train_sha256": _sha(train_payload),
        "holdout_sha256": _sha(holdout_payload),
        "combined_sha256": _sha(combined),
        "fixed_fastapi_benchmark_sha256": FASTAPI_SHA256,
        "existing_benchmark_sha256": EXISTING_SHA256,
        "template_set_hashes": {
            split: _sha("\n".join(row["prompt"] for row in rows))
            for split, rows in (("train", train), ("holdout", holdout))
        },
        "normalized_case_set_hashes": {
            split: _sha(
                "\n".join(
                    row["leakage_guard"]["normalized_case_key"] for row in rows
                )
            )
            for split, rows in (("train", train), ("holdout", holdout))
        },
        "mutant_class_counts": dict(
            sorted(
                Counter(
                    mutant["class"]
                    for row in all_rows
                    for mutant in row["metadata"]["mutants"]
                ).items()
            )
        ),
        "validation_command": (
            "PYTHONPATH=.:src python3 "
            "scripts/build_api_contract_fastapi_candidate_data.py --check"
        ),
        "validation_passed": True,
        "training_performed": False,
        "adapter_created": False,
        "candidate_activated": False,
        "active_by_default": False,
    }


def build_candidate_data(execute_fixtures=False):
    holdout = _split_rows("holdout")
    train = _split_rows("train")
    result = {"holdout": holdout, "train": train}
    result["manifest"] = _manifest(train, holdout)
    validate_candidate_data(result, execute_fixtures=execute_fixtures)
    return result


def _safe_output(output):
    output = Path(output).resolve()
    forbidden = (
        EXISTING_BENCHMARK.resolve(),
        FASTAPI_BENCHMARK.parent.resolve(),
    )
    if output == forbidden[0] or output == forbidden[1] or forbidden[1] in output.parents:
        raise ValueError("refusing protected benchmark output path")
    return output


def write_candidate_data(result, output):
    output = _safe_output(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    expected_files = ("holdout.jsonl", "manifest.json", "train.jsonl")
    try:
        (temporary / "holdout.jsonl").write_text(_jsonl(result["holdout"]))
        (temporary / "train.jsonl").write_text(_jsonl(result["train"]))
        (temporary / "manifest.json").write_text(
            json.dumps(result["manifest"], indent=2, sort_keys=True) + "\n"
        )
        if output.exists():
            if (
                output.is_dir()
                and sorted(path.name for path in output.iterdir()) == list(expected_files)
            ):
                if all(
                    (output / name).read_bytes() == (temporary / name).read_bytes()
                    for name in expected_files
                ):
                    shutil.rmtree(temporary, ignore_errors=True)
                    return output
                shutil.rmtree(output)
                temporary.replace(output)
                return output
            raise FileExistsError(f"output already exists: {output}")
        temporary.replace(output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--skip-fixtures", action="store_true")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args(argv)
    if args.write and not args.check:
        parser.error("--write requires --check")
    result = build_candidate_data(execute_fixtures=not args.skip_fixtures)
    if args.write:
        write_candidate_data(result, args.output)
    print(
        json.dumps(
            {
                "validation_passed": True,
                "rows": {
                    "holdout": len(result["holdout"]),
                    "train": len(result["train"]),
                },
                "checksums": {
                    "holdout": result["manifest"]["holdout_sha256"],
                    "train": result["manifest"]["train_sha256"],
                    "combined": result["manifest"]["combined_sha256"],
                },
                "written": bool(args.write),
                "output": str(args.output) if args.write else None,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
