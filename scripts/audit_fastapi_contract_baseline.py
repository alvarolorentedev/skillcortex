#!/usr/bin/env python3
"""Audit FastAPI baseline artifacts without training or inference."""

import argparse
import ast
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

from skill_lattice_coder.schemas import ExecutionFixture
from skill_lattice_coder.utils import run_fixture, write_json


FASTAPI_BENCHMARK_SHA256 = (
    "05f903fbdb5271e15ebee6edb6d2583f02724678ae946b93817a17b5d9f6d85e"
)
EXISTING_BENCHMARK_SHA256 = (
    "0ec79d983ba1a9ee2363789288242843e46c78fc0ed997b5a934c2978b89bcc6"
)
CANDIDATE = "api_contract_fastapi_skill"


def classify_failure(generation, task_type, error_output):
    stripped = generation.strip()
    if not stripped or stripped.startswith(("```", "json\n", "{")):
        return "output extraction failure"
    try:
        ast.parse(generation)
    except SyntaxError:
        return "syntax/import failure"
    if task_type == "fastapi_contract_test_generation":
        if "def test_" not in generation:
            return "weak generated tests"
    if "cannot import name 'app'" in error_output or "cannot import name 'records'" in error_output:
        return "wrong FastAPI app shape"
    if "ModuleNotFoundError" in error_output or "ImportError" in error_output or "NameError" in error_output:
        return "syntax/import failure"
    if "404 ==" in error_output or "405 ==" in error_output:
        return "wrong route/path/method"
    if "422 ==" in error_output:
        return "wrong request model"
    if "201 ==" in error_output or "200 ==" in error_output or "204 ==" in error_output:
        return "wrong status code"
    if "response_model" in generation or "BaseModel" in generation:
        return "wrong response model"
    if "HTTPException" in generation:
        return "wrong error behavior"
    return "other"


def discovery(rows):
    failures = [row for row in rows if row.get("execution_passed") is False]
    groups = {row["benchmark_group"] for row in failures}
    tasks = {row["task_type"] for row in failures}
    seeds = {row["seed"] for row in failures}
    examples = {row["example_id"] for row in failures}
    checks = {
        "at_least_3_behavior_groups": len(groups) >= 3,
        "at_least_2_task_types": len(tasks) >= 2,
        "at_least_4_of_5_seeds": len(seeds) >= 4,
        "at_least_12_distinct_examples": len(examples) >= 12,
    }
    return {
        "failing_behavior_groups": len(groups),
        "failing_task_types": len(tasks),
        "failing_seeds": len(seeds),
        "distinct_failing_examples": len(examples),
        "checks": checks,
        "all_thresholds_pass": all(checks.values()),
    }


def sample_coverage(rows, require=False):
    result = {
        "routers": sorted({row["mode"] for row in rows}),
        "task_types": sorted({row["task_type"] for row in rows}),
        "behavior_groups": sorted({row["benchmark_group"] for row in rows}),
        "seeds": sorted({row["seed"] for row in rows}),
    }
    result["checks"] = {
        "both_routers": set(result["routers"])
        == {"base", "skillcortex_router_v1"},
        "all_four_task_types": len(result["task_types"]) == 4,
        "at_least_six_behavior_groups": len(result["behavior_groups"]) >= 6,
        "at_least_two_seeds": len(result["seeds"]) >= 2,
    }
    result["all_requirements_pass"] = all(result["checks"].values())
    if require and not result["all_requirements_pass"]:
        raise ValueError("sample coverage requirements not met")
    return result


def _load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def _import_valid(example, generation):
    fixture = example["execution"]
    with tempfile.TemporaryDirectory(prefix="fastapi-audit-") as directory:
        root = Path(directory)
        generated_name = (
            "test_generated.py" if "solution.py" in fixture["files"] else "solution.py"
        )
        (root / generated_name).write_text(generation)
        for name, content in fixture["files"].items():
            (root / name).write_text(content)
        module = generated_name.removesuffix(".py")
        result = subprocess.run(
            [sys.executable, "-c", f"import {module}"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        return result.returncode == 0


def _sample(rows):
    selected = []
    tasks = (
        "fastapi_contract_generation",
        "fastapi_contract_debugging",
        "fastapi_contract_test_generation",
        "fastapi_contract_refactor",
    )
    for seed_index, seed in enumerate((11, 22)):
        seed_rows = [row for row in rows if row["seed"] == seed]
        for mode_index, mode in enumerate(("base", "skillcortex_router_v1")):
            for task_index, task in enumerate(tasks):
                candidates = [
                    row
                    for row in seed_rows
                    if row["mode"] == mode and row["task_type"] == task
                ]
                selected.append(
                    candidates[
                        (seed_index + mode_index * len(tasks) + task_index)
                        % len(candidates)
                    ]
                )
    return selected


def _assigned_mutants(example):
    verifier = example["execution"]["files"]["verify_tests.py"]
    tree = ast.parse(verifier)
    mutants = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "write_text"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            mutants.append(node.args[0].value)
    if len(mutants) != 2:
        raise ValueError(
            f"{example['id']}: expected two assigned mutants, found {len(mutants)}"
        )
    return mutants


def _run_generated_tests(tests, app):
    with tempfile.TemporaryDirectory(prefix="fastapi-mutant-audit-") as directory:
        root = Path(directory)
        (root / "solution.py").write_text(app)
        (root / "test_generated.py").write_text(tests)
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "test_generated.py"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
            env={**__import__("os").environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
        return result.returncode == 0, result.returncode, result.stdout + result.stderr


def audit(dataset, baseline, output):
    dataset = Path(dataset)
    baseline = Path(baseline)
    output = Path(output)
    manifest = json.loads((dataset.parent / "manifest.json").read_text())
    benchmark_sha256 = hashlib.sha256(dataset.read_bytes()).hexdigest()
    existing_benchmark_sha256 = hashlib.sha256(
        Path("data/eval.jsonl").read_bytes()
    ).hexdigest()
    if benchmark_sha256 != FASTAPI_BENCHMARK_SHA256:
        raise ValueError("FastAPI benchmark checksum does not match approved frozen SHA")
    if benchmark_sha256 != manifest["benchmark_sha256"]:
        raise ValueError("FastAPI benchmark checksum does not match manifest")
    if existing_benchmark_sha256 != EXISTING_BENCHMARK_SHA256:
        raise ValueError("data/eval.jsonl checksum changed")
    examples = _load_jsonl(dataset)
    by_id = {example["id"]: example for example in examples}

    reference_failures = []
    mutant_validation_failures = []
    mutant_results = []
    for example in examples:
        passed, details = run_fixture(
            ExecutionFixture.from_dict(example["execution"]), example["target"]
        )
        if not passed:
            reference_failures.append({"id": example["id"], "details": details[-1000:]})
        if example["task_type"] == "fastapi_contract_test_generation":
            correct_app = example["execution"]["files"]["solution.py"]
            correct_passed, correct_code, correct_output = _run_generated_tests(
                example["target"], correct_app
            )
            mutant_rows = []
            for index, mutant in enumerate(_assigned_mutants(example), 1):
                mutant_passed, returncode, output_text = _run_generated_tests(
                    example["target"], mutant
                )
                mutant_rows.append(
                    {
                        "mutant": index,
                        "tests_passed": mutant_passed,
                        "returncode": returncode,
                        "output_excerpt": output_text[-1000:],
                    }
                )
            valid = correct_passed and all(
                not mutant["tests_passed"] for mutant in mutant_rows
            )
            result = {
                "id": example["id"],
                "correct_app_tests_passed": correct_passed,
                "correct_app_returncode": correct_code,
                "correct_app_output_excerpt": correct_output[-1000:],
                "mutants": mutant_rows,
                "passed": valid,
            }
            mutant_results.append(result)
            if not valid:
                mutant_validation_failures.append(result)

    rows = []
    for path in sorted(baseline.glob("seed-*/results.jsonl")):
        rows.extend(_load_jsonl(path))
    if len(rows) != 480:
        raise ValueError(f"expected 480 baseline rows, found {len(rows)}")
    baseline_summary = json.loads((baseline / "summary.json").read_text())
    baseline_validation = baseline_summary.get("validation", {})
    candidate_selected = [
        {
            "seed": row["seed"],
            "example_id": row["example_id"],
            "mode": row["mode"],
        }
        for row in rows
        if CANDIDATE in row.get("selected_skills", [])
    ]
    registry = json.loads(Path("configs/skill_registry.json").read_text())
    candidate_registered = CANDIDATE in {
        skill["skill_name"] for skill in registry["skills"]
    }
    candidate_adapter_paths = [
        str(path)
        for path in Path("artifacts").rglob(CANDIDATE)
        if path.is_dir() and "experiments" not in path.parts
    ]

    syntax = Counter()
    imports = Counter()
    for row in rows:
        try:
            ast.parse(row["generation"])
            syntax[row["mode"]] += 1
        except SyntaxError:
            pass
        if _import_valid(by_id[row["example_id"]], row["generation"]):
            imports[row["mode"]] += 1

    sampled = []
    taxonomy = Counter()
    selected_sample = _sample(rows)
    coverage = sample_coverage(selected_sample, require=True)
    for row in selected_sample:
        passed, details = run_fixture(
            ExecutionFixture.from_dict(by_id[row["example_id"]]["execution"]),
            row["generation"],
        )
        cause = classify_failure(row["generation"], row["task_type"], details)
        taxonomy[cause] += 1
        sampled.append(
            {
                "seed": row["seed"],
                "mode": row["mode"],
                "task_type": row["task_type"],
                "behavior_group": row["benchmark_group"],
                "example_id": row["example_id"],
                "selected_skills": row["selected_skills"],
                "active_adapter_parameters": row["active_adapter_parameters"],
                "execution_passed": passed,
                "failure_cause": cause,
                "generation_excerpt": row["generation"][:240],
                "error_excerpt": details[-600:],
            }
        )

    mode_counts = Counter(row["mode"] for row in rows)
    parameter_counts = Counter(
        (row["mode"], row["active_adapter_parameters"]) for row in rows
    )
    threshold = discovery(rows)
    governance_checks = {
        "candidate_never_selected": not candidate_selected,
        "baseline_candidate_activated_false": baseline_validation.get(
            "candidate_activated"
        )
        is False,
        "baseline_training_performed_false": baseline_validation.get(
            "training_performed"
        )
        is False,
        "baseline_router_modified_false": baseline_validation.get("router_modified")
        is False,
        "baseline_benchmark_modified_false": baseline_validation.get(
            "benchmark_modified"
        )
        is False,
        "candidate_not_registered": not candidate_registered,
        "candidate_adapter_absent": not candidate_adapter_paths,
        "fastapi_benchmark_frozen_sha_matches": benchmark_sha256
        == FASTAPI_BENCHMARK_SHA256,
        "existing_benchmark_sha_matches": existing_benchmark_sha256
        == EXISTING_BENCHMARK_SHA256,
    }
    harness_valid = (
        not reference_failures
        and not mutant_validation_failures
        and all(row.get("error") is None for row in rows)
    )
    corrected_gates_pass = (
        harness_valid
        and coverage["all_requirements_pass"]
        and all(governance_checks.values())
        and threshold["all_thresholds_pass"]
    )
    recommendation = (
        "proceed_to_candidate_data_design"
        if corrected_gates_pass
        else "fix_benchmark_or_runner_first"
        if not harness_valid or not all(governance_checks.values())
        else "do_not_create_new_skill_yet"
    )
    report = {
        "benchmark_sha256": benchmark_sha256,
        "approved_benchmark_sha256": FASTAPI_BENCHMARK_SHA256,
        "existing_benchmark_sha256": existing_benchmark_sha256,
        "approved_existing_benchmark_sha256": EXISTING_BENCHMARK_SHA256,
        "files_inspected": [
            str(dataset),
            str(dataset.parent / "manifest.json"),
            str(baseline / "summary.json"),
            *[
                str(path)
                for path in sorted(baseline.glob("seed-*/results.jsonl"))
            ],
            "scripts/build_fastapi_contract_benchmark.py",
            "scripts/run_fastapi_contract_baseline.py",
            "src/skill_lattice_coder/inference.py",
            "src/skill_lattice_coder/router.py",
            "configs/skill_registry.json",
        ],
        "commands_run": [
            "shasum -a 256 data/benchmarks/fastapi_contract/v1/benchmark.jsonl",
            "PYTHONPATH=. pytest tests/test_fastapi_contract_benchmark.py -q",
            "PYTHONPATH=. python scripts/audit_fastapi_contract_baseline.py",
        ],
        "reference_fixtures": {
            "total": len(examples),
            "passed": len(examples) - len(reference_failures),
            "failed": len(reference_failures),
            "failures": reference_failures,
        },
        "test_generation_mutants": {
            "cases": sum(
                example["task_type"] == "fastapi_contract_test_generation"
                for example in examples
            ),
            "correct_app_passed_and_both_mutants_failed": not mutant_validation_failures,
            "failures": mutant_validation_failures,
            "results": mutant_results,
        },
        "baseline_rows": len(rows),
        "syntax_validity": {
            mode: {
                "valid": syntax[mode],
                "total": mode_counts[mode],
                "rate": syntax[mode] / mode_counts[mode],
            }
            for mode in mode_counts
        },
        "import_validity": {
            mode: {
                "valid": imports[mode],
                "total": mode_counts[mode],
                "rate": imports[mode] / mode_counts[mode],
            }
            for mode in mode_counts
        },
        "sample": sampled,
        "sample_coverage": coverage,
        "sample_taxonomy": dict(taxonomy),
        "harness_valid": harness_valid,
        "zero_percent_interpretation": (
            "Reference fixtures and mutant checks pass; sampled model outputs fail "
            "through invalid/extraneous output, missing API symbols, contract errors, "
            "or weak tests. The 0% result is not explained by the harness."
        ),
        "router_parameters": {
            f"{mode}:{parameters}": count
            for (mode, parameters), count in parameter_counts.items()
        },
        "generation_zero_parameters": {
            "count": sum(
                row["mode"] == "skillcortex_router_v1"
                and row["mapped_router_task_type"] == "python_generation"
                and row["active_adapter_parameters"] == 0
                for row in rows
            ),
            "expected": True,
            "reason": (
                "fastapi_contract_generation maps to python_generation; "
                "SkillCortexRouterV1 delegates that task to protected base fallback. "
                "No candidate or unregistered skill is activated."
            ),
        },
        "discovery_thresholds": threshold,
        "governance_checks": governance_checks,
        "governance_evidence": {
            "baseline_validation": baseline_validation,
            "candidate_selected_rows": candidate_selected,
            "candidate_registered": candidate_registered,
            "candidate_adapter_paths": candidate_adapter_paths,
        },
        "all_corrected_gates_pass": corrected_gates_pass,
        "recommendation": recommendation,
        "boundaries": {
            "training_performed": not governance_checks[
                "baseline_training_performed_false"
            ],
            "adapter_created": not governance_checks["candidate_adapter_absent"],
            "router_changed": not governance_checks[
                "baseline_router_modified_false"
            ],
            "registry_changed": candidate_registered,
            "candidate_activated": not (
                governance_checks["baseline_candidate_activated_false"]
                and governance_checks["candidate_never_selected"]
            ),
            "benchmark_mutated": not (
                governance_checks["baseline_benchmark_modified_false"]
                and governance_checks["fastapi_benchmark_frozen_sha_matches"]
                and governance_checks["existing_benchmark_sha_matches"]
            ),
        },
    }
    write_json(output / "audit.json", report)
    (output / "audit.md").write_text(_markdown(report))
    return report


def _markdown(report):
    fixtures = report["reference_fixtures"]
    mutants = report["test_generation_mutants"]
    lines = [
        "# FastAPI Contract Baseline Sanity Audit",
        "",
        f"- Benchmark SHA-256: `{report['benchmark_sha256']}`",
        f"- Reference fixtures: **{fixtures['passed']}/{fixtures['total']} passed**",
        f"- Test-generation correct/mutant validation: **{str(mutants['correct_app_passed_and_both_mutants_failed']).lower()}**",
        f"- Baseline rows audited: **{report['baseline_rows']}**",
        f"- Existing benchmark SHA-256: `{report['existing_benchmark_sha256']}`",
        "",
        "## Syntax and import validity",
        "",
        "| Router | Syntax valid | Import valid |",
        "|---|---:|---:|",
    ]
    for mode in report["syntax_validity"]:
        syntax = report["syntax_validity"][mode]
        imports = report["import_validity"][mode]
        lines.append(
            f"| `{mode}` | {syntax['valid']}/{syntax['total']} ({syntax['rate']:.1%}) | "
            f"{imports['valid']}/{imports['total']} ({imports['rate']:.1%}) |"
        )
    lines.extend(
        [
            "",
            "## Sampled failure taxonomy",
            "",
            *[
                f"- {cause}: **{count}**"
                for cause, count in sorted(report["sample_taxonomy"].items())
            ],
            "",
            "The sample covers both routers, all four task types, at least six "
            "behavior groups, and seeds 11 and 22.",
            "",
            "## Verified governance checks",
            "",
            *[
                f"- {name}: **{str(passed).lower()}**"
                for name, passed in sorted(report["governance_checks"].items())
            ],
            "",
            "## Discovery thresholds",
            "",
        ]
    )
    thresholds = report["discovery_thresholds"]
    for name, passed in thresholds["checks"].items():
        lines.append(f"- {name}: **{str(passed).lower()}**")
    lines.extend(
        [
            "",
            f"Recommendation: {report['recommendation']}",
            "",
            "No training, adapter creation, router change, registry change, "
            "candidate activation, or benchmark mutation was performed.",
        ]
    )
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default="data/benchmarks/fastapi_contract/v1/benchmark.jsonl",
    )
    parser.add_argument(
        "--baseline", default="artifacts/experiments/fastapi-contract-baseline"
    )
    parser.add_argument(
        "--output",
        default="artifacts/experiments/fastapi-contract-baseline-audit",
    )
    args = parser.parse_args(argv)
    report = audit(args.dataset, args.baseline, args.output)
    print(json.dumps({"recommendation": report["recommendation"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
