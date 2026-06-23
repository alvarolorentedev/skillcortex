import pytest

from scripts.audit_fastapi_contract_baseline import (
    classify_failure,
    discovery,
    sample_coverage,
)


def test_audit_classifies_outputs_and_applies_breadth_thresholds():
    assert classify_failure("", "fastapi_contract_debugging", "") == "output extraction failure"
    assert classify_failure("json\n{}", "fastapi_contract_generation", "NameError") == "output extraction failure"
    assert classify_failure("from fastapi import FastAPI\napp = FastAPI()", "fastapi_contract_test_generation", "") == "weak generated tests"
    assert classify_failure("def broken(:", "fastapi_contract_debugging", "SyntaxError") == "syntax/import failure"

    rows = [
        {
            "seed": seed,
            "example_id": f"example-{example}",
            "task_type": f"task-{example % 2}",
            "benchmark_group": f"group-{example % 3}",
            "execution_passed": False,
        }
        for seed in (11, 22, 33, 44)
        for example in range(12)
    ]
    result = discovery(rows)
    assert result["all_thresholds_pass"] is True
    assert result["distinct_failing_examples"] == 12


def test_sample_coverage_is_enforced():
    rows = [
        {
            "mode": mode,
            "task_type": task,
            "benchmark_group": group,
            "seed": seed,
        }
        for mode in ("base", "skillcortex_router_v1")
        for task in (
            "fastapi_contract_generation",
            "fastapi_contract_debugging",
            "fastapi_contract_test_generation",
            "fastapi_contract_refactor",
        )
        for group in ("a", "b", "c", "d", "e", "f")
        for seed in (11, 22)
    ]
    assert sample_coverage(rows)["all_requirements_pass"] is True

    with pytest.raises(ValueError, match="sample coverage"):
        sample_coverage(rows[:4], require=True)
