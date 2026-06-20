from skill_lattice_coder.metrics import (
    aggregate_results,
    classify_hypothesis,
    paired_execution_comparison,
    extract_code,
    fuzzy_match,
    python_syntax_valid,
)


def test_code_metrics():
    fenced = "Here:\n```python\ndef add(a, b):\n    return a + b\n```"
    assert extract_code(fenced).startswith("def add")
    assert python_syntax_valid(fenced)
    assert not python_syntax_valid("```python\ndef broken(:\n```")
    assert fuzzy_match("abc", "abc") == 1.0


def test_aggregation_and_hypothesis_classification():
    rows = [
        {"mode": "base", "fuzzy_score": 0.2, "active_adapter_parameters": 0},
        {
            "mode": "generic",
            "fuzzy_score": 0.5,
            "active_adapter_parameters": 24_000_000,
        },
        {
            "mode": "lattice",
            "fuzzy_score": 0.7,
            "active_adapter_parameters": 16_000_000,
        },
    ]
    summary = aggregate_results(rows)
    assert summary["lattice"]["count"] == 1
    assert classify_hypothesis(summary) == "supported"
    summary["lattice"]["fuzzy_score"] = 0.4
    assert classify_hypothesis(summary) == "falsified"


def test_hypothesis_uses_execution_not_fuzzy_when_available():
    rows = [
        {
            "mode": mode,
            "fuzzy_score": fuzzy,
            "execution_passed": execution,
            "active_adapter_parameters": parameters,
        }
        for mode, fuzzy, execution, parameters in [
            ("generic", 0.9, False, 24_000_000),
            ("lattice", 0.1, True, 16_000_000),
        ]
    ]
    assert classify_hypothesis(aggregate_results(rows)) == "supported"


def test_paired_execution_comparison_matches_examples():
    rows = []
    for index in range(20):
        rows.extend(
            [
                {
                    "example_id": str(index),
                    "mode": "generic",
                    "execution_passed": index >= 12,
                },
                {
                    "example_id": str(index),
                    "mode": "lattice",
                    "execution_passed": index >= 4,
                },
            ]
        )
    comparison = paired_execution_comparison(rows, samples=1_000)
    assert comparison["count"] == 20
    assert comparison["difference"] == 0.4
    assert comparison["ci_low"] > 0
    assert (
        classify_hypothesis(
            {
                "generic": {
                    "execution_pass_rate": 0.4,
                    "execution_per_million_active_parameters": 0.4,
                },
                "lattice": {
                    "execution_pass_rate": 0.8,
                    "execution_per_million_active_parameters": 0.8,
                },
            },
            comparison,
        )
        == "supported"
    )


def test_paired_execution_comparison_bootstraps_semantic_groups():
    rows = []
    for group in range(10):
        for task in range(3):
            rows.extend(
                [
                    {
                        "example_id": f"generic-{group}-{task}",
                        "benchmark_group": str(group),
                        "mode": "generic",
                        "execution_passed": False,
                    },
                    {
                        "example_id": f"lattice-{group}-{task}",
                        "benchmark_group": str(group),
                        "mode": "lattice",
                        "execution_passed": group < 8,
                    },
                ]
            )
    comparison = paired_execution_comparison(rows, samples=1_000)
    assert comparison["count"] == 10
    assert comparison["difference"] == 0.8
