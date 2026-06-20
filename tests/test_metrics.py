from skill_lattice_coder.metrics import (
    aggregate_results,
    classify_hypothesis,
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
