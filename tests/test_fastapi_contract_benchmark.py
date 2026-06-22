import hashlib
import json
from collections import Counter
from pathlib import Path

from scripts.build_fastapi_contract_benchmark import build_benchmark
from skill_lattice_coder.schemas import ExecutionFixture
from skill_lattice_coder.utils import run_fixture


ROOT = Path(__file__).resolve().parents[1]


def _rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_fastapi_contract_benchmark_is_balanced_deterministic_and_executable(tmp_path):
    benchmark_before = hashlib.sha256((ROOT / "data/eval.jsonl").read_bytes()).hexdigest()
    first = tmp_path / "first"
    second = tmp_path / "second"

    build_benchmark(first)
    build_benchmark(second)

    assert (first / "benchmark.jsonl").read_bytes() == (
        second / "benchmark.jsonl"
    ).read_bytes()
    assert (first / "manifest.json").read_bytes() == (
        second / "manifest.json"
    ).read_bytes()

    rows = _rows(first / "benchmark.jsonl")
    assert len(rows) == 48
    assert len({row["id"] for row in rows}) == 48
    assert set(Counter(row["task_type"] for row in rows).values()) == {12}
    assert set(Counter(row["behavior_group"] for row in rows).values()) == {4}
    assert all(row["metadata"] == {
        "evaluation_only": True,
        "candidate_skill": "api_contract_fastapi_skill",
        "requires_candidate_activation": False,
    } for row in rows)
    assert not list(first.glob("*train*")) and not list(first.glob("*holdout*"))

    failures = []
    for row in rows:
        fixture = ExecutionFixture.from_dict(row["execution"])
        passed, output = run_fixture(fixture, row["target"])
        if not passed:
            failures.append((row["id"], output[-500:]))
    assert failures == []
    assert hashlib.sha256((ROOT / "data/eval.jsonl").read_bytes()).hexdigest() == benchmark_before
