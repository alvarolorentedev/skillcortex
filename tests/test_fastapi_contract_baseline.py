import hashlib
import json
from pathlib import Path

import scripts.run_fastapi_contract_baseline as baseline
from skill_lattice_coder.schemas import GenerationResult


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data/benchmarks/fastapi_contract/v1/benchmark.jsonl"


def test_dry_run_maps_tasks_and_writes_seed_results(monkeypatch, tmp_path):
    calls = []
    before = hashlib.sha256(DATASET.read_bytes()).hexdigest()

    monkeypatch.setattr(
        baseline,
        "infer",
        lambda mode, prompt, **kwargs: calls.append((mode, kwargs))
        or GenerationResult(mode=mode, generation="[dry-run generation]"),
    )

    baseline.run_baseline(
        DATASET,
        output=tmp_path,
        seeds=[11],
        routers=["base", "skillcortex_router_v1"],
        adapter_experiment=tmp_path / "unused",
        dry_run=True,
    )

    rows = [
        json.loads(line)
        for line in (tmp_path / "seed-11/results.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 96
    assert {row["mode"] for row in rows} == {"base", "skillcortex_router_v1"}
    assert {row["task_type"] for row in rows} == {
        "fastapi_contract_generation",
        "fastapi_contract_debugging",
        "fastapi_contract_test_generation",
        "fastapi_contract_refactor",
    }
    routed = [kwargs for mode, kwargs in calls if mode == "lattice"]
    assert {call["task_type"] for call in routed} == {
        "python_generation",
        "debugging",
        "test_generation",
    }
    assert all(call["semantic_family"] == "fastapi_contract" for call in routed)
    assert hashlib.sha256(DATASET.read_bytes()).hexdigest() == before
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["validation"]["training_performed"] is False
    assert summary["validation"]["candidate_activated"] is False
