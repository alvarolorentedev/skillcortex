import hashlib
import json
from collections import Counter
from pathlib import Path

from scripts.build_api_contract_fastapi_candidate_data import (
    build_candidate_data,
    main,
    validate_candidate_data,
    write_candidate_data,
)


ROOT = Path(__file__).resolve().parents[1]
FASTAPI_BENCHMARK = ROOT / "data/benchmarks/fastapi_contract/v1/benchmark.jsonl"
EXISTING_BENCHMARK = ROOT / "data/eval.jsonl"
ROUTER = ROOT / "src/skill_lattice_coder/router.py"
REGISTRY = ROOT / "configs/skill_registry.json"


def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_candidate_data_is_deterministic_balanced_and_independent():
    first = build_candidate_data()
    second = build_candidate_data()

    assert first == second
    rows = first["holdout"] + first["train"]
    assert len(first["train"]) == 96
    assert len(first["holdout"]) == 96
    assert len(rows) == 192
    assert set(Counter(row["task_type"] for row in first["train"]).values()) == {24}
    assert set(Counter(row["task_type"] for row in first["holdout"]).values()) == {24}
    assert set(Counter(row["behavior_group"] for row in first["train"]).values()) == {8}
    assert set(Counter(row["behavior_group"] for row in first["holdout"]).values()) == {8}
    assert set(first["manifest"]["row_counts_by_split"].values()) == {96}
    assert first["manifest"]["validation_passed"] is True

    train_domains = {row["domain"] for row in first["train"]}
    holdout_domains = {row["domain"] for row in first["holdout"]}
    fixed_domains = {
        json.loads(line)["domain"] for line in FASTAPI_BENCHMARK.read_text().splitlines()
    }
    assert train_domains.isdisjoint(holdout_domains)
    assert train_domains.isdisjoint(fixed_domains)
    assert holdout_domains.isdisjoint(fixed_domains)

    required = {
        "id", "split", "candidate_skill", "benchmark_family", "schema_version",
        "task_type", "behavior_group", "domain", "prompt", "target", "execution",
        "metadata", "leakage_guard",
    }
    assert all(set(row) == required for row in rows)
    assert all(row["leakage_guard"]["benchmark_overlap_checked"] for row in rows)
    assert all(row["leakage_guard"]["cross_split_overlap_checked"] for row in rows)
    assert all(
        row["leakage_guard"]["normalized_ast_sha256"]
        == hashlib.sha256(row["target"].encode()).hexdigest()
        for row in rows
    )
    validate_candidate_data(first, execute_fixtures=False)


def test_check_mode_writes_nothing_and_preserves_governance_files(tmp_path):
    before = {
        path: _hash(path)
        for path in (FASTAPI_BENCHMARK, EXISTING_BENCHMARK, ROUTER, REGISTRY)
    }
    output = tmp_path / "candidate"

    assert main(["--check", "--output", str(output), "--skip-fixtures"]) == 0
    assert not output.exists()
    assert all(_hash(path) == digest for path, digest in before.items())
    registry = json.loads(REGISTRY.read_text())
    assert "api_contract_fastapi_skill" not in {
        skill["skill_name"] for skill in registry["skills"]
    }


def test_explicit_write_uses_only_requested_temporary_directory(tmp_path):
    result = build_candidate_data()
    output = tmp_path / "candidate"

    write_candidate_data(result, output)

    assert sorted(path.name for path in output.iterdir()) == [
        "holdout.jsonl",
        "manifest.json",
        "train.jsonl",
    ]
    assert len((output / "train.jsonl").read_text().splitlines()) == 96
    assert len((output / "holdout.jsonl").read_text().splitlines()) == 96


def test_explicit_write_is_idempotent_for_existing_matching_output(tmp_path):
    result = build_candidate_data()
    output = tmp_path / "candidate"

    write_candidate_data(result, output)
    before = {path.name: path.read_bytes() for path in output.iterdir()}

    assert write_candidate_data(result, output) == output
    after = {path.name: path.read_bytes() for path in output.iterdir()}
    assert before == after


def test_explicit_write_replaces_existing_generated_output(tmp_path):
    result = build_candidate_data()
    output = tmp_path / "candidate"

    write_candidate_data(result, output)
    manifest = json.loads((output / "manifest.json").read_text())
    manifest["validation_command"] = "stale command"
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )

    assert write_candidate_data(result, output) == output
    refreshed = json.loads((output / "manifest.json").read_text())
    assert refreshed["validation_command"] == result["manifest"]["validation_command"]
