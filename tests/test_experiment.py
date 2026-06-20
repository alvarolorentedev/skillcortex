import json

from scripts.run_seeds import adapters_ready, summarize


def test_summarize_prefixes_seed_groups(tmp_path):
    for seed, lattice in ((1, True), (2, False)):
        directory = tmp_path / f"seed-{seed}"
        directory.mkdir()
        rows = [
            {
                "example_id": "x",
                "benchmark_group": "family",
                "mode": mode,
                "fuzzy_score": 1.0,
                "execution_passed": passed,
                "active_adapter_parameters": parameters,
                "latency_seconds": 0,
            }
            for mode, passed, parameters in [
                ("generic", False, 24),
                ("lattice", lattice, 16),
            ]
        ]
        (directory / "results.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows)
        )
    summary = summarize(tmp_path, [1, 2])
    assert summary["generic_vs_lattice_execution"]["count"] == 2


def test_summarize_without_execution_is_inconclusive(tmp_path):
    directory = tmp_path / "seed-1"
    directory.mkdir()
    (directory / "results.jsonl").write_text(
        "".join(
            json.dumps(
                {
                    "example_id": "x",
                    "benchmark_group": "family",
                    "mode": mode,
                    "fuzzy_score": 1.0,
                    "execution_passed": None,
                    "active_adapter_parameters": parameters,
                    "latency_seconds": 0,
                }
            )
            + "\n"
            for mode, parameters in (("generic", 24), ("lattice", 16))
        )
    )
    assert summarize(tmp_path, [1])["hypothesis"] == "inconclusive"


def test_adapters_ready_requires_all_four(tmp_path):
    assert not adapters_ready(tmp_path)
    for name in ("python_skill", "debugging_skill", "test_generation_skill", "generic"):
        path = tmp_path / name
        path.mkdir()
        (path / "adapters.safetensors").touch()
    assert adapters_ready(tmp_path)
