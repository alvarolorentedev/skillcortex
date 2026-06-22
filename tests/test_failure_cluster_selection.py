import hashlib
import json
from pathlib import Path

from scripts.select_failure_cluster import (
    build_cluster_selection,
    build_governed_selection,
    main,
)


ROOT = Path(__file__).resolve().parents[1]


def _row(seed, family, task, mode, passed, skills):
    return {
        "seed": seed,
        "example_id": f"{family}-{task}",
        "benchmark_group": family,
        "task_type": task,
        "mode": mode,
        "execution_passed": passed,
        "selected_skills": skills,
    }


def test_cluster_ranking_prefers_repeated_non_python_localized_failures():
    rows = []
    for seed in (11, 22, 33):
        for task in ("debugging", "test_generation"):
            for mode, passed, skills in (
                ("base", False, []),
                ("lattice", False, ["debugging_skill"]),
                ("oracle-lattice", False, ["python_skill", "debugging_skill"]),
                ("protected_skill_router", False, ["debugging_skill", "python_skill"]),
            ):
                rows.append(_row(seed, "localized", task, mode, passed, skills))
        for mode, passed, skills in (
            ("base", True, []),
            ("lattice", False, ["python_skill"]),
            ("oracle-lattice", False, ["python_skill"]),
            ("protected_skill_router", False, []),
        ):
            rows.append(
                _row(seed, "base_fallback", "python_generation", mode, passed, skills)
            )

    result = build_cluster_selection(rows, seeds=[11, 22, 33], benchmark_sha256="abc")

    assert result["recommended_primary_cluster"] == {
        "semantic_family": "localized",
        "task_type": "debugging",
        "candidate_skill_name": "localized_skill",
    }
    first = result["clusters"][0]
    assert first["fail_count"] == 3
    assert first["failure_count_by_seed"] == {"11": 1, "22": 1, "33": 1}
    assert first["family_non_python_fail_count"] == 6
    assert first["protected_router_failed"] is True
    assert first["distinct_benchmark_examples"] == 1
    assert first["enough_repeated_failures_for_candidate_design"] is True
    assert first["enough_independent_examples_for_promotion"] is False
    assert first["likely_skill_specific"] is True
    python = next(
        cluster
        for cluster in result["clusters"]
        if cluster["task_type"] == "python_generation"
    )
    assert python["eligible_for_failure_born_skill"] is False


def test_main_writes_only_cluster_selection_and_preserves_benchmark(tmp_path):
    benchmark = ROOT / "data/eval.jsonl"
    before = hashlib.sha256(benchmark.read_bytes()).hexdigest()

    assert main([
        "--seeds", "11", "22", "33", "44", "55",
        "--dataset", str(benchmark),
        "--validation-experiment", str(
            ROOT / "artifacts/experiments/python-skill-gating-validation"
        ),
        "--output", str(tmp_path),
    ]) == 0

    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "cluster_selection.json",
        "cluster_selection.md",
    ]
    assert hashlib.sha256(benchmark.read_bytes()).hexdigest() == before
    summary = json.loads((tmp_path / "cluster_selection.json").read_text())
    assert len(summary["recommended_primary_cluster"]) == 3
    assert summary["source"]["router"] == "protected_skill_router"
    assert summary["evaluation_leakage_warning"]


def _governed_rows():
    rows = []
    for seed in (11, 22, 33):
        for family in ("alternating", "next_family", "backup_family"):
            for task in ("debugging", "test_generation"):
                example = f"{family}-{task}"
                for mode, passed, skills in (
                    ("base", False, []),
                    (
                        "protected_skill_router_without_failure_born",
                        False,
                        ["debugging_skill", "python_skill"],
                    ),
                    (
                        "skillcortex_router_v1",
                        family == "alternating",
                        ["debugging_skill", "python_skill"],
                    ),
                    (
                        "oracle-lattice",
                        family == "backup_family",
                        ["python_skill", "debugging_skill"],
                    ),
                ):
                    rows.append(
                        {
                            "seed": seed,
                            "example_id": example,
                            "benchmark_group": family,
                            "task_type": task,
                            "mode": mode,
                            "execution_passed": passed,
                            "selected_skills": skills,
                        }
                    )
    return rows


def _registry(max_parameters=2_500_000):
    return {
        "router": "skillcortex_router_v1",
        "skills": [
            {
                "skill_name": "alternating_skill",
                "status": "promoted",
                "origin": "failure_born",
                "semantic_family": "alternating",
                "rank": 8,
                "trainable_parameters": 311296,
            }
        ],
        "capacity_budget": {
            "current_total_adapter_parameters": 1245184,
            "max_total_adapter_parameters": max_parameters,
            "current_promoted_failure_born_skills": 1,
            "max_promoted_failure_born_skills": 5,
        },
    }


def test_governed_selection_excludes_promoted_families_and_reports_capacity():
    result = build_governed_selection(
        _governed_rows(),
        registry=_registry(),
        seeds=[11, 22, 33],
        benchmark_sha256="abc",
    )

    assert result["governance"]["promoted_semantic_families"] == ["alternating"]
    assert result["governance"]["remaining_adapter_parameter_budget"] == 1_254_816
    assert result["governance"]["whether_one_more_rank8_skill_fits"] is True
    assert result["primary_candidate"]["semantic_family"] == "next_family"
    assert len(result["backup_candidates"]) <= 2
    assert all(
        cluster["semantic_family"] != "alternating" for cluster in result["clusters"]
    )
    assert result["recommendation"] == "proceed_to_candidate_design"
    assert result["analysis_only"] is True
    assert result["training_performed"] is False
    assert result["data_generated"] is False


def test_governed_selection_blocks_when_registry_capacity_is_exceeded():
    result = build_governed_selection(
        _governed_rows(),
        registry=_registry(max_parameters=1_245_184),
        seeds=[11, 22, 33],
        benchmark_sha256="abc",
    )

    assert result["primary_candidate"] is None
    assert result["backup_candidates"] == []
    assert result["governance"]["whether_one_more_rank8_skill_fits"] is False
    assert result["recommendation"] == "do_not_create_new_skill_yet"


def test_governed_main_reads_registry_and_preserves_code_and_benchmark(tmp_path):
    benchmark = ROOT / "data/eval.jsonl"
    router = ROOT / "src/skill_lattice_coder/router.py"
    before = {
        "benchmark": hashlib.sha256(benchmark.read_bytes()).hexdigest(),
        "router": hashlib.sha256(router.read_bytes()).hexdigest(),
    }

    assert main([
        "--mode", "governed-phase-2-3",
        "--registry", str(ROOT / "configs/skill_registry.json"),
        "--source-router", "skillcortex_router_v1",
        "--dataset", str(benchmark),
        "--promotion-experiment", str(
            ROOT / "artifacts/experiments/failure-born-skill/alternating_skill"
        ),
        "--validation-experiment", str(
            ROOT / "artifacts/experiments/python-skill-gating-validation"
        ),
        "--output", str(tmp_path),
    ]) == 0

    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "cluster_selection.json",
        "cluster_selection.md",
    ]
    assert hashlib.sha256(benchmark.read_bytes()).hexdigest() == before["benchmark"]
    assert hashlib.sha256(router.read_bytes()).hexdigest() == before["router"]
    summary = json.loads((tmp_path / "cluster_selection.json").read_text())
    assert summary["source"]["router"] == "skillcortex_router_v1"
    assert summary["primary_candidate"]
    assert summary["recommendation"] == "proceed_to_candidate_design"
