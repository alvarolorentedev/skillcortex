import json

import pytest

from skill_lattice_coder.analysis import (
    analyze_composition,
    analyze_python_regression,
    analyze_router,
    load_experiment,
)


MODES = ("base", "generic", "single-skill", "lattice", "oracle-lattice")


def _experiment(tmp_path):
    root = tmp_path / "experiment"
    for seed in (1, 2):
        seed_root = root / f"seed-{seed}"
        seed_root.mkdir(parents=True)
        rows = []
        examples = (
            (
                "python",
                "python_generation",
                "family-python",
                {
                    "base": True,
                    "generic": False,
                    "single-skill": False,
                    "lattice": False,
                    "oracle-lattice": False,
                },
            ),
            (
                "debug",
                "debugging",
                "family-debug",
                {
                    "base": False,
                    "generic": False,
                    "single-skill": False,
                    "lattice": True,
                    "oracle-lattice": True,
                },
            ),
            (
                "test",
                "test_generation",
                "family-test",
                {
                    "base": False,
                    "generic": False,
                    "single-skill": False,
                    "lattice": seed == 2,
                    "oracle-lattice": True,
                },
            ),
        )
        for example_id, task, group, outcomes in examples:
            for mode in MODES:
                skills = {
                    "base": [],
                    "generic": [],
                    "single-skill": [
                        {
                            "python_generation": "python_skill",
                            "debugging": "debugging_skill",
                            "test_generation": "test_generation_skill",
                        }[task]
                    ],
                    "lattice": {
                        "python_generation": ["python_skill"],
                        "debugging": ["debugging_skill", "python_skill"],
                        "test_generation": ["test_generation_skill"],
                    }[task],
                    "oracle-lattice": {
                        "python_generation": ["python_skill"],
                        "debugging": ["python_skill", "debugging_skill"],
                        "test_generation": ["python_skill", "test_generation_skill"],
                    }[task],
                }[mode]
                rows.append(
                    {
                        "example_id": example_id,
                        "task_type": task,
                        "benchmark_group": group,
                        "mode": mode,
                        "execution_passed": outcomes[mode],
                        "syntax_valid": task != "test_generation",
                        "selected_skills": skills,
                        "active_adapter_parameters": len(skills) * 10
                        if skills
                        else (30 if mode == "generic" else 0),
                    }
                )
        (seed_root / "results.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows)
        )
        for name, parameters in (
            ("python_skill", 10),
            ("debugging_skill", 10),
            ("test_generation_skill", 10),
            ("generic", 30),
        ):
            path = seed_root / "adapters" / name
            path.mkdir(parents=True)
            (path / "metadata.json").write_text(
                json.dumps({"trainable_parameters": parameters})
            )
    return root


def test_router_analysis_attributes_oracle_gain_and_parameters(tmp_path):
    root = _experiment(tmp_path)
    data = analyze_router(root)

    assert data["comparison"]["difference"] == pytest.approx(0.5 / 3)
    assert data["comparison"]["candidate_wins"] == 1
    assert data["route_set_agreement_rate"] == 2 / 3
    assert data["by_task"][0]["task_type"] == "test_generation"
    assert data["by_task"][0]["difference"] == 0.5
    assert data["parameters"]["lattice"] == {
        "active_adapter_parameters": 40 / 3,
        "stored_adapter_parameters": 30,
    }
    assert data["worst_families"][0]["benchmark_group"] == "family-test"
    assert (root / "router_analysis.json").exists()
    assert "omits `python_skill`" in (root / "router_analysis.md").read_text()


def test_python_regression_and_composition_reports(tmp_path):
    root = _experiment(tmp_path)

    regression = analyze_python_regression(root)
    assert regression["modes"]["single-skill"]["difference"] == -1
    assert regression["modes"]["single-skill"]["baseline_wins"] == 2
    assert regression["worst_families"][0]["benchmark_group"] == "family-python"

    composition = analyze_composition(root)
    assert composition["comparisons"]["lattice_vs_single_skill"]["difference"] == 0.5
    assert (
        composition["comparisons"]["lattice_vs_single_skill_two_active"][
            "difference"
        ]
        == 1
    )
    assert composition["by_selection"][0]["selected_skills"] == [
        "debugging_skill",
        "python_skill",
    ]
    assert (root / "python_regression_analysis.md").exists()
    assert (root / "composition_analysis.json").exists()


def test_loader_rejects_duplicate_rows_and_inconsistent_metadata(tmp_path):
    root = _experiment(tmp_path)
    results = root / "seed-1" / "results.jsonl"
    first = results.read_text().splitlines()[0]
    results.write_text(results.read_text() + first + "\n")
    with pytest.raises(ValueError, match="duplicate"):
        load_experiment(root)

    root = _experiment(tmp_path / "other")
    metadata = root / "seed-2" / "adapters" / "python_skill" / "metadata.json"
    metadata.write_text(json.dumps({"trainable_parameters": 11}))
    with pytest.raises(ValueError, match="inconsistent"):
        load_experiment(root)
