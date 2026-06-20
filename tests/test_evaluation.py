import json

from skill_lattice_coder.evaluation import evaluate


def test_dry_evaluation_writes_overall_and_task_summaries(tmp_path):
    output = evaluate("data/eval.jsonl", output=tmp_path, dry_run=True)
    summary = json.loads((output / "summary.json").read_text())
    assert summary["hypothesis"] == "inconclusive"
    assert set(summary["modes"]) == {"base", "generic", "single-skill", "lattice"}
    assert set(summary["tasks"]) == {
        "python_generation",
        "debugging",
        "test_generation",
    }
    assert sum(1 for _ in (output / "results.jsonl").open()) == 72
