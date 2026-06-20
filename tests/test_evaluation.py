import json

import skill_lattice_coder.evaluation as evaluation
from skill_lattice_coder.evaluation import evaluate
from skill_lattice_coder.schemas import GenerationResult


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
    assert sum(1 for _ in (output / "results.jsonl").open()) == 600


def test_evaluation_scores_extracted_code(monkeypatch, tmp_path):
    dataset = tmp_path / "eval.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "id": "one",
                "task_type": "python_generation",
                "skills": ["python_skill"],
                "prompt": "Write answer",
                "target": "def answer():\n    return 42",
                "execution": {
                    "files": {
                        "test_solution.py": "from solution import answer\n\ndef test_answer(): assert answer() == 42\n"
                    },
                    "command": ["python", "-m", "pytest", "-q"],
                },
            }
        )
        + "\n"
    )
    monkeypatch.setattr(
        evaluation,
        "infer",
        lambda mode, prompt, **kwargs: GenerationResult(
            mode=mode,
            generation="Here you go:\n```python\ndef answer():\n    return 42\n```\nExplanation",
        ),
    )

    output = evaluate(dataset, output=tmp_path / "out")
    rows = [
        json.loads(line) for line in (output / "results.jsonl").read_text().splitlines()
    ]
    assert all(row["generation"] == "def answer():\n    return 42" for row in rows)
    assert all(row["exact_match"] for row in rows)
    assert all(row["execution_passed"] for row in rows)
