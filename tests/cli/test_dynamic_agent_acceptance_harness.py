import json
import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("repository root not found")


ROOT = _repo_root()


def test_dynamic_agent_acceptance_harness_runs_end_to_end(tmp_path):
    output_root = tmp_path / "dynamic-agent-harness"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_dynamic_agent_acceptance_harness.py",
            "--output-root",
            str(output_root),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    summary = json.loads(completed.stdout)
    assert summary["status"] == "complete"
    assert [scenario["name"] for scenario in summary["scenarios"]] == [
        "fastapi_pydantic",
        "pytest_generation",
        "debugging_bugfix",
        "python_refactor",
    ]
    assert all(scenario["mode"] == "dynamic_agent" for scenario in summary["scenarios"])
    assert all(scenario["agent_execution_status"] == "dry_run_completed" for scenario in summary["scenarios"])
    assert all(scenario["validation_status"] == "passed" for scenario in summary["scenarios"])
    assert all(scenario["repo_unchanged"] for scenario in summary["scenarios"])
    assert output_root.joinpath("skills").exists()
    assert all(Path(scenario["runtime_out"]).exists() for scenario in summary["scenarios"])
    assert all(Path(scenario["trace_out"]).exists() for scenario in summary["scenarios"])
