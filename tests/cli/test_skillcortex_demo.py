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


def _run_skillcortex(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "skillcortex", *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )


def test_skillcortex_root_help_lists_product_commands_and_examples():
    completed = _run_skillcortex("--help")
    assert completed.returncode == 0
    assert "Package, compose, validate, and run Skill Cortex runtime bundles." in completed.stdout
    assert "package-skill" in completed.stdout
    assert "compose-skills" in completed.stdout
    assert "validate-runtime" in completed.stdout
    assert "agent" in completed.stdout
    assert "Examples:" in completed.stdout


def test_skillcortex_product_help_examples_cover_every_command():
    commands = {
        ("train-skill", "--help"): "skillcortex train-skill --skill-id fastapi_contract",
        ("package-skill", "--help"): "skillcortex package-skill --skill-id python_skill",
        ("validate-skill-package", "--help"): "skillcortex validate-skill-package --path",
        ("compose-skills", "--help"): "skillcortex compose-skills --skills",
        ("validate-runtime", "--help"): "skillcortex validate-runtime --runtime",
        ("infer", "--help"): "skillcortex infer --runtime",
        ("serve", "--help"): "skillcortex serve --runtime",
        ("agent", "run", "--help"): "skillcortex agent run --runtime",
    }
    for args, expected in commands.items():
        completed = _run_skillcortex(*args)
        assert completed.returncode == 0, completed.stderr
        assert "Examples:" in completed.stdout
        assert expected in completed.stdout


def test_skillcortex_demo_script_runs_end_to_end(tmp_path):
    output_root = tmp_path / "demo"
    completed = subprocess.run(
        [sys.executable, "scripts/run_skillcortex_demo.py", "--output-root", str(output_root)],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    summary = json.loads(completed.stdout)
    assert summary["status"] == "complete"
    assert [step["name"] for step in summary["steps"]] == [
        "package_python_skill",
        "package_debugging_skill",
        "compose_runtime",
        "validate_runtime",
        "infer_dry_run",
        "agent_run_dry_run",
    ]
    assert output_root.joinpath("python_skill", "skill.yaml").exists()
    assert output_root.joinpath("debugging_skill", "skill.yaml").exists()
    assert output_root.joinpath("runtime", "composition.yaml").exists()
    assert output_root.joinpath("agent-trace.json").exists()

    infer_step = next(step for step in summary["steps"] if step["name"] == "infer_dry_run")
    assert infer_step["status"] == "dry-run"
    agent_step = next(step for step in summary["steps"] if step["name"] == "agent_run_dry_run")
    assert agent_step["status"] == "dry-run"


def test_arbitrary_skill_smoke_script_runs_default_no_model_loop(tmp_path):
    output_root = tmp_path / "fastapi-contract-smoke"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_skillcortex_arbitrary_skill_smoke.py",
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
    assert summary["mode"] == "no-model-package-demo"
    assert [step["name"] for step in summary["steps"]] == [
        "package_fastapi_contract",
        "compose_runtime",
        "validate_runtime",
        "infer_dry_run",
        "agent_run_dry_run",
    ]
    assert output_root.joinpath("fastapi_contract", "skill.yaml").exists()
    assert output_root.joinpath("runtime", "composition.yaml").exists()
    assert output_root.joinpath("agent-trace.json").exists()