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


def _run_slmcortex(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "slmcortex", *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )


def test_slmcortex_root_help_lists_product_commands_and_examples():
    completed = _run_slmcortex("--help")
    assert completed.returncode == 0
    assert "Compose, validate, run, and optionally author Slm Cortex packages." in completed.stdout
    assert "doctor" in completed.stdout
    assert "compose-folder" in completed.stdout
    assert "package-slm" in completed.stdout
    assert "compose-slms" in completed.stdout
    assert "validate-runtime" in completed.stdout
    assert "agent" in completed.stdout
    assert "Examples:" in completed.stdout


def test_slmcortex_product_help_examples_cover_every_command():
    commands = {
        ("doctor", "--help"): "slmcortex doctor --workspace",
        ("compose-folder", "--help"): "slmcortex compose-folder --folder . --task",
        ("train-slm", "--help"): "slmcortex train-slm --slm-id fastapi_contract",
        ("package-slm", "--help"): "slmcortex package-slm --slm-id python_slm",
        ("validate-slm-package", "--help"): "slmcortex validate-slm-package --path",
        ("compose-slms", "--help"): "slmcortex compose-slms --slms",
        ("validate-runtime", "--help"): "slmcortex validate-runtime --runtime",
        ("infer", "--help"): "slmcortex infer --runtime",
        ("serve", "--help"): "slmcortex serve --runtime",
        ("agent", "run", "--help"): "slmcortex agent run --runtime",
    }
    for args, expected in commands.items():
        completed = _run_slmcortex(*args)
        assert completed.returncode == 0, completed.stderr
        assert "Examples:" in completed.stdout
        assert expected in completed.stdout


def test_slmcortex_demo_script_runs_end_to_end(tmp_path):
    output_root = tmp_path / "demo"
    completed = subprocess.run(
        [sys.executable, "scripts/run_slmcortex_demo.py", "--output-root", str(output_root)],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    summary = json.loads(completed.stdout)
    assert summary["status"] == "complete"
    assert [step["name"] for step in summary["steps"]] == [
        "package_python_slm",
        "package_debugging_slm",
        "compose_runtime",
        "validate_runtime",
        "infer_dry_run",
        "agent_run_dry_run",
    ]
    assert output_root.joinpath("python_slm", "slm.yaml").exists()
    assert output_root.joinpath("debugging_slm", "slm.yaml").exists()
    assert output_root.joinpath("runtime", "composition.yaml").exists()
    assert output_root.joinpath("agent-trace.json").exists()

    infer_step = next(step for step in summary["steps"] if step["name"] == "infer_dry_run")
    assert infer_step["status"] == "dry-run"
    agent_step = next(step for step in summary["steps"] if step["name"] == "agent_run_dry_run")
    assert agent_step["status"] == "dry-run"


def test_arbitrary_slm_smoke_script_runs_default_no_model_loop(tmp_path):
    output_root = tmp_path / "fastapi-contract-smoke"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_slmcortex_arbitrary_slm_smoke.py",
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
    assert output_root.joinpath("fastapi_contract", "slm.yaml").exists()
    assert output_root.joinpath("runtime", "composition.yaml").exists()
    assert output_root.joinpath("agent-trace.json").exists()


def test_package_product_smoke_script_runs_external_workspace_loop(tmp_path):
    workspace_root = tmp_path / "workspace"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_package_product_smoke.py",
            "--workspace-root",
            str(workspace_root),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    summary = json.loads(completed.stdout)
    assert summary["status"] == "complete"
    assert summary["mode"] == "external-workspace-compose"
    assert [step["name"] for step in summary["steps"]] == [
        "doctor",
        "package_fastapi_contract",
        "compose_folder",
        "validate_runtime",
        "infer_dry_run",
    ]
    assert workspace_root.joinpath("packages", "fastapi_contract", "slm.yaml").exists()
    assert workspace_root.joinpath("runtimes", "toy-repo", "composition.yaml").exists()
    assert workspace_root.joinpath("exports", "toy-repo.json").exists()
    assert workspace_root.joinpath("logs", "compose-toy-repo.json").exists()


def test_dynamic_adaptive_smoke_script_runs_mock_loop(tmp_path):
    output_root = tmp_path / "dynamic-adaptive-smoke"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_dynamic_adaptive_smoke.py",
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
    assert summary["mode"] == "mock"
    assert summary["branches"] == {
        "local": "local_lora",
        "remote": "remote_lora",
        "plasticity": "plasticity_train",
    }
    assert output_root.joinpath("slms", "fastapi_slm", "slm.yaml").exists()
    assert output_root.joinpath("slms", "sql_remote", "slm.yaml").exists()


def test_dynamic_adaptive_smoke_script_exercises_failure_modes(tmp_path):
    for mode in ("remote-download", "training"):
        completed = subprocess.run(
            [
                sys.executable,
                "scripts/run_dynamic_adaptive_smoke.py",
                "--output-root",
                str(tmp_path / mode),
                "--failure-mode",
                mode,
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )

        assert completed.returncode == 0, completed.stderr
        summary = json.loads(completed.stdout)
        failed = summary["results"]["remote" if mode == "remote-download" else "plasticity"]
        assert failed["route_branch"] == "base_fallback"
        assert failed["adaptation_error"]
