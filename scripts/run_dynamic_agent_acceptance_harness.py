from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "tests" / "fixtures" / "dynamic_agent_acceptance"
BASE_FIXTURES = ROOT / "tests" / "fixtures" / "skillcortex_demo"

SCENARIOS = [
    {
        "name": "fastapi_pydantic",
        "skill_id": "fastapi_contract_skill",
        "display_name": "FastAPI Contract Skill",
        "adapter_dir": ROOT / "artifacts" / "adapters" / "python_skill",
        "allowed_task_types": ["python_generation"],
        "expected_selected_skill_id": "fastapi_contract_skill",
    },
    {
        "name": "pytest_generation",
        "skill_id": "pytest_generation_skill",
        "display_name": "Pytest Generation Skill",
        "adapter_dir": ROOT / "artifacts" / "adapters" / "test_generation_skill",
        "allowed_task_types": ["python_generation", "test_generation"],
        "expected_selected_skill_id": "pytest_generation_skill",
    },
    {
        "name": "debugging_bugfix",
        "skill_id": "debugging_fix_skill",
        "display_name": "Debugging Fix Skill",
        "adapter_dir": ROOT / "artifacts" / "adapters" / "debugging_skill",
        "allowed_task_types": ["python_generation", "debugging"],
        "expected_selected_skill_id": "debugging_fix_skill",
    },
    {
        "name": "python_refactor",
        "skill_id": "python_refactor_skill",
        "display_name": "Python Refactor Skill",
        "adapter_dir": ROOT / "artifacts" / "adapters" / "python_skill",
        "allowed_task_types": ["python_generation"],
        "expected_selected_skill_id": "python_refactor_skill",
    },
]


def _command(*args: str) -> list[str]:
    return [sys.executable, "-m", "skillcortex", *args]


def _run(*args: str) -> dict:
    completed = subprocess.run(
        _command(*args),
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    record = {
        "command": ["python", "-m", "skillcortex", *args],
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }
    if completed.returncode != 0:
        raise RuntimeError(json.dumps(record, indent=2))
    try:
        record["result"] = json.loads(completed.stdout)
    except json.JSONDecodeError:
        record["result"] = None
    return record


def _copy_repo(source: Path, destination: Path) -> Path:
    shutil.copytree(source, destination)
    return destination


def _snapshot_repo(repo: Path) -> dict[str, str]:
    snapshot = {}
    for path in sorted(repo.rglob("*")):
        if path.is_file():
            snapshot[str(path.relative_to(repo))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return snapshot


def _package_checksums(root: Path) -> dict[str, str]:
    checksums = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if relative == "metadata.json":
            continue
        checksums[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return checksums


def _ensure_routing_metadata(package_dir: Path, routing_file: Path, routing_card_file: Path) -> None:
    manifest = yaml.safe_load((package_dir / "skill.yaml").read_text())
    overlay = yaml.safe_load(routing_file.read_text()) or {}
    for key, value in overlay.items():
        manifest[key] = value
    (package_dir / "skill.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False))
    shutil.copy2(routing_card_file, package_dir / "routing_card.json")
    metadata = json.loads((package_dir / "metadata.json").read_text())
    metadata["checksums"] = _package_checksums(package_dir)
    (package_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")


def _build_skill_packages(output_root: Path) -> Path:
    skills_dir = output_root / "skills"
    skills_dir.mkdir(parents=True, exist_ok=True)
    train_dataset = BASE_FIXTURES / "train.jsonl"
    eval_dataset = BASE_FIXTURES / "eval.jsonl"
    eval_summary = BASE_FIXTURES / "eval-summary.json"

    for scenario in SCENARIOS:
        package_dir = skills_dir / scenario["skill_id"]
        routing_root = FIXTURES / scenario["name"]
        _run(
            "package-skill",
            "--skill-id",
            scenario["skill_id"],
            "--name",
            scenario["display_name"],
            "--adapter-dir",
            str(scenario["adapter_dir"]),
            "--train-dataset",
            str(train_dataset),
            "--eval-dataset",
            str(eval_dataset),
            "--eval-summary",
            str(eval_summary),
            "--output",
            str(package_dir),
            "--allowed-task-types",
            *scenario["allowed_task_types"],
            "--activation-scope",
            "task",
        )
        _ensure_routing_metadata(
            package_dir,
            routing_root / "routing.yaml",
            routing_root / "routing_card.json",
        )
        _run("validate-skill-package", "--path", str(package_dir))
    return skills_dir


def _scenario_task(scenario: dict) -> str:
    return (FIXTURES / scenario["name"] / "task.txt").read_text().strip()


def _scenario_repo(scenario: dict, output_root: Path) -> Path:
    source = FIXTURES / scenario["name"] / "repo"
    destination = output_root / "repos" / scenario["name"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    return _copy_repo(source, destination)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the dynamic agent acceptance harness.")
    parser.add_argument("--output-root")
    parsed = parser.parse_args(argv)

    output_root = (
        Path(parsed.output_root).resolve()
        if parsed.output_root
        else Path(tempfile.mkdtemp(prefix="skillcortex-dynamic-agent-"))
    )
    output_root.mkdir(parents=True, exist_ok=True)

    skills_dir = _build_skill_packages(output_root)
    scenarios_output = []
    for scenario in SCENARIOS:
        repo = _scenario_repo(scenario, output_root)
        before = _snapshot_repo(repo)
        runtime_out = output_root / "runtimes" / scenario["name"]
        trace_out = output_root / "traces" / f"{scenario['name']}.json"
        task = _scenario_task(scenario)
        completed = _run(
            "agent",
            "run",
            "--skills-dir",
            str(skills_dir),
            "--repo",
            str(repo),
            "--task",
            task,
            "--dry-run",
            "--compose-runtime-out",
            str(runtime_out),
            "--trace-out",
            str(trace_out),
        )
        result = completed["result"]
        if result["mode"] != "dynamic_agent":
            raise RuntimeError(f"unexpected mode for {scenario['name']}: {result['mode']}")
        selected_skill_id = result["routing_decision"]["selected_skills"][0]["skill_id"]
        if selected_skill_id != scenario["expected_selected_skill_id"]:
            raise RuntimeError(
                f"unexpected routing for {scenario['name']}: {selected_skill_id}"
            )
        if result["agent_execution_status"] != "dry_run_completed":
            raise RuntimeError(
                f"unexpected execution status for {scenario['name']}: {result['agent_execution_status']}"
            )
        if result["validation_status"] != "passed":
            raise RuntimeError(
                f"unexpected validation status for {scenario['name']}: {result['validation_status']}"
            )
        if not runtime_out.exists():
            raise RuntimeError(f"runtime was not written for {scenario['name']}")
        if not trace_out.exists():
            raise RuntimeError(f"trace was not written for {scenario['name']}")
        if not result.get("agent_result"):
            raise RuntimeError(f"missing agent_result for {scenario['name']}")
        agent_selected = result["agent_result"].get("selected_skills") or []
        if _snapshot_repo(repo) != before:
            raise RuntimeError(f"repo changed during dry-run for {scenario['name']}")
        scenarios_output.append(
            {
                "name": scenario["name"],
                "expected_selected_skill_id": scenario["expected_selected_skill_id"],
                "selected_skill_id": selected_skill_id,
                "runtime_out": str(runtime_out),
                "trace_out": str(trace_out),
                "mode": result["mode"],
                "agent_execution_status": result["agent_execution_status"],
                "validation_status": result["validation_status"],
                "agent_selected_skills": agent_selected,
                "repo_unchanged": True,
            }
        )

    summary = {
        "status": "complete",
        "output_root": str(output_root),
        "skills_dir": str(skills_dir),
        "scenarios": scenarios_output,
    }
    json.dump(summary, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
