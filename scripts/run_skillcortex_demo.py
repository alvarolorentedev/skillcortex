import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "tests" / "fixtures" / "skillcortex_demo"


def _command(*args: str) -> list[str]:
    return [sys.executable, "-m", "skillcortex", *args]


def _run(name: str, args: list[str]) -> dict:
    completed = subprocess.run(
        _command(*args),
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    record = {
        "name": name,
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


def _copy_demo_repo(destination: Path) -> Path:
    source = FIXTURES / "toy-repo"
    shutil.copytree(source, destination)
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the no-model Skill Cortex v0.1 demo flow.",
    )
    parser.add_argument(
        "--output-root",
        help="Directory where temporary demo packages, runtime, and traces will be written.",
    )
    parsed = parser.parse_args(argv)

    output_root = (
        Path(parsed.output_root).resolve()
        if parsed.output_root
        else Path(tempfile.mkdtemp(prefix="skillcortex-demo-"))
    )
    output_root.mkdir(parents=True, exist_ok=True)

    python_skill = output_root / "python_skill"
    debugging_skill = output_root / "debugging_skill"
    runtime = output_root / "runtime"
    toy_repo = _copy_demo_repo(output_root / "toy-repo")
    trace_path = output_root / "agent-trace.json"

    dataset_train = FIXTURES / "train.jsonl"
    dataset_eval = FIXTURES / "eval.jsonl"
    eval_summary = FIXTURES / "eval-summary.json"
    request = FIXTURES / "request.json"

    steps = [
        _run(
            "package_python_skill",
            [
                "package-skill",
                "--skill-id",
                "python_skill",
                "--name",
                "Python Skill",
                "--adapter-dir",
                "artifacts/adapters/python_skill",
                "--train-dataset",
                str(dataset_train),
                "--eval-dataset",
                str(dataset_eval),
                "--eval-summary",
                str(eval_summary),
                "--output",
                str(python_skill),
            ],
        ),
        _run(
            "package_debugging_skill",
            [
                "package-skill",
                "--skill-id",
                "debugging_skill",
                "--name",
                "Debugging Skill",
                "--adapter-dir",
                "artifacts/adapters/debugging_skill",
                "--train-dataset",
                str(dataset_train),
                "--eval-dataset",
                str(dataset_eval),
                "--eval-summary",
                str(eval_summary),
                "--output",
                str(debugging_skill),
            ],
        ),
        _run(
            "compose_runtime",
            [
                "compose-skills",
                "--skills",
                f"{python_skill},{debugging_skill}",
                "--strategy",
                "routed",
                "--output",
                str(runtime),
            ],
        ),
        _run(
            "validate_runtime",
            [
                "validate-runtime",
                "--runtime",
                str(runtime),
            ],
        ),
        _run(
            "infer_dry_run",
            [
                "infer",
                "--runtime",
                str(runtime),
                "--request-file",
                str(request),
                "--dry-run",
            ],
        ),
        _run(
            "agent_run_dry_run",
            [
                "agent",
                "run",
                "--runtime",
                str(runtime),
                "--repo",
                str(toy_repo),
                "--task",
                "Fix the failing answer implementation.",
                "--dry-run",
                "--trace-out",
                str(trace_path),
            ],
        ),
    ]

    summary = {
        "status": "complete",
        "output_root": str(output_root),
        "runtime": str(runtime),
        "toy_repo": str(toy_repo),
        "trace_path": str(trace_path),
        "steps": [
            {
                "name": step["name"],
                "command": step["command"],
                "status": (step.get("result") or {}).get("status", "complete"),
            }
            for step in steps
        ],
    }
    json.dump(summary, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())