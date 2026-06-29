import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from slmcortex.packaging.artifacts import package_checksums


ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "tests" / "fixtures" / "slmcortex_demo"


def _command(*args: str) -> list[str]:
    return [sys.executable, "-m", "slmcortex", *args]


def _run(name: str, args: list[str]) -> dict:
    completed = subprocess.run(
        _command(*args),
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    record = {
        "name": name,
        "command": ["python", "-m", "slmcortex", *args],
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }
    if completed.returncode != 0:
        raise RuntimeError(json.dumps(record, indent=2))
    record["result"] = json.loads(completed.stdout)
    return record


def _copy_demo_repo(destination: Path) -> Path:
    shutil.copytree(FIXTURES / "toy-repo", destination)
    (destination / "app.py").write_text(
        "from fastapi import FastAPI\nfrom pydantic import BaseModel\n"
    )
    return destination


def _enrich_fastapi_package(package_path: Path) -> None:
    (package_path / "routing_card.json").write_text(
        json.dumps(
            {
                "positive_examples": [
                    "Create a FastAPI endpoint with Pydantic validation",
                ],
                "negative_examples": ["Fix a React hydration bug"],
            }
        )
        + "\n"
    )
    metadata = json.loads((package_path / "metadata.json").read_text())
    metadata["checksums"] = package_checksums(package_path)
    (package_path / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the Composer-first package foundation smoke flow against an external app workspace.",
    )
    parser.add_argument("--workspace-root")
    parsed = parser.parse_args(argv)

    workspace_root = (
        Path(parsed.workspace_root).resolve()
        if parsed.workspace_root
        else Path(tempfile.mkdtemp(prefix="slmcortex-app-workspace-"))
    )
    workspace_root.mkdir(parents=True, exist_ok=True)
    toy_repo = _copy_demo_repo(workspace_root / "state" / "toy-repo")
    package_path = workspace_root / "packages" / "fastapi_contract"
    runtime_path = workspace_root / "runtimes" / "toy-repo"
    export_descriptor = workspace_root / "exports" / "toy-repo.json"
    eval_summary = FIXTURES / "eval-summary.json"
    request = FIXTURES / "request.json"

    steps = [
        _run("doctor", ["doctor", "--workspace", str(workspace_root)]),
        _run(
            "package_fastapi_contract",
            [
                "package-slm",
                "--slm-id",
                "fastapi_contract",
                "--name",
                "FastAPI Contract Slm",
                "--adapter-dir",
                "artifacts/adapters/python_slm",
                "--train-dataset",
                "data/train.jsonl",
                "--eval-dataset",
                "data/eval.jsonl",
                "--eval-summary",
                str(eval_summary),
                "--description",
                "FastAPI endpoints with Pydantic validation.",
                "--allowed-task-types",
                "python_generation",
                "--activation-scope",
                "task",
                "--output",
                str(package_path),
            ],
        ),
    ]
    _enrich_fastapi_package(package_path)
    steps.extend(
        [
            _run(
            "compose_folder",
            [
                "compose-folder",
                "--workspace",
                str(workspace_root),
                "--folder",
                str(toy_repo),
                "--task",
                "Create a FastAPI endpoint with Pydantic validation",
                "--export-descriptor",
                str(export_descriptor),
            ],
        ),
        _run(
            "validate_runtime",
            [
                "validate-runtime",
                "--runtime",
                str(runtime_path),
            ],
        ),
        _run(
            "infer_dry_run",
            [
                "infer",
                "--runtime",
                str(runtime_path),
                "--request-file",
                str(request),
                "--dry-run",
            ],
        ),
        ]
    )

    summary = {
        "status": "complete",
        "mode": "external-workspace-compose",
        "workspace_root": str(workspace_root),
        "runtime": str(runtime_path),
        "export_descriptor": str(export_descriptor),
        "steps": [
            {
                "name": step["name"],
                "command": step["command"],
                "status": step["result"].get("status", "complete"),
            }
            for step in steps
        ],
    }
    json.dump(summary, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())