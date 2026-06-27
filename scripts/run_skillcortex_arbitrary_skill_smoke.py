import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "tests" / "fixtures" / "skillcortex_demo"
EXAMPLE_ROOT = ROOT / "examples" / "fastapi_contract_tiny"


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
    shutil.copytree(FIXTURES / "toy-repo", destination)
    return destination


def _stage_demo_adapter(destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "adapter_config.json").write_text(
        json.dumps(
            {
                "fine_tune_type": "lora",
                "num_layers": 1,
                "lora_parameters": {
                    "rank": 1,
                    "scale": 1.0,
                    "dropout": 0.0,
                    "keys": ["self_attn.q_proj"],
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    (destination / "metadata.json").write_text(
        json.dumps(
            {
                "source_model": "demo-source-model",
                "base_model": "demo-runtime-model",
                "quantization": "4bit",
                "config": {},
                "dataset_size": 2,
                "elapsed_seconds": 0.0,
                "rank": 1,
                "target_modules": ["self_attn.q_proj"],
                "trainable_parameters": 1,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    (destination / "adapters.safetensors").write_bytes(b"demo")
    return destination


def _write_demo_eval_summary(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "hypothesis": None,
                "modes": {
                    "base": {"count": 2, "fuzzy_score": 0.2},
                    "single-skill": {"count": 2, "fuzzy_score": 1.0},
                },
                "tasks": {
                    "python_generation": {"single-skill": {"count": 1, "fuzzy_score": 1.0}},
                    "debugging": {"single-skill": {"count": 1, "fuzzy_score": 1.0}},
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the arbitrary-skill Skill Cortex smoke flow. Default mode is a no-model, "
            "package-first demo. Pass --real-training to run real local LoRA training."
        ),
    )
    parser.add_argument(
        "--output-root",
        help="Directory where temporary packages, runtime, and traces will be written.",
    )
    parser.add_argument(
        "--real-training",
        action="store_true",
        help=(
            "Run the real local train-skill path. Slow and opt-in. Not intended for CI. "
            "Requires a working local mlx-lm training environment."
        ),
    )
    parsed = parser.parse_args(argv)

    output_root = (
        Path(parsed.output_root).resolve()
        if parsed.output_root
        else Path(tempfile.mkdtemp(prefix="skillcortex-fastapi-contract-"))
    )
    output_root.mkdir(parents=True, exist_ok=True)

    skill_id = "fastapi_contract"
    name = "FastAPI Contract Skill"
    package_output = output_root / skill_id
    runtime = output_root / "runtime"
    toy_repo = _copy_demo_repo(output_root / "toy-repo")
    trace_path = output_root / "agent-trace.json"
    train_dataset = EXAMPLE_ROOT / "train.jsonl"
    eval_dataset = EXAMPLE_ROOT / "eval.jsonl"
    request = EXAMPLE_ROOT / "request.json"

    steps = []
    if parsed.real_training:
        steps.append(
            _run(
                "train_fastapi_contract",
                [
                    "train-skill",
                    "--skill-id",
                    skill_id,
                    "--name",
                    name,
                    "--train-dataset",
                    str(train_dataset),
                    "--eval-dataset",
                    str(eval_dataset),
                    "--output",
                    str(package_output),
                    "--allowed-task-types",
                    "python_generation",
                    "debugging",
                    "--activation-scope",
                    "task",
                    "--force",
                ],
            )
        )
    else:
        adapter_dir = _stage_demo_adapter(output_root / "demo-adapters" / skill_id)
        eval_summary = _write_demo_eval_summary(output_root / "eval-summary.json")
        steps.append(
            _run(
                "package_fastapi_contract",
                [
                    "package-skill",
                    "--skill-id",
                    skill_id,
                    "--name",
                    name,
                    "--adapter-dir",
                    str(adapter_dir),
                    "--train-dataset",
                    str(train_dataset),
                    "--eval-dataset",
                    str(eval_dataset),
                    "--eval-summary",
                    str(eval_summary),
                    "--output",
                    str(package_output),
                    "--allowed-task-types",
                    "python_generation",
                    "debugging",
                    "--activation-scope",
                    "task",
                ],
            )
        )

    steps.extend(
        [
            _run(
                "compose_runtime",
                [
                    "compose-skills",
                    "--skills",
                    str(package_output),
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
    )

    summary = {
        "status": "complete",
        "mode": "real-training" if parsed.real_training else "no-model-package-demo",
        "output_root": str(output_root),
        "package_output": str(package_output),
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