import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


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
    try:
        record["result"] = json.loads(completed.stdout)
    except json.JSONDecodeError:
        record["result"] = None
    return record


def _copy_demo_repo(destination: Path) -> Path:
    source = FIXTURES / "toy-repo"
    shutil.copytree(source, destination)
    return destination


def _stage_demo_adapter(slm_id: str, destination: Path) -> Path:
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
                "dataset_size": 1,
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the no-model SLM Cortex v0.1 demo flow.",
    )
    parser.add_argument(
        "--output-root",
        help="Directory where temporary demo packages, runtime, and traces will be written.",
    )
    parsed = parser.parse_args(argv)

    output_root = (
        Path(parsed.output_root).resolve()
        if parsed.output_root
        else Path(tempfile.mkdtemp(prefix="slmcortex-demo-"))
    )
    output_root.mkdir(parents=True, exist_ok=True)

    python_slm = output_root / "python_slm"
    debugging_slm = output_root / "debugging_slm"
    demo_adapters = output_root / "demo-adapters"
    runtime = output_root / "runtime"
    toy_repo = _copy_demo_repo(output_root / "toy-repo")
    trace_path = output_root / "agent-trace.json"
    python_adapter = _stage_demo_adapter("python_slm", demo_adapters / "python_slm")
    debugging_adapter = _stage_demo_adapter("debugging_slm", demo_adapters / "debugging_slm")

    dataset_train = FIXTURES / "train.jsonl"
    dataset_eval = FIXTURES / "eval.jsonl"
    eval_summary = FIXTURES / "eval-summary.json"
    request = FIXTURES / "request.json"

    steps = [
        _run(
            "package_python_slm",
            [
                "package-slm",
                "--slm-id",
                "python_slm",
                "--name",
                "Python Slm",
                "--adapter-dir",
                str(python_adapter),
                "--train-dataset",
                str(dataset_train),
                "--eval-dataset",
                str(dataset_eval),
                "--eval-summary",
                str(eval_summary),
                "--output",
                str(python_slm),
            ],
        ),
        _run(
            "package_debugging_slm",
            [
                "package-slm",
                "--slm-id",
                "debugging_slm",
                "--name",
                "Debugging Slm",
                "--adapter-dir",
                str(debugging_adapter),
                "--train-dataset",
                str(dataset_train),
                "--eval-dataset",
                str(dataset_eval),
                "--eval-summary",
                str(eval_summary),
                "--output",
                str(debugging_slm),
            ],
        ),
        _run(
            "compose_runtime",
            [
                "compose-slms",
                "--slms",
                f"{python_slm},{debugging_slm}",
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
