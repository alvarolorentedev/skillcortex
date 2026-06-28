from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections import Counter
from pathlib import Path

from slmcortex.runtime.dynamic import DynamicRouteDecision, DynamicRuntime


CASES = [
    ("Fix a FastAPI validation bug", "local_lora"),
    ("Refactor a FastAPI endpoint", "local_lora"),
    ("Polish a coding helper", "remote_lora"),
    ("Improve a coding workflow", "remote_lora"),
    ("train a custom adapter for a new task", "plasticity_train"),
    ("Train on this unfamiliar local workflow", "plasticity_train"),
    ("Summarize project notes", "base_fallback"),
    ("Summarize release notes", "base_fallback"),
]


def run_benchmark(slms_dir: Path) -> dict:
    previous = os.environ.get("SLMCORTEX_BASE_CONFIG")
    os.environ["SLMCORTEX_BASE_CONFIG"] = str(Path("configs/prototype.yaml").resolve())
    rows = []
    try:
        with tempfile.TemporaryDirectory(prefix="slmcortex-router-bench-") as directory:
            benchmark_slms = slms_dir if any(slms_dir.glob("*/slm.yaml")) else _demo_slms(Path(directory))
            runtime = DynamicRuntime.load(benchmark_slms, allow_remote_loras=False)
            for prompt, expected in CASES:
                decision = _decision(runtime, prompt)
                actual = runtime._route_branch(decision)
                rows.append({"prompt": prompt, "expected": expected, "actual": actual, "passed": actual == expected})
    finally:
        # Restore the SLMCORTEX env var we modified. Previous held the
        # original value of `SLMCORTEX_BASE_CONFIG`.
        if previous is None:
            os.environ.pop("SLMCORTEX_BASE_CONFIG", None)
        else:
            os.environ["SLMCORTEX_BASE_CONFIG"] = previous
    return {
        "total": len(rows),
        "passed": sum(row["passed"] for row in rows),
        "accuracy": sum(row["passed"] for row in rows) / len(rows),
        "expected": dict(sorted(Counter(row["expected"] for row in rows).items())),
        "actual": dict(sorted(Counter(row["actual"] for row in rows).items())),
        "cases": rows,
    }


def _demo_slms(root: Path) -> Path:
    from slmcortex.packaging import package_slm

    eval_summary = root / "eval.json"
    eval_summary.write_text(json.dumps({"modes": {}, "tasks": {}}) + "\n")
    package_slm(
        slm_id="fastapi_slm",
        name="FastAPI Slm",
        adapter_dir=Path("artifacts/adapters/python_slm"),
        output=root / "slms" / "fastapi_slm",
        train_dataset=Path("data/train.jsonl"),
        eval_dataset=Path("data/eval.jsonl"),
        eval_summary=eval_summary,
        version="0.1.0",
        description="FastAPI endpoint validation",
        composition={
            "capabilities": {"allowed_task_types": ["python_generation"]},
            "activation": {
                "default_route_type": "adapter",
                "scope": "task",
                "semantic_families": ["fastapi"],
            },
            "compatibility": {"compatible_slms": [], "incompatible_slms": []},
            "routing": {"tasks": {}},
        },
        force=True,
    )
    return root / "slms"


def _decision(runtime: DynamicRuntime, prompt: str) -> DynamicRouteDecision:
    if "train" in prompt.lower():
        return DynamicRouteDecision(
            base_model="benchmark",
            selected_slms=[],
            remote_loras=[],
            task_type="python_generation",
            semantic_family=None,
            train_new_lora=True,
            reason="benchmark training branch",
        )
    return runtime._rule_router([{"role": "user", "content": prompt}], list(runtime.slms.values()))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark dynamic router branch choices without downloads or training.")
    parser.add_argument("--slms-dir", default="slms")
    parsed = parser.parse_args(argv)
    print(json.dumps(run_benchmark(Path(parsed.slms_dir)), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
