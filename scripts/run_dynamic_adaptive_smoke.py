from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def _write_eval(path: Path) -> Path:
    path.write_text(json.dumps({"modes": {}, "tasks": {}}) + "\n")
    return path


def _package(output_root: Path, skill_id: str, description: str, output: Path | None = None) -> Path:
    output = output or output_root / "skills" / skill_id
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "slmcortex",
            "package-skill",
            "--skill-id",
            skill_id,
            "--name",
            skill_id.replace("_", " ").title(),
            "--adapter-dir",
            str(ROOT / "artifacts" / "adapters" / "python_skill"),
            "--train-dataset",
            str(ROOT / "data" / "train.jsonl"),
            "--eval-dataset",
            str(ROOT / "data" / "eval.jsonl"),
            "--eval-summary",
            str(_write_eval(output_root / f"{skill_id}-eval.json")),
            "--output",
            str(output),
            "--description",
            description,
            "--allowed-task-types",
            "python_generation",
            "--semantic-families",
            skill_id.split("_")[0],
            "--activation-scope",
            "task",
            "--force",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        raise RuntimeError(completed.stderr)
    return output


def _mock_runtime(output_root: Path, failure_mode: str | None = None):
    from slmcortex.runtime.dynamic import DynamicRuntime

    _package(output_root, "fastapi_skill", "FastAPI endpoint validation")
    runtime = DynamicRuntime.load(output_root / "skills", allow_remote_loras=True)

    def fake_resolve(source, skill_id, name=None):
        if failure_mode == "remote-download":
            raise ValueError("mock remote download failed")
        _package(output_root, skill_id, f"Imported mock LoRA from {source}")
        runtime.registry.reload()
        return runtime.registry.local[skill_id]

    def fake_train(**kwargs):
        if failure_mode == "training":
            raise ValueError("mock training failed")
        _package(output_root, kwargs["skill"], "Mock trained plasticity LoRA", output=kwargs["output"])
        return {"status": "complete", "skill_id": kwargs["skill"]}

    import slmcortex.runtime.dynamic as dynamic

    runtime.registry.resolve_remote = fake_resolve
    dynamic.train_skill_package = fake_train
    dynamic.load_model = lambda model_name=None, adapter=None: ("mock-model", "mock-tokenizer")
    dynamic.generate_text = lambda *args, **kwargs: ("mock generation", 1, 1)
    return runtime


def _write_mock_config(output_root: Path) -> Path:
    path = output_root / "mock-prototype.yaml"
    path.write_text(
        "\n".join(
            [
                "source_model: smoke-source",
                "model: mlx-test-base",
                "default_runtime_model: mlx-test-base",
                "router_model: mlx-test-router",
                f"lora_cache_dir: {output_root / 'lora-cache'}",
                "allow_remote_lora_downloads: true",
                "allowed_hf_publishers: []",
                "max_download_bytes: 2000000000",
                f"remote_lora_train_dataset: {ROOT / 'data' / 'train.jsonl'}",
                f"remote_lora_eval_dataset: {ROOT / 'data' / 'eval.jsonl'}",
                "training_enabled: true",
                f"plasticity_train_dataset: {ROOT / 'data' / 'train.jsonl'}",
                f"plasticity_eval_dataset: {ROOT / 'data' / 'eval.jsonl'}",
                f"plasticity_publish_dir: {output_root / 'skills'}",
                "max_plasticity_loras: 8",
                "remote_lora_catalog:",
                "  - skill_id: sql_remote",
                "    source: hf://owner/sql",
                "    name: SQL Remote",
                "    description: SQL query tuning",
                "    cues: [sql]",
                "    task_types: [python_generation]",
                "    semantic_families: [sql]",
                "max_tokens: 8",
                "temperature: 0.0",
                "",
            ]
        )
    )
    return path


def _write_real_config(output_root: Path, remote_source: str) -> Path:
    path = output_root / "real-prototype.yaml"
    owner = remote_source.removeprefix("hf://").split("/", 1)[0]
    path.write_text(
        "\n".join(
            [
                "source_model: Qwen/Qwen2.5-Coder-1.5B-Instruct",
                "model: mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit",
                "default_runtime_model: mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit",
                "router_model: mlx-community/Qwen2.5-Coder-0.5B-Instruct-4bit",
                f"lora_cache_dir: {output_root / 'lora-cache'}",
                "allow_remote_lora_downloads: true",
                f"allowed_hf_publishers: [{owner}]",
                "max_download_bytes: 2000000000",
                f"remote_lora_train_dataset: {ROOT / 'data' / 'train.jsonl'}",
                f"remote_lora_eval_dataset: {ROOT / 'data' / 'eval.jsonl'}",
                "training_enabled: true",
                f"plasticity_train_dataset: {ROOT / 'data' / 'train.jsonl'}",
                f"plasticity_eval_dataset: {ROOT / 'data' / 'eval.jsonl'}",
                f"plasticity_publish_dir: {output_root / 'skills'}",
                "max_plasticity_loras: 8",
                "remote_lora_catalog:",
                "  - skill_id: sql_remote",
                f"    source: {remote_source}",
                "    cues: [sql]",
                "max_tokens: 8",
                "temperature: 0.0",
                "",
            ]
        )
    )
    return path


def _run(runtime: DynamicRuntime, prompt: str) -> dict:
    return runtime.infer(prompt=prompt, max_tokens=8, temperature=0.0)


def main(argv: list[str] | None = None) -> int:
    from slmcortex.runtime.dynamic import DynamicRuntime, DynamicRouteDecision

    parser = argparse.ArgumentParser(description="Run dynamic adaptive local/remote/plasticity smoke.")
    parser.add_argument("--output-root")
    parser.add_argument("--real", action="store_true", help="Use real runtime fetch/train/model paths.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "prototype.yaml"))
    parser.add_argument("--remote-source", help=argparse.SUPPRESS)
    parser.add_argument("--failure-mode", choices=("remote-download", "training"))
    parsed = parser.parse_args(argv)

    output_root = Path(parsed.output_root or tempfile.mkdtemp(prefix="slmcortex-dynamic-smoke-")).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    os.environ["SLMCORTEX_BASE_CONFIG"] = parsed.config if parsed.real else str(_write_mock_config(output_root))

    if parsed.real:
        _package(output_root, "fastapi_skill", "FastAPI endpoint validation")
        runtime = DynamicRuntime.load(output_root / "skills", allow_remote_loras=True)
    else:
        runtime = _mock_runtime(output_root, parsed.failure_mode)

    def router(messages, skills):
        text = messages[-1]["content"].lower()
        if "train" in text:
            return DynamicRouteDecision(
                base_model="mlx-test-base",
                selected_skills=[],
                remote_loras=[],
                task_type="python_generation",
                semantic_family="custom",
                train_new_lora=True,
                reason="smoke training branch",
            )
        return runtime._rule_router(messages, skills)

    runtime._router_model = router
    results = {
        "local": _run(runtime, "Fix a FastAPI validation bug"),
        "remote": _run(runtime, "Tune a SQL query"),
        "plasticity": _run(runtime, "train a custom adapter"),
    }
    summary = {
        "status": "complete",
        "mode": "real" if parsed.real else "mock",
        "output_root": str(output_root),
        "branches": {name: result["route_branch"] for name, result in results.items()},
        "selected_skills": {name: result["selected_skills"] for name, result in results.items()},
        "results": results,
    }
    json.dump(summary, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
