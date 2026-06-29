import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from slmcortex.packaging.artifacts import adapter_weights_path, package_checksums
from slmcortex.shared.config import (
    adapter_format_for_backend,
    adapter_weight_name_for_format,
    base_config,
    resolve_backend,
    training_config,
)


ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "tests" / "fixtures" / "slmcortex_demo"


def _run(name: str, command: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> dict:
    completed = subprocess.run(
        command,
        cwd=cwd or ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    record = {
        "name": name,
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }
    if completed.returncode != 0:
        raise RuntimeError(json.dumps(record, indent=2))
    return record


def _launcher_command(launcher_path: Path, *args: str) -> list[str]:
    if launcher_path.suffix.lower() == ".ps1":
        return ["powershell", "-ExecutionPolicy", "Bypass", "-File", str(launcher_path), *args]
    return [str(launcher_path), *args]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Create an isolated virtual environment, install Slm Cortex, and launch the Composer-first entry point.",
    )
    parser.add_argument("--package-source", default=str(ROOT))
    parser.add_argument("--workspace-root")
    parser.add_argument("--install-root")
    parsed = parser.parse_args(argv)

    install_root = (
        Path(parsed.install_root).resolve()
        if parsed.install_root
        else Path(tempfile.mkdtemp(prefix="slmcortex-install-"))
    )
    workspace_root = (
        Path(parsed.workspace_root).resolve()
        if parsed.workspace_root
        else Path(tempfile.mkdtemp(prefix="slmcortex-installed-workspace-"))
    )
    adapter_dir = _resolve_smoke_adapter_dir(workspace_root)
    installer_path, launcher_path, composer_launcher_path = _installer_contract(install_root)
    env = dict(os.environ)
    env["SLMCORTEX_INSTALL_ROOT"] = str(install_root)
    toy_repo = workspace_root / "state" / "repo"
    _copy_demo_repo(toy_repo)
    package_path = workspace_root / "packages" / "fastapi_contract"
    export_descriptor = workspace_root / "exports" / "repo.json"
    support_bundle = workspace_root / "diagnostics" / "support" / "doctor-support.json"

    steps = []
    install_command = installer_path + [parsed.package_source]
    if len(installer_path) > 1 and not Path(installer_path[1]).exists():
        install_command = [
            sys.executable,
            str(ROOT / "scripts" / "install_from_source.py"),
            parsed.package_source,
            "--install-root",
            str(install_root),
        ]
    steps.append(_run("install_package", install_command, cwd=ROOT, env=env))
    steps.append(_run("launch_help", _launcher_command(launcher_path, "--help")))
    steps.append(_run("composer_launcher_help", _launcher_command(composer_launcher_path, "--help")))
    steps.append(
        _run(
            "doctor",
            _launcher_command(launcher_path, "doctor", "--workspace", str(workspace_root)),
        )
    )
    steps.append(
        _run(
            "doctor_support_bundle",
            [
                *_launcher_command(launcher_path, "doctor", "--workspace", str(workspace_root)),
                "--export-support-bundle",
                "--support-bundle-path",
                str(support_bundle),
            ],
        )
    )
    steps.append(
        _run(
            "package_fastapi_contract",
            [
                *_launcher_command(launcher_path, "package-slm", "--slm-id", "fastapi_contract", "--name", "FastAPI Contract Slm"),
                "--adapter-dir",
                str(adapter_dir),
                "--output",
                str(package_path),
                "--train-dataset",
                str(ROOT / "data" / "train.jsonl"),
                "--eval-dataset",
                str(ROOT / "data" / "eval.jsonl"),
                "--eval-summary",
                str(FIXTURES / "eval-summary.json"),
                "--description",
                "FastAPI endpoints with Pydantic validation.",
                "--allowed-task-types",
                "python_generation",
                "--activation-scope",
                "task",
            ],
        )
    )
    _enrich_fastapi_package(package_path)
    steps.append(
        _run(
            "compose_folder",
            [
                *_launcher_command(
                    launcher_path,
                    "compose-folder",
                    "--workspace",
                    str(workspace_root),
                    "--folder",
                    str(toy_repo),
                    "--task",
                    "Create a FastAPI endpoint with Pydantic validation",
                ),
                "--export-descriptor",
                str(export_descriptor),
            ],
        )
    )
    steps.append(
        _run(
            "composer_app_export",
            [
                *_launcher_command(
                    composer_launcher_path,
                    "--workspace",
                    str(workspace_root),
                    "--folder",
                    str(toy_repo),
                    "--task",
                    "Create a FastAPI endpoint with Pydantic validation",
                    "--outcome",
                    "export_bundle",
                ),
                "--export-descriptor",
                str(export_descriptor),
                "--export-logs",
                "--overwrite",
            ],
        )
    )

    summary = {
        "status": "complete",
        "install_root": str(install_root),
        "workspace_root": str(workspace_root),
        "steps": [
            {
                "name": step["name"],
                "command": step["command"],
            }
            for step in steps
        ],
    }
    json.dump(summary, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


def _copy_demo_repo(destination: Path) -> Path:
    import shutil

    shutil.copytree(FIXTURES / "toy-repo", destination)
    (destination / "app.py").write_text(
        "from fastapi import FastAPI\nfrom pydantic import BaseModel\n"
    )
    return destination


def _resolve_smoke_adapter_dir(workspace_root: Path) -> Path:
    adapter_dir = ROOT / "artifacts" / "adapters" / "python_slm"
    try:
        adapter_weights_path(adapter_dir)
        return adapter_dir
    except FileNotFoundError:
        fallback = workspace_root / ".smoke-adapter"
        config = base_config()
        training = training_config()
        backend = resolve_backend(config)
        adapter_format = adapter_format_for_backend(backend)
        runtime_model = str(config.get("model") or config.get("default_runtime_model") or "")
        if backend == "gguf" and not runtime_model.endswith(".gguf"):
            runtime_model = "smoke-test.gguf"
        fallback.mkdir(parents=True, exist_ok=True)
        fallback.joinpath(adapter_weight_name_for_format(adapter_format)).write_bytes(
            b"slmcortex smoke adapter\n"
        )
        fallback.joinpath("adapter_config.json").write_text(
            json.dumps(
                {
                    "fine_tune_type": "lora",
                    "lora_parameters": {
                        "dropout": 0.0,
                        "keys": list(training["target_modules"]),
                        "rank": int(training["slm_rank"]),
                        "scale": 20.0,
                    },
                    "model": runtime_model,
                    "num_layers": int(training["lora_layers"]),
                    "seed": int(training["seed"]),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        fallback.joinpath("metadata.json").write_text(
            json.dumps(
                {
                    "backend": backend,
                    "base_model": runtime_model,
                    "config": training,
                    "format": adapter_format,
                    "quantization": "4bit",
                    "rank": int(training["slm_rank"]),
                    "seed": int(training_config()["seed"]),
                    "source_model": str(config.get("source_model") or "synthetic-smoke-source"),
                    "target_modules": list(training["target_modules"]),
                    "trainable_parameters": 1,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        print(
            f"Using synthetic smoke adapter at {fallback} because adapter weights were not found at {adapter_dir}",
            file=sys.stderr,
        )
        return fallback


def _enrich_fastapi_package(package_path: Path) -> None:
    package_path.joinpath("routing_card.json").write_text(
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
    metadata = json.loads(package_path.joinpath("metadata.json").read_text())
    metadata["checksums"] = package_checksums(package_path)
    package_path.joinpath("metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")


def _installer_contract(install_root: Path) -> tuple[list[str], Path, Path]:
    if os.name == "nt":
        script = ROOT / "artifacts" / "installers" / "install-slmcortex-windows.ps1"
        return (
            ["powershell", "-ExecutionPolicy", "Bypass", "-File", str(script)],
            install_root / "Scripts" / "slmcortex.ps1",
            install_root / "Scripts" / "slmcortex-composer.ps1",
        )
    system = os.uname().sysname
    if system == "Darwin":
        script = ROOT / "artifacts" / "installers" / "install-slmcortex-macos.sh"
    else:
        script = ROOT / "artifacts" / "installers" / "install-slmcortex-linux.sh"
    return (["sh", str(script)], install_root / "slmcortex", install_root / "slmcortex-composer")


if __name__ == "__main__":
    raise SystemExit(main())