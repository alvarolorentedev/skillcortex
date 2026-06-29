from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from .config import BACKEND_DEPENDENCIES, backend_supported_on_platform


def provision_backend(
    *,
    backend: str,
    workspace_root: Path | None = None,
    dry_run: bool = False,
) -> dict:
    from .product import environment_diagnostics

    if backend not in BACKEND_DEPENDENCIES:
        raise ValueError(f"unknown backend: {backend}")
    before = environment_diagnostics(workspace_root=workspace_root, product_mode="composer")
    command = backend_provisioning_command(backend)
    if not backend_supported_on_platform(backend):
        return {
            "status": "blocked",
            "exit_code": 2,
            "backend": backend,
            "supported_platform": False,
            "command": command,
            "dependencies": list(BACKEND_DEPENDENCIES[backend]),
            "capabilities_before": _capabilities(before),
            "capabilities_unlocked": [],
            "warnings": [f"{backend} is not supported on this platform"],
        }
    if dry_run:
        return {
            "status": "dry-run",
            "exit_code": 0,
            "backend": backend,
            "supported_platform": True,
            "command": command,
            "dependencies": list(BACKEND_DEPENDENCIES[backend]),
            "capabilities_before": _capabilities(before),
            "capabilities_unlocked": ["local_run", "local_inference"],
            "warnings": [],
        }
    completed = subprocess.run(command, capture_output=True, text=True)
    after = environment_diagnostics(workspace_root=workspace_root, product_mode="composer")
    unlocked = []
    if backend in (after.get("available_runtime_backends") or []) and backend not in (
        before.get("available_runtime_backends") or []
    ):
        unlocked = ["local_run", "local_inference"]
    return {
        "status": "complete" if completed.returncode == 0 else "failed",
        "exit_code": 0 if completed.returncode == 0 else 2,
        "backend": backend,
        "supported_platform": True,
        "command": command,
        "dependencies": list(BACKEND_DEPENDENCIES[backend]),
        "capabilities_before": _capabilities(before),
        "capabilities_after": _capabilities(after),
        "capabilities_unlocked": unlocked,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "warnings": [] if completed.returncode == 0 else [f"{backend} provisioning failed"],
    }


def backend_provisioning_command(backend: str) -> list[str]:
    if backend not in BACKEND_DEPENDENCIES:
        raise ValueError(f"unknown backend: {backend}")
    return [sys.executable, "-m", "pip", "install", *BACKEND_DEPENDENCIES[backend]]


def _capabilities(diagnostics: dict) -> dict:
    return {
        "available_runtime_backends": list(diagnostics.get("available_runtime_backends") or []),
        "composer_ready": bool(diagnostics.get("composer_ready")),
    }