from __future__ import annotations

import importlib.util
import os
import platform
from dataclasses import dataclass
from pathlib import Path

from .config import mlx_supported


PRODUCT_MODES = ("composer", "factory")


@dataclass(slots=True)
class AppWorkspace:
    root: Path
    state_dir: Path
    packages_dir: Path
    runtimes_dir: Path
    exports_dir: Path
    logs_dir: Path
    diagnostics_dir: Path

    def as_dict(self) -> dict[str, str]:
        return {
            "root": str(self.root),
            "state_dir": str(self.state_dir),
            "packages_dir": str(self.packages_dir),
            "runtimes_dir": str(self.runtimes_dir),
            "exports_dir": str(self.exports_dir),
            "logs_dir": str(self.logs_dir),
            "diagnostics_dir": str(self.diagnostics_dir),
        }


def default_app_workspace_root() -> Path:
    system = platform.system().lower()
    home = Path.home()
    if system == "darwin":
        return home / "Library" / "Application Support" / "SlmCortex"
    if system == "windows":
        appdata = Path(os.environ.get("APPDATA") or (home / "AppData" / "Roaming"))
        return appdata / "SlmCortex"
    state_home = Path(os.environ.get("XDG_STATE_HOME") or (home / ".local" / "state"))
    return state_home / "slmcortex"


def resolve_app_workspace(root: Path | None = None) -> AppWorkspace:
    workspace_root = (root or default_app_workspace_root()).expanduser().resolve()
    return AppWorkspace(
        root=workspace_root,
        state_dir=workspace_root / "state",
        packages_dir=workspace_root / "packages",
        runtimes_dir=workspace_root / "runtimes",
        exports_dir=workspace_root / "exports",
        logs_dir=workspace_root / "logs",
        diagnostics_dir=workspace_root / "diagnostics",
    )


def ensure_app_workspace(root: Path | None = None) -> AppWorkspace:
    workspace = resolve_app_workspace(root)
    for path in (
        workspace.root,
        workspace.state_dir,
        workspace.packages_dir,
        workspace.runtimes_dir,
        workspace.exports_dir,
        workspace.logs_dir,
        workspace.diagnostics_dir,
    ):
        path.mkdir(parents=True, exist_ok=True)
    return workspace


def runtime_name_for_folder(folder: Path, runtime_name: str | None = None) -> str:
    if runtime_name:
        candidate = runtime_name.strip()
    else:
        candidate = folder.resolve().name.strip()
    normalized = "".join(char.lower() if char.isalnum() else "-" for char in candidate)
    normalized = "-".join(part for part in normalized.split("-") if part)
    return normalized or "composed-runtime"


def environment_diagnostics(
    *,
    workspace_root: Path | None = None,
    product_mode: str = "composer",
) -> dict:
    if product_mode not in PRODUCT_MODES:
        raise ValueError(f"unknown product mode: {product_mode}")
    workspace = resolve_app_workspace(workspace_root)
    system = platform.system().lower()
    machine = platform.machine().lower()
    python_version = platform.python_version()
    backend_rows = [
        {
            "backend": "mlx",
            "supported_platform": mlx_supported(),
            "dependency_available": _module_available("mlx_lm"),
        },
        {
            "backend": "gguf",
            "supported_platform": True,
            "dependency_available": _module_available("llama_cpp"),
        },
    ]
    for row in backend_rows:
        row["available"] = row["supported_platform"] and row["dependency_available"]
    optional_factory_dependencies = [
        _dependency_row("peft"),
        _dependency_row("transformers"),
        _dependency_row("safetensors"),
        _dependency_row("torch"),
    ]
    available_runtime_backends = [row["backend"] for row in backend_rows if row["available"]]
    default_backend = "mlx" if mlx_supported() else "gguf"
    composer_ready = True
    summary_lines = [
        f"Product mode: {product_mode}",
        f"Platform: {system}/{machine} with Python {python_version}",
        f"Default runtime backend: {default_backend}",
        (
            "Runtime backends available: " + ", ".join(available_runtime_backends)
            if available_runtime_backends
            else "Runtime backends available: none detected; dry-run composition still works"
        ),
        "Factory extras installed: "
        + ", ".join(
            row["name"] for row in optional_factory_dependencies if row["available"]
        )
        if any(row["available"] for row in optional_factory_dependencies)
        else "Factory extras installed: none",
    ]
    warnings = []
    if not available_runtime_backends:
        warnings.append(
            "no runtime backend dependencies detected; composition and dry-run flows still work"
        )
    if product_mode == "factory" and not all(
        row["available"] for row in optional_factory_dependencies
    ):
        warnings.append(
            "factory mode is available, but optional training dependencies are missing"
        )
    return {
        "status": "complete",
        "product_mode": product_mode,
        "platform": {
            "system": system,
            "machine": machine,
            "python_version": python_version,
        },
        "workspace": workspace.as_dict(),
        "default_runtime_backend": default_backend,
        "available_runtime_backends": available_runtime_backends,
        "composer_ready": composer_ready,
        "backends": backend_rows,
        "optional_factory_dependencies": optional_factory_dependencies,
        "summary": "\n".join(summary_lines),
        "summary_lines": summary_lines,
        "warnings": warnings,
    }


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _dependency_row(name: str) -> dict[str, object]:
    return {
        "name": name,
        "available": _module_available(name),
    }