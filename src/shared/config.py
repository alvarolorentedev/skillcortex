from __future__ import annotations

import importlib.resources
import os
import platform
from pathlib import Path

from .io import read_yaml


REPO_ROOT = Path(__file__).resolve().parents[2]


def _bundled_root() -> Path | None:
    try:
        return Path(str(importlib.resources.files("slmcortex_resources"))).resolve()
    except (ModuleNotFoundError, FileNotFoundError):
        return None


def _resource_root() -> Path | None:
    bundled = _bundled_root()
    if bundled is not None:
        return bundled
    candidate = (REPO_ROOT / "src" / "slmcortex_resources").resolve()
    if candidate.exists():
        return candidate
    return None


def _resolve_root() -> Path:
    if (REPO_ROOT / "pyproject.toml").exists():
        return REPO_ROOT
    return _resource_root() or REPO_ROOT


def _resolve_dir(env_var: str, relative: str, *, require_exists: bool = False) -> Path:
    env_path = os.environ.get(env_var)
    if env_path:
        return Path(env_path).expanduser().resolve()
    candidate = REPO_ROOT / relative
    if candidate.exists() or not require_exists:
        return candidate
    bundled = _bundled_root()
    if bundled is not None:
        return bundled / relative
    return candidate


def _resolve_config_dir() -> Path:
    env_path = os.environ.get("SLMCORTEX_CONFIG_DIR")
    if env_path:
        return Path(env_path).expanduser().resolve()
    resource_root = _resource_root()
    if resource_root is not None:
        return resource_root / "configs"
    return REPO_ROOT / "src" / "slmcortex_resources" / "configs"


ROOT = _resolve_root()
CONFIG_DIR = _resolve_config_dir()
DATA_DIR = _resolve_dir("SLMCORTEX_DATA_DIR", "data")
ARTIFACT_DIR = _resolve_dir("SLMCORTEX_ARTIFACT_DIR", "artifacts")

BACKEND_DEPENDENCIES = {
    "mlx": ["mlx-lm>=0.31,<0.32"],
    "gguf": [
        "llama-cpp-python>=0.3,<0.4",
        "peft>=0.18,<0.19",
        "safetensors>=0.6,<0.7",
        "torch>=2.7,<3",
        "transformers>=5,<6",
    ],
}


def base_config() -> dict:
    env_path = os.environ.get("SLMCORTEX_BASE_CONFIG")
    config = read_yaml(Path(env_path) if env_path else CONFIG_DIR / "base.yaml")
    config.setdefault("backend", "auto")
    return config


def training_config() -> dict:
    return read_yaml(CONFIG_DIR / "training.yaml")


def mlx_supported() -> bool:
    return platform.system().lower() == "darwin" and platform.machine().lower() in {
        "arm64",
        "aarch64",
    }


def backend_supported_on_platform(backend: str) -> bool:
    if backend == "mlx":
        return mlx_supported()
    if backend == "gguf":
        return True
    raise ValueError(f"unknown backend: {backend}")


def resolve_backend(config: dict | None = None) -> str:
    backend = str((config or base_config()).get("backend") or "auto").lower()
    if backend == "auto":
        return "mlx" if mlx_supported() else "gguf"
    if backend == "mlx":
        if not mlx_supported():
            raise ValueError("MLX backend requires macOS arm64")
        return "mlx"
    if backend == "gguf":
        return "gguf"
    raise ValueError("backend must be one of auto, mlx, gguf")


def adapter_format_for_backend(backend: str) -> str:
    return {"mlx": "mlx-lora", "gguf": "gguf-lora"}[backend]


def adapter_weight_name_for_format(adapter_format: str) -> str:
    return {"mlx-lora": "adapters.safetensors", "gguf-lora": "adapter.gguf"}[adapter_format]


def validate_runtime_model(config: dict | None = None) -> str:
    resolved = resolve_backend(config)
    value = config or base_config()
    model = str(value.get("default_runtime_model") or value.get("model") or "")
    if resolved == "gguf" and not model.endswith(".gguf"):
        raise ValueError("GGUF backend requires a .gguf runtime model")
    if resolved == "mlx" and model.endswith(".gguf"):
        raise ValueError("MLX backend does not support .gguf runtime models")
    return resolved
