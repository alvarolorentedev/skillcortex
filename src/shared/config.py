from __future__ import annotations

import os
import platform
from pathlib import Path

from .io import read_yaml


ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "configs"
DATA_DIR = ROOT / "data"
ARTIFACT_DIR = ROOT / "artifacts"


def base_config() -> dict:
    # Use the new `SLMCORTEX_BASE_CONFIG` environment variable.
    # Backward-compatibility with `SLMCORTEX_BASE_CONFIG` has been removed.
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
