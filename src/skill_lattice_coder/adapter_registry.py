import json
from pathlib import Path

from .config import ARTIFACT_DIR
from .schemas import SKILLS


def adapter_path(name: str) -> Path:
    if name != "generic" and name not in SKILLS:
        raise ValueError(f"unknown adapter: {name}")
    return ARTIFACT_DIR / "adapters" / name


def adapter_metadata(name: str) -> dict:
    path = adapter_path(name) / "metadata.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def require_adapter(name: str) -> Path:
    path = adapter_path(name)
    if not (path / "adapters.safetensors").exists():
        raise FileNotFoundError(f"adapter not found: {path}")
    return path
