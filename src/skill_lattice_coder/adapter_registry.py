import json
from pathlib import Path

from .config import ARTIFACT_DIR
from .schemas import SKILLS


def adapter_path(name: str, root: str | Path | None = None) -> Path:
    if name != "generic" and name not in SKILLS:
        raise ValueError(f"unknown adapter: {name}")
    return Path(root) / name if root else ARTIFACT_DIR / "adapters" / name


def adapter_metadata(name: str, root: str | Path | None = None) -> dict:
    path = adapter_path(name, root) / "metadata.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def require_adapter(name: str, root: str | Path | None = None) -> Path:
    path = adapter_path(name, root)
    if not (path / "adapters.safetensors").exists():
        raise FileNotFoundError(f"adapter not found: {path}")
    return path
