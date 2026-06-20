from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "configs"
DATA_DIR = ROOT / "data"
ARTIFACT_DIR = ROOT / "artifacts"


def load_yaml(path: str | Path) -> dict:
    with Path(path).open() as handle:
        value = yaml.safe_load(handle) or {}
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return value


def base_config() -> dict:
    return load_yaml(CONFIG_DIR / "base.yaml")


def training_config() -> dict:
    return load_yaml(CONFIG_DIR / "training.yaml")
