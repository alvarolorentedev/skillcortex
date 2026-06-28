from __future__ import annotations

from pathlib import Path

from ..packaging.composition import validate_composition_metadata
from ..packaging.validation import validate_slm_package
from ..shared.hashing import package_fingerprint
from ..shared.io import read_json, read_yaml


def load_slm_package(path: Path) -> dict:
    root = path.resolve()
    validate_slm_package(root)
    slm_manifest = read_yaml(root / "slm.yaml")
    metadata = read_json(root / "metadata.json")
    adapter_config_path = root / "adapter" / "adapter_config.json"
    adapter_config = read_json(adapter_config_path) if adapter_config_path.exists() else {}
    composition = load_composition(slm_manifest, metadata)
    return {
        "path": root,
        "slm_id": slm_manifest["slm_id"],
        "name": slm_manifest["name"],
        "version": slm_manifest["version"],
        "manifest": slm_manifest,
        "metadata": metadata,
        "adapter_config": adapter_config,
        "composition": composition,
        "fingerprint": package_fingerprint(slm_manifest, metadata),
    }


def load_composition(slm_manifest: dict, metadata: dict) -> dict:
    manifest_composition = slm_manifest.get("composition")
    metadata_composition = metadata.get("composition")
    if manifest_composition is None and metadata_composition is None:
        raise ValueError(
            "package missing composition metadata: composition.capabilities.allowed_task_types"
        )
    if manifest_composition is None or metadata_composition is None:
        raise ValueError("package composition metadata must exist in both slm.yaml and metadata.json")
    if manifest_composition != metadata_composition:
        raise ValueError("manifest mismatch for composition metadata")
    validate_composition_metadata(manifest_composition)
    return manifest_composition


def validate_unique_slm_ids(loaded: list[dict]) -> None:
    seen = set()
    for item in loaded:
        if item["slm_id"] in seen:
            raise ValueError(f"duplicate slm package: {item['slm_id']}")
        seen.add(item["slm_id"])
