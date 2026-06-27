from __future__ import annotations

from pathlib import Path

from ..packaging import validate_composition_metadata, validate_skill_package
from ..shared.hashing import package_fingerprint
from ..shared.io import read_json, read_yaml


def load_skill_package(path: Path) -> dict:
    root = path.resolve()
    validate_skill_package(root)
    skill_manifest = read_yaml(root / "skill.yaml")
    metadata = read_json(root / "metadata.json")
    adapter_config = read_json(root / "adapter" / "adapter_config.json")
    composition = load_composition(skill_manifest, metadata)
    return {
        "path": root,
        "skill_id": skill_manifest["skill_id"],
        "name": skill_manifest["name"],
        "version": skill_manifest["version"],
        "manifest": skill_manifest,
        "metadata": metadata,
        "adapter_config": adapter_config,
        "composition": composition,
        "fingerprint": package_fingerprint(skill_manifest, metadata),
    }


def load_composition(skill_manifest: dict, metadata: dict) -> dict:
    manifest_composition = skill_manifest.get("composition")
    metadata_composition = metadata.get("composition")
    if manifest_composition is None and metadata_composition is None:
        raise ValueError(
            "package missing composition metadata: composition.capabilities.allowed_task_types"
        )
    if manifest_composition is None or metadata_composition is None:
        raise ValueError("package composition metadata must exist in both skill.yaml and metadata.json")
    if manifest_composition != metadata_composition:
        raise ValueError("manifest mismatch for composition metadata")
    validate_composition_metadata(manifest_composition)
    return manifest_composition


def validate_unique_skill_ids(loaded: list[dict]) -> None:
    seen = set()
    for item in loaded:
        if item["skill_id"] in seen:
            raise ValueError(f"duplicate skill package: {item['skill_id']}")
        seen.add(item["skill_id"])
