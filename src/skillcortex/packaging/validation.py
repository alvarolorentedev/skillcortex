from __future__ import annotations

from pathlib import Path

from ..shared.hashing import sha256
from ..shared.io import read_json, read_yaml
from .artifacts import REQUIRED_PACKAGE_FILES, validate_package_checksums
from .composition import validate_composition_metadata


def validate_skill_package(path: Path) -> dict:
    root = path.resolve()
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"package not found: {root}")
    missing = [name for name in REQUIRED_PACKAGE_FILES if not (root / name).exists()]
    if missing:
        raise ValueError(f"package is missing required files: {missing[0]}")

    skill_manifest = read_yaml(root / "skill.yaml")
    metadata = read_json(root / "metadata.json")
    training = read_json(root / "training_config.json")
    evaluation = read_json(root / "eval.json")

    if skill_manifest.get("schema_version") != "1":
        raise ValueError("skill.yaml schema_version must be '1'")
    if metadata.get("schema_version") != "1":
        raise ValueError("metadata.json schema_version must be '1'")
    if metadata.get("status") != "complete":
        raise ValueError("metadata.json status must be 'complete'")
    if skill_manifest.get("status") != "complete":
        raise ValueError("skill.yaml status must be 'complete'")

    for field in ("skill_id", "name", "version"):
        if skill_manifest.get(field) != metadata.get(field):
            raise ValueError(f"manifest mismatch for {field}")

    skill_base = skill_manifest.get("base") or {}
    metadata_base = metadata.get("base") or {}
    if skill_base.get("source_model") != metadata_base.get("source_model"):
        raise ValueError("manifest mismatch for base source_model")
    if skill_base.get("runtime_model") != metadata_base.get("runtime_model"):
        raise ValueError("manifest mismatch for base runtime_model")
    if (skill_manifest.get("adapter") or {}).get("trainable_parameters") != (
        metadata.get("adapter") or {}
    ).get("trainable_parameters"):
        raise ValueError("manifest mismatch for trainable_parameters")

    if training.get("seed") != (metadata.get("training") or {}).get("seed"):
        raise ValueError("training_config.json seed must match metadata.json")
    if evaluation.get("summary_path") != "eval.json":
        raise ValueError("eval.json summary_path must reference itself")
    if evaluation.get("dataset", {}).get("sha256") != (
        metadata.get("datasets") or {}
    ).get("eval", {}).get("sha256"):
        raise ValueError("eval dataset hash must match metadata.json")
    if not metadata.get("checksums"):
        raise ValueError("metadata.json must record package checksums")
    validate_package_checksums(root, metadata["checksums"])

    protected = metadata.get("protected_inputs") or {}
    if not protected.get("all_unchanged"):
        raise ValueError("protected input snapshot must confirm inputs were unchanged")
    for file_path, snapshot in sorted((protected.get("files") or {}).items()):
        current = Path(file_path)
        if current.exists() and sha256(current) != snapshot.get("after_sha256"):
            raise ValueError(f"protected input changed since packaging: {file_path}")

    examples_path = (skill_manifest.get("examples") or {}).get("path")
    if examples_path and not (root / examples_path).exists():
        raise ValueError("examples.jsonl is declared but missing")
    if skill_manifest.get("composition") != metadata.get("composition"):
        if not (skill_manifest.get("composition") is None and metadata.get("composition") is None):
            raise ValueError("manifest mismatch for composition metadata")
    if skill_manifest.get("composition") is not None:
        validate_composition_metadata(skill_manifest["composition"])
    return {
        "status": "valid",
        "path": str(root),
        "skill_id": skill_manifest["skill_id"],
        "version": skill_manifest["version"],
    }
