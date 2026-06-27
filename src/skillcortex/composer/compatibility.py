from __future__ import annotations

from pathlib import Path

from .adapters import validate_adapter_configs, validate_adapter_metadata
from ..shared.io import read_json


def build_compatibility_report(loaded: list[dict], enrichment: dict) -> dict:
    errors = []
    warnings = []
    manifests = [item["manifest"] for item in loaded]
    first_base = manifests[0].get("base") or {}
    for key in ("source_model", "runtime_model", "quantization"):
        expected = first_base.get(key)
        if any((manifest.get("base") or {}).get(key) != expected for manifest in manifests[1:]):
            errors.append(f"incompatible package base {key}")
    try:
        validate_adapter_metadata(
            [
                {
                    "base_model": item["metadata"]["base"]["runtime_model"],
                    "target_modules": item["metadata"]["adapter"]["target_modules"],
                    "quantization": item["metadata"]["base"]["quantization"],
                }
                for item in loaded
            ]
        )
    except ValueError as error:
        errors.append(str(error))
    try:
        validate_adapter_configs([item["adapter_config"] for item in loaded])
    except ValueError as error:
        errors.append(str(error))
    selected = {item["skill_id"] for item in loaded}
    for item in loaded:
        incompatible = set(((item["composition"].get("compatibility") or {}).get("incompatible_skills") or []))
        overlap = sorted(selected & incompatible)
        if overlap:
            errors.append(f"skill {item['skill_id']} is incompatible with selected skill {overlap[0]}")
    for entry in enrichment.get("matched_skills", []):
        package_tasks = set(entry["package_allowed_task_types"])
        registry_tasks = set(entry["registry_allowed_task_types"])
        if registry_tasks and package_tasks != registry_tasks:
            warnings.append(
                f"registry enrichment differs from package metadata for {entry['skill_id']} allowed_task_types"
            )
    return {
        "schema_version": "1",
        "status": "valid" if not errors else "invalid",
        "skills": [item["skill_id"] for item in loaded],
        "errors": errors,
        "warnings": warnings,
        "optional_enrichment_used": enrichment["enabled"] and bool(enrichment["matched_skills"]),
        "registry_enrichment": enrichment,
    }


def load_registry_enrichment(registry: Path | None, loaded: list[dict]) -> dict:
    if registry is None:
        return {
            "enabled": False,
            "path": None,
            "matched_skills": [],
            "source_of_truth": "package",
            "override_applied": False,
        }
    payload = read_json(registry.resolve())
    skills = payload.get("skills") or []
    by_name = {
        item.get("skill_name"): item
        for item in skills
        if isinstance(item, dict) and item.get("skill_name")
    }
    matched = []
    for item in loaded:
        registry_skill = by_name.get(item["skill_id"])
        if registry_skill is None:
            continue
        matched.append(
            {
                "skill_id": item["skill_id"],
                "registry_status": registry_skill.get("status"),
                "registry_origin": registry_skill.get("origin"),
                "registry_router": registry_skill.get("router"),
                "registry_activation_scope": registry_skill.get("activation_scope"),
                "registry_allowed_task_types": registry_skill.get("allowed_task_types") or [],
                "package_allowed_task_types": item["composition"]["capabilities"].get("allowed_task_types") or [],
            }
        )
    return {
        "enabled": True,
        "path": str(registry.resolve()),
        "matched_skills": matched,
        "source_of_truth": "package",
        "override_applied": False,
    }
