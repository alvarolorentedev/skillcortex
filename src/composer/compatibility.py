from __future__ import annotations

from pathlib import Path

from .adapters import validate_adapter_configs, validate_adapter_metadata
from ..shared.config import adapter_format_for_backend
from ..shared.io import read_json


def build_compatibility_report(loaded: list[dict], enrichment: dict) -> dict:
    errors = []
    warnings = []
    manifests = [item["manifest"] for item in loaded]
    first_base = manifests[0].get("base") or {}
    for key in ("source_model", "runtime_model", "quantization", "backend"):
        expected = first_base.get(key)
        if any((manifest.get("base") or {}).get(key) != expected for manifest in manifests[1:]):
            errors.append(f"incompatible package base {key}")
    for item in loaded:
        backend = (item["metadata"].get("base") or {}).get("backend") or "mlx"
        adapter_format = (item["metadata"].get("adapter") or {}).get("format")
        if adapter_format != adapter_format_for_backend(backend):
            errors.append(f"adapter format {adapter_format} is incompatible with backend {backend}")
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
    if all((item["metadata"].get("adapter") or {}).get("format") == "mlx-lora" for item in loaded):
        try:
            validate_adapter_configs([item["adapter_config"] for item in loaded])
        except ValueError as error:
            errors.append(str(error))
    selected = {item["slm_id"] for item in loaded}
    for item in loaded:
        incompatible = set(((item["composition"].get("compatibility") or {}).get("incompatible_slms") or []))
        overlap = sorted(selected & incompatible)
        if overlap:
            errors.append(f"slm {item['slm_id']} is incompatible with selected slm {overlap[0]}")
    for entry in enrichment.get("matched_slms", []):
        package_tasks = set(entry["package_allowed_task_types"])
        registry_tasks = set(entry["registry_allowed_task_types"])
        if registry_tasks and package_tasks != registry_tasks:
            warnings.append(
                f"registry enrichment differs from package metadata for {entry['slm_id']} allowed_task_types"
            )
    return {
        "schema_version": "1",
        "status": "valid" if not errors else "invalid",
        "slms": [item["slm_id"] for item in loaded],
        "errors": errors,
        "warnings": warnings,
        "optional_enrichment_used": enrichment["enabled"] and bool(enrichment["matched_slms"]),
        "registry_enrichment": enrichment,
    }


def load_registry_enrichment(registry: Path | None, loaded: list[dict]) -> dict:
    if registry is None:
        return {
            "enabled": False,
            "path": None,
            "matched_slms": [],
            "source_of_truth": "package",
            "override_applied": False,
        }
    payload = read_json(registry.resolve())
    slms = payload.get("slms") or []
    by_name = {
        item.get("slm_name"): item
        for item in slms
        if isinstance(item, dict) and item.get("slm_name")
    }
    matched = []
    for item in loaded:
        registry_slm = by_name.get(item["slm_id"])
        if registry_slm is None:
            continue
        matched.append(
            {
                "slm_id": item["slm_id"],
                "registry_status": registry_slm.get("status"),
                "registry_origin": registry_slm.get("origin"),
                "registry_router": registry_slm.get("router"),
                "registry_activation_scope": registry_slm.get("activation_scope"),
                "registry_allowed_task_types": registry_slm.get("allowed_task_types") or [],
                "package_allowed_task_types": item["composition"]["capabilities"].get("allowed_task_types") or [],
            }
        )
    return {
        "enabled": True,
        "path": str(registry.resolve()),
        "matched_slms": matched,
        "source_of_truth": "package",
        "override_applied": False,
    }
