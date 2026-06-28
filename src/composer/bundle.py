from __future__ import annotations

import json
from pathlib import Path

import yaml

from ..shared.hashing import sha256


def build_bundle(loaded: list[dict], routes: list[dict], enrichment: dict) -> dict:
    base = loaded[0]["metadata"]["base"]
    return {
        "schema_version": "1",
        "composition_type": "slm_runtime_bundle",
        "strategy": "routed",
        "status": "complete",
        "runtime": {
            "source_model": base["source_model"],
            "runtime_model": base["runtime_model"],
            "quantization": base["quantization"],
            "backend": base.get("backend") or "mlx",
        },
        "slms": [
            {
                "slm_id": item["slm_id"],
                "name": item["name"],
                "version": item["version"],
                "package_path": str(item["path"]),
                "fingerprint": item["fingerprint"],
                "composition": item["composition"],
                "adapter": item["metadata"]["adapter"],
            }
            for item in sorted(loaded, key=lambda value: value["slm_id"])
        ],
        "routes": routes,
        "provenance": {"registry_enrichment": enrichment},
    }


def build_budget_report(loaded: list[dict], routes: list[dict]) -> dict:
    stored_parameters = sum(int(item["metadata"]["adapter"]["trainable_parameters"]) for item in loaded)
    bytes_by_slm = {
        item["slm_id"]: (item["path"] / item["metadata"]["adapter"]["files"]["weights"]).stat().st_size
        for item in loaded
    }
    params_by_slm = {
        item["slm_id"]: int(item["metadata"]["adapter"]["trainable_parameters"])
        for item in loaded
    }
    max_active_parameters = max(sum(params_by_slm[slm_id] for slm_id in route["selected_slms"]) for route in routes)
    max_active_bytes = max(sum(bytes_by_slm[slm_id] for slm_id in route["selected_slms"]) for route in routes)
    return {
        "schema_version": "1",
        "stored_adapter_count": len(loaded),
        "stored_adapter_parameters": stored_parameters,
        "stored_adapter_bytes": sum(bytes_by_slm.values()),
        "max_active_adapter_count": max(len(route["selected_slms"]) for route in routes),
        "max_active_adapter_parameters": max_active_parameters,
        "max_active_adapter_bytes": max_active_bytes,
        "memory_estimate_method": "active packaged adapter file bytes",
    }


def bundle_files(bundle: dict, compatibility: dict, budget: dict) -> dict[str, str]:
    active_slms = {
        "schema_version": "1",
        "slms": [
            {
                "slm_id": item["slm_id"],
                "name": item["name"],
                "version": item["version"],
                "package_path": item["package_path"],
                "allowed_task_types": item["composition"]["capabilities"]["allowed_task_types"],
                "activation": item["composition"]["activation"],
                "trainable_parameters": item["adapter"]["trainable_parameters"],
                "route_membership": [
                    route["route_id"]
                    for route in bundle["routes"]
                    if item["slm_id"] in route["selected_slms"]
                ],
            }
            for item in bundle["slms"]
        ],
    }
    router_config = {
        "schema_version": "1",
        "strategy": bundle["strategy"],
        "routes": bundle["routes"],
    }
    return {
        "composition.yaml": yaml.safe_dump(bundle, sort_keys=False),
        "router_config.json": json.dumps(router_config, indent=2, sort_keys=True) + "\n",
        "active_slms.json": json.dumps(active_slms, indent=2, sort_keys=True) + "\n",
        "compatibility_report.json": json.dumps(compatibility, indent=2, sort_keys=True) + "\n",
        "budget_report.json": json.dumps(budget, indent=2, sort_keys=True) + "\n",
        "README.md": build_readme(bundle, compatibility, budget),
    }


def write_checksums(staging: Path, files: dict[str, str], loaded: list[dict], enrichment: dict) -> None:
    checksums = {relative: sha256(staging / relative) for relative in sorted(files)}
    (staging / "checksums.json").write_text(
        json.dumps(
            {
                "schema_version": "1",
                "files": checksums,
                "source_packages": [
                    {
                        "slm_id": item["slm_id"],
                        "path": str(item["path"]),
                        "fingerprint": item["fingerprint"],
                    }
                    for item in loaded
                ],
                "registry_enrichment": enrichment,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def build_readme(bundle: dict, compatibility: dict, budget: dict) -> str:
    lines = [
        "# SLMCortex Runtime Bundle",
        "",
        f"- Strategy: `{bundle['strategy']}`",
        f"- Slms: {', '.join(item['slm_id'] for item in bundle['slms'])}",
        f"- Runtime model: `{bundle['runtime']['runtime_model']}`",
        f"- Compatibility status: `{compatibility['status']}`",
        f"- Max active adapter parameters: **{budget['max_active_adapter_parameters']}**",
        f"- Optional registry enrichment used: `{compatibility['optional_enrichment_used']}`",
        "",
    ]
    return "\n".join(lines)
