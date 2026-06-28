from __future__ import annotations

from pathlib import Path
from typing import Any

from ..contracts import TASK_TYPES
from ..packaging.validation import validate_slm_package
from ..shared.hashing import package_fingerprint, sha256
from ..shared.io import read_json, read_yaml
from ..shared.config import adapter_format_for_backend, validate_runtime_model
from .models import REQUIRED_RUNTIME_FILES, RuntimeBundle, RuntimeSlm


def load_runtime_bundle(runtime_path: Path) -> RuntimeBundle:
    root = runtime_path.resolve()
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"runtime bundle not found: {root}")
    missing = [name for name in REQUIRED_RUNTIME_FILES if not (root / name).exists()]
    if missing:
        raise ValueError(f"runtime bundle is missing required files: {missing[0]}")

    composition = read_yaml(root / "composition.yaml")
    router_config = read_json(root / "router_config.json")
    active_slms = read_json(root / "active_slms.json")
    compatibility_report = read_json(root / "compatibility_report.json")
    budget_report = read_json(root / "budget_report.json")
    checksums = read_json(root / "checksums.json")

    validate_bundle_metadata(composition, router_config, active_slms, compatibility_report, checksums)
    validate_bundle_checksums(root, checksums)
    slms = load_runtime_slms(composition, checksums)
    validate_router_routes(router_config.get("routes") or [], slms)
    validate_active_slms(active_slms, slms, router_config.get("routes") or [])

    runtime = composition.get("runtime") or {}
    backend = runtime.get("backend") or "mlx"
    validate_runtime_model({"backend": backend, "model": runtime["runtime_model"]})
    return RuntimeBundle(
        path=root,
        name=root.name,
        runtime_model=runtime["runtime_model"],
        source_model=runtime["source_model"],
        quantization=runtime["quantization"],
        backend=backend,
        strategy=composition["strategy"],
        routes=list(router_config["routes"]),
        slms=slms,
        compatibility_report=compatibility_report,
        budget_report=budget_report,
        checksums=checksums,
    )


def load_runtime_slms(composition: dict[str, Any], checksums: dict[str, Any]) -> dict[str, RuntimeSlm]:
    slm_entries = composition.get("slms") or []
    fingerprints_by_slm = {
        item["slm_id"]: item["fingerprint"] for item in checksums.get("source_packages") or []
    }
    slms: dict[str, RuntimeSlm] = {}
    for item in slm_entries:
        package_path = Path(item["package_path"]).resolve()
        validate_slm_package(package_path)
        metadata = read_json(package_path / "metadata.json")
        manifest = read_yaml(package_path / "slm.yaml")
        fingerprint = package_fingerprint(manifest, metadata)
        if fingerprint != item.get("fingerprint"):
            raise ValueError(f"runtime package fingerprint mismatch for {item['slm_id']}")
        if fingerprints_by_slm.get(item["slm_id"]) != fingerprint:
            raise ValueError(f"runtime checksum fingerprint mismatch for {item['slm_id']}")
        adapter_files = (metadata.get("adapter") or {}).get("files") or {}
        adapter_relative = adapter_files.get("weights") or "adapter/adapters.safetensors"
        adapter_format = (metadata.get("adapter") or {}).get("format") or "mlx-lora"
        backend = (metadata.get("base") or {}).get("backend") or "mlx"
        if adapter_format != adapter_format_for_backend(backend):
            raise ValueError(f"adapter format {adapter_format} is incompatible with backend {backend}")
        adapter_weight = package_path / adapter_relative
        slms[item["slm_id"]] = RuntimeSlm(
            slm_id=item["slm_id"],
            name=item["name"],
            version=item["version"],
            package_path=package_path,
            adapter_path=adapter_weight if adapter_format == "gguf-lora" else adapter_weight.parent,
            fingerprint=fingerprint,
            allowed_task_types=list(item["composition"]["capabilities"]["allowed_task_types"]),
            activation=dict(item["composition"]["activation"]),
            trainable_parameters=int(item["adapter"]["trainable_parameters"]),
            adapter_format=adapter_format,
        )
    return slms


def validate_active_slms(
    active_slms: dict[str, Any],
    slms: dict[str, RuntimeSlm],
    routes: list[dict[str, Any]],
) -> None:
    route_ids = {route["route_id"] for route in routes}
    active_ids = {item["slm_id"] for item in active_slms.get("slms") or []}
    if active_ids != set(slms):
        raise ValueError("active_slms.json must match composition slms")
    for item in active_slms.get("slms") or []:
        membership = set(item.get("route_membership") or [])
        if not membership.issubset(route_ids):
            raise ValueError(f"unknown route membership for {item['slm_id']}")


def validate_bundle_checksums(root: Path, checksums: dict[str, Any]) -> None:
    for relative, expected in sorted((checksums.get("files") or {}).items()):
        candidate = root / relative
        if not candidate.exists():
            raise ValueError(f"runtime bundle checksum target is missing: {relative}")
        if sha256(candidate) != expected:
            raise ValueError(f"runtime bundle checksum mismatch: {relative}")


def validate_bundle_metadata(
    composition: dict[str, Any],
    router_config: dict[str, Any],
    active_slms: dict[str, Any],
    compatibility_report: dict[str, Any],
    checksums: dict[str, Any],
) -> None:
    if composition.get("schema_version") != "1":
        raise ValueError("composition.yaml schema_version must be '1'")
    if composition.get("composition_type") != "slm_runtime_bundle":
        raise ValueError("composition.yaml composition_type must be 'slm_runtime_bundle'")
    if composition.get("status") != "complete":
        raise ValueError("composition.yaml status must be 'complete'")
    if router_config.get("schema_version") != "1":
        raise ValueError("router_config.json schema_version must be '1'")
    if active_slms.get("schema_version") != "1":
        raise ValueError("active_slms.json schema_version must be '1'")
    if compatibility_report.get("schema_version") != "1":
        raise ValueError("compatibility_report.json schema_version must be '1'")
    if compatibility_report.get("status") != "valid":
        errors = compatibility_report.get("errors") or ["unknown runtime compatibility error"]
        raise ValueError(errors[0])
    if checksums.get("schema_version") != "1":
        raise ValueError("checksums.json schema_version must be '1'")
    if router_config.get("strategy") != composition.get("strategy"):
        raise ValueError("runtime bundle strategy mismatch")
    if list(router_config.get("routes") or []) != list(composition.get("routes") or []):
        raise ValueError("router_config.json routes must match composition.yaml")


def validate_router_routes(routes: list[dict[str, Any]], slms: dict[str, RuntimeSlm]) -> None:
    if not routes:
        raise ValueError("runtime bundle routes are required")
    default_routes = set()
    for route in routes:
        task_type = route.get("task_type")
        if task_type not in TASK_TYPES:
            raise ValueError(f"unknown runtime task_type: {task_type}")
        unknown = [slm_id for slm_id in (route.get("selected_slms") or []) if slm_id not in slms]
        if unknown:
            raise ValueError(f"route references unknown slm: {unknown[0]}")
        if route.get("semantic_family") is None:
            default_routes.add(task_type)
    missing_defaults = [task_type for task_type in TASK_TYPES if task_type not in default_routes]
    if missing_defaults:
        raise ValueError(f"runtime bundle is missing default route: {missing_defaults[0]}")
