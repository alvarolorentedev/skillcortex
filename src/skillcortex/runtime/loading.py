from __future__ import annotations

from pathlib import Path
from typing import Any

from ..contracts import TASK_TYPES
from ..packaging import validate_skill_package
from ..shared.hashing import package_fingerprint, sha256
from ..shared.io import read_json, read_yaml
from .models import REQUIRED_RUNTIME_FILES, RuntimeBundle, RuntimeSkill


def load_runtime_bundle(runtime_path: Path) -> RuntimeBundle:
    root = runtime_path.resolve()
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"runtime bundle not found: {root}")
    missing = [name for name in REQUIRED_RUNTIME_FILES if not (root / name).exists()]
    if missing:
        raise ValueError(f"runtime bundle is missing required files: {missing[0]}")

    composition = read_yaml(root / "composition.yaml")
    router_config = read_json(root / "router_config.json")
    active_skills = read_json(root / "active_skills.json")
    compatibility_report = read_json(root / "compatibility_report.json")
    budget_report = read_json(root / "budget_report.json")
    checksums = read_json(root / "checksums.json")

    validate_bundle_metadata(composition, router_config, active_skills, compatibility_report, checksums)
    validate_bundle_checksums(root, checksums)
    skills = load_runtime_skills(composition, checksums)
    validate_router_routes(router_config.get("routes") or [], skills)
    validate_active_skills(active_skills, skills, router_config.get("routes") or [])

    runtime = composition.get("runtime") or {}
    return RuntimeBundle(
        path=root,
        name=root.name,
        runtime_model=runtime["runtime_model"],
        source_model=runtime["source_model"],
        quantization=runtime["quantization"],
        strategy=composition["strategy"],
        routes=list(router_config["routes"]),
        skills=skills,
        compatibility_report=compatibility_report,
        budget_report=budget_report,
        checksums=checksums,
    )


def load_runtime_skills(composition: dict[str, Any], checksums: dict[str, Any]) -> dict[str, RuntimeSkill]:
    skill_entries = composition.get("skills") or []
    fingerprints_by_skill = {
        item["skill_id"]: item["fingerprint"] for item in checksums.get("source_packages") or []
    }
    skills: dict[str, RuntimeSkill] = {}
    for item in skill_entries:
        package_path = Path(item["package_path"]).resolve()
        validate_skill_package(package_path)
        metadata = read_json(package_path / "metadata.json")
        manifest = read_yaml(package_path / "skill.yaml")
        fingerprint = package_fingerprint(manifest, metadata)
        if fingerprint != item.get("fingerprint"):
            raise ValueError(f"runtime package fingerprint mismatch for {item['skill_id']}")
        if fingerprints_by_skill.get(item["skill_id"]) != fingerprint:
            raise ValueError(f"runtime checksum fingerprint mismatch for {item['skill_id']}")
        adapter_files = (metadata.get("adapter") or {}).get("files") or {}
        adapter_relative = adapter_files.get("weights") or "adapter/adapters.safetensors"
        skills[item["skill_id"]] = RuntimeSkill(
            skill_id=item["skill_id"],
            name=item["name"],
            version=item["version"],
            package_path=package_path,
            adapter_path=(package_path / adapter_relative).parent,
            fingerprint=fingerprint,
            allowed_task_types=list(item["composition"]["capabilities"]["allowed_task_types"]),
            activation=dict(item["composition"]["activation"]),
            trainable_parameters=int(item["adapter"]["trainable_parameters"]),
        )
    return skills


def validate_active_skills(
    active_skills: dict[str, Any],
    skills: dict[str, RuntimeSkill],
    routes: list[dict[str, Any]],
) -> None:
    route_ids = {route["route_id"] for route in routes}
    active_ids = {item["skill_id"] for item in active_skills.get("skills") or []}
    if active_ids != set(skills):
        raise ValueError("active_skills.json must match composition skills")
    for item in active_skills.get("skills") or []:
        membership = set(item.get("route_membership") or [])
        if not membership.issubset(route_ids):
            raise ValueError(f"unknown route membership for {item['skill_id']}")


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
    active_skills: dict[str, Any],
    compatibility_report: dict[str, Any],
    checksums: dict[str, Any],
) -> None:
    if composition.get("schema_version") != "1":
        raise ValueError("composition.yaml schema_version must be '1'")
    if composition.get("composition_type") != "skill_runtime_bundle":
        raise ValueError("composition.yaml composition_type must be 'skill_runtime_bundle'")
    if composition.get("status") != "complete":
        raise ValueError("composition.yaml status must be 'complete'")
    if router_config.get("schema_version") != "1":
        raise ValueError("router_config.json schema_version must be '1'")
    if active_skills.get("schema_version") != "1":
        raise ValueError("active_skills.json schema_version must be '1'")
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


def validate_router_routes(routes: list[dict[str, Any]], skills: dict[str, RuntimeSkill]) -> None:
    if not routes:
        raise ValueError("runtime bundle routes are required")
    default_routes = set()
    for route in routes:
        task_type = route.get("task_type")
        if task_type not in TASK_TYPES:
            raise ValueError(f"unknown runtime task_type: {task_type}")
        unknown = [skill_id for skill_id in (route.get("selected_skills") or []) if skill_id not in skills]
        if unknown:
            raise ValueError(f"route references unknown skill: {unknown[0]}")
        if route.get("semantic_family") is None:
            default_routes.add(task_type)
    missing_defaults = [task_type for task_type in TASK_TYPES if task_type not in default_routes]
    if missing_defaults:
        raise ValueError(f"runtime bundle is missing default route: {missing_defaults[0]}")
