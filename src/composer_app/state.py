from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..shared.hashing import package_fingerprint, sha256
from ..shared.io import load_json_if_exists, read_json, read_yaml
from ..shared.product import APP_WORKSPACE_SCHEMA_VERSION
from .time import utc_now


def load_state(path: Path) -> dict[str, Any]:
    state = load_json_if_exists(path)
    if not state:
        return new_state()
    state = migrate_state(path, state)
    if not isinstance(state.get("projects"), dict):
        state["projects"] = {}
    return state


def write_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")


def new_state() -> dict[str, Any]:
    return {
        "schema_version": APP_WORKSPACE_SCHEMA_VERSION,
        "projects": {},
        "migrations": [],
    }


def migrate_state(path: Path, state: dict[str, Any]) -> dict[str, Any]:
    version = str(state.get("schema_version") or "1")
    if version == APP_WORKSPACE_SCHEMA_VERSION:
        state.setdefault("migrations", [])
        return state
    migrated = dict(state)
    migrations = list(migrated.get("migrations") or [])
    if version == "1":
        backup_path = backup_state_file(path, state, version)
        migrated["schema_version"] = APP_WORKSPACE_SCHEMA_VERSION
        migrated.setdefault("projects", {})
        for project in migrated["projects"].values():
            if isinstance(project, dict):
                project.setdefault("selected_packages", [])
                project.setdefault("runtime_bundle", {})
        migrations.append(
            {
                "from": "1",
                "to": APP_WORKSPACE_SCHEMA_VERSION,
                "applied_at": utc_now(),
                "status": "complete",
                "backup_path": backup_path,
            }
        )
        migrated["migrations"] = migrations
        return migrated
    raise ValueError(
        f"unsupported app workspace state schema_version: {version}; expected {APP_WORKSPACE_SCHEMA_VERSION}. Restore a known-good backup or use a clean workspace root."
    )


def backup_state_file(path: Path, state: dict[str, Any], version: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    target = path.with_name(f"{path.stem}.schema-v{version}.bak.json")
    if target.exists():
        stamp = utc_now().replace(":", "").replace("-", "")
        target = path.with_name(f"{path.stem}.schema-v{version}-{stamp}.bak.json")
    target.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    return str(target.resolve())


def record_project_state(
    state: dict[str, Any],
    *,
    repo: Path,
    runtime_name: str,
    task: str,
    scan_summary: dict[str, Any],
    compose_result: dict[str, Any],
    first_run: bool,
) -> dict[str, Any]:
    projects = state.setdefault("projects", {})
    projects[str(repo)] = {
        "runtime_name": runtime_name,
        "task": task,
        "scan_summary": scan_summary,
        "last_status": compose_result.get("status"),
        "runtime_path": (compose_result.get("runtime") or {}).get("path"),
        "selected_packages": selected_package_records(compose_result),
        "runtime_bundle": runtime_bundle_record(compose_result),
        "updated_at": utc_now(),
    }
    state["schema_version"] = APP_WORKSPACE_SCHEMA_VERSION
    state["onboarding_completed"] = True
    state["last_opened_project"] = str(repo)
    state["updated_at"] = utc_now()
    if first_run:
        state["onboarding_completed_at"] = utc_now()
    return state


def selected_package_records(compose_result: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for item in compose_result.get("selected_slms") or []:
        path = Path(item)
        try:
            slm_manifest = read_yaml(path / "slm.yaml")
            metadata = read_json(path / "metadata.json")
            records.append(
                {
                    "path": str(path.resolve()),
                    "slm_id": slm_manifest.get("slm_id"),
                    "version": slm_manifest.get("version"),
                    "fingerprint": package_fingerprint(slm_manifest, metadata),
                }
            )
        except (FileNotFoundError, ValueError, KeyError):
            records.append({"path": str(path.resolve())})
    return records


def runtime_bundle_record(compose_result: dict[str, Any]) -> dict[str, Any]:
    runtime = compose_result.get("runtime") or {}
    runtime_path_value = runtime.get("path")
    if not runtime_path_value:
        return {}
    runtime_path = Path(runtime_path_value)
    record = {
        "path": str(runtime_path.resolve()),
        "validation_status": runtime.get("validation_status"),
        "composition_status": runtime.get("composition_status"),
    }
    checksums_path = runtime_path / "checksums.json"
    if checksums_path.exists():
        record["checksums_sha256"] = sha256(checksums_path)
    return record
