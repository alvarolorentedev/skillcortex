from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..composer import compose_slm_packages
from ..shared.product import (
    PRODUCT_MODES,
    ensure_app_workspace,
    environment_diagnostics,
    runtime_name_for_folder,
)
from .discovery import SlmCatalog
from .scanning import infer_task_hints, scan_repo_context


def compose_from_route(
    *,
    slms_dir: Path,
    repo: Path,
    task: str,
    runtime_out: Path,
    explain: bool = False,
    allow_base: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    from . import route_task

    routing_decision = route_task(slms_dir=slms_dir, repo=repo, task=task, explain=explain)
    catalog = SlmCatalog.discover(slms_dir)
    by_slm_id = {slm.slm_id: slm for slm in catalog.slms}
    selected = list(routing_decision["selected_slms"])
    warnings = list(routing_decision["warnings"])
    errors = list(routing_decision["errors"])
    if not selected:
        if not allow_base:
            raise ValueError("no slm selected; pass --allow-base or improve routing metadata")
        warnings.append("base fallback allowed; no runtime was composed")
        return {
            "routing_decision": routing_decision,
            "selected_slms": [],
            "runtime_out": None,
            "composition_strategy": "routed",
            "composition_status": "skipped",
            "validation_status": "not_run",
            "warnings": warnings,
            "errors": errors,
        }

    selected_paths = []
    for item in selected:
        slm = by_slm_id.get(item["slm_id"])
        if slm is None:
            raise ValueError(f"selected slm not found in catalog: {item['slm_id']}")
        selected_paths.append(slm.path)

    try:
        compose_slm_packages(slms=selected_paths, strategy="routed", output=runtime_out, force=overwrite)
    except FileExistsError:
        raise
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        context = ", ".join(
            f"{item['slm_id']}={path}" for item, path in zip(selected, selected_paths, strict=True)
        )
        raise ValueError(f"selected slm package is not composable ({context}): {error}") from error

    from . import validate_runtime_bundle

    validation_status = validate_runtime_bundle(runtime_out).get("status")
    if validation_status not in {"valid", "passed"}:
        raise ValueError(f"runtime validation failed: {validation_status}")
    return {
        "routing_decision": routing_decision,
        "selected_slms": [str(path) for path in selected_paths],
        "runtime_out": str(runtime_out.resolve()),
        "composition_strategy": "routed",
        "composition_status": "written",
        "validation_status": "passed",
        "warnings": warnings,
        "errors": errors,
    }


def compose_from_folder(
    *,
    folder: Path,
    task: str,
    workspace_root: Path | None = None,
    slms_dir: Path | None = None,
    runtime_name: str | None = None,
    export_descriptor: Path | None = None,
    allow_base: bool = False,
    overwrite: bool = False,
    product_mode: str = "composer",
) -> dict[str, Any]:
    if product_mode not in PRODUCT_MODES:
        raise ValueError(f"unknown product mode: {product_mode}")
    workspace = ensure_app_workspace(workspace_root)
    repo = folder.resolve()
    packages_root = (slms_dir or workspace.packages_dir).resolve()
    runtime_slug = runtime_name_for_folder(repo, runtime_name)
    runtime_out = workspace.runtimes_dir / runtime_slug
    diagnostics = environment_diagnostics(workspace_root=workspace.root, product_mode=product_mode)
    task_hints: list[dict[str, str]] = []
    routing_decision: dict[str, Any] | None = None
    try:
        repo_context = scan_repo_context(repo)
        task_hints = infer_task_hints(repo_context)
        composition = compose_from_route(
            slms_dir=packages_root,
            repo=repo,
            task=task,
            runtime_out=runtime_out,
            explain=True,
            allow_base=allow_base,
            overwrite=overwrite,
        )
        routing_decision = composition["routing_decision"]
        result = _compose_success(
            repo=repo,
            task=task,
            workspace=workspace,
            packages_root=packages_root,
            runtime_slug=runtime_slug,
            task_hints=task_hints,
            diagnostics=diagnostics,
            routing_decision=routing_decision,
            composition=composition,
            export_descriptor=export_descriptor,
            product_mode=product_mode,
        )
    except (FileNotFoundError, FileExistsError, ValueError, RuntimeError) as error:
        result = _compose_failure(
            repo=repo,
            task=task,
            workspace=workspace,
            packages_root=packages_root,
            runtime_slug=runtime_slug,
            runtime_out=runtime_out,
            task_hints=task_hints,
            diagnostics=diagnostics,
            routing_decision=routing_decision,
            error=error,
            product_mode=product_mode,
        )
    _write_compose_log(workspace.logs_dir, runtime_slug, result)
    return result


def _compose_success(**kwargs) -> dict[str, Any]:
    composition = kwargs["composition"]
    workspace = kwargs["workspace"]
    routing_decision = kwargs["routing_decision"]
    return {
        "schema_version": "1",
        "status": "complete",
        "exit_code": 0,
        "operation": "compose_from_folder",
        "product_mode": kwargs["product_mode"],
        "folder": str(kwargs["repo"]),
        "task": kwargs["task"],
        "workspace": workspace.as_dict(),
        "slms_dir": str(kwargs["packages_root"]),
        "runtime_name": kwargs["runtime_slug"],
        "task_hints": kwargs["task_hints"],
        "diagnostics": kwargs["diagnostics"],
        "routing_decision": routing_decision,
        "selected_slms": composition["selected_slms"],
        "runtime": {
            "path": composition["runtime_out"],
            "composition_strategy": composition["composition_strategy"],
            "composition_status": composition["composition_status"],
            "validation_status": composition["validation_status"],
        },
        "export_bundle": _write_export_descriptor(
            export_descriptor=kwargs["export_descriptor"],
            workspace_root=workspace.exports_dir,
            runtime_name=kwargs["runtime_slug"],
            task=kwargs["task"],
            runtime_out=composition["runtime_out"],
            selected_slms=composition["selected_slms"],
            routing_decision=routing_decision,
        ),
        "warnings": composition["warnings"],
        "errors": composition["errors"],
    }


def _compose_failure(**kwargs) -> dict[str, Any]:
    workspace = kwargs["workspace"]
    return {
        "schema_version": "1",
        "status": "failed",
        "exit_code": 2,
        "operation": "compose_from_folder",
        "product_mode": kwargs["product_mode"],
        "folder": str(kwargs["repo"]),
        "task": kwargs["task"],
        "workspace": workspace.as_dict(),
        "slms_dir": str(kwargs["packages_root"]),
        "runtime_name": kwargs["runtime_slug"],
        "task_hints": kwargs["task_hints"],
        "diagnostics": kwargs["diagnostics"],
        "routing_decision": kwargs["routing_decision"],
        "runtime": {
            "path": str(kwargs["runtime_out"].resolve()),
            "composition_strategy": "routed",
            "composition_status": "failed",
            "validation_status": "not_run",
        },
        "export_bundle": None,
        "warnings": kwargs["diagnostics"]["warnings"],
        "errors": [str(kwargs["error"])],
        "error_code": _error_code(kwargs["error"]),
    }


def _write_export_descriptor(
    *,
    export_descriptor: Path | None,
    workspace_root: Path,
    runtime_name: str,
    task: str,
    runtime_out: str | None,
    selected_slms: list[str],
    routing_decision: dict[str, Any],
) -> dict[str, Any] | None:
    if export_descriptor is None and runtime_out is None:
        return None
    target = export_descriptor.resolve() if export_descriptor else (workspace_root / f"{runtime_name}.json").resolve()
    descriptor = {
        "schema_version": "1",
        "runtime_name": runtime_name,
        "task": task,
        "runtime_out": runtime_out,
        "selected_slms": selected_slms,
        "routing_mode": routing_decision.get("routing_mode"),
        "fallback": routing_decision.get("fallback"),
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(descriptor, indent=2) + "\n")
    return {"status": "written", "descriptor_path": str(target)}


def _write_compose_log(logs_dir: Path, runtime_name: str, result: dict[str, Any]) -> None:
    logs_dir.mkdir(parents=True, exist_ok=True)
    (logs_dir / f"compose-{runtime_name}.json").write_text(json.dumps(result, indent=2) + "\n")


def _error_code(error: Exception) -> str:
    if isinstance(error, FileNotFoundError):
        return "not_found"
    if isinstance(error, FileExistsError):
        return "already_exists"
    if isinstance(error, RuntimeError):
        return "runtime_error"
    return "invalid_request"
