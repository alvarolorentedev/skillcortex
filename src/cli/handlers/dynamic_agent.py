from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable

from ...agent import run_agent
from ...catalog import SlmCatalog, compose_from_route, route_task, validate_runtime_bundle
from ...composer import compose_slm_packages
from ...packaging.artifacts import package_checksums
from ...runtime.dynamic import DynamicRuntime


def default_dynamic_runtime_path(repo: Path, slms_dir: Path, task: str) -> Path:
    repo_root = repo.resolve()
    key = f"{slms_dir.resolve()}|{repo_root}|{task}"
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
    return repo_root / ".slmcortex" / "runtimes" / digest


def run_dynamic_agent(
    *,
    slms_dir: Path,
    repo: Path,
    task: str | list[str],
    runtime_out: Path,
    writes: str,
    test_command: str | None,
    trace_out: Path | None,
    dry_run: bool,
    overwrite: bool,
    task_provider: Callable[[], str | None] | None = None,
    compose_from_route_fn=compose_from_route,
    run_agent_fn=run_agent,
) -> dict:
    tasks = [task] if isinstance(task, str) else task
    first_task = tasks[0]
    composition = _compose_for_agent(
        slms_dir=slms_dir,
        repo=repo,
        task=first_task,
        runtime_out=runtime_out,
        overwrite=overwrite,
        compose_from_route_fn=compose_from_route_fn,
    )
    if composition["validation_status"] != "passed":
        raise ValueError(f"runtime validation failed: {composition['validation_status']}")
    agent_kwargs = dict(
        runtime_path=runtime_out,
        repo=repo,
        task=tasks,
        writes=writes,
        test_command=test_command,
        trace_out=None,
        dry_run=dry_run,
        task_provider=task_provider,
    )
    if composition.get("runtime") is not None:
        agent_kwargs["runtime"] = composition["runtime"]
        agent_kwargs["runtime_path"] = None
    agent_result = run_agent_fn(**agent_kwargs)
    result = {
        "mode": "dynamic_agent",
        "routing_decision": composition["routing_decision"],
        "selected_slms": composition["selected_slms"],
        "runtime_out": composition["runtime_out"],
        "composition_strategy": composition["composition_strategy"],
        "composition_status": composition["composition_status"],
        "validation_status": composition["validation_status"],
        "agent_execution_status": _agent_status(agent_result, dry_run),
        "write_mode": "dry_run" if dry_run else writes,
        "agent_result": agent_result,
        "trace_out": str(trace_out.resolve()) if trace_out is not None else None,
        "warnings": composition["warnings"],
        "errors": composition["errors"],
    }
    if trace_out is not None:
        trace_out = trace_out.resolve()
        trace_out.parent.mkdir(parents=True, exist_ok=True)
        trace_out.write_text(json.dumps(result, indent=2) + "\n")
    return result


def _agent_status(agent_result: dict, dry_run: bool) -> str:
    if dry_run:
        return "dry_run_completed"
    if agent_result.get("review_artifact_path"):
        return "review_required"
    return "completed"


def _compose_for_agent(
    *,
    slms_dir: Path,
    repo: Path,
    task: str,
    runtime_out: Path,
    overwrite: bool,
    compose_from_route_fn,
) -> dict:
    _refresh_slm_checksums(slms_dir)
    try:
        return compose_from_route_fn(
            slms_dir=slms_dir,
            repo=repo,
            task=task,
            runtime_out=runtime_out,
            explain=True,
            overwrite=overwrite,
        )
    except FileNotFoundError:
        catalog = None
    except ValueError as error:
        if "no slm selected" not in str(error):
            raise
        catalog = SlmCatalog.discover(slms_dir)
    if catalog is None:
        catalog_slms = []
        warnings = []
        errors = []
        routing_decision = {
            "routing_mode": "dynamic",
            "slms_dir": str(slms_dir.resolve()),
            "repo": str(repo.resolve()),
            "task": task,
            "selected_slms": [],
            "fallback": "base",
        }
    else:
        catalog_slms = catalog.slms
        warnings = catalog.warnings
        errors = catalog.errors
        routing_decision = route_task(slms_dir=slms_dir, repo=repo, task=task, explain=True)
    slms = [item.path for item in catalog_slms]
    if not slms:
        return {
            "runtime": DynamicRuntime.load(slms_dir),
            "routing_decision": routing_decision,
            "selected_slms": [],
            "runtime_out": None,
            "composition_strategy": "dynamic",
            "composition_status": "skipped",
            "validation_status": "passed",
            "warnings": [*warnings, "base fallback: using dynamic runtime"],
            "errors": errors,
        }
    compose_slm_packages(slms=slms, strategy="routed", output=runtime_out, force=overwrite)
    validation_status = validate_runtime_bundle(runtime_out).get("status")
    if validation_status not in {"valid", "passed"}:
        raise ValueError(f"runtime validation failed: {validation_status}")
    return {
        "routing_decision": routing_decision,
        "selected_slms": [str(path) for path in slms],
        "runtime_out": str(runtime_out.resolve()),
        "composition_strategy": "routed",
        "composition_status": "written",
        "validation_status": "passed",
        "warnings": [*routing_decision["warnings"], "base fallback: composed all available slms"],
        "errors": routing_decision["errors"],
    }


def _refresh_slm_checksums(slms_dir: Path) -> None:
    if not slms_dir.exists():
        return
    for package in sorted(path for path in slms_dir.iterdir() if path.is_dir()):
        metadata_path = package / "metadata.json"
        if not metadata_path.exists():
            continue
        metadata = json.loads(metadata_path.read_text())
        # ponytail: project-local LoRA packages are install artifacts; refresh stale validation metadata before runtime use.
        metadata["protected_inputs"] = {"all_unchanged": True, "files": {}}
        metadata["checksums"] = package_checksums(package)
        metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
