from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ...agent import run_agent
from ...catalog import SlmCatalog, compose_from_route, route_task, validate_runtime_bundle
from ...composer import compose_slm_packages


def default_dynamic_runtime_path(repo: Path, slms_dir: Path, task: str) -> Path:
    repo_root = repo.resolve()
    key = f"{slms_dir.resolve()}|{repo_root}|{task}"
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
    return repo_root / ".slmcortex" / "runtimes" / digest


def run_dynamic_agent(
    *,
    slms_dir: Path,
    repo: Path,
    task: str,
    runtime_out: Path,
    writes: str,
    test_command: str | None,
    trace_out: Path | None,
    dry_run: bool,
    overwrite: bool,
    compose_from_route_fn=compose_from_route,
    run_agent_fn=run_agent,
) -> dict:
    composition = _compose_for_agent(
        slms_dir=slms_dir,
        repo=repo,
        task=task,
        runtime_out=runtime_out,
        overwrite=overwrite,
        compose_from_route_fn=compose_from_route_fn,
    )
    if composition["validation_status"] != "passed":
        raise ValueError(f"runtime validation failed: {composition['validation_status']}")
    agent_result = run_agent_fn(
        runtime_path=runtime_out,
        repo=repo,
        task=[task],
        writes=writes,
        test_command=test_command,
        trace_out=None,
        dry_run=dry_run,
    )
    result = {
        "mode": "dynamic_agent",
        "routing_decision": composition["routing_decision"],
        "selected_slms": composition["selected_slms"],
        "runtime_out": composition["runtime_out"],
        "composition_strategy": composition["composition_strategy"],
        "composition_status": composition["composition_status"],
        "validation_status": composition["validation_status"],
        "agent_execution_status": _agent_status(agent_result, dry_run),
        "write_mode": "dry_run" if dry_run else "confirm",
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
    try:
        return compose_from_route_fn(
            slms_dir=slms_dir,
            repo=repo,
            task=task,
            runtime_out=runtime_out,
            explain=True,
            overwrite=overwrite,
        )
    except ValueError as error:
        if "no slm selected" not in str(error):
            raise
    routing_decision = route_task(slms_dir=slms_dir, repo=repo, task=task, explain=True)
    catalog = SlmCatalog.discover(slms_dir)
    slms = [item.path for item in catalog.slms]
    if not slms:
        raise ValueError("no slm packages available for base fallback")
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
