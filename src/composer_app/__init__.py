from __future__ import annotations

from pathlib import Path
from typing import Any

from ..catalog import compose_from_folder, infer_task_hints, scan_repo_context
from ..shared.product import APP_STATE_FILE, ensure_app_workspace, environment_diagnostics, runtime_name_for_folder
from .outcomes import capability_summary, resolve_outcome, scan_warnings, translate_product_error
from .state import load_state, record_project_state, write_state
from .support import write_support_bundle, write_workspace_support_bundle


ONBOARDING_MESSAGE = (
    "Start by selecting a local folder. Slm Cortex scans the codebase, recommends the right "
    "packages, composes a runtime, and then lets you run locally or export the bundle. "
    "Training, packaging, and registry workflows remain available as advanced capabilities, "
    "but they are not required for the default path."
)


def run_composer_app(
    *,
    folder: Path,
    workspace_root: Path | None = None,
    slms_dir: Path | None = None,
    task: str | None = None,
    runtime_name: str | None = None,
    outcome: str = "local_run",
    run_target: str = "compatibility_server",
    prompt: str | None = None,
    export_descriptor: Path | None = None,
    export_logs: bool = False,
    allow_base: bool = False,
    overwrite: bool = False,
    host: str = "127.0.0.1",
    port: int = 8000,
    writes: str = "confirm",
    test_command: str | None = None,
    trace_out: Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    workspace = ensure_app_workspace(workspace_root)
    repo = folder.resolve()
    diagnostics = environment_diagnostics(workspace_root=workspace.root, product_mode="composer")
    capabilities = capability_summary(diagnostics)
    state_path = workspace.state_dir / APP_STATE_FILE
    try:
        state = load_state(state_path)
    except ValueError as error:
        return _state_error(
            workspace=workspace,
            repo=repo,
            runtime_name=runtime_name,
            task=task,
            outcome=outcome,
            diagnostics=diagnostics,
            capabilities=capabilities,
            state_path=state_path,
            error=error,
        )

    first_run = not bool(state.get("onboarding_completed"))
    runtime_slug = runtime_name_for_folder(repo, runtime_name)
    scan_summary = scan_repo_context(repo)
    warnings_from_scan = scan_warnings(scan_summary)
    task_hints = infer_task_hints(scan_summary)
    resolved_task = (task or task_hints[0]["suggested_task"]).strip()
    reopened = str(repo) in (state.get("projects") or {})
    descriptor_target = export_descriptor
    if descriptor_target is None and outcome == "export_bundle":
        descriptor_target = workspace.exports_dir / f"{runtime_slug}.json"

    compose_result = compose_from_folder(
        folder=repo,
        task=resolved_task,
        workspace_root=workspace.root,
        slms_dir=slms_dir,
        runtime_name=runtime_slug,
        export_descriptor=descriptor_target,
        allow_base=allow_base,
        overwrite=overwrite or reopened,
        product_mode="composer",
    )
    state = record_project_state(
        state,
        repo=repo,
        runtime_name=runtime_slug,
        task=resolved_task,
        scan_summary=scan_summary,
        compose_result=compose_result,
        first_run=first_run,
    )
    write_state(state_path, state)

    product_error = None
    support_bundle = None
    compose_log_path = workspace.logs_dir / f"compose-{runtime_slug}.json"
    if compose_result["status"] == "complete":
        outcome_result = resolve_outcome(
            compose_result=compose_result,
            outcome=outcome,
            run_target=run_target,
            prompt=prompt,
            host=host,
            port=port,
            writes=writes,
            test_command=test_command,
            trace_out=trace_out,
            dry_run=dry_run,
            dry_run_only=capabilities["dry_run_only"],
        )
        warnings = [*warnings_from_scan, *compose_result.get("warnings", [])]
        if outcome == "local_run" and capabilities["dry_run_only"]:
            warnings.append(
                "local run is currently limited to dry-run only because no supported runtime backend was detected"
            )
        errors = list(compose_result.get("errors", []))
        status = "complete"
        exit_code = 0
    else:
        outcome_result = {"requested": outcome, "status": "blocked", "run_target": run_target if outcome == "local_run" else None}
        warnings = [*warnings_from_scan, *compose_result.get("warnings", [])]
        errors = list(compose_result.get("errors", []))
        product_error = translate_product_error(
            errors[0] if errors else "composer app workflow failed",
            capabilities=capabilities,
            outcome=outcome,
        )
        status = "failed"
        exit_code = 2
        if export_logs or status == "failed":
            support_bundle = write_workspace_support_bundle(
                workspace=workspace,
                runtime_name=runtime_slug,
                compose_result=compose_result,
                state=state,
                diagnostics=diagnostics,
                scan_summary=scan_summary,
                product_error=product_error,
            )

    if export_logs and support_bundle is None:
        support_bundle = write_workspace_support_bundle(
            workspace=workspace,
            runtime_name=runtime_slug,
            compose_result=compose_result,
            state=state,
            diagnostics=diagnostics,
            scan_summary=scan_summary,
            product_error=product_error,
        )

    return _app_result(
        status=status,
        exit_code=exit_code,
        first_run=first_run,
        state=state,
        state_path=state_path,
        repo=repo,
        reopened=reopened,
        resolved_task=resolved_task,
        task_hints=task_hints,
        scan_summary=scan_summary,
        scan_warnings=warnings_from_scan,
        workspace=workspace,
        runtime_slug=runtime_slug,
        compose_result=compose_result,
        outcome_result=outcome_result,
        product_error=product_error,
        compose_log_path=compose_log_path,
        support_bundle=support_bundle,
        diagnostics=diagnostics,
        warnings=warnings,
        errors=errors,
        capabilities=capabilities,
    )


def _state_error(**kwargs) -> dict[str, Any]:
    runtime_slug = runtime_name_for_folder(kwargs["repo"], kwargs["runtime_name"])
    product_error = {
        "code": "state_schema_incompatible",
        "summary": "Composer App could not load the existing workspace state.",
        "likely_cause": str(kwargs["error"]),
        "recommended_next_action": "Restore the latest state backup from the workspace state directory or retry with a clean workspace root after exporting a doctor support bundle.",
    }
    support_bundle = write_support_bundle(
        workspace_root=kwargs["workspace"].root,
        runtime_name="state-schema-error",
        compose_result=None,
        state=None,
        diagnostics=kwargs["diagnostics"],
        scan_summary=None,
        product_error=product_error,
    )
    return _app_result(
        status="failed",
        exit_code=2,
        first_run=False,
        state={"onboarding_completed": False},
        state_path=kwargs["state_path"],
        repo=kwargs["repo"],
        reopened=False,
        resolved_task=kwargs["task"],
        task_hints=[],
        scan_summary={},
        scan_warnings=[],
        workspace=kwargs["workspace"],
        runtime_slug=runtime_slug,
        compose_result={"runtime": None, "routing_decision": None, "selected_slms": [], "export_bundle": None},
        outcome_result={"requested": kwargs["outcome"], "status": "blocked", "run_target": None},
        product_error=product_error,
        compose_log_path=kwargs["workspace"].logs_dir / f"compose-{runtime_slug}.json",
        support_bundle=support_bundle,
        diagnostics=kwargs["diagnostics"],
        warnings=kwargs["diagnostics"].get("warnings", []),
        errors=[str(kwargs["error"])],
        capabilities=kwargs["capabilities"],
    )


def _app_result(**kwargs) -> dict[str, Any]:
    compose_result = kwargs["compose_result"]
    return {
        "schema_version": "1",
        "status": kwargs["status"],
        "exit_code": kwargs["exit_code"],
        "operation": "composer_app",
        "product_mode": "composer",
        "onboarding": {
            "first_run": kwargs["first_run"],
            "completed": bool(kwargs["state"].get("onboarding_completed")),
            "message": ONBOARDING_MESSAGE,
            "capabilities": kwargs["capabilities"],
            "state_path": str(kwargs["state_path"].resolve()),
        },
        "project": {
            "folder": str(kwargs["repo"]),
            "reopened": kwargs["reopened"],
            "task": kwargs["resolved_task"],
            "task_hints": kwargs["task_hints"],
            "scan_summary": kwargs["scan_summary"],
            "scan_warnings": kwargs["scan_warnings"],
        },
        "workspace": kwargs["workspace"].as_dict(),
        "composition": {
            "runtime_name": kwargs["runtime_slug"],
            "runtime": compose_result.get("runtime"),
            "routing_decision": compose_result.get("routing_decision"),
            "selected_slms": compose_result.get("selected_slms", []),
            "export_bundle": compose_result.get("export_bundle"),
        },
        "outcome": kwargs["outcome_result"],
        "product_error": kwargs["product_error"],
        "support": {
            "logs_dir": str(kwargs["workspace"].logs_dir.resolve()),
            "compose_log_path": str(kwargs["compose_log_path"].resolve()),
            "support_bundle": kwargs["support_bundle"],
        },
        "diagnostics": kwargs["diagnostics"],
        "warnings": kwargs["warnings"],
        "errors": kwargs["errors"],
    }


__all__ = ["run_composer_app", "write_support_bundle"]
