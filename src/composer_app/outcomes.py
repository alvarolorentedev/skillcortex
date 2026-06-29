from __future__ import annotations

from pathlib import Path
from typing import Any

from ..agent import run_agent
from ..catalog import MAX_REPO_FILES, MAX_TOTAL_BYTES
from ..runtime import SlmRuntime, serve_runtime


def capability_summary(diagnostics: dict[str, Any]) -> dict[str, Any]:
    backends = diagnostics.get("available_runtime_backends") or []
    runtime_ready = bool(backends)
    run_targets = ["compatibility_server", "agent_flow"] if runtime_ready else []
    return {
        "local_run": {
            "available": bool(run_targets),
            "status": "available" if runtime_ready else "dry-run-only",
            "supported_targets": run_targets,
        },
        "local_inference": {
            "available": runtime_ready,
            "status": "available" if runtime_ready else "dry-run-only",
            "supported_targets": ["inference"] if runtime_ready else [],
        },
        "dry_run_only": not runtime_ready,
        "available_runtime_backends": backends,
    }


def scan_warnings(scan_summary: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    if scan_summary.get("files_scanned", 0) >= MAX_REPO_FILES:
        warnings.append(
            "folder scan hit the file limit; narrow the folder or remove generated content for better routing"
        )
    if scan_summary.get("bytes_scanned", 0) >= MAX_TOTAL_BYTES:
        warnings.append(
            "folder scan hit the byte limit; routing used a bounded summary instead of the full repository"
        )
    if not scan_summary.get("language_signals"):
        warnings.append(
            "no supported language signals were detected; add source files or choose a more specific project folder"
        )
    return warnings


def resolve_outcome(
    *,
    compose_result: dict[str, Any],
    outcome: str,
    run_target: str,
    prompt: str | None,
    host: str,
    port: int,
    writes: str,
    test_command: str | None,
    trace_out: Path | None,
    dry_run: bool,
    dry_run_only: bool,
) -> dict[str, Any]:
    runtime_path = Path(compose_result["runtime"]["path"])
    if outcome == "export_bundle":
        return {
            "requested": outcome,
            "status": "written",
            "bundle_path": str(runtime_path.resolve()),
            "summary_path": str((runtime_path / "README.md").resolve()),
            "descriptor": compose_result.get("export_bundle"),
            "dry_run_only": dry_run_only,
        }
    if run_target == "agent_flow":
        resolved_trace_out = trace_out or (runtime_path.parent.parent / "logs" / f"agent-{runtime_path.name}.json")
        agent_result = run_agent(
            runtime_path=runtime_path,
            repo=Path(compose_result["folder"]),
            task=[compose_result["task"]],
            writes=writes,
            test_command=test_command,
            trace_out=resolved_trace_out,
            dry_run=True if dry_run_only else dry_run,
        )
        return _run_result(outcome, run_target, runtime_path, {"agent": agent_result}, agent_result["status"], dry_run_only)
    if run_target == "compatibility_server":
        server_result = serve_runtime(
            runtime_path=runtime_path,
            slms_dir=None,
            host=host,
            port=port,
            dry_run=True if dry_run_only else dry_run,
        )
        return _run_result(outcome, run_target, runtime_path, {"server": server_result}, server_result["status"], dry_run_only)
    inference_result = SlmRuntime.load(runtime_path).infer(
        prompt=prompt or compose_result["task"],
        dry_run=True if dry_run_only else dry_run,
    )
    return _run_result(outcome, run_target, runtime_path, {"inference": inference_result}, inference_result["status"], dry_run_only)


def translate_product_error(message: str, *, capabilities: dict[str, Any], outcome: str) -> dict[str, Any]:
    lowered = message.lower()
    if "slms directory not found" in lowered or "missing slm.yaml" in lowered:
        return {
            "code": "missing_package_metadata",
            "summary": "Composer could not find installable packages for this project.",
            "likely_cause": "The app workspace package catalog is empty or one of the packages is missing required metadata.",
            "recommended_next_action": "Install or copy validated slm packages into the app workspace packages folder, then retry the compose flow.",
        }
    if "no slm selected" in lowered or "incompatible with selected slm" in lowered:
        return {
            "code": "incompatible_selection",
            "summary": "Composer could not find a compatible slm selection for this folder.",
            "likely_cause": "The detected repository signals or package compatibility rules did not permit a safe slm selection.",
            "recommended_next_action": "Choose a more specific task prompt, install a package that matches this codebase, or allow a base fallback if that is acceptable.",
        }
    if "validation failed" in lowered:
        return {
            "code": "validation_failed",
            "summary": "The runtime bundle did not pass validation.",
            "likely_cause": "One of the selected packages or emitted runtime files is incomplete or inconsistent.",
            "recommended_next_action": "Rebuild the runtime, then validate the source packages before composing again.",
        }
    if "backend" in lowered and (
        "requires" in lowered or "does not support" in lowered or "must be one of" in lowered
    ):
        return {
            "code": "unsupported_backend_choice",
            "summary": "The selected runtime backend is not supported for this composition.",
            "likely_cause": message,
            "recommended_next_action": "Choose a compatible backend or runtime model, or switch to export mode until a supported local backend is available.",
        }
    if outcome == "local_run" and capabilities.get("dry_run_only"):
        return {
            "code": "backend_unavailable",
            "summary": "Local run is not available on this machine yet.",
            "likely_cause": "No supported runtime backend dependency was detected for the current platform.",
            "recommended_next_action": "Install an available runtime backend or use export mode until local inference support is installed.",
        }
    return {
        "code": "invalid_request",
        "summary": "Composer App could not complete this workflow.",
        "likely_cause": message,
        "recommended_next_action": "Review the exported support bundle and the compose log, then retry with a smaller folder or corrected package setup.",
    }


def _run_result(
    outcome: str,
    run_target: str,
    runtime_path: Path,
    payload: dict[str, Any],
    status: str,
    dry_run_only: bool,
) -> dict[str, Any]:
    return {
        "requested": outcome,
        "status": status,
        "run_target": run_target,
        "runtime_path": str(runtime_path.resolve()),
        **payload,
        "dry_run_only": dry_run_only,
    }
