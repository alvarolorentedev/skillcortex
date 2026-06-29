from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..shared.product import ensure_app_workspace
from .time import utc_now


def write_support_bundle(
    *,
    workspace_root: Path | None,
    runtime_name: str,
    compose_result: dict[str, Any] | None,
    state: dict[str, Any] | None,
    diagnostics: dict[str, Any],
    scan_summary: dict[str, Any] | None,
    product_error: dict[str, Any] | None,
    bundle_path: Path | None = None,
) -> str:
    workspace = ensure_app_workspace(workspace_root)
    return write_workspace_support_bundle(
        workspace=workspace,
        runtime_name=runtime_name,
        compose_result=compose_result,
        state=state,
        diagnostics=diagnostics,
        scan_summary=scan_summary,
        product_error=product_error,
        bundle_path=bundle_path,
    )


def write_workspace_support_bundle(
    *,
    workspace,
    runtime_name: str,
    compose_result: dict[str, Any] | None,
    state: dict[str, Any] | None,
    diagnostics: dict[str, Any],
    scan_summary: dict[str, Any] | None,
    product_error: dict[str, Any] | None,
    bundle_path: Path | None = None,
) -> str:
    support_dir = bundle_path.parent if bundle_path else workspace.diagnostics_dir / "support"
    support_dir.mkdir(parents=True, exist_ok=True)
    target = bundle_path or support_dir / f"{runtime_name}-support.json"
    payload = {
        "schema_version": "1",
        "generated_at": utc_now(),
        "tool": diagnostics.get("tool"),
        "workspace_schema_version": diagnostics.get("workspace_schema_version"),
        "product_error": product_error,
        "compose_result": compose_result,
        "diagnostics": diagnostics,
        "scan_summary": scan_summary,
        "state": state,
        "recent_errors": (compose_result or {}).get("errors") or diagnostics.get("warnings") or [],
        "redactions": {
            "env_vars": "excluded",
            "file_contents": "excluded",
        },
    }
    target.write_text(json.dumps(payload, indent=2) + "\n")
    return str(target.resolve())
