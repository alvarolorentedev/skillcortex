from __future__ import annotations

from pathlib import Path
from typing import Any

from .composition import compose_from_folder, compose_from_route
from .discovery import SlmCatalog
from .scanning import MAX_REPO_FILES, MAX_TOTAL_BYTES, infer_task_hints, scan_repo_context
from .scoring import ROUTE_THRESHOLD, candidate
from .types import CatalogResult, RoutingCard, SlmRecord


def route_task(
    *,
    slms_dir: Path,
    repo: Path,
    task: str,
    explain: bool = False,
    current_base_model: str | None = None,
) -> dict[str, Any]:
    catalog = SlmCatalog.discover(slms_dir)
    repo_root = repo.resolve()
    if not repo_root.exists() or not repo_root.is_dir():
        raise FileNotFoundError(f"repo not found: {repo_root}")
    repo_context = scan_repo_context(repo_root)
    candidates = [
        candidate(slm, task, repo_context, current_base_model=current_base_model)
        for slm in catalog.slms
    ]
    candidates.sort(key=lambda item: (-item["score"], item["slm_id"]))
    selected = []
    for item in candidates:
        if item["compatible"] and item["score"] >= ROUTE_THRESHOLD:
            item["selected"] = True
            selected.append(
                {
                    "slm_id": item["slm_id"],
                    "score": item["score"],
                    "reason": "Strongest capability match.",
                }
            )
            break
    if not explain:
        for item in candidates:
            item["score_breakdown"] = {}
    return {
        "routing_mode": "capability",
        "slms_dir": str(slms_dir.resolve()),
        "repo": str(repo_root),
        "task": task,
        "repo_context": repo_context,
        "selected_slms": selected,
        "candidates": candidates,
        "fallback": "base",
        "errors": catalog.errors,
        "warnings": catalog.warnings,
    }


__all__ = [
    "CatalogResult",
    "MAX_REPO_FILES",
    "MAX_TOTAL_BYTES",
    "ROUTE_THRESHOLD",
    "RoutingCard",
    "SlmCatalog",
    "SlmRecord",
    "compose_from_folder",
    "compose_from_route",
    "infer_task_hints",
    "route_task",
    "scan_repo_context",
]
