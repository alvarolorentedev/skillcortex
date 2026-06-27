from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from .actions import extract_actions


def normalize_task_inputs(task: str | list[str] | None, *, allow_empty: bool = False) -> list[str]:
    if task is None:
        tasks = []
    elif isinstance(task, str):
        tasks = [task]
    else:
        tasks = task
    normalized = [item.strip() for item in tasks if isinstance(item, str) and item.strip()]
    if not normalized and not allow_empty:
        raise ValueError("at least one task is required")
    return normalized


def task_sequence(
    task: str | list[str] | None,
    *,
    task_provider: Callable[[], str | None] | None = None,
):
    queued = normalize_task_inputs(task, allow_empty=True)
    yielded = False
    for item in queued:
        yielded = True
        yield item
    if task_provider is not None:
        while True:
            next_task = task_provider()
            if next_task is None:
                break
            normalized = next_task.strip()
            if not normalized:
                continue
            yielded = True
            yield normalized
    if not yielded:
        raise ValueError("at least one task is required")


def inference_step(
    step_type: str,
    result: dict[str, Any],
    *,
    files_read: list[str],
    files_changed: list[str] | None = None,
    tool_name: str | None = None,
    tool_result_summary: str | None = None,
    proposed_diff: str | None = None,
    write_status: str | None = None,
    review_artifact_path: str | None = None,
    mode_label: str | None = None,
) -> dict[str, Any]:
    return {
        "step_type": step_type,
        "selected_skills": result.get("selected_skills") or [],
        "route_type": result.get("route_type"),
        "route_reason": result.get("reason"),
        "tool_name": tool_name,
        "files_read": files_read,
        "files_changed": files_changed or [],
        "status": result.get("status") or "complete",
        "generation": result.get("generation"),
        "result_summary": tool_result_summary or result.get("generation") or result.get("reason"),
        "proposed_diff": proposed_diff,
        "write_status": write_status,
        "review_artifact_path": review_artifact_path,
        "mode_label": mode_label,
    }


def materialize_inference_action(
    sandbox,
    result: dict[str, Any],
    default_path: str,
    dry_run: bool,
    *,
    review_path: Path | None = None,
) -> dict[str, Any]:
    if dry_run:
        return {
            "kind": "dry-run",
            "write_status": "dry-run",
            "files_changed": [],
            "diff": "",
            "review_artifact_path": None,
            "summary": "Dry-run route/plan only: skipped artifact materialization.",
            "actions": [],
        }
    actions = extract_actions(result.get("generation") or "", default_path)
    return sandbox.materialize_actions(actions, review_path=review_path)


def append_step(trace: dict[str, Any], step: dict[str, Any]) -> None:
    step["step_index"] = len(trace["steps"]) + 1
    trace["steps"].append(step)


def default_review_path(repo: Path, run_id: str) -> Path:
    return repo / ".skillcortex" / "reviews" / f"{run_id}.patch"


def step_messages(
    task: str,
    instruction: str,
    *,
    repo_files: list[str],
    repo_context: dict[str, str],
    execution_history: str | None = None,
    validation_output: str | None = None,
) -> list[dict[str, str]]:
    content = [
        f"Task: {task}",
        instruction,
        "Repository files:",
        "\n".join(repo_files[:20]),
        "Repository excerpts:",
        "\n\n".join(f"[{path}]\n{text}" for path, text in repo_context.items()),
    ]
    if execution_history:
        content.extend(["Previous execution context:", execution_history])
    if validation_output:
        content.extend(["Validation output:", validation_output])
    return [{"role": "user", "content": "\n\n".join(content)}]
