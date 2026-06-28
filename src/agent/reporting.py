from __future__ import annotations

from typing import Any


def merge_task_statuses(task_results: list[dict[str, Any]], *, dry_run: bool) -> str:
    if dry_run:
        return "dry-run"
    statuses = [final_status(result["steps"], result["validation"], dry_run) for result in task_results]
    if "validation_failed" in statuses:
        return "validation_failed"
    if "review_required" in statuses:
        return "review_required"
    if "applied" in statuses:
        return "applied"
    return statuses[-1]


def multi_task_summary(task_results: list[dict[str, Any]], *, writes: str, dry_run: bool) -> str:
    if len(task_results) == 1:
        return final_summary(task_results[0]["steps"], task_results[0]["validation"], writes, dry_run=dry_run)
    selected: list[tuple[Any, ...]] = []
    for result in task_results:
        for step in result["steps"]:
            slms = tuple(step.get("selected_slms") or [])
            if slms not in selected:
                selected.append(slms)
    if dry_run:
        return (
            f"Executed {len(task_results)} tasks and {sum(len(result['steps']) for result in task_results)} steps in route/plan only dry-run mode. "
            f"Observed {len(selected)} distinct slm selections. "
            "Generation, writes, and validation were skipped."
        )
    validations = [result["validation"]["status"] for result in task_results]
    return (
        f"Executed {len(task_results)} tasks and {sum(len(result['steps']) for result in task_results)} steps with writes mode '{writes}'. "
        f"Observed {len(selected)} distinct slm selections. "
        f"Validation statuses: {', '.join(validations)}."
    )


def task_trace_payload(task_result: dict[str, Any]) -> dict[str, Any]:
    final_materialization = task_result["final_materialization"]
    return {
        "task": task_result["task"],
        "task_index": task_result["task_index"],
        "status": final_status(task_result["steps"], task_result["validation"], False),
        "validation": task_result["validation"],
        "review_artifact_path": final_materialization.get("review_artifact_path"),
        "generated_patch": final_materialization["diff"],
        "generated_actions": final_materialization.get("actions"),
        "steps": task_result["steps"],
    }


def task_result_payload(task_result: dict[str, Any]) -> dict[str, Any]:
    final_materialization = task_result["final_materialization"]
    return {
        "task": task_result["task"],
        "task_index": task_result["task_index"],
        "status": final_status(task_result["steps"], task_result["validation"], False),
        "validation": task_result["validation"],
        "review_artifact_path": final_materialization.get("review_artifact_path"),
        "generated_actions": final_materialization.get("actions"),
        "last_proposed_diff": final_materialization["diff"],
        "steps": task_result["steps"],
    }


def final_summary(steps: list[dict[str, Any]], validation: dict[str, Any], writes: str, *, dry_run: bool) -> str:
    selected = [tuple(step.get("selected_slms") or []) for step in steps if step.get("selected_slms") is not None]
    unique = [list(item) for index, item in enumerate(selected) if item not in selected[:index]]
    if dry_run:
        return (
            f"Executed {len(steps)} steps in route/plan only dry-run mode. "
            f"Observed {len(unique)} distinct slm selections. "
            "Generation, writes, and validation were skipped."
        )
    return (
        f"Executed {len(steps)} steps with writes mode '{writes}'. "
        f"Observed {len(unique)} distinct slm selections. "
        f"Validation status: {validation['status']}."
    )


def final_status(steps: list[dict[str, Any]], validation: dict[str, Any], dry_run: bool) -> str:
    if dry_run:
        return "dry-run"
    if validation["status"] == "failed":
        return "validation_failed"
    write_statuses = {step.get("write_status") for step in steps if step.get("write_status")}
    if "applied" in write_statuses:
        return "applied"
    if "review_required" in write_statuses:
        return "review_required"
    return "complete"


def execution_history(task_results: list[dict[str, Any]], *, max_items: int = 5) -> str | None:
    if not task_results:
        return None
    lines: list[str] = []
    for result in task_results[-max_items:]:
        final_materialization = result["final_materialization"]
        changed_files = final_materialization.get("files_changed") or []
        changed_files_label = ", ".join(changed_files) if changed_files else "none"
        summary = final_materialization.get("summary") or result["steps"][-1].get("result_summary") or ""
        lines.extend([
            f"Task {result['task_index']}: {result['task']}",
            f"Status: {final_status(result['steps'], result['validation'], False)}",
            f"Changed files: {changed_files_label}",
            f"Validation: {result['validation'].get('status')}",
            f"Summary: {summary}",
        ])
        diff_text = (final_materialization.get("diff") or "").strip()
        if diff_text:
            lines.append("Last diff excerpt:")
            lines.append(truncate_text(diff_text, 800))
    return "\n".join(lines)


def truncate_text(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def validation_summary(validation: dict[str, Any]) -> str:
    return (validation.get("stderr") or validation.get("stdout") or validation.get("status") or "")[:4000]
