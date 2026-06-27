from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

from .context import choose_patch_target, default_read_targets, recent_changed_files
from .reporting import validation_summary
from .steps import append_step, default_review_path, inference_step, materialize_inference_action, step_messages


def run_single_task(
    *,
    runtime,
    sandbox,
    repo_root: Path,
    task: str,
    writes: str,
    test_command: str | None,
    dry_run: bool,
    run_id: str,
    task_index: int,
    prior_task_results: list[dict[str, Any]] | None = None,
    validation_runner: Callable[[str | None, Path, bool], dict[str, Any]],
    execution_history: str | None = None,
) -> dict[str, Any]:
    task_steps: list[dict[str, Any]] = []
    repo_files = sandbox.list_files()
    patch_target = choose_patch_target(repo_files, task)
    read_targets = default_read_targets(repo_files, patch_target=patch_target, prior_task_results=prior_task_results or [])
    focus_targets = {patch_target, *recent_changed_files(prior_task_results or [])}
    repo_context = {
        path: sandbox.read_file(path, max_chars=12000 if path in focus_targets else 4000)
        for path in read_targets
    }
    append_step(
        {"steps": task_steps},
        {
            "step_type": "inspect_repo",
            "selected_skills": [],
            "route_type": None,
            "route_reason": None,
            "tool_name": "list_files+read_file",
            "files_read": read_targets,
            "files_changed": [],
            "status": "complete",
            "result_summary": f"Inspected {len(repo_files)} files and read {len(read_targets)} files.",
            "task": task,
            "task_index": task_index,
        },
    )
    plan_result = runtime.infer(
        messages=step_messages(
            task,
            "Create a short execution plan for a local coding task. Return concise numbered steps only.",
            repo_files=repo_files,
            repo_context=repo_context,
            execution_history=execution_history,
        ),
        task_type="python_generation",
        dry_run=dry_run,
    )
    plan_step = inference_step("plan", plan_result, files_read=read_targets, mode_label="route/plan only" if dry_run else None)
    plan_step["task"] = task
    plan_step["task_index"] = task_index
    append_step({"steps": task_steps}, plan_step)
    patch_result = runtime.infer(
        messages=step_messages(
            task,
            (
                "Produce JSON-only coding actions for the next step. Return either a JSON array of action objects or an object with an 'actions' array. "
                "Each action must use kind=file_replace, proposed_diff, or no_change. Prefer proposed_diff for edits to existing files and preserve unchanged code. "
                "Use file_replace for new files or intentional full-file rewrites only, and never emit a partial file body for an existing file. "
                "Do not rewrite runtime bundles, datasets, skill packages, or other generated artifacts unless the task explicitly asks for that. "
                f"Prefer editing {patch_target} only when no better path is clear, and create a new source file when the repo has no suitable code target."
            ),
            repo_files=repo_files,
            repo_context=repo_context,
            execution_history=execution_history,
        ),
        task_type="python_generation",
        dry_run=dry_run,
    )
    review_path = default_review_path(repo_root, f"{run_id}-task-{task_index}") if writes == "confirm" and not dry_run else None
    patch_materialization = materialize_inference_action(sandbox, patch_result, patch_target, dry_run, review_path=review_path)
    patch_step = inference_step(
        "propose_patch",
        patch_result,
        files_read=read_targets,
        files_changed=patch_materialization["files_changed"],
        tool_name="propose_diff" if patch_materialization["write_status"] != "applied" else "apply_patch",
        tool_result_summary=patch_materialization["summary"],
        proposed_diff=patch_materialization["diff"],
        write_status=patch_materialization["write_status"],
        review_artifact_path=patch_materialization.get("review_artifact_path"),
        mode_label="route/plan only" if dry_run else None,
    )
    patch_step["task"] = task
    patch_step["task_index"] = task_index
    append_step({"steps": task_steps}, patch_step)
    validation = validation_runner(test_command, repo_root, dry_run)
    append_step(
        {"steps": task_steps},
        {
            "step_type": "run_validation",
            "selected_skills": [],
            "route_type": None,
            "route_reason": None,
            "tool_name": "run_validation",
            "tool_args": validation["command"],
            "files_read": [],
            "files_changed": [],
            "validation_exit_code": validation["exit_code"],
            "status": validation["status"],
            "result_summary": (validation["stderr"] or validation["stdout"] or validation["status"])[:400],
            "task": task,
            "task_index": task_index,
        },
    )
    debug_materialization = None
    if validation["status"] == "failed":
        debug_result = runtime.infer(
            messages=step_messages(
                task,
                (
                    "Debug the failed validation and propose the next code change as JSON-only actions. Return either a JSON array of action objects or an object with an 'actions' array. "
                    "Each action must use kind=file_replace, proposed_diff, or no_change. Prefer proposed_diff for edits to existing files and preserve unchanged code. "
                    "Use file_replace for new files or intentional full-file rewrites only, and never emit a partial file body for an existing file. "
                    "Do not rewrite runtime bundles, datasets, skill packages, or other generated artifacts unless the task explicitly asks for that."
                ),
                repo_files=repo_files,
                repo_context=repo_context,
                execution_history=execution_history,
                validation_output=validation_summary(validation),
            ),
            task_type="debugging",
            dry_run=dry_run,
        )
        debug_review_path = default_review_path(repo_root, f"{run_id}-task-{task_index}-debug") if writes == "confirm" and not dry_run else None
        debug_materialization = materialize_inference_action(sandbox, debug_result, patch_target, dry_run, review_path=debug_review_path)
        debug_step = inference_step(
            "debug_failure",
            debug_result,
            files_read=read_targets,
            files_changed=debug_materialization["files_changed"],
            tool_name="propose_diff" if debug_materialization["write_status"] != "applied" else "apply_patch",
            tool_result_summary=debug_materialization["summary"],
            proposed_diff=debug_materialization["diff"],
            write_status=debug_materialization["write_status"],
            review_artifact_path=debug_materialization.get("review_artifact_path"),
        )
        debug_step["task"] = task
        debug_step["task_index"] = task_index
        append_step({"steps": task_steps}, debug_step)
    return {
        "task": task,
        "task_index": task_index,
        "steps": task_steps,
        "validation": validation,
        "final_materialization": debug_materialization or patch_materialization,
    }
