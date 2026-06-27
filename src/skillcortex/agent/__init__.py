import json
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any, Callable

from ..runtime import SkillRuntime
from .reporting import execution_history, final_status, merge_task_statuses, multi_task_summary, task_result_payload, task_trace_payload
from .sandbox import WRITE_MODES, ToolSandbox
from .service import run_single_task
from .steps import task_sequence


def run_agent(
    *,
    runtime_path: Path,
    repo: Path,
    task: str | list[str] | None = None,
    writes: str = "confirm",
    test_command: str | None = None,
    trace_out: Path | None = None,
    dry_run: bool = False,
    task_provider: Callable[[], str | None] | None = None,
) -> dict[str, Any]:
    if writes not in WRITE_MODES:
        raise ValueError(f"unknown writes mode: {writes}")
    repo_root = repo.resolve()
    if not repo_root.exists() or not repo_root.is_dir():
        raise FileNotFoundError(f"repo not found: {repo_root}")
    runtime = SkillRuntime.load(runtime_path)
    runtime.validate()
    sandbox = ToolSandbox(repo_root, writes)
    tasks: list[str] = []
    trace = {
        "schema_version": "1",
        "run_id": f"agent-{int(time.time() * 1000)}",
        "task": None,
        "tasks": tasks,
        "repo": str(repo_root),
        "runtime": str(runtime_path.resolve()),
        "writes_mode": writes,
        "execution_mode": "dry-run-route-plan-only" if dry_run else f"write-mode-{writes}",
        "test_command": test_command,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "steps": [],
    }
    task_results = []
    for index, task_input in enumerate(task_sequence(task, task_provider=task_provider), start=1):
        if trace["task"] is None:
            trace["task"] = task_input
        tasks.append(task_input)
        task_result = run_single_task(
            runtime=runtime,
            sandbox=sandbox,
            repo_root=repo_root,
            task=task_input,
            writes=writes,
            test_command=test_command,
            dry_run=dry_run,
            run_id=trace["run_id"],
            task_index=index,
            prior_task_results=task_results,
            validation_runner=_validation_result_for_mode,
            execution_history=execution_history(task_results),
        )
        task_results.append(task_result)
        trace["steps"].extend(task_result["steps"])
    if not task_results:
        raise ValueError("at least one task is required")
    final_status_value = merge_task_statuses(task_results, dry_run=dry_run)
    final_summary = multi_task_summary(task_results, writes=writes, dry_run=dry_run)
    final_materialization = task_results[-1]["final_materialization"]
    validations = [result["validation"] for result in task_results]
    trace["ended_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    trace["status"] = final_status_value
    trace["final_summary"] = final_summary
    trace["review_artifact_path"] = final_materialization.get("review_artifact_path")
    trace["generated_patch"] = final_materialization["diff"]
    trace["generated_actions"] = final_materialization.get("actions")
    trace["validation"] = validations[-1]
    trace["validation_results"] = validations
    trace["task_results"] = [task_trace_payload(result) for result in task_results]
    if trace_out is not None:
        trace_out = trace_out.resolve()
        trace_out.parent.mkdir(parents=True, exist_ok=True)
        trace_out.write_text(json.dumps(trace, indent=2) + "\n")
    result: dict[str, Any] = {
        "status": final_status_value,
        "task": tasks[0],
        "tasks": tasks,
        "writes_mode": writes,
        "execution_mode": trace["execution_mode"],
        "repo": str(repo_root),
        "runtime": str(runtime_path.resolve()),
        "trace_path": str(trace_out.resolve()) if trace_out is not None else None,
        "step_count": len(trace["steps"]),
        "final_summary": final_summary,
        "steps": trace["steps"],
        "validation": validations[-1],
        "validation_results": validations,
        "review_artifact_path": final_materialization.get("review_artifact_path"),
        "generated_actions": final_materialization.get("actions"),
        "last_proposed_diff": final_materialization["diff"],
        "task_results": [task_result_payload(result) for result in task_results],
    }
    if len(tasks) == 1:
        result["generated_actions"] = final_materialization.get("actions")
        result["last_proposed_diff"] = final_materialization["diff"]
    return result


def _validation_result_for_mode(command: str | None, repo: Path, dry_run: bool) -> dict[str, Any]:
    if dry_run:
        return {
            "status": "skipped",
            "command": command,
            "exit_code": None,
            "stdout": "",
            "stderr": "dry-run route/plan only: validation skipped",
        }
    if not command:
        return {"status": "skipped", "command": None, "exit_code": None, "stdout": "", "stderr": ""}
    return _run_validation_command(command, repo)


def _run_validation_command(command: str, repo: Path) -> dict[str, Any]:
    completed = subprocess.run(
        shlex.split(command),
        cwd=repo,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return {
        "status": "passed" if completed.returncode == 0 else "failed",
        "command": command,
        "exit_code": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }
