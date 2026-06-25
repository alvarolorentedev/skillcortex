import difflib
import json
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any

from .runtime import SkillRuntime


WRITE_MODES = ("off", "confirm", "on")
SKIP_DIRS = {".git", ".venv", "__pycache__", ".pytest_cache", ".ruff_cache"}


class ToolSandbox:
    def __init__(self, repo: Path, writes_mode: str):
        self.repo = repo.resolve()
        self.writes_mode = writes_mode

    def list_files(self, *, limit: int = 200) -> list[str]:
        files = []
        for path in sorted(self.repo.rglob("*")):
            if any(part in SKIP_DIRS for part in path.parts):
                continue
            if path.is_file():
                files.append(path.relative_to(self.repo).as_posix())
            if len(files) >= limit:
                break
        return files

    def read_file(self, relative_path: str, *, max_chars: int = 4000) -> str:
        path = self._resolve(relative_path)
        return path.read_text()[:max_chars]

    def materialize_action(self, action: dict[str, Any]) -> dict[str, Any]:
        kind = action.get("kind") or "proposed_diff"
        if kind == "no_change":
            return {
                "kind": kind,
                "write_status": "skipped",
                "files_changed": [],
                "diff": "",
                "summary": action.get("summary") or "No change proposed.",
            }
        if kind == "proposed_diff":
            return {
                "kind": kind,
                "write_status": "not_applicable",
                "files_changed": [],
                "diff": action.get("diff") or "",
                "summary": action.get("summary") or "Proposed diff.",
            }
        if kind != "file_replace":
            raise ValueError(f"unknown action kind: {kind}")
        relative_path = action.get("path")
        content = action.get("content")
        if not isinstance(relative_path, str) or not relative_path.strip():
            raise ValueError("file_replace action requires a non-empty path")
        if not isinstance(content, str):
            raise ValueError("file_replace action requires string content")
        path = self._resolve(relative_path)
        before = path.read_text() if path.exists() else ""
        diff = _unified_diff(before, content, relative_path)
        write_status = "proposed"
        if self.writes_mode == "on":
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
            write_status = "applied"
        elif self.writes_mode == "confirm":
            write_status = "approval_required"
        return {
            "kind": kind,
            "write_status": write_status,
            "files_changed": [relative_path],
            "diff": diff,
            "summary": action.get("summary") or f"Replace {relative_path}.",
        }

    def _resolve(self, relative_path: str) -> Path:
        candidate = (self.repo / relative_path).resolve()
        if self.repo not in (candidate, *candidate.parents):
            raise ValueError(f"path escapes repo root: {relative_path}")
        return candidate


def run_agent(
    *,
    runtime_path: Path,
    repo: Path,
    task: str,
    writes: str = "confirm",
    test_command: str | None = None,
    trace_out: Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    if writes not in WRITE_MODES:
        raise ValueError(f"unknown writes mode: {writes}")
    repo_root = repo.resolve()
    if not repo_root.exists() or not repo_root.is_dir():
        raise FileNotFoundError(f"repo not found: {repo_root}")

    runtime = SkillRuntime.load(runtime_path)
    runtime.validate()
    sandbox = ToolSandbox(repo_root, writes)
    trace = {
        "schema_version": "1",
        "run_id": f"agent-{int(time.time() * 1000)}",
        "task": task,
        "repo": str(repo_root),
        "runtime": str(runtime_path.resolve()),
        "writes_mode": writes,
        "test_command": test_command,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "steps": [],
    }

    repo_files = sandbox.list_files()
    read_targets = _default_read_targets(repo_files)
    repo_context = {path: sandbox.read_file(path) for path in read_targets}
    trace["steps"].append(
        {
            "step_index": len(trace["steps"]) + 1,
            "step_type": "inspect_repo",
            "selected_skills": [],
            "route_type": None,
            "route_reason": None,
            "tool_name": "list_files+read_file",
            "files_read": read_targets,
            "files_changed": [],
            "status": "complete",
            "result_summary": f"Inspected {len(repo_files)} files and read {len(read_targets)} files.",
        }
    )

    plan_result = runtime.infer(
        messages=_step_messages(
            task,
            (
                "Create a short execution plan for a local coding task. "
                "Return concise numbered steps only."
            ),
            repo_files=repo_files,
            repo_context=repo_context,
        ),
        task_type="python_generation",
        dry_run=dry_run,
    )
    trace["steps"].append(_inference_step("plan", plan_result, files_read=read_targets))

    patch_target = _choose_patch_target(repo_files)
    patch_result = runtime.infer(
        messages=_step_messages(
            task,
            (
                "Produce a JSON action for the next coding step. "
                "Return one JSON object with kind=file_replace, proposed_diff, or no_change. "
                f"Prefer editing {patch_target}."
            ),
            repo_files=repo_files,
            repo_context=repo_context,
        ),
        task_type="python_generation",
        dry_run=dry_run,
    )
    patch_materialization = _materialize_inference_action(sandbox, patch_result, patch_target, dry_run)
    trace["steps"].append(
        _inference_step(
            "propose_patch",
            patch_result,
            files_read=read_targets,
            files_changed=patch_materialization["files_changed"],
            tool_name="propose_diff" if patch_materialization["write_status"] != "applied" else "apply_patch",
            tool_result_summary=patch_materialization["summary"],
            proposed_diff=patch_materialization["diff"],
            write_status=patch_materialization["write_status"],
        )
    )

    validation = _run_validation_command(test_command, repo_root) if test_command else {
        "status": "skipped",
        "command": None,
        "exit_code": None,
        "stdout": "",
        "stderr": "",
    }
    trace["steps"].append(
        {
            "step_index": len(trace["steps"]) + 1,
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
        }
    )

    debug_materialization = None
    if validation["status"] == "failed":
        debug_result = runtime.infer(
            messages=_step_messages(
                task,
                (
                    "Debug the failed validation and propose the next code change as JSON. "
                    "Return one JSON object with kind=file_replace, proposed_diff, or no_change."
                ),
                repo_files=repo_files,
                repo_context=repo_context,
                validation_output=_validation_summary(validation),
            ),
            task_type="debugging",
            dry_run=dry_run,
        )
        debug_materialization = _materialize_inference_action(sandbox, debug_result, patch_target, dry_run)
        trace["steps"].append(
            _inference_step(
                "debug_failure",
                debug_result,
                files_read=read_targets,
                files_changed=debug_materialization["files_changed"],
                tool_name="propose_diff" if debug_materialization["write_status"] != "applied" else "apply_patch",
                tool_result_summary=debug_materialization["summary"],
                proposed_diff=debug_materialization["diff"],
                write_status=debug_materialization["write_status"],
            )
        )

    final_summary = _final_summary(trace["steps"], validation, writes)
    trace["ended_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    trace["status"] = "complete"
    trace["final_summary"] = final_summary
    if trace_out is not None:
        trace_out = trace_out.resolve()
        trace_out.parent.mkdir(parents=True, exist_ok=True)
        trace_out.write_text(json.dumps(trace, indent=2) + "\n")

    return {
        "status": "complete",
        "task": task,
        "writes_mode": writes,
        "repo": str(repo_root),
        "runtime": str(runtime_path.resolve()),
        "trace_path": str(trace_out.resolve()) if trace_out is not None else None,
        "step_count": len(trace["steps"]),
        "final_summary": final_summary,
        "steps": trace["steps"],
        "validation": validation,
        "last_proposed_diff": (debug_materialization or patch_materialization)["diff"],
    }


def _choose_patch_target(files: list[str]) -> str:
    for path in files:
        if path.endswith(".py") and not path.startswith("tests/"):
            return path
    return files[0] if files else "README.md"


def _default_read_targets(files: list[str]) -> list[str]:
    preferred = [path for path in files if path.endswith((".py", ".md", ".txt", ".json", ".yaml", ".yml"))]
    return preferred[:5]


def _extract_action(generation: str, default_path: str) -> dict[str, Any]:
    text = (generation or "").strip()
    if not text:
        return {"kind": "no_change", "summary": "Empty model output."}
    try:
        value = json.loads(text)
        if isinstance(value, dict):
            if value.get("kind") == "file_replace" and "path" not in value:
                value["path"] = default_path
            return value
    except json.JSONDecodeError:
        pass
    return {
        "kind": "proposed_diff",
        "diff": text,
        "summary": f"Unstructured model output for {default_path}.",
    }


def _final_summary(steps: list[dict[str, Any]], validation: dict[str, Any], writes: str) -> str:
    selected = [tuple(step.get("selected_skills") or []) for step in steps if step.get("selected_skills") is not None]
    unique = [list(item) for index, item in enumerate(selected) if item not in selected[:index]]
    return (
        f"Executed {len(steps)} steps with writes mode '{writes}'. "
        f"Observed {len(unique)} distinct skill selections. "
        f"Validation status: {validation['status']}."
    )


def _inference_step(
    step_type: str,
    result: dict[str, Any],
    *,
    files_read: list[str],
    files_changed: list[str] | None = None,
    tool_name: str | None = None,
    tool_result_summary: str | None = None,
    proposed_diff: str | None = None,
    write_status: str | None = None,
) -> dict[str, Any]:
    return {
        "step_index": 0,
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
    }


def _materialize_inference_action(
    sandbox: ToolSandbox,
    result: dict[str, Any],
    default_path: str,
    dry_run: bool,
) -> dict[str, Any]:
    if dry_run:
        return {
            "kind": "dry-run",
            "write_status": "dry-run",
            "files_changed": [],
            "diff": "",
            "summary": "Dry-run skipped artifact materialization.",
        }
    action = _extract_action(result.get("generation") or "", default_path)
    return sandbox.materialize_action(action)


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


def _step_messages(
    task: str,
    instruction: str,
    *,
    repo_files: list[str],
    repo_context: dict[str, str],
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
    if validation_output:
        content.extend(["Validation output:", validation_output])
    return [{"role": "user", "content": "\n\n".join(content)}]


def _unified_diff(before: str, after: str, relative_path: str) -> str:
    return "".join(
        difflib.unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile=f"a/{relative_path}",
            tofile=f"b/{relative_path}",
        )
    )


def _validation_summary(validation: dict[str, Any]) -> str:
    return (validation.get("stderr") or validation.get("stdout") or validation.get("status") or "")[:4000]