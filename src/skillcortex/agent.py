import difflib
import json
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any

from .runtime import SkillRuntime


WRITE_MODES = ("off", "confirm", "on")
SKIP_DIRS = {".git", ".venv", ".skillcortex", "__pycache__", ".pytest_cache", ".ruff_cache"}
ARTIFACT_DIR_PREFIXES = ("datasets/", "runtime/", "skills/", "tmp/")
CODE_FILE_SUFFIXES = (".py", ".pyi", ".ts", ".tsx", ".js", ".jsx")
TEXT_FILE_SUFFIXES = CODE_FILE_SUFFIXES + (".md", ".txt", ".json", ".yaml", ".yml")


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

    def materialize_action(self, action: dict[str, Any], *, review_path: Path | None = None) -> dict[str, Any]:
        kind = action.get("kind") or "proposed_diff"
        if kind == "no_change":
            return {
                "kind": kind,
                "write_status": "skipped",
                "files_changed": [],
                "diff": "",
                "review_artifact_path": None,
                "summary": action.get("summary") or "No change proposed.",
            }
        if kind == "proposed_diff":
            diff = action.get("diff") or ""
            files_changed = _files_from_unified_diff(diff)
            artifact_path = None
            write_status = "proposed"
            if self.writes_mode == "on":
                _apply_unified_diff(self.repo, diff)
                write_status = "applied"
            elif self.writes_mode == "confirm":
                artifact_path = _write_review_artifact(review_path, diff)
                write_status = "review_required"
            return {
                "kind": kind,
                "write_status": write_status,
                "files_changed": files_changed,
                "diff": diff,
                "review_artifact_path": str(artifact_path) if artifact_path is not None else None,
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
        artifact_path = None
        write_status = "proposed"
        if self.writes_mode == "on":
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
            write_status = "applied"
        elif self.writes_mode == "confirm":
            artifact_path = _write_review_artifact(review_path, diff)
            write_status = "review_required"
        return {
            "kind": kind,
            "write_status": write_status,
            "files_changed": [relative_path],
            "diff": diff,
            "review_artifact_path": str(artifact_path) if artifact_path is not None else None,
            "summary": action.get("summary") or f"Replace {relative_path}.",
        }

    def materialize_actions(self, actions: list[dict[str, Any]], *, review_path: Path | None = None) -> dict[str, Any]:
        if not actions:
            return {
                "kind": "no_change",
                "write_status": "skipped",
                "files_changed": [],
                "diff": "",
                "review_artifact_path": None,
                "summary": "No actions proposed.",
                "actions": [],
            }
        parts = []
        files_changed: list[str] = []
        write_status = "skipped"
        summaries: list[str] = []
        action_results: list[dict[str, Any]] = []
        review_artifact_path = None
        for action in actions:
            result = self.materialize_action(action)
            action_results.append(result)
            if result["diff"]:
                parts.append(result["diff"])
            for path in result["files_changed"]:
                if path not in files_changed:
                    files_changed.append(path)
            if result["summary"]:
                summaries.append(result["summary"])
            write_status = _merge_write_status(write_status, result["write_status"])
        diff = "\n".join(part.rstrip("\n") for part in parts if part).strip()
        if diff:
            diff += "\n"
        if self.writes_mode == "confirm" and review_path is not None and diff:
            artifact = _write_review_artifact(review_path, diff)
            review_artifact_path = str(artifact) if artifact is not None else None
            write_status = "review_required"
        return {
            "kind": "action_list" if len(actions) > 1 else action_results[0]["kind"],
            "write_status": write_status,
            "files_changed": files_changed,
            "diff": diff,
            "review_artifact_path": review_artifact_path,
            "summary": " ".join(summaries).strip() or f"Materialized {len(actions)} action(s).",
            "actions": actions,
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
        "execution_mode": "dry-run-route-plan-only" if dry_run else f"write-mode-{writes}",
        "test_command": test_command,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "steps": [],
    }

    repo_files = sandbox.list_files()
    read_targets = _default_read_targets(repo_files)
    repo_context = {path: sandbox.read_file(path) for path in read_targets}
    _append_step(
        trace,
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
        },
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
    _append_step(
        trace,
        _inference_step(
            "plan",
            plan_result,
            files_read=read_targets,
            mode_label="route/plan only" if dry_run else None,
        ),
    )

    patch_target = _choose_patch_target(repo_files, task)
    patch_result = runtime.infer(
        messages=_step_messages(
            task,
            (
                "Produce JSON-only coding actions for the next step. "
                "Return either a JSON array of action objects or an object with an 'actions' array. "
                "Each action must use kind=file_replace, proposed_diff, or no_change. "
                "Use explicit path and content fields for file_replace actions whenever possible. "
                "Do not rewrite runtime bundles, datasets, skill packages, or other generated artifacts unless the task explicitly asks for that. "
                f"Prefer editing {patch_target} only when no better path is clear, and create a new source file when the repo has no suitable code target."
            ),
            repo_files=repo_files,
            repo_context=repo_context,
        ),
        task_type="python_generation",
        dry_run=dry_run,
    )
    review_path = _default_review_path(repo_root, trace["run_id"]) if writes == "confirm" and not dry_run else None
    patch_materialization = _materialize_inference_action(
        sandbox,
        patch_result,
        patch_target,
        dry_run,
        review_path=review_path,
    )
    _append_step(
        trace,
        _inference_step(
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
        ),
    )

    validation = _validation_result_for_mode(test_command, repo_root, dry_run)
    _append_step(
        trace,
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
        },
    )

    debug_materialization = None
    if validation["status"] == "failed":
        debug_result = runtime.infer(
            messages=_step_messages(
                task,
                (
                    "Debug the failed validation and propose the next code change as JSON-only actions. "
                    "Return either a JSON array of action objects or an object with an 'actions' array. "
                    "Each action must use kind=file_replace, proposed_diff, or no_change. "
                    "Use explicit path and content fields for file_replace actions whenever possible. "
                    "Do not rewrite runtime bundles, datasets, skill packages, or other generated artifacts unless the task explicitly asks for that."
                ),
                repo_files=repo_files,
                repo_context=repo_context,
                validation_output=_validation_summary(validation),
            ),
            task_type="debugging",
            dry_run=dry_run,
        )
        debug_review_path = _default_review_path(repo_root, f"{trace['run_id']}-debug") if writes == "confirm" and not dry_run else None
        debug_materialization = _materialize_inference_action(
            sandbox,
            debug_result,
            patch_target,
            dry_run,
            review_path=debug_review_path,
        )
        _append_step(
            trace,
            _inference_step(
                "debug_failure",
                debug_result,
                files_read=read_targets,
                files_changed=debug_materialization["files_changed"],
                tool_name="propose_diff" if debug_materialization["write_status"] != "applied" else "apply_patch",
                tool_result_summary=debug_materialization["summary"],
                proposed_diff=debug_materialization["diff"],
                write_status=debug_materialization["write_status"],
                review_artifact_path=debug_materialization.get("review_artifact_path"),
            ),
        )

    final_status = _final_status(trace["steps"], validation, dry_run)
    final_summary = _final_summary(trace["steps"], validation, writes, dry_run=dry_run)
    final_materialization = debug_materialization or patch_materialization
    trace["ended_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    trace["status"] = final_status
    trace["final_summary"] = final_summary
    trace["review_artifact_path"] = final_materialization.get("review_artifact_path")
    trace["generated_patch"] = final_materialization["diff"]
    trace["generated_actions"] = final_materialization.get("actions")
    trace["validation"] = validation
    if trace_out is not None:
        trace_out = trace_out.resolve()
        trace_out.parent.mkdir(parents=True, exist_ok=True)
        trace_out.write_text(json.dumps(trace, indent=2) + "\n")

    return {
        "status": final_status,
        "task": task,
        "writes_mode": writes,
        "execution_mode": trace["execution_mode"],
        "repo": str(repo_root),
        "runtime": str(runtime_path.resolve()),
        "trace_path": str(trace_out.resolve()) if trace_out is not None else None,
        "step_count": len(trace["steps"]),
        "final_summary": final_summary,
        "steps": trace["steps"],
        "validation": validation,
        "review_artifact_path": final_materialization.get("review_artifact_path"),
        "generated_actions": final_materialization.get("actions"),
        "last_proposed_diff": final_materialization["diff"],
    }


def _choose_patch_target(files: list[str], task: str) -> str:
    for path in _preferred_source_files(files):
        if path.endswith(".py") and not path.startswith("tests/"):
            return path
    preferred = _preferred_source_files(files)
    if preferred:
        return preferred[0]
    return _default_new_source_path(task)


def _default_read_targets(files: list[str]) -> list[str]:
    preferred = _preferred_source_files(files)
    return preferred[:5]


def _preferred_source_files(files: list[str]) -> list[str]:
    source_files = [
        path for path in files
        if path.endswith(TEXT_FILE_SUFFIXES)
        and not _is_artifact_path(path)
    ]
    if source_files:
        code_files = [path for path in source_files if path.endswith(CODE_FILE_SUFFIXES)]
        return code_files + [path for path in source_files if path not in code_files]
    return []


def _is_artifact_path(path: str) -> bool:
    return path.startswith(ARTIFACT_DIR_PREFIXES)


def _default_new_source_path(task: str) -> str:
    lowered = task.lower()
    if any(token in lowered for token in ("fastapi", "endpoint", "router", "api")):
        return "app.py"
    return "main.py"


def _extract_actions(generation: str, default_path: str) -> list[dict[str, Any]]:
    text = _strip_code_fence((generation or "").strip())
    if not text:
        return [{"kind": "no_change", "summary": "Empty model output."}]
    try:
        value = json.loads(text)
        if isinstance(value, dict) and isinstance(value.get("actions"), list):
            return _normalize_actions(value["actions"], default_path)
        if isinstance(value, list):
            return _normalize_actions(value, default_path)
        if isinstance(value, dict):
            return _normalize_actions([value], default_path)
    except json.JSONDecodeError:
        pass
    if _looks_like_unified_diff(text):
        return [{
            "kind": "proposed_diff",
            "diff": text,
            "summary": f"Unstructured diff output for {default_path}.",
        }]
    content = _extract_code_content(text)
    return [{
        "kind": "file_replace",
        "path": default_path,
        "content": content,
        "summary": f"Unstructured model output converted to file update for {default_path}.",
    }]


def _normalize_actions(actions: list[Any], default_path: str) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for item in actions:
        if not isinstance(item, dict):
            continue
        action = dict(item)
        if action.get("kind") == "file_replace" and "path" not in action:
            action["path"] = default_path
        normalized.append(action)
    return normalized or [{"kind": "no_change", "summary": "Empty action list."}]


def _final_summary(steps: list[dict[str, Any]], validation: dict[str, Any], writes: str, *, dry_run: bool) -> str:
    selected = [tuple(step.get("selected_skills") or []) for step in steps if step.get("selected_skills") is not None]
    unique = [list(item) for index, item in enumerate(selected) if item not in selected[:index]]
    if dry_run:
        return (
            f"Executed {len(steps)} steps in route/plan only dry-run mode. "
            f"Observed {len(unique)} distinct skill selections. "
            "Generation, writes, and validation were skipped."
        )
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


def _materialize_inference_action(
    sandbox: ToolSandbox,
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
    actions = _extract_actions(result.get("generation") or "", default_path)
    return sandbox.materialize_actions(actions, review_path=review_path)


def _append_step(trace: dict[str, Any], step: dict[str, Any]) -> None:
    step["step_index"] = len(trace["steps"]) + 1
    trace["steps"].append(step)


def _default_review_path(repo: Path, run_id: str) -> Path:
    return repo / ".skillcortex" / "reviews" / f"{run_id}.patch"


def _write_review_artifact(review_path: Path | None, diff: str) -> Path | None:
    if review_path is None or not diff:
        return None
    review_path.parent.mkdir(parents=True, exist_ok=True)
    review_path.write_text(diff)
    return review_path.resolve()


def _merge_write_status(current: str, new: str) -> str:
    priority = {
        "skipped": 0,
        "not_applicable": 0,
        "proposed": 1,
        "review_required": 2,
        "approval_required": 2,
        "applied": 3,
    }
    return new if priority.get(new, 0) >= priority.get(current, 0) else current


def _files_from_unified_diff(diff: str) -> list[str]:
    files: list[str] = []
    for line in diff.splitlines():
        if line.startswith("+++ b/"):
            path = line[6:]
            if path and path not in files:
                files.append(path)
    return files


def _looks_like_unified_diff(text: str) -> bool:
    lines = text.splitlines()
    if len(lines) < 3:
        return False
    return lines[0].startswith("--- ") and lines[1].startswith("+++ ") and any(
        line.startswith("@@") for line in lines[2:]
    )


def _extract_code_content(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3:
            return "\n".join(lines[1:-1]).strip() + "\n"
    return stripped + ("\n" if stripped and not stripped.endswith("\n") else "")


def _strip_code_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3:
            return "\n".join(lines[1:-1]).strip()
    return stripped


def _apply_unified_diff(repo: Path, diff: str) -> None:
    if not diff.strip():
        return
    patches = _parse_unified_diff(diff)
    if not patches:
        raise ValueError("proposed_diff action requires a unified diff payload")
    for patch in patches:
        _apply_single_patch(repo, patch)


def _parse_unified_diff(diff: str) -> list[dict[str, Any]]:
    lines = diff.splitlines()
    index = 0
    patches: list[dict[str, Any]] = []
    while index < len(lines):
        line = lines[index]
        if not line.startswith("--- "):
            index += 1
            continue
        if index + 1 >= len(lines) or not lines[index + 1].startswith("+++ "):
            raise ValueError("malformed unified diff: missing destination header")
        from_file = lines[index][4:]
        to_file = lines[index + 1][4:]
        index += 2
        hunks: list[dict[str, Any]] = []
        while index < len(lines) and lines[index].startswith("@@"):
            header = lines[index]
            old_range, new_range = header.split("@@")[1].strip().split(" ")
            old_start, old_length = _parse_range(old_range)
            new_start, new_length = _parse_range(new_range)
            index += 1
            hunk_lines: list[tuple[str, str]] = []
            while index < len(lines):
                current = lines[index]
                if current.startswith("@@") or current.startswith("--- "):
                    break
                marker = current[0] if current else " "
                value = current[1:] if current else ""
                if marker not in {" ", "+", "-"}:
                    raise ValueError(f"unsupported diff line: {current}")
                hunk_lines.append((marker, value))
                index += 1
            hunks.append(
                {
                    "old_start": old_start,
                    "old_length": old_length,
                    "new_start": new_start,
                    "new_length": new_length,
                    "lines": hunk_lines,
                }
            )
        patches.append({"from_file": from_file, "to_file": to_file, "hunks": hunks})
    return patches


def _parse_range(token: str) -> tuple[int, int]:
    token = token[1:]
    if "," in token:
        start, length = token.split(",", 1)
        return int(start), int(length)
    return int(token), 1


def _apply_single_patch(repo: Path, patch: dict[str, Any]) -> None:
    relative_path = _normalize_diff_path(patch["to_file"])
    path = (repo / relative_path).resolve()
    if repo not in (path, *path.parents):
        raise ValueError(f"patch escapes repo root: {relative_path}")
    original_exists = path.exists()
    original_text = path.read_text() if original_exists else ""
    original_lines = original_text.splitlines()
    result_lines: list[str] = []
    source_index = 0
    for hunk in patch["hunks"]:
        target_index = max(hunk["old_start"] - 1, 0)
        result_lines.extend(original_lines[source_index:target_index])
        source_index = target_index
        for marker, value in hunk["lines"]:
            if marker == " ":
                if source_index >= len(original_lines) or original_lines[source_index] != value:
                    raise ValueError(f"diff context mismatch for {relative_path}")
                result_lines.append(value)
                source_index += 1
            elif marker == "-":
                if source_index >= len(original_lines) or original_lines[source_index] != value:
                    raise ValueError(f"diff removal mismatch for {relative_path}")
                source_index += 1
            elif marker == "+":
                result_lines.append(value)
    result_lines.extend(original_lines[source_index:])
    path.parent.mkdir(parents=True, exist_ok=True)
    trailing_newline = original_text.endswith("\n") if original_exists else True
    path.write_text("\n".join(result_lines) + ("\n" if result_lines or trailing_newline else ""))


def _normalize_diff_path(path: str) -> str:
    normalized = path.strip()
    if normalized.startswith("a/") or normalized.startswith("b/"):
        normalized = normalized[2:]
    return normalized


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
        return {
            "status": "skipped",
            "command": None,
            "exit_code": None,
            "stdout": "",
            "stderr": "",
        }
    return _run_validation_command(command, repo)


def _final_status(steps: list[dict[str, Any]], validation: dict[str, Any], dry_run: bool) -> str:
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