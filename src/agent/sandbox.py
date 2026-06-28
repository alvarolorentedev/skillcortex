from __future__ import annotations

from pathlib import Path
from typing import Any

from .diffing import (
    apply_unified_diff,
    files_from_unified_diff,
    looks_like_truncating_file_replace,
    merge_write_status,
    unified_diff,
    write_review_artifact,
)


WRITE_MODES = ("off", "confirm", "on")
SKIP_DIRS = {".git", ".venv", ".slmcortex", "__pycache__", ".pytest_cache", ".ruff_cache"}
ARTIFACT_DIR_PREFIXES = ("datasets/", "runtime/", "slms/", "tmp/")
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
            files_changed = files_from_unified_diff(diff)
            artifact_path = None
            write_status = "proposed"
            if self.writes_mode == "on":
                apply_unified_diff(self.repo, diff)
                write_status = "applied"
            elif self.writes_mode == "confirm":
                artifact_path = write_review_artifact(review_path, diff)
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
        if looks_like_truncating_file_replace(before, content):
            raise ValueError(
                f"file_replace for {relative_path} appears to truncate an existing file; use proposed_diff or include the full updated file"
            )
        diff = unified_diff(before, content, relative_path)
        artifact_path = None
        write_status = "proposed"
        if self.writes_mode == "on":
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
            write_status = "applied"
        elif self.writes_mode == "confirm":
            artifact_path = write_review_artifact(review_path, diff)
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
            write_status = merge_write_status(write_status, result["write_status"])
        diff = "\n".join(part.rstrip("\n") for part in parts if part).strip()
        if diff:
            diff += "\n"
        if self.writes_mode == "confirm" and review_path is not None and diff:
            artifact = write_review_artifact(review_path, diff)
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
