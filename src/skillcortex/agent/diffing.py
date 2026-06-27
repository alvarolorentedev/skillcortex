from __future__ import annotations

import difflib
from pathlib import Path
from typing import Any


def write_review_artifact(review_path: Path | None, diff: str) -> Path | None:
    if review_path is None or not diff:
        return None
    review_path.parent.mkdir(parents=True, exist_ok=True)
    review_path.write_text(diff)
    return review_path.resolve()


def merge_write_status(current: str, new: str) -> str:
    priority = {
        "skipped": 0,
        "not_applicable": 0,
        "proposed": 1,
        "review_required": 2,
        "approval_required": 2,
        "applied": 3,
    }
    return new if priority.get(new, 0) >= priority.get(current, 0) else current


def files_from_unified_diff(diff: str) -> list[str]:
    files: list[str] = []
    for line in diff.splitlines():
        if line.startswith("+++ b/"):
            path = line[6:]
            if path and path not in files:
                files.append(path)
    return files


def looks_like_truncating_file_replace(before: str, after: str) -> bool:
    before_text = before.rstrip("\n")
    after_text = after.rstrip("\n")
    if not before_text or not after_text or before_text == after_text:
        return False
    return before_text.startswith(after_text)


def apply_unified_diff(repo: Path, diff: str) -> None:
    if not diff.strip():
        return
    patches = parse_unified_diff(diff)
    if not patches:
        raise ValueError("proposed_diff action requires a unified diff payload")
    for patch in patches:
        apply_single_patch(repo, patch)


def parse_unified_diff(diff: str) -> list[dict[str, Any]]:
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
            old_start, old_length = parse_range(old_range)
            new_start, new_length = parse_range(new_range)
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


def parse_range(token: str) -> tuple[int, int]:
    token = token[1:]
    if "," in token:
        start, length = token.split(",", 1)
        return int(start), int(length)
    return int(token), 1


def apply_single_patch(repo: Path, patch: dict[str, Any]) -> None:
    relative_path = normalize_diff_path(patch["to_file"])
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


def normalize_diff_path(path: str) -> str:
    normalized = path.strip()
    if normalized.startswith("a/") or normalized.startswith("b/"):
        normalized = normalized[2:]
    return normalized


def unified_diff(before: str, after: str, relative_path: str) -> str:
    return "".join(
        difflib.unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile=f"a/{relative_path}",
            tofile=f"b/{relative_path}",
        )
    )
