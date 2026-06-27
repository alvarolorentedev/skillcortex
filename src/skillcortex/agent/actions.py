from __future__ import annotations

import json
from typing import Any


def extract_actions(generation: str, default_path: str) -> list[dict[str, Any]]:
    text = strip_code_fence((generation or "").strip())
    if not text:
        return [{"kind": "no_change", "summary": "Empty model output."}]
    try:
        value = json.loads(text)
        if isinstance(value, dict) and isinstance(value.get("actions"), list):
            return normalize_actions(value["actions"], default_path)
        if isinstance(value, list):
            return normalize_actions(value, default_path)
        if isinstance(value, dict):
            return normalize_actions([value], default_path)
    except json.JSONDecodeError:
        pass
    if looks_like_unified_diff(text):
        return [{
            "kind": "proposed_diff",
            "diff": text,
            "summary": f"Unstructured diff output for {default_path}.",
        }]
    content = extract_code_content(text)
    return [{
        "kind": "file_replace",
        "path": default_path,
        "content": content,
        "summary": f"Unstructured model output converted to file update for {default_path}.",
    }]


def normalize_actions(actions: list[Any], default_path: str) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for item in actions:
        if not isinstance(item, dict):
            continue
        action = dict(item)
        if action.get("kind") == "file_replace" and "path" not in action:
            action["path"] = default_path
        normalized.append(action)
    return normalized or [{"kind": "no_change", "summary": "Empty action list."}]


def looks_like_unified_diff(text: str) -> bool:
    lines = text.splitlines()
    if len(lines) < 3:
        return False
    return lines[0].startswith("--- ") and lines[1].startswith("+++ ") and any(
        line.startswith("@@") for line in lines[2:]
    )


def looks_like_truncating_prefix_rewrite(before: str, after: str) -> bool:
    stripped_before = before.rstrip()
    stripped_after = after.rstrip()
    if not stripped_after or stripped_after == stripped_before:
        return False
    return stripped_before.startswith(stripped_after)


def extract_code_content(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3:
            return "\n".join(lines[1:-1]).strip() + "\n"
    return stripped + ("\n" if stripped and not stripped.endswith("\n") else "")


def strip_code_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3:
            return "\n".join(lines[1:-1]).strip()
    return stripped
