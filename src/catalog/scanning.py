from __future__ import annotations

import json
import tomllib
from pathlib import Path
from typing import Any

from ..agent.sandbox import SKIP_DIRS


MAX_REPO_FILES = 200
MAX_BYTES_PER_FILE = 16_384
MAX_TOTAL_BYTES = 262_144
EXTRA_SKIP_DIRS = {"node_modules", "venv", "dist", "build", ".mypy_cache"}
REPO_SKIP_DIRS = set(SKIP_DIRS) | EXTRA_SKIP_DIRS
FRAMEWORK_SIGNALS = {
    "fastapi",
    "pydantic",
    "pytest",
    "django",
    "flask",
    "react",
    "next",
    "vue",
    "sqlalchemy",
}
LANGUAGE_BY_SUFFIX = {
    ".py": "python",
    ".pyi": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".sql": "sql",
}


def scan_repo_context(repo: Path) -> dict[str, Any]:
    language_signals: set[str] = set()
    framework_signals: set[str] = set()
    scanned_files: list[str] = []
    total_bytes = 0
    skipped_binary = 0
    for path in sorted(repo.rglob("*")):
        if len(scanned_files) >= MAX_REPO_FILES or total_bytes >= MAX_TOTAL_BYTES:
            break
        if path.is_dir() or any(part in REPO_SKIP_DIRS for part in path.relative_to(repo).parts):
            continue
        relative = path.relative_to(repo).as_posix()
        suffix = path.suffix.lower()
        if suffix in LANGUAGE_BY_SUFFIX:
            language_signals.add(LANGUAGE_BY_SUFFIX[suffix])
        try:
            raw = path.read_bytes()[:MAX_BYTES_PER_FILE]
        except OSError:
            continue
        if b"\x00" in raw:
            skipped_binary += 1
            continue
        total_bytes += len(raw)
        text = raw.decode("utf-8", errors="ignore").lower()
        scanned_files.append(relative)
        _collect_frameworks(relative.lower(), text, framework_signals)
    return {
        "language_signals": sorted(language_signals),
        "framework_signals": sorted(framework_signals),
        "files_scanned": len(scanned_files),
        "bytes_scanned": total_bytes,
        "skipped_binary_files": skipped_binary,
        "scan_limits": {
            "max_files": MAX_REPO_FILES,
            "max_bytes_per_file": MAX_BYTES_PER_FILE,
            "max_total_bytes": MAX_TOTAL_BYTES,
        },
        "scanned_files": scanned_files,
    }


def infer_task_hints(repo_context: dict[str, Any]) -> list[dict[str, str]]:
    frameworks = set(repo_context.get("framework_signals") or [])
    languages = set(repo_context.get("language_signals") or [])
    hints: list[dict[str, str]] = []
    if "fastapi" in frameworks:
        hints.append(
            {
                "label": "api-backend",
                "task_type": "python_generation",
                "suggested_task": "Create or update a FastAPI endpoint with request validation.",
            }
        )
    if "react" in frameworks or "next" in frameworks:
        hints.append(
            {
                "label": "frontend-ui",
                "task_type": "python_generation",
                "suggested_task": "Update the UI flow while preserving the existing component contract.",
            }
        )
    if "python" in languages and not hints:
        hints.append(
            {
                "label": "python-general",
                "task_type": "python_generation",
                "suggested_task": "Inspect the folder and compose the best available Python coding runtime.",
            }
        )
    if not hints:
        hints.append(
            {
                "label": "generic",
                "task_type": "python_generation",
                "suggested_task": "Inspect the folder and compose the best available coding runtime.",
            }
        )
    return hints


def _collect_frameworks(relative: str, text: str, signals: set[str]) -> None:
    if relative.endswith("pyproject.toml"):
        try:
            payload = tomllib.loads(text)
        except tomllib.TOMLDecodeError:
            payload = {}
        text = f"{text} {json.dumps(payload)}"
    for signal in FRAMEWORK_SIGNALS:
        if signal in text or signal in relative:
            signals.add(signal)
