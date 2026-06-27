from __future__ import annotations

from .sandbox import ARTIFACT_DIR_PREFIXES, CODE_FILE_SUFFIXES, TEXT_FILE_SUFFIXES


def choose_patch_target(files: list[str], task: str) -> str:
    for path in preferred_source_files(files):
        if path.endswith(".py") and not path.startswith("tests/"):
            return path
    preferred = preferred_source_files(files)
    if preferred:
        return preferred[0]
    return default_new_source_path(task)


def default_read_targets(
    files: list[str],
    *,
    patch_target: str | None = None,
    prior_task_results: list[dict] | None = None,
) -> list[str]:
    preferred = preferred_source_files(files)
    targets: list[str] = []
    for candidate in [patch_target, *recent_changed_files(prior_task_results or [])]:
        if candidate and candidate in preferred and candidate not in targets:
            targets.append(candidate)
    for candidate in preferred:
        if candidate not in targets:
            targets.append(candidate)
        if len(targets) >= 5:
            break
    return targets


def recent_changed_files(task_results: list[dict], *, limit: int = 3) -> list[str]:
    changed: list[str] = []
    for result in reversed(task_results):
        for path in result["final_materialization"].get("files_changed") or []:
            if path not in changed:
                changed.append(path)
            if len(changed) >= limit:
                return changed
    return changed


def preferred_source_files(files: list[str]) -> list[str]:
    source_files = [path for path in files if path.endswith(TEXT_FILE_SUFFIXES) and not is_artifact_path(path)]
    if source_files:
        code_files = [path for path in source_files if path.endswith(CODE_FILE_SUFFIXES)]
        return code_files + [path for path in source_files if path not in code_files]
    return []


def is_artifact_path(path: str) -> bool:
    return path.startswith(ARTIFACT_DIR_PREFIXES)


def default_new_source_path(task: str) -> str:
    lowered = task.lower()
    if any(token in lowered for token in ("fastapi", "endpoint", "router", "api")):
        return "app.py"
    return "main.py"
