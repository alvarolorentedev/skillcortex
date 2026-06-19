from __future__ import annotations

from pathlib import Path

from repo_brain.config import state_dir
from repo_brain.patches import parse_patch
from repo_brain.skills.localize import localize
from repo_brain.storage.sqlite_store import SQLiteStore


def suggest_tests(
    root: Path, task: str = "", patch_text: str | None = None
) -> list[dict[str, object]]:
    store = SQLiteStore(state_dir(root) / "index.sqlite3")
    candidates = localize(root, task) if task else []
    paths = {candidate.path for candidate in candidates[:20]}
    if patch_text:
        paths.update(parse_patch(patch_text).files)
    suggestions: list[dict[str, object]] = []
    for test in store.tests():
        candidate_match = test.file_path in paths
        stem_match = any(Path(path).stem.replace("test_", "") in test.file_path for path in paths)
        if not (candidate_match or stem_match or not paths):
            continue
        suggestions.append(
            {
                "command": test.command,
                "reason": "localized test" if candidate_match else "matching source/test name",
                "scope": "focused",
                "confidence": 0.9 if candidate_match else 0.65,
                "estimated_cost": "low",
            }
        )
    return suggestions

