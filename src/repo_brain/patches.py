from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import PurePosixPath


@dataclass(frozen=True)
class PatchInfo:
    files: tuple[str, ...]
    added_lines: int
    deleted_lines: int
    unsafe_reason: str | None = None


def parse_patch(text: str) -> PatchInfo:
    if "GIT binary patch" in text or "Binary files " in text:
        return PatchInfo((), 0, 0, "binary patches are unsupported")
    paths: list[str] = []
    for match in re.finditer(r"^\+\+\+ (?:b/)?(.+)$", text, re.MULTILINE):
        path = match.group(1).strip()
        if path == "/dev/null":
            continue
        pure = PurePosixPath(path)
        if pure.is_absolute() or ".." in pure.parts:
            return PatchInfo((), 0, 0, f"unsafe patch path: {path}")
        paths.append(str(pure))
    if not paths:
        return PatchInfo((), 0, 0, "patch contains no supported file changes")
    added = sum(
        1
        for line in text.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    )
    deleted = sum(
        1
        for line in text.splitlines()
        if line.startswith("-") and not line.startswith("---")
    )
    return PatchInfo(tuple(dict.fromkeys(paths)), added, deleted)

