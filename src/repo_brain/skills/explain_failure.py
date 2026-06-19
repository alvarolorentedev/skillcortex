from __future__ import annotations

import re
from pathlib import Path

from repo_brain.models import FailureExplanation

PYTHON_FRAME = re.compile(r'File "([^"]+)", line (\d+), in ([^\n]+)')
TYPESCRIPT = re.compile(r"([^\s(]+\.[cm]?[jt]sx?)\((\d+),(\d+)\): error (TS\d+): (.+)")
PYTEST = re.compile(r"^E\s+(.+)$", re.MULTILINE)


def explain_failure(root: Path, log: str) -> FailureExplanation:
    del root
    frames: list[dict[str, object]] = []
    if "Traceback (most recent call last):" in log:
        for path, line, function in PYTHON_FRAME.findall(log):
            frames.append({"path": path, "line": int(line), "function": function.strip()})
        message = next(
            (
                line.strip()
                for line in reversed(log.splitlines())
                if re.match(r"^[A-Za-z_][\w.]*Error:", line.strip())
            ),
            "Python exception",
        )
        return FailureExplanation(
            "python_exception",
            message,
            tuple(frames),
            (),
            ("Inspect the first application frame.", "Run the smallest reproducing test."),
        )
    match = TYPESCRIPT.search(log)
    if match:
        path, line, column, code, message = match.groups()
        frames.append({"path": path, "line": int(line), "column": int(column)})
        return FailureExplanation(
            "typescript",
            f"{code}: {message}",
            tuple(frames),
            (),
            ("Inspect the reported type assignment.",),
        )
    pytest_match = PYTEST.search(log)
    if pytest_match:
        return FailureExplanation(
            "test_failure",
            pytest_match.group(1),
            (),
            (),
            ("Run the failing test alone with verbose output.",),
        )
    lines = [line.strip() for line in log.splitlines() if line.strip()]
    return FailureExplanation(
        "unknown",
        lines[-1] if lines else "No failure message found",
        (),
        (),
        ("Collect a complete compiler or test log.",),
    )
