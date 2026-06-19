from __future__ import annotations

import shlex
import subprocess
import time
from pathlib import Path

from repo_brain.models import ValidationCheck


def run_command(command: str | list[str], cwd: Path, timeout: int = 120) -> ValidationCheck:
    arguments = shlex.split(command) if isinstance(command, str) else command
    started = time.monotonic()
    try:
        result = subprocess.run(
            arguments,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        status = "passed" if result.returncode == 0 else "failed"
        output = (result.stdout + result.stderr)[-20_000:]
    except subprocess.TimeoutExpired as exc:
        status = "timeout"
        output = f"command timed out after {timeout}s: {exc}"
    duration = int((time.monotonic() - started) * 1000)
    return ValidationCheck("command", " ".join(arguments), status, duration, output)

