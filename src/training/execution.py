from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from .types import ExecutionFixture


def run_fixture(fixture: ExecutionFixture, generated_code: str) -> tuple[bool, str]:
    with tempfile.TemporaryDirectory(prefix="slmcortex-eval-") as directory:
        root = Path(directory)
        generated_filename = (
            "test_generated.py" if "solution.py" in fixture.files else "solution.py"
        )
        (root / generated_filename).write_text(generated_code)
        for name, content in fixture.files.items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
        try:
            command = list(fixture.command)
            if (
                command[:3] == ["python", "-m", "pytest"]
                and importlib.util.find_spec("pytest") is None
                and shutil.which("uv")
            ):
                command = [
                    shutil.which("uv"),
                    "run",
                    "--project",
                    str(Path(__file__).resolve().parents[2]),
                    "--extra",
                    "test",
                    *command,
                ]
            elif command[0] == "python":
                command[0] = sys.executable
            result = subprocess.run(
                command,
                cwd=root,
                capture_output=True,
                text=True,
                timeout=fixture.timeout_seconds,
                check=False,
            )
            return result.returncode == 0, result.stdout + result.stderr
        except subprocess.TimeoutExpired:
            return False, "execution timed out"
