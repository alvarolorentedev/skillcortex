import argparse
import json
import os
import subprocess
import sys
import tempfile
import venv
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def _venv_python(venv_root: Path) -> Path:
    if os.name == "nt":
        return venv_root / "Scripts" / "python.exe"
    return venv_root / "bin" / "python"


def _run(name: str, command: list[str], *, cwd: Path | None = None) -> dict:
    completed = subprocess.run(
        command,
        cwd=cwd or ROOT,
        capture_output=True,
        text=True,
    )
    record = {
        "name": name,
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }
    if completed.returncode != 0:
        raise RuntimeError(json.dumps(record, indent=2))
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Create an isolated virtual environment, install Slm Cortex, and launch the Composer-first entry point.",
    )
    parser.add_argument("--package-source", default=str(ROOT))
    parser.add_argument("--workspace-root")
    parser.add_argument("--install-root")
    parsed = parser.parse_args(argv)

    install_root = (
        Path(parsed.install_root).resolve()
        if parsed.install_root
        else Path(tempfile.mkdtemp(prefix="slmcortex-install-"))
    )
    workspace_root = (
        Path(parsed.workspace_root).resolve()
        if parsed.workspace_root
        else Path(tempfile.mkdtemp(prefix="slmcortex-installed-workspace-"))
    )
    venv_root = install_root / "venv"
    venv.EnvBuilder(with_pip=True, clear=True).create(venv_root)
    python = _venv_python(venv_root)

    steps = [
        _run("install_package", [str(python), "-m", "pip", "install", parsed.package_source]),
        _run("launch_help", [str(python), "-m", "slmcortex", "--help"]),
        _run(
            "doctor",
            [str(python), "-m", "slmcortex", "doctor", "--workspace", str(workspace_root)],
        ),
    ]

    summary = {
        "status": "complete",
        "install_root": str(install_root),
        "workspace_root": str(workspace_root),
        "steps": [
            {
                "name": step["name"],
                "command": step["command"],
            }
            for step in steps
        ],
    }
    json.dump(summary, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())