from __future__ import annotations

import argparse
import os
import shutil
import stat
import subprocess
import sys
import textwrap
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INSTALL_ROOT_ENV_VARS = (
    "SLMCORTEX_INSTALL_ROOT",
    "INSTALL_ROOT",
    "SLMCORTEX_HOME",
    "SLMCORTEX_PREFIX",
    "PREFIX",
    "DESTDIR",
)
LAUNCHER_SPECS = (
    ("slmcortex", "default"),
    ("slmcortex-composer", "composer"),
    ("composer", "composer"),
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Install slmcortex from a local source tree.")
    parser.add_argument("package_source", help="Path to the package source that should be installed.")
    parser.add_argument(
        "--install-root",
        help="Override the destination root used for the created virtual environment and launchers.",
    )
    return parser.parse_args()


def _resolve_install_root(parsed: argparse.Namespace) -> Path:
    if parsed.install_root:
        return Path(parsed.install_root).expanduser().resolve()
    for env_var in INSTALL_ROOT_ENV_VARS:
        value = os.environ.get(env_var)
        if value:
            return Path(value).expanduser().resolve()
    raise SystemExit(
        "Missing install root. Pass --install-root or set one of: " + ", ".join(INSTALL_ROOT_ENV_VARS)
    )


def _venv_paths(install_root: Path) -> tuple[Path, Path]:
    venv_root = install_root / "venv"
    scripts_dir = venv_root / ("Scripts" if os.name == "nt" else "bin")
    python_bin = scripts_dir / ("python.exe" if os.name == "nt" else "python")
    return scripts_dir, python_bin


def _run(command: list[str], cwd: Path | None = None) -> None:
    completed = subprocess.run(command, cwd=cwd, check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _make_executable(path: Path) -> None:
    current_mode = path.stat().st_mode
    path.chmod(current_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _write_runtime_helper(install_root: Path) -> None:
    helper_path = install_root / "libexec" / "launch_installed.py"
    helper_source = textwrap.dedent(
        """
        from __future__ import annotations

        import os
        import subprocess
        import sys
        from pathlib import Path

        LAUNCHER_NAMES = {
            "default": ["slmcortex", "slmcortex-composer", "composer"],
            "composer": ["slmcortex-composer", "composer", "slmcortex"],
        }


        def _venv_paths(install_root: Path) -> tuple[Path, Path]:
            scripts_dir = install_root / "venv" / ("Scripts" if os.name == "nt" else "bin")
            python_bin = scripts_dir / ("python.exe" if os.name == "nt" else "python")
            return scripts_dir, python_bin


        def main() -> int:
            mode = sys.argv[1] if len(sys.argv) > 1 else "default"
            args = sys.argv[2:] if len(sys.argv) > 2 else []
            install_root = Path(os.environ.get("SLMCORTEX_INSTALL_ROOT") or Path(__file__).resolve().parents[1]).resolve()
            scripts_dir, python_bin = _venv_paths(install_root)

            for launcher_name in LAUNCHER_NAMES.get(mode, LAUNCHER_NAMES["default"]):
                launcher_path = scripts_dir / launcher_name
                if launcher_path.exists():
                    command = [str(launcher_path), *args]
                    if mode == "composer" and launcher_name == "slmcortex":
                        command = [str(launcher_path), "composer-app", *args]
                    raise SystemExit(subprocess.call(command))
                if os.name == "nt":
                    command_path = scripts_dir / f"{launcher_name}.exe"
                    if command_path.exists():
                        command = [str(command_path), *args]
                        if mode == "composer" and launcher_name == "slmcortex":
                            command = [str(command_path), "composer-app", *args]
                        raise SystemExit(subprocess.call(command))
                    batch_path = scripts_dir / f"{launcher_name}.cmd"
                    if batch_path.exists():
                        command = [str(batch_path), *args]
                        if mode == "composer" and launcher_name == "slmcortex":
                            command = [str(batch_path), "composer-app", *args]
                        raise SystemExit(subprocess.call(command, shell=True))

            command = [str(python_bin), "-m", "slmcortex"]
            if mode == "composer":
                command.append("composer-app")
            raise SystemExit(subprocess.call([*command, *args]))


        if __name__ == "__main__":
            raise SystemExit(main())
        """
    ).lstrip()
    _write_text(helper_path, helper_source)


def _write_unix_launcher(path: Path, mode: str, root_levels_up: int) -> None:
    upward = "/".join([".."] * root_levels_up) or "."
    content = textwrap.dedent(
        f"""#!/usr/bin/env sh
        set -eu

        LAUNCHER_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
        INSTALL_ROOT=$(CDPATH= cd -- "$LAUNCHER_DIR/{upward}" && pwd)
        export SLMCORTEX_INSTALL_ROOT="${{SLMCORTEX_INSTALL_ROOT:-$INSTALL_ROOT}}"

        exec "$INSTALL_ROOT/venv/bin/python" "$INSTALL_ROOT/libexec/launch_installed.py" {mode} "$@"
        """
    )
    _write_text(path, content)
    _make_executable(path)


def _write_windows_launcher(path: Path, mode: str, root_levels_up: int) -> None:
    relative_root = "\\".join([".."] * root_levels_up) or "."
    content = textwrap.dedent(
        f"""
        $launcherDir = Split-Path -Parent $MyInvocation.MyCommand.Path
        $installRoot = (Resolve-Path (Join-Path $launcherDir "{relative_root}")).Path
        if (-not $env:SLMCORTEX_INSTALL_ROOT) {{
            $env:SLMCORTEX_INSTALL_ROOT = $installRoot
        }}

        & (Join-Path $installRoot "venv\\Scripts\\python.exe") (Join-Path $installRoot "libexec\\launch_installed.py") {mode} @args
        exit $LASTEXITCODE
        """
    ).lstrip()
    _write_text(path, content)


def _create_launchers(install_root: Path) -> None:
    bin_dir = install_root / "bin"
    scripts_dir = install_root / "Scripts"

    for launcher_name, mode in LAUNCHER_SPECS:
        _write_unix_launcher(bin_dir / launcher_name, mode, 1)
        _write_unix_launcher(install_root / launcher_name, mode, 0)
        _write_windows_launcher(scripts_dir / f"{launcher_name}.ps1", mode, 1)


def main() -> int:
    parsed = _parse_args()
    install_root = _resolve_install_root(parsed)
    package_source = Path(parsed.package_source).expanduser()
    if not package_source.is_absolute():
        package_source = (Path.cwd() / package_source).resolve()

    install_root.mkdir(parents=True, exist_ok=True)

    venv_root = install_root / "venv"
    if venv_root.exists():
        shutil.rmtree(venv_root)

    venv.EnvBuilder(with_pip=True, system_site_packages=True, clear=True).create(venv_root)
    scripts_dir, python_bin = _venv_paths(install_root)

    _run([str(python_bin), "-m", "pip", "install", "--force-reinstall", str(package_source)], cwd=ROOT)

    _write_runtime_helper(install_root)
    _create_launchers(install_root)

    print(f"Installed slmcortex into {install_root}")
    print(f"Virtual environment scripts live in {scripts_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())