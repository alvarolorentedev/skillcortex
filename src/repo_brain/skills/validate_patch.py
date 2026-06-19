from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from repo_brain.analyzers.javascript import JavaScriptAnalyzer
from repo_brain.commands import run_command
from repo_brain.models import ValidationCheck, ValidationReport
from repo_brain.patches import parse_patch
from repo_brain.skills.suggest_tests import suggest_tests


def _ignore(_: str, names: list[str]) -> set[str]:
    return {
        name
        for name in names
        if name
        in {
            ".git",
            ".repo-brain",
            ".venv",
            "node_modules",
            "__pycache__",
            ".pytest_cache",
        }
    }


def validate_patch(
    root: Path,
    patch_path: Path,
    *,
    run_tests: bool = False,
    command: str | None = None,
) -> ValidationReport:
    try:
        text = patch_path.read_text()
    except OSError as exc:
        check = ValidationCheck("patch", "", "failed", 0, str(exc))
        return ValidationReport(False, (check,), (), False)
    info = parse_patch(text)
    if info.unsafe_reason:
        check = ValidationCheck("patch_safety", "", "failed", 0, info.unsafe_reason)
        return ValidationReport(False, (check,), info.files, False)
    checks: list[ValidationCheck] = []
    preflight = run_command(["git", "apply", "--check", str(patch_path.resolve())], root)
    checks.append(
        ValidationCheck(
            "git_apply_check",
            preflight.command,
            preflight.status,
            preflight.duration_ms,
            preflight.output,
        )
    )
    if preflight.status != "passed":
        return ValidationReport(False, tuple(checks), info.files, False)
    with tempfile.TemporaryDirectory(prefix="repo-brain-") as temporary:
        sandbox = Path(temporary) / "repo"
        shutil.copytree(root, sandbox, ignore=_ignore)
        applied = run_command(["git", "apply", str(patch_path.resolve())], sandbox)
        checks.append(
            ValidationCheck(
                "apply_sandbox",
                applied.command,
                applied.status,
                applied.duration_ms,
                applied.output,
            )
        )
        if applied.status != "passed":
            return ValidationReport(False, tuple(checks), info.files, False)
        python_files = [path for path in info.files if path.endswith((".py", ".pyi"))]
        if python_files:
            syntax = run_command(["python", "-m", "py_compile", *python_files], sandbox)
            checks.append(
                ValidationCheck(
                    "python_syntax",
                    syntax.command,
                    syntax.status,
                    syntax.duration_ms,
                    syntax.output,
                )
            )
        javascript = JavaScriptAnalyzer()
        for relative in info.files:
            if not relative.endswith((".js", ".jsx", ".ts", ".tsx")):
                continue
            path = sandbox / relative
            analysis = javascript.analyze(path, path.read_text(errors="replace"))
            status = "failed" if analysis.diagnostics else "passed"
            checks.append(
                ValidationCheck(
                    "javascript_syntax",
                    f"tree-sitter {relative}",
                    status,
                    0,
                    "\n".join(analysis.diagnostics),
                )
            )
        commands: list[str] = []
        if run_tests:
            commands.extend(
                str(item["command"]) for item in suggest_tests(root, patch_text=text)[:5]
            )
        if command:
            commands.append(command)
        checks.extend(run_command(item, sandbox) for item in commands)
    passed = all(check.status == "passed" for check in checks)
    return ValidationReport(True, tuple(checks), info.files, passed)

