from __future__ import annotations

from pathlib import Path

from repo_brain.config import state_dir
from repo_brain.models import RiskFinding, RiskReport
from repo_brain.patches import parse_patch
from repo_brain.storage.sqlite_store import SQLiteStore

SECURITY_AREAS = {
    "authentication": ("auth", "authorization"),
    "cryptography": ("crypto",),
    "migration": ("migration",),
    "billing": ("billing",),
    "secrets": ("secret", "password", "token"),
}
CONFIG_NAMES = {
    "pyproject.toml",
    "package.json",
    "package-lock.json",
    "uv.lock",
    "requirements.txt",
    "tsconfig.json",
}


def score_risk(root: Path, patch_text: str) -> RiskReport:
    info = parse_patch(patch_text)
    findings: list[RiskFinding] = []

    def add(code: str, severity: str, score: int, message: str, paths: tuple[str, ...]) -> None:
        findings.append(RiskFinding(code, severity, score, message, paths))

    paths = info.files
    lower_paths = " ".join(paths).lower()
    if any(Path(path).name in CONFIG_NAMES for path in paths):
        add("config", "medium", 12, "Dependency or build configuration changed.", paths)
    for area, markers in SECURITY_AREAS.items():
        if any(marker in lower_paths for marker in markers):
            add(
                f"sensitive_{area}",
                "high",
                20,
                f"{area.title()} area changed.",
                paths,
            )
    if len(paths) > 5:
        add("many_files", "medium", 8, "Patch changes more than five files.", paths)
    if info.added_lines + info.deleted_lines > 200:
        add("large_patch", "medium", 8, "Patch changes more than 200 lines.", paths)
    if "/dev/null" in patch_text:
        add("add_delete", "medium", 8, "Patch adds or deletes files.", paths)
    if any("generated" in path or "vendor" in path for path in paths):
        add("generated", "medium", 10, "Generated or vendor content changed.", paths)
    test_only = bool(paths) and all(
        Path(path).name.startswith("test_")
        or ".test." in path
        or ".spec." in path
        or "__tests__" in Path(path).parts
        for path in paths
    )
    if test_only:
        add("test_only", "low", -10, "Patch changes tests only.", paths)
    affected_symbols: tuple[str, ...] = ()
    impacted_tests: tuple[str, ...] = ()
    confidence = 0.35
    database = state_dir(root) / "index.sqlite3"
    if database.exists():
        store = SQLiteStore(database)
        symbols = [symbol for symbol in store.symbols() if symbol.file_path in paths]
        affected_symbols = tuple(symbol.qualified_name for symbol in symbols)
        public = [symbol for symbol in symbols if not symbol.name.startswith("_")]
        if public:
            add(
                "public_api",
                "high",
                15,
                "Public symbols are affected.",
                tuple(sorted({symbol.file_path for symbol in public})),
            )
        tests = [test.command for test in store.tests() if test.file_path in paths]
        impacted_tests = tuple(tests)
        if not test_only and not tests:
            add("no_linked_tests", "medium", 12, "No linked tests changed.", paths)
        confidence = min(1.0, 0.5 + (0.4 if symbols else 0.0))
    score = max(0, min(100, sum(finding.score for finding in findings)))
    band = "low" if score < 25 else "medium" if score < 50 else "high" if score < 75 else "critical"
    return RiskReport(
        score,
        band,
        confidence,
        tuple(sorted(findings, key=lambda finding: (-finding.score, finding.code))),
        affected_symbols,
        impacted_tests,
    )
