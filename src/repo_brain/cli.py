from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from repo_brain import __version__
from repo_brain.indexer import index_repository
from repo_brain.repository import RepositoryError, resolve_repository
from repo_brain.skills.evidence import build_evidence
from repo_brain.skills.explain_failure import explain_failure
from repo_brain.skills.localize import localize
from repo_brain.skills.repo_map import repository_map
from repo_brain.skills.suggest_tests import suggest_tests
from repo_brain.skills.validate_patch import validate_patch


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="repo-brain")
    parser.add_argument("--repo", type=Path)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    index = subparsers.add_parser("index")
    index.add_argument("path", nargs="?", type=Path)
    subparsers.add_parser("map")
    localization = subparsers.add_parser("localize")
    localization.add_argument("task")
    evidence = subparsers.add_parser("evidence")
    evidence.add_argument("task")
    evidence.add_argument("--max-chars", type=int, default=16_000)
    test_suggestions = subparsers.add_parser("suggest-tests")
    test_suggestions.add_argument("task")
    test_suggestions.add_argument("--patch", type=Path)
    validation = subparsers.add_parser("validate")
    validation.add_argument("--patch", type=Path, required=True)
    validation.add_argument("--run-tests", action="store_true")
    validation.add_argument("--command", dest="explicit_command")
    failure = subparsers.add_parser("explain-failure")
    failure.add_argument("--log", type=Path, required=True)
    return parser


def _envelope(
    command: str,
    repository: str,
    data: object,
    warnings: list[str],
    errors: list[str],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "command": command,
        "repository": repository,
        "data": data,
        "warnings": warnings,
        "errors": errors,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    selected = getattr(args, "path", None) or args.repo
    try:
        root = resolve_repository(selected)
    except RepositoryError as exc:
        print(str(exc), file=sys.stderr)
        return 3
    if args.command == "map":
        map_data = repository_map(root)
        if args.json:
            print(json.dumps(_envelope("map", str(root), map_data, [], []), sort_keys=True))
        else:
            print(json.dumps(map_data, indent=2, sort_keys=True))
        return 0
    if args.command == "localize":
        localization_data = [asdict(item) for item in localize(root, args.task)]
        if args.json:
            print(
                json.dumps(
                    _envelope("localize", str(root), localization_data, [], []),
                    sort_keys=True,
                )
            )
        else:
            for item in localization_data:
                print(f"{item['score']:>6}  {item['path']}  ({'; '.join(item['reasons'])})")
        return 0
    if args.command == "evidence":
        evidence_data = build_evidence(root, args.task, args.max_chars)
        if args.json:
            print(
                json.dumps(
                    _envelope(
                        "evidence",
                        str(root),
                        asdict(evidence_data.bundle),
                        list(evidence_data.warnings),
                        [],
                    ),
                    sort_keys=True,
                )
            )
        else:
            print(evidence_data.render())
        return 0
    if args.command == "suggest-tests":
        patch_text = args.patch.read_text() if args.patch else None
        suggestions = suggest_tests(root, args.task, patch_text)
        if args.json:
            print(
                json.dumps(
                    _envelope("suggest-tests", str(root), suggestions, [], []),
                    sort_keys=True,
                )
            )
        else:
            for item in suggestions:
                print(f"{item['command']}  # {item['reason']}")
        return 0
    if args.command == "validate":
        report = validate_patch(
            root,
            args.patch,
            run_tests=args.run_tests,
            command=args.explicit_command,
        )
        report_data = asdict(report)
        if args.json:
            print(
                json.dumps(
                    _envelope("validate", str(root), report_data, [], []),
                    sort_keys=True,
                )
            )
        else:
            print(json.dumps(report_data, indent=2, sort_keys=True))
        return 0 if report.passed else 5
    if args.command == "explain-failure":
        try:
            log = args.log.read_text(errors="replace")
        except OSError as exc:
            print(str(exc), file=sys.stderr)
            return 4
        explanation = explain_failure(root, log)
        explanation_data = asdict(explanation)
        if args.json:
            print(
                json.dumps(
                    _envelope("explain-failure", str(root), explanation_data, [], []),
                    sort_keys=True,
                )
            )
        else:
            print(json.dumps(explanation_data, indent=2, sort_keys=True))
        return 0
    result = index_repository(root)
    data: dict[str, Any] = asdict(result)
    warnings = list(result.warnings)
    errors = list(result.errors)
    data.pop("warnings")
    data.pop("errors")
    if args.json:
        print(json.dumps(_envelope("index", str(root), data, warnings, errors), sort_keys=True))
    elif errors:
        print("\n".join(errors), file=sys.stderr)
    else:
        print(
            f"indexed {data['scanned']} files: {data['added']} added, "
            f"{data['updated']} updated, {data['unchanged']} unchanged, "
            f"{data['removed']} removed, {data['ignored']} ignored, {data['failed']} failed"
        )
    return 3 if errors else 0
