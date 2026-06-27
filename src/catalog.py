from __future__ import annotations

import json
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .agent.sandbox import SKIP_DIRS
from .composer import compose_skill_packages
from .runtime import validate_runtime_bundle
from .shared.io import read_json, read_yaml


MAX_REPO_FILES = 200
MAX_BYTES_PER_FILE = 16_384
MAX_TOTAL_BYTES = 262_144
ROUTE_THRESHOLD = 0.25
EXTRA_SKIP_DIRS = {"node_modules", "venv", "dist", "build", ".mypy_cache"}
REPO_SKIP_DIRS = set(SKIP_DIRS) | EXTRA_SKIP_DIRS
FRAMEWORK_SIGNALS = {
    "fastapi",
    "pydantic",
    "pytest",
    "django",
    "flask",
    "react",
    "next",
    "vue",
    "sqlalchemy",
}
LANGUAGE_BY_SUFFIX = {
    ".py": "python",
    ".pyi": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".sql": "sql",
}


@dataclass(slots=True)
class RoutingCard:
    summary: str = ""
    embedding_text: str = ""
    positive_examples: list[str] = field(default_factory=list)
    negative_examples: list[str] = field(default_factory=list)
    observed_success_contexts: list[dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class SkillRecord:
    skill_id: str
    name: str
    path: Path
    description: str = ""
    capabilities: list[str] = field(default_factory=list)
    activation_cues: list[str] = field(default_factory=list)
    avoid_when: list[str] = field(default_factory=list)
    task_type_hint: str | None = None
    base_model: str | None = None
    adapter_path: Path | None = None
    routing_card: RoutingCard = field(default_factory=RoutingCard)
    eval_summary: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class CatalogResult:
    skills: list[SkillRecord]
    errors: list[str]
    warnings: list[str]


class SkillCatalog:
    @staticmethod
    def discover(skills_dir: Path) -> CatalogResult:
        root = skills_dir.resolve()
        if not root.exists() or not root.is_dir():
            raise FileNotFoundError(f"skills directory not found: {root}")
        skills: list[SkillRecord] = []
        errors: list[str] = []
        warnings: list[str] = []
        for package in sorted(path for path in root.iterdir() if path.is_dir()):
            manifest_path = package / "skill.yaml"
            if not manifest_path.exists():
                errors.append(f"{package.name}: missing skill.yaml")
                continue
            try:
                manifest = read_yaml(manifest_path)
                skills.append(_skill_from_manifest(package, manifest, warnings))
            except ValueError as error:
                errors.append(f"{package.name}: {error}")
        return CatalogResult(skills=skills, errors=errors, warnings=warnings)


def route_task(
    *,
    skills_dir: Path,
    repo: Path,
    task: str,
    explain: bool = False,
    current_base_model: str | None = None,
) -> dict[str, Any]:
    catalog = SkillCatalog.discover(skills_dir)
    repo_root = repo.resolve()
    if not repo_root.exists() or not repo_root.is_dir():
        raise FileNotFoundError(f"repo not found: {repo_root}")
    repo_context = scan_repo_context(repo_root)
    candidates = [
        _candidate(skill, task, repo_context, current_base_model=current_base_model)
        for skill in catalog.skills
    ]
    candidates.sort(key=lambda item: (-item["score"], item["skill_id"]))
    selected = []
    for item in candidates:
        if item["compatible"] and item["score"] >= ROUTE_THRESHOLD:
            item["selected"] = True
            selected.append(
                {
                    "skill_id": item["skill_id"],
                    "score": item["score"],
                    "reason": "Strongest capability match.",
                }
            )
            break
    if not explain:
        for item in candidates:
            item["score_breakdown"] = {}
    return {
        "routing_mode": "capability",
        "skills_dir": str(skills_dir.resolve()),
        "repo": str(repo_root),
        "task": task,
        "repo_context": repo_context,
        "selected_skills": selected,
        "candidates": candidates,
        "fallback": "base",
        "errors": catalog.errors,
        "warnings": catalog.warnings,
    }


def compose_from_route(
    *,
    skills_dir: Path,
    repo: Path,
    task: str,
    runtime_out: Path,
    explain: bool = False,
    allow_base: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    routing_decision = route_task(
        skills_dir=skills_dir,
        repo=repo,
        task=task,
        explain=explain,
    )
    catalog = SkillCatalog.discover(skills_dir)
    by_skill_id = {skill.skill_id: skill for skill in catalog.skills}
    selected = list(routing_decision["selected_skills"])
    warnings = list(routing_decision["warnings"])
    errors = list(routing_decision["errors"])
    if not selected:
        if not allow_base:
            raise ValueError("no skill selected; pass --allow-base or improve routing metadata")
        warnings.append("base fallback allowed; no runtime was composed")
        return {
            "routing_decision": routing_decision,
            "selected_skills": [],
            "runtime_out": None,
            "composition_strategy": "routed",
            "composition_status": "skipped",
            "validation_status": "not_run",
            "warnings": warnings,
            "errors": errors,
        }

    selected_paths = []
    for item in selected:
        skill_id = item["skill_id"]
        skill = by_skill_id.get(skill_id)
        if skill is None:
            raise ValueError(f"selected skill not found in catalog: {skill_id}")
        selected_paths.append(skill.path)

    try:
        compose_skill_packages(
            skills=selected_paths,
            strategy="routed",
            output=runtime_out,
            force=overwrite,
        )
    except FileExistsError:
        raise
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        context = ", ".join(
            f"{item['skill_id']}={path}" for item, path in zip(selected, selected_paths, strict=True)
        )
        raise ValueError(f"selected skill package is not composable ({context}): {error}") from error

    validation = validate_runtime_bundle(runtime_out)
    validation_status = validation.get("status")
    if validation_status not in {"valid", "passed"}:
        raise ValueError(f"runtime validation failed: {validation_status}")
    return {
        "routing_decision": routing_decision,
        "selected_skills": [str(path) for path in selected_paths],
        "runtime_out": str(runtime_out.resolve()),
        "composition_strategy": "routed",
        "composition_status": "written",
        "validation_status": "passed",
        "warnings": warnings,
        "errors": errors,
    }

def scan_repo_context(repo: Path) -> dict[str, Any]:
    language_signals: set[str] = set()
    framework_signals: set[str] = set()
    scanned_files: list[str] = []
    total_bytes = 0
    skipped_binary = 0
    for path in sorted(repo.rglob("*")):
        if len(scanned_files) >= MAX_REPO_FILES or total_bytes >= MAX_TOTAL_BYTES:
            break
        if path.is_dir() or any(part in REPO_SKIP_DIRS for part in path.relative_to(repo).parts):
            continue
        relative = path.relative_to(repo).as_posix()
        suffix = path.suffix.lower()
        if suffix in LANGUAGE_BY_SUFFIX:
            language_signals.add(LANGUAGE_BY_SUFFIX[suffix])
        try:
            raw = path.read_bytes()[:MAX_BYTES_PER_FILE]
        except OSError:
            continue
        if b"\x00" in raw:
            skipped_binary += 1
            continue
        total_bytes += len(raw)
        text = raw.decode("utf-8", errors="ignore").lower()
        scanned_files.append(relative)
        _collect_frameworks(relative.lower(), text, framework_signals)
    return {
        "language_signals": sorted(language_signals),
        "framework_signals": sorted(framework_signals),
        "files_scanned": len(scanned_files),
        "bytes_scanned": total_bytes,
        "skipped_binary_files": skipped_binary,
        "scan_limits": {
            "max_files": MAX_REPO_FILES,
            "max_bytes_per_file": MAX_BYTES_PER_FILE,
            "max_total_bytes": MAX_TOTAL_BYTES,
        },
        "scanned_files": scanned_files,
    }


def _skill_from_manifest(package: Path, manifest: dict[str, Any], warnings: list[str]) -> SkillRecord:
    skill_id = _required_text(manifest, "skill_id")
    name = _required_text(manifest, "name")
    base = manifest.get("base") or {}
    adapter = manifest.get("adapter") or {}
    task_type_hint = manifest.get("task_type_hint") or manifest.get("task_type")
    composition = manifest.get("composition") or {}
    allowed_task_types = ((composition.get("capabilities") or {}).get("allowed_task_types") or [])
    if not task_type_hint and len(allowed_task_types) == 1:
        task_type_hint = allowed_task_types[0]
    routing_card = _load_routing_card(package / "routing_card.json", warnings)
    eval_summary = _load_optional_json(package / "eval_summary.json", warnings)
    return SkillRecord(
        skill_id=skill_id,
        name=name,
        path=package.resolve(),
        description=str(manifest.get("description") or ""),
        capabilities=_text_list(manifest.get("capabilities"), f"{skill_id}: capabilities", warnings),
        activation_cues=_text_list(manifest.get("activation_cues"), f"{skill_id}: activation_cues", warnings),
        avoid_when=_text_list(manifest.get("avoid_when"), f"{skill_id}: avoid_when", warnings),
        task_type_hint=str(task_type_hint) if task_type_hint else None,
        base_model=manifest.get("base_model") or base.get("runtime_model") or base.get("source_model"),
        adapter_path=package / (manifest.get("adapter_path") or adapter.get("path") or "adapter"),
        routing_card=routing_card,
        eval_summary=eval_summary,
    )


def _candidate(
    skill: SkillRecord,
    task: str,
    repo_context: dict[str, Any],
    *,
    current_base_model: str | None,
) -> dict[str, Any]:
    task_text = task.lower()
    repo_signals = set(repo_context["language_signals"]) | set(repo_context["framework_signals"])
    matched: set[str] = set()
    negative: set[str] = set()
    breakdown = {
        "task": 0.0,
        "repo": 0.0,
        "positive_examples": 0.0,
        "negative": 0.0,
        "eval": 0.0,
        "task_type_hint": 0.0,
        "base_model": 0.0,
    }
    positive_terms = [skill.description, *skill.capabilities, *skill.activation_cues]
    for term in positive_terms:
        signal = _matching_signal(term, task_text)
        if signal:
            matched.add(signal)
            breakdown["task"] += 0.12
        repo_signal = _repo_signal(term, repo_signals)
        if repo_signal:
            matched.add(repo_signal)
            breakdown["repo"] += 0.08
    for example in skill.routing_card.positive_examples:
        signal = _matching_signal(example, task_text)
        if signal:
            matched.add(signal)
            breakdown["positive_examples"] += 0.08
    for term in [*skill.avoid_when, *skill.routing_card.negative_examples]:
        signal = _matching_signal(term, task_text)
        if signal:
            negative.add(signal)
            breakdown["negative"] -= 0.18
    if skill.task_type_hint and _matching_signal(skill.task_type_hint.replace("_", " "), task_text):
        matched.add(skill.task_type_hint)
        breakdown["task_type_hint"] = 0.04
    breakdown["eval"] = _eval_bonus(skill.eval_summary)
    compatible = True
    if current_base_model and skill.base_model and current_base_model != skill.base_model:
        compatible = False
        negative.add("base_model")
        breakdown["base_model"] = -1.0
    score = max(0.0, min(1.0, sum(breakdown.values())))
    if not compatible:
        score = 0.0
    reason = _reason(score, compatible, matched, negative)
    return {
        "skill_id": skill.skill_id,
        "score": round(score, 4),
        "selected": False,
        "compatible": compatible,
        "matched_signals": sorted(matched),
        "negative_signals": sorted(negative),
        "score_breakdown": {key: round(value, 4) for key, value in breakdown.items() if value},
        "reason": reason,
    }


def _collect_frameworks(relative: str, text: str, signals: set[str]) -> None:
    if relative.endswith("pyproject.toml"):
        try:
            payload = tomllib.loads(text)
        except tomllib.TOMLDecodeError:
            payload = {}
        text = f"{text} {json.dumps(payload)}"
    for signal in FRAMEWORK_SIGNALS:
        if signal in text or signal in relative:
            signals.add(signal)


def _required_text(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value.strip()


def _text_list(value: Any, label: str, warnings: list[str]) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        warnings.append(f"{label} must be a list of strings; ignoring")
        return []
    return [item.strip() for item in value if item.strip()]


def _load_routing_card(path: Path, warnings: list[str]) -> RoutingCard:
    payload = _load_optional_json(path, warnings)
    return RoutingCard(
        summary=str(payload.get("summary") or ""),
        embedding_text=str(payload.get("embedding_text") or ""),
        positive_examples=_text_list(payload.get("positive_examples"), f"{path}: positive_examples", warnings),
        negative_examples=_text_list(payload.get("negative_examples"), f"{path}: negative_examples", warnings),
        observed_success_contexts=list(payload.get("observed_success_contexts") or []),
    )


def _load_optional_json(path: Path, warnings: list[str]) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return read_json(path)
    except ValueError as error:
        warnings.append(f"{path.name}: {error}")
        return {}


def _matching_signal(term: str, text: str) -> str | None:
    words = _words(term)
    if not words:
        return None
    if len(words) > 1 and " ".join(words) in text:
        return " ".join(words)
    for word in words:
        if len(word) >= 3 and re.search(rf"\b{re.escape(word)}\b", text):
            return word
    return None


def _repo_signal(term: str, repo_signals: set[str]) -> str | None:
    for word in _words(term):
        if word in repo_signals:
            return word
    return None


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _eval_bonus(summary: dict[str, Any]) -> float:
    values = []
    for mode in (summary.get("modes") or {}).values():
        if isinstance(mode, dict):
            for key in ("validation_pass_rate", "execution_pass_rate", "fuzzy_score"):
                value = mode.get(key)
                if isinstance(value, int | float):
                    values.append(float(value))
    return min(values or [0.0]) * 0.05


def _reason(score: float, compatible: bool, matched: set[str], negative: set[str]) -> str:
    if not compatible:
        return "Skill base model is incompatible with the current base model."
    if score >= ROUTE_THRESHOLD and matched:
        return "Matched capability signals: " + ", ".join(sorted(matched)[:6]) + "."
    if negative:
        return "Negative routing signals outweighed capability matches."
    return "Insufficient capability evidence for this task."
