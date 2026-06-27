from __future__ import annotations

import argparse
from pathlib import Path

from ..contracts import SKILLS


COMPOSITION_SCOPES = ("task", "semantic_family")


def parser_kwargs(description: str, examples: str | None = None) -> dict:
    kwargs = {
        "description": description,
        "formatter_class": argparse.RawDescriptionHelpFormatter,
    }
    if examples:
        kwargs["epilog"] = f"Examples:\n{examples}"
    return kwargs


def csv_paths(value: str) -> list[Path]:
    return [Path(item.strip()) for item in value.split(",") if item.strip()]


def default_dataset_outputs(skill_id: str) -> tuple[Path, Path]:
    root = Path("datasets") / skill_id
    return root / "train.jsonl", root / "eval.jsonl"


def infer_payload(parsed: argparse.Namespace) -> dict:
    from ..runtime import load_chat_request

    if bool(parsed.prompt) == bool(parsed.request_file):
        raise ValueError("exactly one of --prompt or --request-file is required")
    if parsed.request_file:
        return load_chat_request(Path(parsed.request_file))
    payload = {
        "messages": ([{"role": "system", "content": parsed.system}] if parsed.system else [])
        + [{"role": "user", "content": parsed.prompt}],
        "task_type": parsed.task_type,
        "semantic_family": parsed.semantic_family,
        "skill_override": parsed.skill_override,
        "max_tokens": parsed.max_tokens,
        "temperature": parsed.temperature,
    }
    return load_chat_request_payload(payload)


def load_chat_request_payload(payload: dict) -> dict:
    from ..runtime import normalize_chat_request

    return normalize_chat_request(payload)


def package_composition(parsed: argparse.Namespace) -> dict | None:
    if not any(
        (
            parsed.allowed_task_types,
            parsed.activation_scope,
            parsed.semantic_families,
            parsed.compatible_skills,
            parsed.incompatible_skills,
        )
    ):
        return None
    if not parsed.allowed_task_types:
        raise ValueError("--allowed-task-types is required when composition metadata is provided")
    if not parsed.activation_scope:
        raise ValueError("--activation-scope is required when composition metadata is provided")
    return {
        "capabilities": {
            "allowed_task_types": list(parsed.allowed_task_types),
        },
        "activation": {
            "default_route_type": "adapter",
            "scope": parsed.activation_scope,
            "semantic_families": list(parsed.semantic_families or []),
        },
        "compatibility": {
            "compatible_skills": list(parsed.compatible_skills or []),
            "incompatible_skills": list(parsed.incompatible_skills or []),
        },
        "routing": {
            "tasks": {},
        },
    }


def train_skill_composition(parsed: argparse.Namespace) -> tuple[dict | None, dict[str, object]]:
    if not parsed.skill_id:
        return package_composition(parsed), {}

    defaults_applied: dict[str, object] = {}
    allowed_task_types = list(parsed.allowed_task_types or [])
    activation_scope = parsed.activation_scope
    if not allowed_task_types:
        allowed_task_types = ["python_generation"]
        defaults_applied["allowed_task_types"] = list(allowed_task_types)
    if activation_scope is None:
        activation_scope = "task"
        defaults_applied["activation_scope"] = activation_scope
    return {
        "capabilities": {
            "allowed_task_types": allowed_task_types,
        },
        "activation": {
            "default_route_type": "adapter",
            "scope": activation_scope,
            "semantic_families": list(parsed.semantic_families or []),
        },
        "compatibility": {
            "compatible_skills": list(parsed.compatible_skills or []),
            "incompatible_skills": list(parsed.incompatible_skills or []),
        },
        "routing": {
            "tasks": {},
        },
    }, defaults_applied


def resolve_train_skill(parsed: argparse.Namespace) -> tuple[str, str, dict | None, dict[str, object]]:
    preset_skill = parsed.skill
    explicit_skill_id = parsed.skill_id
    if explicit_skill_id and preset_skill and explicit_skill_id != preset_skill:
        raise ValueError("provide either a preset positional skill or a matching --skill-id")
    if explicit_skill_id:
        composition, defaults_applied = train_skill_composition(parsed)
        return "generic", explicit_skill_id, composition, defaults_applied
    if preset_skill is None:
        raise ValueError("train-skill requires either a preset skill or --skill-id")
    if preset_skill not in SKILLS:
        raise ValueError(
            f"unknown built-in preset: {preset_skill}; use --skill-id for arbitrary skills"
        )
    return "preset", preset_skill, None, {}
