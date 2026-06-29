from __future__ import annotations

import argparse
from pathlib import Path

from ..contracts import PRESET_SLMS


COMPOSITION_SCOPES = ("task", "semantic_family")


def parser_kwargs(description: str, examples: str | None = None, summary: str | None = None) -> dict:
    kwargs = {
        "description": description,
        "formatter_class": argparse.RawDescriptionHelpFormatter,
    }
    if examples:
        kwargs["epilog"] = f"Examples:\n{examples}"
    if summary:
        kwargs["help"] = summary
    return kwargs


def csv_paths(value: str) -> list[Path]:
    return [Path(item.strip()) for item in value.split(",") if item.strip()]


def default_dataset_outputs(slm_id: str) -> tuple[Path, Path]:
    root = Path("datasets") / slm_id
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
        "slm_override": parsed.slm_override,
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
            parsed.compatible_slms,
            parsed.incompatible_slms,
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
            "compatible_slms": list(parsed.compatible_slms or []),
            "incompatible_slms": list(parsed.incompatible_slms or []),
        },
        "routing": {
            "tasks": {},
        },
    }


def train_slm_composition(parsed: argparse.Namespace) -> tuple[dict | None, dict[str, object]]:
    if not parsed.slm_id:
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
            "compatible_slms": list(parsed.compatible_slms or []),
            "incompatible_slms": list(parsed.incompatible_slms or []),
        },
        "routing": {
            "tasks": {},
        },
    }, defaults_applied


def resolve_train_slm(parsed: argparse.Namespace) -> tuple[str, str, dict | None, dict[str, object]]:
    preset_slm = parsed.slm
    explicit_slm_id = parsed.slm_id
    if explicit_slm_id and preset_slm and explicit_slm_id != preset_slm:
        raise ValueError("provide either a preset positional slm or a matching --slm-id")
    if explicit_slm_id:
        composition, defaults_applied = train_slm_composition(parsed)
        return "generic", explicit_slm_id, composition, defaults_applied
    if preset_slm is None:
        raise ValueError("train-slm requires either a preset slm or --slm-id")
    if preset_slm not in PRESET_SLMS:
        raise ValueError(
            f"unknown built-in preset: {preset_slm}; use --slm-id for arbitrary slms"
        )
    return "preset", preset_slm, None, {}
