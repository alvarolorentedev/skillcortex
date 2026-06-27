from __future__ import annotations

from pathlib import Path
from typing import Any

from ..contracts import TASK_TYPES
from ..shared.io import read_json


def coerce_messages(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    coerced = []
    for message in messages:
        role = message.get("role")
        content = message.get("content")
        if role not in {"system", "user", "assistant"}:
            raise ValueError("message role must be one of system, user, assistant")
        if not isinstance(content, str):
            raise ValueError("message content must be a string")
        coerced.append({"role": role, "content": content})
    return coerced


def normalize_messages(
    *,
    prompt: str | None,
    system: str | None,
    messages: list[dict[str, str]] | None,
) -> list[dict[str, str]]:
    if messages is not None:
        return coerce_messages(messages)
    if prompt is None:
        raise ValueError("prompt or messages is required")
    normalized = []
    if system:
        normalized.append({"role": "system", "content": system})
    normalized.append({"role": "user", "content": prompt})
    return normalized


def normalize_chat_request(payload: dict[str, Any], *, runtime_name: str | None = None) -> dict[str, Any]:
    model_name = payload.get("model")
    if runtime_name and model_name and model_name != runtime_name:
        raise ValueError(f"unknown model: {model_name}")
    if payload.get("stream"):
        raise ValueError("streaming is not supported")
    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a non-empty list")
    task_type = payload.get("task_type")
    if task_type is not None and task_type not in TASK_TYPES:
        raise ValueError(f"unknown task_type: {task_type}")
    return {
        "model": model_name,
        "messages": coerce_messages(messages),
        "task_type": task_type,
        "semantic_family": payload.get("semantic_family"),
        "skill_override": payload.get("skill_override"),
        "max_tokens": payload.get("max_tokens"),
        "temperature": payload.get("temperature"),
    }


def load_chat_request(path: Path, *, runtime_name: str | None = None) -> dict[str, Any]:
    payload = read_json(path.resolve())
    return normalize_chat_request(payload, runtime_name=runtime_name)
