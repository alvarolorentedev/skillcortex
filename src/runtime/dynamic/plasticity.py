from __future__ import annotations

import hashlib
import importlib
import json
import shutil
import tempfile
from pathlib import Path
from typing import Callable

from ...packaging.validation import validate_slm_package
from ...shared.config import base_config
from .types import DynamicRouteDecision


def train_slm_package(**kwargs):
    from ...packaging import train_slm_package as package_train_slm

    return package_train_slm(**kwargs)


def ensure_plasticity_lora(
    runtime,
    messages: list[dict[str, str]],
    decision: DynamicRouteDecision,
    *,
    config_loader=base_config,
    trainer=train_slm_package,
    validator=validate_slm_package,
) -> str:
    config = config_loader()
    if not config.get("training_enabled"):
        raise ValueError("dynamic plasticity training is disabled")
    publish_dir = config.get("plasticity_publish_dir")
    if not publish_dir:
        raise ValueError("dynamic plasticity training requires plasticity_publish_dir")
    text = "\n".join(message["content"] for message in messages if message["role"] == "user")
    slm_id = f"plasticity_{hashlib.sha256(text.encode()).hexdigest()[:8]}"
    if slm_id in runtime.slms:
        return slm_id
    output = Path(publish_dir) / slm_id
    if output.exists():
        validator(output)
        runtime.reload()
        if slm_id in runtime.slms:
            return slm_id
    _check_limit(runtime, config)
    with tempfile.TemporaryDirectory(prefix=f"slmcortex-{slm_id}-publish-") as directory:
        train_dataset = _write_train_dataset(Path(directory), text, decision, slm_id)
        if train_dataset is None:
            train_dataset = _write_live_or_fallback_dataset(Path(directory), messages, decision, slm_id, config)
        staging = Path(directory) / slm_id
        trainer(
            slm=slm_id,
            mode="generic",
            output=staging,
            train_dataset=train_dataset,
            eval_dataset=Path(config.get("plasticity_eval_dataset") or train_dataset),
            name=slm_id.replace("_", " ").title(),
            version="0.1.0",
            description=f"On-demand plasticity LoRA for {decision.reason}.",
            composition={
                "capabilities": {"allowed_task_types": [decision.task_type or "python_generation"]},
                "activation": {
                    "default_route_type": "adapter",
                    "scope": "task",
                    "semantic_families": [decision.semantic_family] if decision.semantic_family else [],
                },
                "compatibility": {"compatible_slms": [], "incompatible_slms": []},
                "routing": {"tasks": {}},
            },
            force=True,
            dry_run=False,
        )
        validator(staging)
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(staging), str(output))
    validator(output)
    runtime.reload()
    if slm_id not in runtime.slms:
        raise ValueError(f"trained plasticity LoRA did not produce a valid package: {slm_id}")
    return slm_id


def _check_limit(runtime, config: dict) -> None:
    limit = config.get("max_plasticity_loras")
    if limit is None:
        return
    existing = sum(1 for existing_id in runtime.slms if existing_id.startswith("plasticity_"))
    if existing >= int(limit):
        raise ValueError("plasticity slm cap reached")


def _write_train_dataset(directory: Path, text: str, decision: DynamicRouteDecision, slm_id: str) -> Path | None:
    if not text.strip():
        return None
    train_dataset = Path(directory) / "task-train.jsonl"
    train_row = {
        "id": slm_id,
        "task_type": decision.task_type or "python_generation",
        "prompt": text,
        "target": "Adapt to this task.",
        "semantic_family": decision.semantic_family,
        "metadata": {"source": "dynamic_plasticity"},
    }
    train_dataset.write_text(json.dumps(train_row, sort_keys=True) + "\n")
    return train_dataset


def _write_live_or_fallback_dataset(
    directory: Path,
    messages: list[dict[str, str]],
    decision: DynamicRouteDecision,
    slm_id: str,
    config: dict,
) -> Path:
    train_dataset = directory / "task-train.jsonl"
    live_source_handler = config.get("plasticity_live_source_handler")
    if live_source_handler:
        handler = _load_live_source_handler(live_source_handler)
        rows = handler(messages=messages, decision=decision, slm_id=slm_id)
        serialized_rows = [json.dumps(row, sort_keys=True) for row in rows]
        if not serialized_rows:
            raise ValueError("plasticity live source returned no training rows")
        train_dataset.write_text("\n".join(serialized_rows) + "\n")
        return train_dataset
    fallback_train_dataset = config.get("plasticity_train_dataset")
    if fallback_train_dataset:
        return Path(fallback_train_dataset)
    raise ValueError("dynamic plasticity training requires a user prompt")


def _load_live_source_handler(handler: object) -> Callable[..., object]:
    if callable(handler):
        return handler
    if not isinstance(handler, str) or not handler.strip():
        raise ValueError("plasticity_live_source_handler must be a callable or module path")
    module_name, sep, attr_name = handler.partition(":")
    if not sep or not attr_name:
        raise ValueError("plasticity_live_source_handler must use module:attribute syntax")
    resolved = getattr(importlib.import_module(module_name), attr_name)
    if not callable(resolved):
        raise ValueError("plasticity_live_source_handler must resolve to a callable")
    return resolved
