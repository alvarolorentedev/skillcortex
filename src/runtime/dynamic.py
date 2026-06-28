from __future__ import annotations

import hashlib
import importlib
import json
import re
import shutil
import tempfile
import threading
import urllib.request
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from ..composer.adapters import temporary_composed_adapter
from ..packaging.validation import validate_slm_package
from ..shared.config import base_config
from .generation import generate_text, load_model
from .request import normalize_messages
from .registry import AdapterRegistry, ResolvedAdapter


@dataclass(slots=True)
class DynamicRouteDecision:
    base_model: str
    selected_slms: list[str]
    remote_loras: list[str]
    task_type: str | None
    semantic_family: str | None
    train_new_lora: bool
    reason: str


Router = Callable[[list[dict[str, str]], list[ResolvedAdapter]], DynamicRouteDecision]


def train_slm_package(**kwargs):
    from ..packaging import train_slm_package as package_train_slm

    return package_train_slm(**kwargs)


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


def _fetch_remote_lora_catalog(catalog_url: str) -> list[dict]:
    with urllib.request.urlopen(catalog_url) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("entries") or payload.get("remote_lora_catalog") or []
    if not isinstance(payload, list):
        raise ValueError("remote catalog payload must be a list of entries")
    return [entry for entry in payload if isinstance(entry, dict)]


class DynamicRuntime:
    def __init__(self, registry: AdapterRegistry):
        self.registry = registry
        self.slms = registry.local
        self._cache: dict[tuple[str, tuple[str, ...]], tuple[object, object]] = {}
        self._lock = threading.Lock()

    @classmethod
    def load(
        cls,
        slms_dir: Path,
        *,
        allow_remote_loras: bool = False,
        cache_dir: Path | None = None,
    ) -> "DynamicRuntime":
        return cls(
            AdapterRegistry.load(
                slms_dir,
                allow_remote=allow_remote_loras,
                cache_dir=cache_dir,
            )
        )

    def reload(self) -> None:
        self.registry.reload()
        self.slms = self.registry.local

    def infer(
        self,
        *,
        prompt: str | None = None,
        system: str | None = None,
        messages: list[dict[str, str]] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        dry_run: bool = False,
    ) -> dict:
        resolved_messages = normalize_messages(prompt=prompt, system=system, messages=messages)
        adaptation_error = None
        try:
            decision = self.route(
                resolved_messages,
                router=None if dry_run else self._router_model,
            )
        except ValueError as error:
            if dry_run:
                raise
            adaptation_error = str(error)
            decision = self._base_fallback_decision(adaptation_error)
        if dry_run:
            return self._result("dry-run", decision)
        model, tokenizer = self._get_model(decision.base_model, tuple(decision.selected_slms))
        generation, prompt_tokens, generated_tokens = generate_text(
            model,
            tokenizer,
            messages=resolved_messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        result = self._result("complete", decision, adaptation_error=adaptation_error)
        if adaptation_error:
            result["adaptation_error"] = adaptation_error
        result.update(
            {
                "generation": generation,
                "prompt_tokens": prompt_tokens,
                "generated_tokens": generated_tokens,
            }
        )
        return result

    def route(
        self,
        messages: list[dict[str, str]],
        *,
        router: Router | None = None,
    ) -> DynamicRouteDecision:
        decision = (router or self._rule_router)(messages, list(self.slms.values()))
        self._validate_decision(decision)
        if decision.train_new_lora:
            decision.selected_slms = [self._ensure_plasticity_lora(messages, decision)]
        unknown = [slm_id for slm_id in decision.selected_slms if slm_id not in self.slms]
        if unknown and not decision.remote_loras:
            raise ValueError(f"unknown dynamic slm: {unknown[0]}")
        if unknown:
            resolved = [
                self.registry.resolve_remote(source, slm_id)
                for source, slm_id in zip(decision.remote_loras, unknown, strict=False)
            ]
            self.reload()
            decision.selected_slms = [slm.slm_id for slm in resolved]
        self._normalize_selected_slms(decision)
        return decision

    def _normalize_selected_slms(self, decision: DynamicRouteDecision) -> None:
        if len(decision.selected_slms) <= 1:
            return
        selected = [self.slms[slm_id] for slm_id in decision.selected_slms if slm_id in self.slms]
        if not selected or any(slm.adapter_format != "gguf-lora" for slm in selected):
            return
        decision.selected_slms = [selected[0].slm_id]
        if decision.remote_loras:
            decision.remote_loras = decision.remote_loras[:1]
        decision.reason = (
            f"{decision.reason}; gguf single-adapter fallback selected {selected[0].slm_id} "
            "because adapter merge is not configured"
        )

    def _get_model(self, base_model: str, selected_slms: tuple[str, ...]) -> tuple[object, object]:
        key = (
            base_model,
            tuple(f"{slm_id}:{self.slms[slm_id].fingerprint}" for slm_id in selected_slms),
        )
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        with self._lock:
            cached = self._cache.get(key)
            if cached is not None:
                return cached
            adapter_paths = [
                (
                    self.slms[slm_id].adapter_path
                    if self.slms[slm_id].adapter_format == "gguf-lora"
                    else self.slms[slm_id].adapter_path.parent
                )
                for slm_id in selected_slms
            ]
            if not adapter_paths:
                model, tokenizer = load_model(model_name=base_model)
            elif self.slms[selected_slms[0]].adapter_format == "gguf-lora":
                model, tokenizer = load_model(adapter=adapter_paths[0], model_name=base_model)
            else:
                adapter_context = (
                    temporary_composed_adapter(adapter_paths)
                    if len(adapter_paths) > 1
                    else nullcontext(adapter_paths[0])
                )
                with adapter_context as adapter:
                    model, tokenizer = load_model(adapter=adapter, model_name=base_model)
            self._cache[key] = (model, tokenizer)
            return model, tokenizer

    def _rule_router(
        self,
        messages: list[dict[str, str]],
        slms: list[ResolvedAdapter],
    ) -> DynamicRouteDecision:
        text = "\n".join(message["content"] for message in messages if message["role"] == "user")
        words = set(re.findall(r"[a-z0-9]+", text.lower()))
        scored = []
        for slm in slms:
            haystack = " ".join(
                [slm.description, *slm.capabilities, *slm.activation_cues]
            ).lower()
            score = sum(1 for word in words if len(word) >= 3 and word in haystack)
            if score:
                scored.append((score, slm.slm_id))
        selected = [slm_id for _score, slm_id in sorted(scored, reverse=True)[:1]]
        config = base_config()
        remote_loras: list[str] = []
        if not selected:
            selected, remote_loras = self._remote_catalog_match(words, config)
        return DynamicRouteDecision(
            base_model=config.get("default_runtime_model") or config["model"],
            selected_slms=selected,
            remote_loras=remote_loras,
            task_type="python_generation",
            semantic_family=None,
            train_new_lora=False,
            reason="matched local LoRA" if selected and not remote_loras else (
                "matched remote LoRA catalog" if remote_loras else "base fallback"
            ),
        )

    def _remote_catalog_match(self, words: set[str], config: dict) -> tuple[list[str], list[str]]:
        scored = []
        remote_entries = list(config.get("remote_lora_catalog") or [])
        catalog_url = config.get("remote_lora_catalog_url")
        if isinstance(catalog_url, str) and catalog_url.strip():
            remote_entries.extend(_fetch_remote_lora_catalog(catalog_url))
        for entry in remote_entries:
            if not isinstance(entry, dict):
                continue
            slm_id = entry.get("slm_id")
            source = entry.get("source")
            cues = entry.get("cues") or []
            if not isinstance(slm_id, str) or not isinstance(source, str):
                continue
            fields = [entry.get("name"), entry.get("description"), *cues]
            fields.extend(entry.get("task_types") or [])
            fields.extend(entry.get("semantic_families") or [])
            score = sum(
                1
                for field in fields
                if isinstance(field, str)
                and any(word in field.lower() or field.lower() in word for word in words)
            )
            if score:
                scored.append((score, slm_id, source))
        if not scored:
            return [], []
        _score, slm_id, source = sorted(scored, reverse=True)[0]
        return [slm_id], [source]

    def _validate_decision(self, decision: DynamicRouteDecision) -> None:
        if decision.train_new_lora and (decision.selected_slms or decision.remote_loras):
            raise ValueError("ambiguous dynamic route: train_new_lora cannot combine with selected_slms or remote_loras")
        if decision.remote_loras and len(decision.remote_loras) != len(decision.selected_slms):
            raise ValueError("ambiguous dynamic route: remote_loras must map one-to-one with selected_slms")

    def _ensure_plasticity_lora(
        self,
        messages: list[dict[str, str]],
        decision: DynamicRouteDecision,
    ) -> str:
        config = base_config()
        if not config.get("training_enabled"):
            raise ValueError("dynamic plasticity training is disabled")
        publish_dir = config.get("plasticity_publish_dir")
        if not publish_dir:
            raise ValueError("dynamic plasticity training requires plasticity_publish_dir")
        text = "\n".join(message["content"] for message in messages if message["role"] == "user")
        train_row = {
            "id": "",
            "task_type": decision.task_type or "python_generation",
            "prompt": text,
            "target": "Adapt to this task.",
            "semantic_family": decision.semantic_family,
            "metadata": {"source": "dynamic_plasticity"},
        }
        slm_id = f"plasticity_{hashlib.sha256(text.encode()).hexdigest()[:8]}"
        train_row["id"] = slm_id
        if slm_id in self.slms:
            return slm_id
        output = Path(publish_dir) / slm_id
        if output.exists():
            validate_slm_package(output)
            self.reload()
            if slm_id in self.slms:
                return slm_id
        limit = config.get("max_plasticity_loras")
        if limit is not None:
            existing = sum(1 for existing_id in self.slms if existing_id.startswith("plasticity_"))
            if existing >= int(limit):
                raise ValueError("plasticity slm cap reached")
        fallback_train_dataset = config.get("plasticity_train_dataset")
        live_source_handler = config.get("plasticity_live_source_handler")
        with tempfile.TemporaryDirectory(prefix=f"slmcortex-{slm_id}-publish-") as directory:
            train_dataset = Path(directory) / "task-train.jsonl"
            if text.strip():
                train_dataset.write_text(json.dumps(train_row, sort_keys=True) + "\n")
            elif live_source_handler:
                handler = _load_live_source_handler(live_source_handler)
                rows = handler(messages=messages, decision=decision, slm_id=slm_id)
                serialized_rows = []
                for row in rows:
                    serialized_rows.append(json.dumps(row, sort_keys=True))
                if not serialized_rows:
                    raise ValueError("plasticity live source returned no training rows")
                train_dataset.write_text("\n".join(serialized_rows) + "\n")
            elif fallback_train_dataset:
                train_dataset = Path(fallback_train_dataset)
            else:
                raise ValueError("dynamic plasticity training requires a user prompt")
            eval_dataset = Path(config.get("plasticity_eval_dataset") or train_dataset)
            staging = Path(directory) / slm_id
            train_slm_package(
                slm=slm_id,
                mode="generic",
                output=staging,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
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
            validate_slm_package(staging)
            output.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(staging), str(output))
        validate_slm_package(output)
        self.reload()
        if slm_id not in self.slms:
            raise ValueError(f"trained plasticity LoRA did not produce a valid package: {slm_id}")
        return slm_id

    def _base_fallback_decision(self, reason: str) -> DynamicRouteDecision:
        config = base_config()
        return DynamicRouteDecision(
            base_model=str(config.get("default_runtime_model") or config["model"]),
            selected_slms=[],
            remote_loras=[],
            task_type=None,
            semantic_family=None,
            train_new_lora=False,
            reason=f"base fallback after adaptation error: {reason}",
        )

    def _router_model(
        self,
        messages: list[dict[str, str]],
        slms: list[ResolvedAdapter],
    ) -> DynamicRouteDecision:
        config = base_config()
        model, tokenizer = load_model(model_name=config.get("router_model") or config["model"])
        catalog = [
            {
                "slm_id": slm.slm_id,
                "description": slm.description,
                "capabilities": slm.capabilities,
                "base_model": slm.base_model,
            }
            for slm in slms
        ]
        prompt = (
            "Return JSON with base_model, selected_slms, remote_loras, task_type, semantic_family, "
            "train_new_lora, reason.\n"
            f"Available LoRAs: {json.dumps(catalog)}\n"
            f"Messages: {json.dumps(messages)}"
        )
        try:
            raw, _, _ = generate_text(model, tokenizer, prompt=prompt)
            payload = json.loads(raw)
            if not isinstance(payload, dict):
                raise ValueError("router payload must be an object")
            return DynamicRouteDecision(
                base_model=str(payload.get("base_model") or config.get("default_runtime_model") or config["model"]),
                selected_slms=list(payload.get("selected_slms") or payload.get("selected_loras") or []),
                remote_loras=list(payload.get("remote_loras") or []),
                task_type=payload.get("task_type"),
                semantic_family=payload.get("semantic_family"),
                train_new_lora=bool(payload.get("train_new_lora")),
                reason=str(payload.get("reason") or "router model"),
            )
        except (TypeError, ValueError, json.JSONDecodeError):
            return DynamicRouteDecision(
                base_model=str(config.get("default_runtime_model") or config["model"]),
                selected_slms=[],
                remote_loras=[],
                task_type=None,
                semantic_family=None,
                train_new_lora=False,
                reason="router fallback",
            )

    def _result(
        self,
        status: str,
        decision: DynamicRouteDecision,
        *,
        adaptation_error: str | None = None,
    ) -> dict:
        active = [
            self.slms[slm_id]
            for slm_id in decision.selected_slms
        ]
        branch = self._route_branch(decision)
        return {
            "status": status,
            "runtime": "dynamic",
            "base_model": decision.base_model,
            "task_type": decision.task_type,
            "semantic_family": decision.semantic_family,
            "selected_slms": decision.selected_slms,
            "remote_loras": decision.remote_loras,
            "train_new_lora": decision.train_new_lora,
            "reason": decision.reason,
            "route_branch": branch,
            "route_trace": {
                "router_output": {
                    "selected_slms": decision.selected_slms,
                    "remote_loras": decision.remote_loras,
                    "train_new_lora": decision.train_new_lora,
                    "reason": decision.reason,
                },
                "branch": branch,
                "final_selected_slms": decision.selected_slms,
            },
            "adaptation_summary": {
                "branch": branch,
                "reason": decision.reason,
                "fetched_sources": decision.remote_loras,
                "trained_slm": decision.selected_slms[0] if decision.train_new_lora and decision.selected_slms else None,
                "fallback_error": adaptation_error,
                "final_selected_slms": decision.selected_slms,
            },
            "active_adapter_count": len(active),
        }

    def _route_branch(self, decision: DynamicRouteDecision) -> str:
        if decision.train_new_lora:
            return "plasticity_train"
        if decision.remote_loras:
            return "remote_lora"
        if decision.selected_slms:
            return "local_lora"
        return "base_fallback"
