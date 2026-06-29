from __future__ import annotations

import hashlib
import json
import re
import threading
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

from ...composer.adapters import temporary_composed_adapter
from ...packaging.validation import validate_slm_package
from ...shared.config import base_config
from ..generation import generate_text, load_model
from ..registry import AdapterRegistry, ResolvedAdapter
from ..request import normalize_chat_request, normalize_messages
from .plasticity import ensure_plasticity_lora, train_slm_package
from .remote import fetch_remote_lora_catalog as _fetch_remote_lora_catalog
from .remote import remote_catalog_match
from .types import DynamicRouteDecision, Router


class DynamicRuntime:
    def __init__(self, registry: AdapterRegistry):
        self.registry = registry
        self.slms = registry.local
        self.bundle = SimpleNamespace(name="dynamic")
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
        return cls(AdapterRegistry.load(slms_dir, allow_remote=allow_remote_loras, cache_dir=cache_dir))

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
            decision = self.route(resolved_messages, router=None if dry_run else self._router_model)
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

    def chat_completion(self, payload: dict) -> dict:
        normalized = normalize_chat_request(payload, runtime_name=self.bundle.name)
        result = self.infer(
            messages=normalized["messages"],
            max_tokens=normalized["max_tokens"],
            temperature=normalized["temperature"],
            dry_run=False,
        )
        prompt_tokens = result.get("prompt_tokens") or 0
        generated_tokens = result.get("generated_tokens") or 0
        return {
            "id": f"chatcmpl-slmcortex-{hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]}",
            "object": "chat.completion",
            "created": 0,
            "model": self.bundle.name,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": result["generation"]},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": generated_tokens,
                "total_tokens": prompt_tokens + generated_tokens,
            },
        }

    def route(self, messages: list[dict[str, str]], *, router: Router | None = None) -> DynamicRouteDecision:
        decision = (router or self._rule_router)(messages, list(self.slms.values()))
        self._validate_decision(decision)
        if decision.train_new_lora:
            decision.selected_slms = [
                ensure_plasticity_lora(
                    self,
                    messages,
                    decision,
                    config_loader=base_config,
                    trainer=train_slm_package,
                    validator=validate_slm_package,
                )
            ]
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

    def _rule_router(self, messages: list[dict[str, str]], slms: list[ResolvedAdapter]) -> DynamicRouteDecision:
        text = "\n".join(message["content"] for message in messages if message["role"] == "user")
        words = set(re.findall(r"[a-z0-9]+", text.lower()))
        scored = []
        for slm in slms:
            haystack = " ".join([slm.description, *slm.capabilities, *slm.activation_cues]).lower()
            score = sum(1 for word in words if len(word) >= 3 and word in haystack)
            if score:
                scored.append((score, slm.slm_id))
        selected = [slm_id for _score, slm_id in sorted(scored, reverse=True)[:1]]
        config = base_config()
        remote_loras: list[str] = []
        if not selected:
            selected, remote_loras = remote_catalog_match(
                words,
                config,
                fetch_catalog=_fetch_remote_lora_catalog,
            )
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

    def _validate_decision(self, decision: DynamicRouteDecision) -> None:
        if decision.train_new_lora and (decision.selected_slms or decision.remote_loras):
            raise ValueError("ambiguous dynamic route: train_new_lora cannot combine with selected_slms or remote_loras")
        if decision.remote_loras and len(decision.remote_loras) != len(decision.selected_slms):
            raise ValueError("ambiguous dynamic route: remote_loras must map one-to-one with selected_slms")

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

    def _router_model(self, messages: list[dict[str, str]], slms: list[ResolvedAdapter]) -> DynamicRouteDecision:
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
        active = [self.slms[slm_id] for slm_id in decision.selected_slms]
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


__all__ = ["DynamicRouteDecision", "DynamicRuntime", "Router", "train_slm_package"]
