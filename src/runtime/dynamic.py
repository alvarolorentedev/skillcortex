from __future__ import annotations

import json
import re
import threading
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from ..composer.adapters import temporary_composed_adapter
from ..shared.config import base_config
from .generation import generate_text, load_model
from .request import normalize_messages
from .registry import AdapterRegistry, ResolvedAdapter


@dataclass(slots=True)
class DynamicRouteDecision:
    base_model: str
    selected_skills: list[str]
    remote_loras: list[str]
    task_type: str | None
    semantic_family: str | None
    train_new_lora: bool
    reason: str


Router = Callable[[list[dict[str, str]], list[ResolvedAdapter]], DynamicRouteDecision]


class DynamicRuntime:
    def __init__(self, registry: AdapterRegistry):
        self.registry = registry
        self.skills = registry.local
        self._cache: dict[tuple[str, tuple[str, ...]], tuple[object, object]] = {}
        self._lock = threading.Lock()

    @classmethod
    def load(
        cls,
        skills_dir: Path,
        *,
        allow_remote_loras: bool = False,
        cache_dir: Path | None = None,
    ) -> "DynamicRuntime":
        return cls(
            AdapterRegistry.load(
                skills_dir,
                allow_remote=allow_remote_loras,
                cache_dir=cache_dir,
            )
        )

    def reload(self) -> None:
        self.registry.reload()
        self.skills = self.registry.local

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
        decision = self.route(
            resolved_messages,
            router=None if dry_run else self._router_model,
        )
        if dry_run:
            return self._result("dry-run", decision)
        model, tokenizer = self._get_model(decision.base_model, tuple(decision.selected_skills))
        generation, prompt_tokens, generated_tokens = generate_text(
            model,
            tokenizer,
            messages=resolved_messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        result = self._result("complete", decision)
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
        decision = (router or self._rule_router)(messages, list(self.skills.values()))
        unknown = [skill_id for skill_id in decision.selected_skills if skill_id not in self.skills]
        if unknown and not decision.remote_loras:
            raise ValueError(f"unknown dynamic skill: {unknown[0]}")
        if unknown:
            resolved = [
                self.registry.resolve_remote(source, skill_id)
                for source, skill_id in zip(decision.remote_loras, unknown, strict=False)
            ]
            self.reload()
            decision.selected_skills = [skill.skill_id for skill in resolved]
        return decision

    def _get_model(self, base_model: str, selected_skills: tuple[str, ...]) -> tuple[object, object]:
        key = (
            base_model,
            tuple(f"{skill_id}:{self.skills[skill_id].fingerprint}" for skill_id in selected_skills),
        )
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        with self._lock:
            cached = self._cache.get(key)
            if cached is not None:
                return cached
            adapter_paths = [self.skills[skill_id].adapter_path.parent for skill_id in selected_skills]
            if not adapter_paths:
                model, tokenizer = load_model(model_name=base_model)
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
        skills: list[ResolvedAdapter],
    ) -> DynamicRouteDecision:
        text = "\n".join(message["content"] for message in messages if message["role"] == "user")
        words = set(re.findall(r"[a-z0-9]+", text.lower()))
        scored = []
        for skill in skills:
            haystack = " ".join(
                [skill.description, *skill.capabilities, *skill.activation_cues]
            ).lower()
            score = sum(1 for word in words if len(word) >= 3 and word in haystack)
            if score:
                scored.append((score, skill.skill_id))
        selected = [skill_id for _score, skill_id in sorted(scored, reverse=True)[:1]]
        config = base_config()
        return DynamicRouteDecision(
            base_model=config.get("default_runtime_model") or config["model"],
            selected_skills=selected,
            remote_loras=[],
            task_type="python_generation",
            semantic_family=None,
            train_new_lora=False,
            reason="matched local LoRA" if selected else "base fallback",
        )

    def _router_model(
        self,
        messages: list[dict[str, str]],
        skills: list[ResolvedAdapter],
    ) -> DynamicRouteDecision:
        config = base_config()
        model, tokenizer = load_model(model_name=config.get("router_model") or config["model"])
        catalog = [
            {
                "skill_id": skill.skill_id,
                "description": skill.description,
                "capabilities": skill.capabilities,
                "base_model": skill.base_model,
            }
            for skill in skills
        ]
        prompt = (
            "Return JSON with base_model, selected_skills, remote_loras, task_type, semantic_family, "
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
                selected_skills=list(payload.get("selected_skills") or payload.get("selected_loras") or []),
                remote_loras=list(payload.get("remote_loras") or []),
                task_type=payload.get("task_type"),
                semantic_family=payload.get("semantic_family"),
                train_new_lora=bool(payload.get("train_new_lora")),
                reason=str(payload.get("reason") or "router model"),
            )
        except (TypeError, ValueError, json.JSONDecodeError):
            return DynamicRouteDecision(
                base_model=str(config.get("default_runtime_model") or config["model"]),
                selected_skills=[],
                remote_loras=[],
                task_type=None,
                semantic_family=None,
                train_new_lora=False,
                reason="router fallback",
            )

    def _result(self, status: str, decision: DynamicRouteDecision) -> dict:
        active = [
            self.skills[skill_id]
            for skill_id in decision.selected_skills
        ]
        return {
            "status": status,
            "runtime": "dynamic",
            "base_model": decision.base_model,
            "task_type": decision.task_type,
            "semantic_family": decision.semantic_family,
            "selected_skills": decision.selected_skills,
            "remote_loras": decision.remote_loras,
            "train_new_lora": decision.train_new_lora,
            "reason": decision.reason,
            "active_adapter_count": len(active),
        }
