import threading
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any
from wsgiref.simple_server import make_server

from .http import OpenAICompatApp
from .generation import generate_text, load_model
from .loading import load_runtime_bundle
from .models import REQUIRED_RUNTIME_FILES, RuntimeBundle, RuntimeRouteDecision, RuntimeSlm
from .request import load_chat_request, normalize_chat_request, normalize_messages
from .routing import build_route_decision
from .router_rules import route_text
from .dynamic import DynamicRuntime, DynamicRouteDecision
from ..composer.adapters import temporary_composed_adapter


class SlmRuntime:
    def __init__(self, bundle: RuntimeBundle):
        self.bundle = bundle
        self._cache: dict[tuple[str, ...], tuple[object, object]] = {}
        self._lock = threading.Lock()

    @classmethod
    def load(cls, runtime_path: Path) -> "SlmRuntime":
        return cls(load_runtime_bundle(runtime_path))

    def validate(self) -> dict[str, Any]:
        return {
            "status": "valid",
            "runtime": self.bundle.name,
            "path": str(self.bundle.path),
            "slms": sorted(self.bundle.slms),
            "runtime_model": self.bundle.runtime_model,
            "strategy": self.bundle.strategy,
        }

    def infer(
        self,
        *,
        prompt: str | None = None,
        system: str | None = None,
        messages: list[dict[str, str]] | None = None,
        task_type: str | None = None,
        semantic_family: str | None = None,
        slm_override: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        resolved_messages = normalize_messages(prompt=prompt, system=system, messages=messages)
        decision = self._normalize_selected_slms(
            self.route(
                messages=resolved_messages,
                task_type=task_type,
                semantic_family=semantic_family,
                slm_override=slm_override,
            )
        )
        active_parameters = sum(
            self.bundle.slms[slm_id].trainable_parameters
            for slm_id in decision.selected_slms
        )
        if dry_run:
            return {
                "status": "dry-run",
                "runtime": self.bundle.name,
                "task_type": decision.reason.split("task_type=", 1)[1].split(" ", 1)[0],
                "semantic_family": semantic_family,
                "route_type": decision.route_type,
                "selected_slms": decision.selected_slms,
                "reason": decision.reason,
                "active_adapter_count": len(decision.selected_slms),
                "active_adapter_parameters": active_parameters,
            }

        start = time.perf_counter()
        model, tokenizer = self._get_model(tuple(decision.selected_slms))
        generation, prompt_tokens, generated_tokens = generate_text(
            model,
            tokenizer,
            messages=resolved_messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return {
            "status": "complete",
            "runtime": self.bundle.name,
            "task_type": decision.reason.split("task_type=", 1)[1].split(" ", 1)[0],
            "semantic_family": semantic_family,
            "route_type": decision.route_type,
            "selected_slms": decision.selected_slms,
            "reason": decision.reason,
            "generation": generation,
            "latency_seconds": time.perf_counter() - start,
            "prompt_tokens": prompt_tokens,
            "generated_tokens": generated_tokens,
            "active_adapter_count": len(decision.selected_slms),
            "active_adapter_parameters": active_parameters,
        }

    def route(
        self,
        *,
        messages: list[dict[str, str]],
        task_type: str | None = None,
        semantic_family: str | None = None,
        slm_override: str | None = None,
    ) -> RuntimeRouteDecision:
        return build_route_decision(
            self.bundle.routes,
            messages,
            task_type=task_type,
            semantic_family=semantic_family,
            slm_override=slm_override,
            available_slms=set(self.bundle.slms),
            route_text=route_text,
        )

    def _normalize_selected_slms(self, decision: RuntimeRouteDecision) -> RuntimeRouteDecision:
        if self.bundle.backend != "gguf" or len(decision.selected_slms) <= 1:
            return decision
        selected = [decision.selected_slms[0]]
        return RuntimeRouteDecision(
            selected_slms=selected,
            confidence=decision.confidence,
            reason=(
                f"{decision.reason}; gguf single-adapter fallback selected {selected[0]} "
                "because adapter merge is not configured"
            ),
            route_type=decision.route_type,
        )

    def chat_completion(self, payload: dict[str, Any]) -> dict[str, Any]:
        normalized = normalize_chat_request(payload, runtime_name=self.bundle.name)
        result = self.infer(
            messages=normalized["messages"],
            task_type=normalized["task_type"],
            semantic_family=normalized["semantic_family"],
            slm_override=normalized["slm_override"],
            max_tokens=normalized["max_tokens"],
            temperature=normalized["temperature"],
            dry_run=False,
        )
        prompt_tokens = result.get("prompt_tokens") or 0
        generated_tokens = result.get("generated_tokens") or 0
        return {
            "id": f"chatcmpl-slmcortex-{int(time.time() * 1000)}",
            "object": "chat.completion",
            "created": int(time.time()),
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

    def _get_model(self, selected_slms: tuple[str, ...]) -> tuple[object, object]:
        cached = self._cache.get(selected_slms)
        if cached is not None:
            return cached
        with self._lock:
            cached = self._cache.get(selected_slms)
            if cached is not None:
                return cached
            adapter_paths = [self.bundle.slms[slm_id].adapter_path for slm_id in selected_slms]
            if not adapter_paths:
                model, tokenizer = load_model(model_name=self.bundle.runtime_model)
            elif self.bundle.backend == "gguf":
                model, tokenizer = load_model(adapter=adapter_paths[0], model_name=self.bundle.runtime_model)
            else:
                adapter_context = (
                    temporary_composed_adapter(adapter_paths)
                    if len(adapter_paths) > 1
                    else nullcontext(adapter_paths[0])
                )
                with adapter_context as adapter_path:
                    model, tokenizer = load_model(
                        adapter=adapter_path,
                        model_name=self.bundle.runtime_model,
                    )
            self._cache[selected_slms] = (model, tokenizer)
            return model, tokenizer


def validate_runtime_bundle(runtime_path: Path) -> dict[str, Any]:
    return SlmRuntime.load(runtime_path).validate()


def serve_runtime(
    *,
    runtime_path: Path,
    host: str,
    port: int,
    dry_run: bool = False,
) -> dict[str, Any]:
    runtime = SlmRuntime.load(runtime_path)
    if dry_run:
        return {
            "status": "dry-run",
            "runtime": runtime.bundle.name,
            "host": host,
            "port": port,
            "model": runtime.bundle.runtime_model,
            "slms": sorted(runtime.bundle.slms),
        }
    app = OpenAICompatApp(runtime)
    with make_server(host, port, app) as server:
        print(f"Serving SlmCortex runtime '{runtime.bundle.name}' on http://{host}:{port}")
        server.serve_forever()
    return {"status": "stopped"}
