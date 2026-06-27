from __future__ import annotations

import json
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..contracts import KNOWN_SKILLS, MODES, ROUTER_POLICIES, SKILLS
from ..shared.config import ARTIFACT_DIR, base_config
from .router_rules import (
    ProtectedRouterPlusAlternatingSkill,
    ProtectedSkillRouter,
    RouteDecision,
    RuleRouter,
    SkillCortexRouterV1,
)


def adapter_path(name: str, root: str | Path | None = None) -> Path:
    if name != "generic" and name not in KNOWN_SKILLS:
        raise ValueError(f"unknown adapter: {name}")
    return Path(root) / name if root else ARTIFACT_DIR / "adapters" / name


def adapter_metadata(name: str, root: str | Path | None = None) -> dict:
    path = adapter_path(name, root) / "metadata.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def require_adapter(name: str, root: str | Path | None = None) -> Path:
    path = adapter_path(name, root)
    if not (path / "adapters.safetensors").exists():
        raise FileNotFoundError(f"adapter not found: {path}")
    return path


def load_model(adapter: Path | None = None, model_name: str | None = None):
    from mlx_lm import load

    return load(
        model_name or base_config()["model"],
        adapter_path=str(adapter) if adapter else None,
    )


def generate_text(
    model: object,
    tokenizer: object,
    prompt: str | None = None,
    *,
    messages: list[dict[str, str]] | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
) -> tuple[str, int, int]:
    from mlx_lm import generate
    from mlx_lm.sample_utils import make_sampler

    config = base_config()
    if prompt is not None and messages is not None:
        raise ValueError("provide either prompt or messages, not both")
    if prompt is None and not messages:
        raise ValueError("prompt or messages is required")
    resolved_messages = messages or [{"role": "user", "content": prompt or ""}]
    formatted = tokenizer.apply_chat_template(
        resolved_messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    tokenizer.eos_token_ids.add(tokenizer.convert_tokens_to_ids("<|im_end|>"))
    output = generate(
        model,
        tokenizer,
        prompt=formatted,
        max_tokens=max_tokens or config["max_tokens"],
        sampler=make_sampler(config["temperature"] if temperature is None else temperature),
        verbose=False,
    )
    output = output.split("<|im_end|>", 1)[0].rstrip()
    return output, len(tokenizer.encode(formatted)), len(tokenizer.encode(output))


@dataclass(slots=True)
class GenerationResult:
    mode: str
    generation: str
    selected_skills: list[str] = field(default_factory=list)
    route: RouteDecision | None = None
    latency_seconds: float = 0.0
    prompt_tokens: int | None = None
    generated_tokens: int | None = None
    peak_memory_bytes: int | None = None
    active_adapter_count: int = 0
    active_adapter_parameters: int = 0
    error: str | None = None

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"unknown mode: {self.mode}")
        unknown = set(self.selected_skills) - set(KNOWN_SKILLS)
        if unknown:
            raise ValueError(f"unknown skill: {sorted(unknown)[0]}")
        if self.active_adapter_count < 0 or self.active_adapter_parameters < 0:
            raise ValueError("adapter statistics must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def infer(
    mode: str,
    prompt: str,
    *,
    skill: str | None = None,
    skills: list[str] | None = None,
    task_type: str | None = None,
    semantic_family: str | None = None,
    router_policy: str | None = None,
    composition_weights: list[float] | None = None,
    dry_run: bool = False,
    adapter_root: str | Path | None = None,
    model_cache: dict | None = None,
) -> GenerationResult:
    if mode not in MODES:
        raise ValueError(f"unknown mode: {mode}")
    selected: list[str] = []
    route = None
    if mode == "generic":
        adapter_names = ["generic"]
    elif mode == "single-skill":
        if skill not in SKILLS:
            raise ValueError("--skill is required for single-skill mode")
        selected, adapter_names = [skill], [skill]
    elif mode == "lattice":
        if router_policy == "legacy_rule_router":
            route = RuleRouter().route(prompt)
        elif router_policy in (None, "skillcortex_router_v1"):
            route = SkillCortexRouterV1().route(task_type, semantic_family)
        elif router_policy == "protected_router_plus_alternating_skill":
            route = ProtectedRouterPlusAlternatingSkill().route(task_type, semantic_family)
        elif router_policy in (
            "python_only_for_test_generation",
            "protected_skill_router",
            "protected_skill_router_without_failure_born",
            "weighted_task_composition",
            "reverse_weighted_task_composition",
        ):
            route = ProtectedSkillRouter().route(task_type)
        else:
            raise ValueError(
                f"unknown router_policy: {router_policy}; expected {ROUTER_POLICIES}"
            )
        selected = route.selected_skills
        adapter_names = selected
    elif mode == "oracle-lattice":
        selected = list(dict.fromkeys(skills or []))[:2]
        if not selected or any(name not in SKILLS for name in selected):
            raise ValueError("oracle-lattice requires known skills")
        adapter_names = selected
    else:
        adapter_names = []

    if len(adapter_names) == 3 and router_policy not in (
        None,
        "skillcortex_router_v1",
        "protected_router_plus_alternating_skill",
    ):
        raise ValueError("three-adapter composition is quarantined to alternating_skill")
    parameters = sum(
        int(adapter_metadata(name, adapter_root).get("trainable_parameters") or 0)
        for name in adapter_names
    )
    if dry_run:
        return GenerationResult(
            mode=mode,
            generation="[dry-run generation]",
            selected_skills=selected,
            route=route,
            active_adapter_count=len(adapter_names),
            active_adapter_parameters=parameters,
        )

    cache_key = (
        (tuple(adapter_names), tuple(composition_weights))
        if composition_weights
        else tuple(adapter_names)
    )
    cached = model_cache.get(cache_key) if model_cache is not None else None
    try:
        import mlx.core as mx

        mx.reset_peak_memory()
    except ImportError:
        mx = None
    start = time.perf_counter()
    if cached:
        model, tokenizer = cached
    else:
        from ..composer.adapters import temporary_composed_adapter

        paths = [require_adapter(name, adapter_root) for name in adapter_names]
        adapter_context = (
            (
                temporary_composed_adapter(paths, composition_weights)
                if composition_weights
                else temporary_composed_adapter(paths)
            )
            if len(paths) > 1
            else nullcontext(paths[0] if paths else None)
        )
        with adapter_context as adapter:
            model, tokenizer = load_model(adapter)
        if model_cache is not None:
            model_cache[cache_key] = (model, tokenizer)
    generation, prompt_tokens, generated_tokens = generate_text(model, tokenizer, prompt)
    latency = time.perf_counter() - start
    peak_memory = int(mx.get_peak_memory()) if mx else None
    return GenerationResult(
        mode=mode,
        generation=generation,
        selected_skills=selected,
        route=route,
        latency_seconds=latency,
        prompt_tokens=prompt_tokens,
        generated_tokens=generated_tokens,
        peak_memory_bytes=peak_memory,
        active_adapter_count=len(adapter_names),
        active_adapter_parameters=parameters,
    )
