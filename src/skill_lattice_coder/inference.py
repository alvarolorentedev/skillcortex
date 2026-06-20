import time
from contextlib import nullcontext
from pathlib import Path

from .adapter_registry import adapter_metadata, require_adapter
from .compose import temporary_composed_adapter
from .model_loader import generate_text, load_model
from .router import RuleRouter
from .schemas import GenerationResult, MODES, SKILLS


def infer(
    mode: str,
    prompt: str,
    *,
    skill: str | None = None,
    skills: list[str] | None = None,
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
        route = RuleRouter().route(prompt)
        selected = route.selected_skills
        adapter_names = selected
    elif mode == "oracle-lattice":
        selected = list(dict.fromkeys(skills or []))[:2]
        if not selected or any(name not in SKILLS for name in selected):
            raise ValueError("oracle-lattice requires known skills")
        adapter_names = selected
    else:
        adapter_names = []

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

    cache_key = tuple(adapter_names)
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
        paths = [require_adapter(name, adapter_root) for name in adapter_names]
        adapter_context = (
            temporary_composed_adapter(paths)
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
