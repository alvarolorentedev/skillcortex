import json
import subprocess
import time
from pathlib import Path

from .backends import (
    aggregate_results,
    extract_code,
    fuzzy_match,
    generate_text,
    load_model,
    python_syntax_valid,
    run_fixture,
    saved_parameter_count,
    training_command,
    training_config,
    training_metadata,
)
from .data import load_product_jsonl
from .reporting import evaluation_report


def train_product_slm_to_run_directory(
    *,
    slm_id: str,
    train_dataset: Path,
    run_directory: Path,
    seed: int | None,
    force: bool,
) -> tuple[Path, dict]:
    run_directory.mkdir(parents=True, exist_ok=True)
    examples = load_product_jsonl(train_dataset)
    training_directory = run_directory / "training-data"
    adapter_directory = run_directory / "adapters" / slm_id
    if adapter_directory.exists() and any(adapter_directory.iterdir()) and not force:
        raise FileExistsError(f"{adapter_directory} exists; pass --force to replace it")
    if adapter_directory.exists():
        import shutil

        shutil.rmtree(adapter_directory)
    from .data import write_product_mlx_dataset

    dataset_directory = write_product_mlx_dataset(examples, training_directory)
    command = training_command(dataset_directory, adapter_directory, rank=8, seed=seed)
    start = time.perf_counter()
    subprocess.run(command, check=True)
    metadata = training_metadata(
        slm_id,
        examples,
        rank=8,
        elapsed=time.perf_counter() - start,
        seed=seed,
        iterations=training_config()["iterations"],
    )
    metadata["trainable_parameters"] = saved_parameter_count(adapter_directory)
    metadata["training_command"] = command
    (adapter_directory / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return adapter_directory, metadata


def evaluate_product_slm_adapter(
    *,
    slm_id: str,
    dataset: Path,
    output: Path,
    adapter_dir: Path,
) -> Path:
    examples = load_product_jsonl(dataset)
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    model_cache: dict = {}
    raw_path = output / "results.jsonl"
    with raw_path.open("w") as handle:
        for example in examples:
            for mode, resolved_adapter in (("base", None), ("single-slm", adapter_dir)):
                try:
                    generation = generate_for_example(example.prompt, adapter_dir=resolved_adapter, model_cache=model_cache)
                    text = extract_code(generation["generation"])
                    syntax = python_syntax_valid(text) if example.task_type != "test_generation" else None
                    execution = None
                    if example.execution:
                        execution, _ = run_fixture(example.execution, text)
                    row = {
                        "example_id": example.id,
                        "task_type": example.task_type,
                        "mode": mode,
                        "generation": text,
                        "exact_match": text.strip() == example.target.strip(),
                        "fuzzy_score": fuzzy_match(text, example.target),
                        "syntax_valid": syntax,
                        "execution_passed": execution,
                        "latency_seconds": generation["latency_seconds"],
                        "selected_slms": [slm_id] if resolved_adapter else [],
                        "active_adapter_count": 1 if resolved_adapter else 0,
                        "active_adapter_parameters": generation["active_adapter_parameters"],
                        "prompt_tokens": generation["prompt_tokens"],
                        "generated_tokens": generation["generated_tokens"],
                        "peak_memory_bytes": generation["peak_memory_bytes"],
                        "benchmark_group": example.group,
                    }
                except Exception as error:
                    row = {
                        "example_id": example.id,
                        "task_type": example.task_type,
                        "mode": mode,
                        "generation": "",
                        "exact_match": False,
                        "fuzzy_score": 0,
                        "syntax_valid": None,
                        "execution_passed": None,
                        "latency_seconds": 0,
                        "selected_slms": [slm_id] if resolved_adapter else [],
                        "active_adapter_count": 1 if resolved_adapter else 0,
                        "active_adapter_parameters": 0,
                        "prompt_tokens": None,
                        "generated_tokens": None,
                        "peak_memory_bytes": None,
                        "error": str(error),
                        "benchmark_group": example.group,
                    }
                rows.append(row)
                handle.write(json.dumps(row) + "\n")
    summary = aggregate_results(rows)
    tasks = {task: aggregate_results([row for row in rows if row["task_type"] == task]) for task in sorted({row["task_type"] for row in rows})}
    payload = {"hypothesis": None, "modes": summary, "tasks": tasks}
    (output / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    (output / "report.md").write_text(evaluation_report(slm_id, summary, tasks))
    return output / "summary.json"


def generate_for_example(prompt: str, *, adapter_dir: Path | None, model_cache: dict) -> dict:
    cache_key = str(adapter_dir.resolve()) if adapter_dir is not None else "__base__"
    cached = model_cache.get(cache_key)
    try:
        import mlx.core as mx

        mx.reset_peak_memory()
    except ImportError:
        mx = None
    start = time.perf_counter()
    if cached is None:
        model, tokenizer = load_model(adapter_dir)
        model_cache[cache_key] = (model, tokenizer)
    else:
        model, tokenizer = cached
    generation, prompt_tokens, generated_tokens = generate_text(model, tokenizer, prompt)
    return {
        "generation": generation,
        "latency_seconds": time.perf_counter() - start,
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated_tokens,
        "peak_memory_bytes": int(mx.get_peak_memory()) if mx else None,
        "active_adapter_parameters": saved_parameter_count(adapter_dir) if adapter_dir is not None else 0,
    }
