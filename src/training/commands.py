from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import yaml

from ..contracts import PRESET_SLMS
from ..shared.config import base_config, resolve_backend, training_config, validate_runtime_model
from .data import dataset_hash


def build_slm_command(
    slm: str,
    data_directory: str | Path,
    output_directory: str | Path,
    *,
    seed: int | None = None,
) -> list[str]:
    if slm not in PRESET_SLMS:
        raise ValueError(f"unknown slm: {slm}")
    return training_command(data_directory, output_directory, rank=8, seed=seed)


def training_command(
    data_directory: str | Path,
    output_directory: str | Path,
    rank: int,
    *,
    seed: int | None = None,
    iterations: int | None = None,
    learning_rate: float | None = None,
) -> list[str]:
    base = base_config()
    backend = validate_runtime_model(base)
    training = training_config()
    config_path = Path(data_directory) / f"training-rank-{rank}.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "backend": backend,
        "model": base["model"],
        "source_model": base.get("source_model") or base["model"],
        "train": True,
        "data": str(data_directory),
        "adapter_path": str(output_directory),
        "fine_tune_type": "lora",
        "mask_prompt": True,
        "batch_size": training["batch_size"],
        "iters": training["iterations"] if iterations is None else iterations,
        "learning_rate": training["learning_rate"] if learning_rate is None else learning_rate,
        "num_layers": training["lora_layers"],
        "seed": training["seed"] if seed is None else seed,
        "gguf_converter": base.get("gguf_converter"),
        "lora_parameters": {
            "rank": rank,
            "dropout": 0.0,
            "scale": 20.0,
            "keys": training["target_modules"],
        },
    }
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False))
    if backend == "gguf":
        return [sys.executable, "-m", "slmcortex.training.gguf_lora", "--config", str(config_path)]
    return [sys.executable, "-m", "mlx_lm", "lora", "--config", str(config_path)]


def training_metadata(
    slm_id: str,
    examples: list[Any],
    *,
    rank: int,
    elapsed: float,
    seed: int | None = None,
    iterations: int | None = None,
) -> dict:
    base = base_config()
    backend = resolve_backend(base)
    training = training_config()
    return {
        "adapter": slm_id,
        "base_model": base["model"],
        "source_model": base["source_model"],
        "quantization": "4bit",
        "backend": backend,
        "format": "gguf-lora" if backend == "gguf" else "mlx-lora",
        "dataset_size": len(examples),
        "dataset_hash": dataset_hash(examples),
        "rank": rank,
        "target_modules": training["target_modules"],
        "seed": training["seed"] if seed is None else seed,
        "iterations": iterations or training["iterations"],
        "elapsed_seconds": elapsed,
        "trainable_parameters": None,
        "config": training,
    }


def saved_parameter_count(output: Path) -> int:
    if (output / "adapter.gguf").exists():
        return 0
    import mlx.core as mx

    arrays = mx.load(str(output / "adapters.safetensors"))
    return sum(array.size for array in arrays.values())
