import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import yaml

from .adapter_registry import adapter_path
from .config import DATA_DIR, base_config, training_config
from .data import dataset_hash, load_jsonl, select_for_skill, write_mlx_dataset
from .schemas import SKILLS


def build_skill_command(
    skill: str,
    data_directory: str | Path,
    output_directory: str | Path,
    *,
    seed: int | None = None,
) -> list[str]:
    if skill not in SKILLS:
        raise ValueError(f"unknown skill: {skill}")
    return _training_command(data_directory, output_directory, rank=8, seed=seed)


def _training_command(
    data_directory: str | Path,
    output_directory: str | Path,
    rank: int,
    *,
    seed: int | None = None,
    iterations: int | None = None,
    learning_rate: float | None = None,
) -> list[str]:
    base = base_config()
    training = training_config()
    config_path = Path(data_directory) / f"training-rank-{rank}.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        yaml.safe_dump(
            {
                "model": base["model"],
                "train": True,
                "data": str(data_directory),
                "adapter_path": str(output_directory),
                "fine_tune_type": "lora",
                "mask_prompt": True,
                "batch_size": training["batch_size"],
                "iters": training["iterations"] if iterations is None else iterations,
                "learning_rate": (
                    training["learning_rate"]
                    if learning_rate is None
                    else learning_rate
                ),
                "num_layers": training["lora_layers"],
                "seed": training["seed"] if seed is None else seed,
                "lora_parameters": {
                    "rank": rank,
                    "dropout": 0.0,
                    "scale": 20.0,
                    "keys": training["target_modules"],
                },
            },
            sort_keys=False,
        )
    )
    return [
        sys.executable,
        "-m",
        "mlx_lm",
        "lora",
        "--config",
        str(config_path),
    ]


def train_skill(
    skill: str,
    *,
    dry_run: bool = False,
    force: bool = False,
    seed: int | None = None,
    adapter_root: str | Path | None = None,
) -> dict:
    examples = select_for_skill(load_jsonl(DATA_DIR / "train.jsonl"), skill)
    output = adapter_path(skill, adapter_root)
    with tempfile.TemporaryDirectory(prefix=f"skill-lattice-{skill}-") as directory:
        dataset = write_mlx_dataset(examples, Path(directory) / "data")
        command = build_skill_command(skill, dataset, output, seed=seed)
        if dry_run:
            return {"skill": skill, "examples": len(examples), "command": command}
        if output.exists() and any(output.iterdir()) and not force:
            raise FileExistsError(f"{output} exists; pass --force to replace it")
        if output.exists():
            shutil.rmtree(output)
        start = time.perf_counter()
        subprocess.run(command, check=True)
        metadata = _metadata(
            skill,
            examples,
            rank=8,
            elapsed=time.perf_counter() - start,
            seed=seed,
            iterations=training_config()["iterations"],
        )
        metadata["trainable_parameters"] = _saved_parameter_count(output)
        (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        return metadata


def _metadata(
    name: str,
    examples: list,
    rank: int,
    elapsed: float,
    *,
    seed: int | None = None,
    iterations: int | None = None,
) -> dict:
    base = base_config()
    training = training_config()
    return {
        "adapter": name,
        "base_model": base["model"],
        "source_model": base["source_model"],
        "quantization": "4bit",
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


def _saved_parameter_count(output: Path) -> int:
    import mlx.core as mx

    arrays = mx.load(str(output / "adapters.safetensors"))
    return sum(array.size for array in arrays.values())
