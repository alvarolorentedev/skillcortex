import json
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from .adapter_registry import adapter_path
from .config import DATA_DIR
from .data import load_jsonl, write_mlx_dataset
from .train_skill import _metadata, _saved_parameter_count, _training_command


def build_generic_command(
    data_directory: str | Path, output_directory: str | Path
) -> list[str]:
    return _training_command(data_directory, output_directory, rank=24)


def train_generic(*, dry_run: bool = False, force: bool = False) -> dict:
    examples = load_jsonl(DATA_DIR / "train.jsonl")
    output = adapter_path("generic")
    with tempfile.TemporaryDirectory(prefix="skill-lattice-generic-") as directory:
        dataset = write_mlx_dataset(examples, Path(directory) / "data")
        command = build_generic_command(dataset, output)
        if dry_run:
            return {"adapter": "generic", "examples": len(examples), "command": command}
        if output.exists() and any(output.iterdir()) and not force:
            raise FileExistsError(f"{output} exists; pass --force to replace it")
        if output.exists():
            shutil.rmtree(output)
        start = time.perf_counter()
        subprocess.run(command, check=True)
        metadata = _metadata(
            "generic", examples, rank=24, elapsed=time.perf_counter() - start
        )
        metadata["trainable_parameters"] = _saved_parameter_count(output)
        (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        return metadata
