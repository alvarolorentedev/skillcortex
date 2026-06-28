from __future__ import annotations

import json
import platform
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

from .. import __version__
from ..shared.config import adapter_weight_name_for_format, resolve_backend
from ..shared.config import base_config, training_config
from ..shared.hashing import sha256
from .artifacts import line_count
from .composition import default_composition


def build_manifests(
    *,
    slm_id: str,
    name: str,
    version: str,
    description: str,
    adapter_dir: Path,
    adapter_metadata: dict,
    train_dataset: Path,
    eval_dataset: Path,
    eval_summary: Path,
    eval_payload: dict,
    examples: Path | None,
    composition: dict | None,
    protected_inputs: dict | None,
    source_artifacts: dict | None,
    training_details: dict | None,
) -> dict[str, object]:
    created_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    base = base_config()
    backend = adapter_metadata.get("backend") or resolve_backend(base)
    adapter_format = adapter_metadata.get("format") or adapter_metadata.get("adapter_format") or (
        "gguf-lora" if backend == "gguf" else "mlx-lora"
    )
    weight_path = f"adapter/{adapter_weight_name_for_format(adapter_format)}"
    training_defaults = training_config()
    rank = int(adapter_metadata.get("rank") or training_defaults["slm_rank"])
    target_modules = adapter_metadata.get("target_modules") or training_defaults["target_modules"]
    trainable_parameters = int(adapter_metadata.get("trainable_parameters") or 0)
    train_dataset_hash = sha256(train_dataset)
    eval_dataset_hash = sha256(eval_dataset)
    examples_count = line_count(examples) if examples else 0
    resolved_composition = composition or default_composition(slm_id)
    package_training = {
        "seed": int(adapter_metadata.get("seed") or training_defaults["seed"]),
        "batch_size": int(
            (adapter_metadata.get("config") or {}).get("batch_size")
            or training_defaults["batch_size"]
        ),
        "iterations": int(adapter_metadata.get("iterations") or training_defaults["iterations"]),
        "learning_rate": float(
            (adapter_metadata.get("config") or {}).get("learning_rate")
            or training_defaults["learning_rate"]
        ),
        "lora_layers": int(
            (adapter_metadata.get("config") or {}).get("lora_layers")
            or training_defaults["lora_layers"]
        ),
        "rank": rank,
        "target_modules": target_modules,
        "mask_prompt": True,
    }
    evaluation = {
        "schema_version": "1",
        "dataset": {"path": str(eval_dataset), "sha256": eval_dataset_hash},
        "summary_path": "eval.json",
        "source_summary_path": str(eval_summary),
        "modes": eval_payload.get("modes") or {},
        "tasks": eval_payload.get("tasks"),
        "hypothesis": eval_payload.get("hypothesis"),
    }
    metadata = {
        "schema_version": "1",
        "package_type": "slm",
        "status": "complete",
        "slm_id": slm_id,
        "name": name,
        "version": version,
        "created_at": created_at,
        "tool": {"name": "slmcortex", "version": __version__},
        "environment": {
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
        },
        "base": {
            "source_model": adapter_metadata.get("source_model") or base["source_model"],
            "runtime_model": adapter_metadata.get("base_model") or base["model"],
            "quantization": adapter_metadata.get("quantization") or "4bit",
            "backend": backend,
        },
        "adapter": {
            "format": adapter_format,
            "rank": rank,
            "target_modules": target_modules,
            "trainable_parameters": trainable_parameters,
            "files": {
                "weights": weight_path,
                "config": "adapter/adapter_config.json",
            },
        },
        "training": {
            **package_training,
            "elapsed_seconds": (training_details or {}).get("elapsed_seconds")
            or adapter_metadata.get("elapsed_seconds"),
            "command": (training_details or {}).get("command"),
        },
        "datasets": {
            "train": {
                "path": str(train_dataset),
                "sha256": train_dataset_hash,
                "example_count": adapter_metadata.get("dataset_size"),
                **({"role": "packaging_reference"} if (source_artifacts or {}).get("source") else {}),
            },
            "eval": {
                "path": str(eval_dataset),
                "sha256": eval_dataset_hash,
                **({"role": "packaging_reference"} if (source_artifacts or {}).get("source") else {}),
            },
        },
        "evaluation": {
            "summary_path": "eval.json",
            "source_summary_path": str(eval_summary),
        },
        "source_artifacts": source_artifacts or {"adapter_source_dir": str(adapter_dir)},
        "composition": resolved_composition,
        "protected_inputs": protected_inputs,
    }
    slm_yaml = {
        "schema_version": "1",
        "package_type": "slm",
        "slm_id": slm_id,
        "name": name,
        "version": version,
        "description": description,
        "status": "complete",
        "base": metadata["base"],
        "adapter": {
            "format": adapter_format,
            "path": weight_path,
            "config_path": "adapter/adapter_config.json",
            "rank": rank,
            "target_modules": target_modules,
            "trainable_parameters": trainable_parameters,
        },
        "data": {
            "train_dataset_path": str(train_dataset),
            "train_dataset_sha256": train_dataset_hash,
            "eval_dataset_path": str(eval_dataset),
            "eval_dataset_sha256": eval_dataset_hash,
        },
        "evaluation": {"summary_path": "eval.json"},
        "provenance": {
            "metadata_path": "metadata.json",
            "training_config_path": "training_config.json",
        },
    }
    if resolved_composition is not None:
        slm_yaml["composition"] = resolved_composition
    if examples is not None:
        slm_yaml["examples"] = {"path": "examples.jsonl", "count": examples_count}
    readme = build_readme(name, slm_id, version, description, metadata, evaluation)
    return {
        "slm.yaml": yaml.safe_dump(slm_yaml, sort_keys=False),
        "metadata.json": json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        "training_config.json": json.dumps(package_training, indent=2, sort_keys=True) + "\n",
        "eval.json": json.dumps(evaluation, indent=2, sort_keys=True) + "\n",
        "README.md": readme,
    }


def write_package(
    *,
    staging: Path,
    adapter_weights: Path,
    adapter_config: Path | None,
    manifests: dict[str, str],
    examples: Path | None,
) -> None:
    (staging / "adapter").mkdir(parents=True, exist_ok=True)
    shutil.copy2(adapter_weights, staging / "adapter" / adapter_weights.name)
    if adapter_config is not None:
        shutil.copy2(adapter_config, staging / "adapter" / "adapter_config.json")
    for relative_path, content in manifests.items():
        destination = staging / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(content)
    if examples is not None:
        shutil.copy2(examples, staging / "examples.jsonl")


def build_readme(
    name: str,
    slm_id: str,
    version: str,
    description: str,
    metadata: dict,
    evaluation: dict,
) -> str:
    return "\n".join(
        [
            f"# {name}",
            "",
            description,
            "",
            "## Package",
            "",
            f"- Slm ID: `{slm_id}`",
            f"- Version: `{version}`",
            f"- Source model: `{metadata['base']['source_model']}`",
            f"- Runtime model: `{metadata['base']['runtime_model']}`",
            f"- Trainable parameters: **{metadata['adapter']['trainable_parameters']}**",
            "",
            "## Evaluation",
            "",
            f"- Eval dataset: `{evaluation['dataset']['path']}`",
            f"- Eval dataset SHA-256: `{evaluation['dataset']['sha256']}`",
            f"- Package manifest is deterministic and validated with per-file checksums.",
            "",
        ]
    ) + "\n"
