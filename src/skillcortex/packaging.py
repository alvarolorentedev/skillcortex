import hashlib
import json
import platform
import subprocess
import shutil
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

from skill_lattice_coder import __version__
from skill_lattice_coder.config import ARTIFACT_DIR, CONFIG_DIR, ROOT, base_config, training_config
from skill_lattice_coder.data import load_jsonl, select_for_skill, write_mlx_dataset
from skill_lattice_coder.inference import infer
from skill_lattice_coder.metrics import aggregate_results, extract_code, fuzzy_match, python_syntax_valid
from skill_lattice_coder.schemas import EvaluationResult
from skill_lattice_coder.train_skill import _metadata as research_metadata
from skill_lattice_coder.train_skill import _saved_parameter_count, build_skill_command
from skill_lattice_coder.utils import run_fixture


REQUIRED_PACKAGE_FILES = (
    "skill.yaml",
    "metadata.json",
    "training_config.json",
    "eval.json",
    "README.md",
    "adapter/adapters.safetensors",
)

CHECKSUM_EXCLUDES = {"metadata.json"}


def train_skill_package(
    *,
    skill: str,
    output: Path,
    train_dataset: Path,
    eval_dataset: Path,
    name: str | None = None,
    version: str = "0.1.0",
    description: str | None = None,
    examples: Path | None = None,
    seed: int | None = None,
    force: bool = False,
    dry_run: bool = False,
) -> dict:
    output = output.resolve()
    train_dataset = train_dataset.resolve()
    eval_dataset = eval_dataset.resolve()
    if examples is not None:
        examples = examples.resolve()
    run_directory = output.parent / f".{output.name}.run"
    protected_before = _snapshot_files(
        _protected_input_paths(train_dataset=train_dataset, eval_dataset=eval_dataset)
    )
    resolved_name = name or skill.replace("_", " ").title()
    resolved_description = description or f"LoRA coding skill package for {resolved_name}."
    if dry_run:
        return {
            "status": "dry-run",
            "skill": skill,
            "output": str(output),
            "run_directory": str(run_directory),
            "protected_inputs": len(protected_before),
        }

    if output.exists() and any(output.iterdir()) and not force:
        raise FileExistsError(f"{output} exists; pass --force to replace it")
    if run_directory.exists() and force:
        shutil.rmtree(run_directory)
    if output.exists() and force:
        shutil.rmtree(output)

    adapter_dir, metadata = _train_skill_to_run_directory(
        skill=skill,
        train_dataset=train_dataset,
        run_directory=run_directory,
        seed=seed,
        force=force,
    )
    eval_summary = _evaluate_skill_adapter(
        skill=skill,
        dataset=eval_dataset,
        output=run_directory / "evaluation",
        adapter_root=run_directory / "adapters",
    )
    protected = _freeze_protected_inputs(protected_before)
    package_result = package_skill(
        skill_id=skill,
        name=resolved_name,
        adapter_dir=adapter_dir,
        output=output,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        eval_summary=eval_summary,
        version=version,
        description=resolved_description,
        examples=examples,
        force=force,
        protected_inputs=protected,
        source_artifacts={
            "run_directory": str(run_directory),
            "adapter_source_dir": str(adapter_dir),
            "evaluation_summary": str(eval_summary),
        },
        training_details={
            "command": metadata.get("training_command"),
            "elapsed_seconds": metadata.get("elapsed_seconds"),
        },
    )
    package_result["run_directory"] = str(run_directory)
    return package_result


def package_skill(
    *,
    skill_id: str,
    name: str,
    adapter_dir: Path,
    output: Path,
    train_dataset: Path,
    eval_dataset: Path,
    eval_summary: Path,
    version: str,
    description: str | None = None,
    examples: Path | None = None,
    force: bool = False,
    dry_run: bool = False,
    protected_inputs: dict | None = None,
    source_artifacts: dict | None = None,
    training_details: dict | None = None,
) -> dict:
    skill_id = _normalized_skill_id(skill_id)
    _nonempty("name", name)
    _nonempty("version", version)
    adapter_dir = adapter_dir.resolve()
    output = output.resolve()
    train_dataset = train_dataset.resolve()
    eval_dataset = eval_dataset.resolve()
    eval_summary = eval_summary.resolve()
    if examples is not None:
        examples = examples.resolve()

    adapter_weights = _adapter_weights_path(adapter_dir)
    adapter_metadata = _load_json_if_exists(adapter_dir / "metadata.json")
    adapter_config = adapter_dir / "adapter_config.json"
    eval_payload = _read_json(eval_summary)
    resolved_description = description or f"LoRA coding skill package for {name}."
    output_exists = output.exists()
    if output_exists and any(output.iterdir()) and not force:
        raise FileExistsError(f"{output} exists; pass --force to replace it")
    protected_before = _snapshot_files(
        _protected_input_paths(train_dataset=train_dataset, eval_dataset=eval_dataset)
    )

    manifests = _build_manifests(
        skill_id=skill_id,
        name=name,
        version=version,
        description=resolved_description,
        adapter_dir=adapter_dir,
        adapter_metadata=adapter_metadata,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        eval_summary=eval_summary,
        eval_payload=eval_payload,
        examples=examples,
        protected_inputs=protected_inputs,
        source_artifacts=source_artifacts,
        training_details=training_details,
    )
    if dry_run:
        return {
            "status": "dry-run",
            "output": str(output),
            "skill_id": skill_id,
            "files": sorted(manifests),
        }

    if output_exists:
        shutil.rmtree(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f"skillcortex-{skill_id}-") as directory:
        staging = Path(directory) / output.name
        _write_package(
            staging=staging,
            adapter_weights=adapter_weights,
            adapter_config=adapter_config if adapter_config.exists() else None,
            manifests=manifests,
            examples=examples,
        )
        metadata = _read_json(staging / "metadata.json")
        metadata["protected_inputs"] = protected_inputs or _freeze_protected_inputs(
            protected_before
        )
        metadata["checksums"] = _package_checksums(staging)
        (staging / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n"
        )
        validate_skill_package(staging)
        shutil.move(str(staging), str(output))
    return {
        "status": "complete",
        "output": str(output),
        "skill_id": skill_id,
        "version": version,
    }


def validate_skill_package(path: Path) -> dict:
    root = path.resolve()
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"package not found: {root}")
    missing = [name for name in REQUIRED_PACKAGE_FILES if not (root / name).exists()]
    if missing:
        raise ValueError(f"package is missing required files: {missing[0]}")

    skill_manifest = _read_yaml(root / "skill.yaml")
    metadata = _read_json(root / "metadata.json")
    training = _read_json(root / "training_config.json")
    evaluation = _read_json(root / "eval.json")

    if skill_manifest.get("schema_version") != "1":
        raise ValueError("skill.yaml schema_version must be '1'")
    if metadata.get("schema_version") != "1":
        raise ValueError("metadata.json schema_version must be '1'")
    if metadata.get("status") != "complete":
        raise ValueError("metadata.json status must be 'complete'")
    if skill_manifest.get("status") != "complete":
        raise ValueError("skill.yaml status must be 'complete'")

    for field in ("skill_id", "name", "version"):
        if skill_manifest.get(field) != metadata.get(field):
            raise ValueError(f"manifest mismatch for {field}")

    skill_base = skill_manifest.get("base") or {}
    metadata_base = metadata.get("base") or {}
    if skill_base.get("source_model") != metadata_base.get("source_model"):
        raise ValueError("manifest mismatch for base source_model")
    if skill_base.get("runtime_model") != metadata_base.get("runtime_model"):
        raise ValueError("manifest mismatch for base runtime_model")
    if (skill_manifest.get("adapter") or {}).get("trainable_parameters") != (
        metadata.get("adapter") or {}
    ).get("trainable_parameters"):
        raise ValueError("manifest mismatch for trainable_parameters")

    if training.get("seed") != (metadata.get("training") or {}).get("seed"):
        raise ValueError("training_config.json seed must match metadata.json")
    if evaluation.get("summary_path") != "eval.json":
        raise ValueError("eval.json summary_path must reference itself")
    if evaluation.get("dataset", {}).get("sha256") != (
        metadata.get("datasets") or {}
    ).get("eval", {}).get("sha256"):
        raise ValueError("eval dataset hash must match metadata.json")
    if not metadata.get("checksums"):
        raise ValueError("metadata.json must record package checksums")
    _validate_package_checksums(root, metadata["checksums"])

    protected = metadata.get("protected_inputs") or {}
    if not protected.get("all_unchanged"):
        raise ValueError("protected input snapshot must confirm inputs were unchanged")
    for file_path, snapshot in sorted((protected.get("files") or {}).items()):
        current = Path(file_path)
        if current.exists() and _sha256(current) != snapshot.get("after_sha256"):
            raise ValueError(f"protected input changed since packaging: {file_path}")

    examples_path = (skill_manifest.get("examples") or {}).get("path")
    if examples_path and not (root / examples_path).exists():
        raise ValueError("examples.jsonl is declared but missing")
    return {
        "status": "valid",
        "path": str(root),
        "skill_id": skill_manifest["skill_id"],
        "version": skill_manifest["version"],
    }


def _build_manifests(
    *,
    skill_id: str,
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
    protected_inputs: dict | None,
    source_artifacts: dict | None,
    training_details: dict | None,
) -> dict[str, object]:
    created_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    base = base_config()
    training_defaults = training_config()
    rank = int(adapter_metadata.get("rank") or training_defaults["skill_rank"])
    target_modules = adapter_metadata.get("target_modules") or training_defaults[
        "target_modules"
    ]
    trainable_parameters = int(adapter_metadata.get("trainable_parameters") or 0)
    train_dataset_hash = _sha256(train_dataset)
    eval_dataset_hash = _sha256(eval_dataset)
    examples_count = _line_count(examples) if examples else 0
    package_training = {
        "seed": int(adapter_metadata.get("seed") or training_defaults["seed"]),
        "batch_size": int(
            (adapter_metadata.get("config") or {}).get("batch_size")
            or training_defaults["batch_size"]
        ),
        "iterations": int(
            adapter_metadata.get("iterations") or training_defaults["iterations"]
        ),
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
        "dataset": {
            "path": str(eval_dataset),
            "sha256": eval_dataset_hash,
        },
        "summary_path": "eval.json",
        "source_summary_path": str(eval_summary),
        "modes": eval_payload.get("modes") or {},
        "tasks": eval_payload.get("tasks"),
        "hypothesis": eval_payload.get("hypothesis"),
    }
    metadata = {
        "schema_version": "1",
        "package_type": "skill",
        "status": "complete",
        "skill_id": skill_id,
        "name": name,
        "version": version,
        "created_at": created_at,
        "tool": {"name": "skillcortex", "version": __version__},
        "environment": {
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
        },
        "base": {
            "source_model": adapter_metadata.get("source_model")
            or base["source_model"],
            "runtime_model": adapter_metadata.get("base_model") or base["model"],
            "quantization": adapter_metadata.get("quantization") or "4bit",
        },
        "adapter": {
            "format": "mlx-lora",
            "rank": rank,
            "target_modules": target_modules,
            "trainable_parameters": trainable_parameters,
            "files": {
                "weights": "adapter/adapters.safetensors",
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
            },
            "eval": {
                "path": str(eval_dataset),
                "sha256": eval_dataset_hash,
            },
        },
        "evaluation": {
            "summary_path": "eval.json",
            "source_summary_path": str(eval_summary),
        },
        "source_artifacts": source_artifacts
        or {
            "adapter_source_dir": str(adapter_dir),
        },
        "protected_inputs": protected_inputs,
    }
    skill_yaml = {
        "schema_version": "1",
        "package_type": "skill",
        "skill_id": skill_id,
        "name": name,
        "version": version,
        "description": description,
        "status": "complete",
        "base": metadata["base"],
        "adapter": {
            "format": "mlx-lora",
            "path": "adapter/adapters.safetensors",
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
    if examples is not None:
        skill_yaml["examples"] = {"path": "examples.jsonl", "count": examples_count}
    readme = _build_readme(name, skill_id, version, description, metadata, evaluation)
    return {
        "skill.yaml": yaml.safe_dump(skill_yaml, sort_keys=False),
        "metadata.json": json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        "training_config.json": json.dumps(package_training, indent=2, sort_keys=True)
        + "\n",
        "eval.json": json.dumps(evaluation, indent=2, sort_keys=True) + "\n",
        "README.md": readme,
    }


def _write_package(
    *,
    staging: Path,
    adapter_weights: Path,
    adapter_config: Path | None,
    manifests: dict[str, str],
    examples: Path | None,
) -> None:
    (staging / "adapter").mkdir(parents=True, exist_ok=True)
    shutil.copy2(adapter_weights, staging / "adapter" / "adapters.safetensors")
    if adapter_config is not None:
        shutil.copy2(adapter_config, staging / "adapter" / "adapter_config.json")
    for relative_path, content in manifests.items():
        destination = staging / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(content)
    if examples is not None:
        shutil.copy2(examples, staging / "examples.jsonl")


def _train_skill_to_run_directory(
    *,
    skill: str,
    train_dataset: Path,
    run_directory: Path,
    seed: int | None,
    force: bool,
) -> tuple[Path, dict]:
    run_directory.mkdir(parents=True, exist_ok=True)
    examples = select_for_skill(load_jsonl(train_dataset), skill)
    training_directory = run_directory / "training-data"
    adapter_directory = run_directory / "adapters" / skill
    if adapter_directory.exists() and any(adapter_directory.iterdir()) and not force:
        raise FileExistsError(f"{adapter_directory} exists; pass --force to replace it")
    if adapter_directory.exists():
        shutil.rmtree(adapter_directory)
    dataset_directory = write_mlx_dataset(examples, training_directory)
    command = build_skill_command(skill, dataset_directory, adapter_directory, seed=seed)
    start = time.perf_counter()
    subprocess.run(command, check=True)
    metadata = research_metadata(
        skill,
        examples,
        rank=8,
        elapsed=time.perf_counter() - start,
        seed=seed,
        iterations=training_config()["iterations"],
    )
    metadata["trainable_parameters"] = _saved_parameter_count(adapter_directory)
    metadata["training_command"] = command
    (adapter_directory / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    return adapter_directory, metadata


def _evaluate_skill_adapter(
    *,
    skill: str,
    dataset: Path,
    output: Path,
    adapter_root: Path,
) -> Path:
    examples = select_for_skill(load_jsonl(dataset), skill)
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    model_cache: dict = {}
    raw_path = output / "results.jsonl"
    with raw_path.open("w") as handle:
        for example in examples:
            for mode in ("base", "single-skill"):
                try:
                    generation = (
                        infer("base", example.prompt, model_cache=model_cache)
                        if mode == "base"
                        else infer(
                            "single-skill",
                            example.prompt,
                            skill=skill,
                            adapter_root=adapter_root,
                            model_cache=model_cache,
                        )
                    )
                    text = extract_code(generation.generation)
                    syntax = (
                        python_syntax_valid(text)
                        if example.task_type != "test_generation"
                        else None
                    )
                    execution = None
                    if example.execution:
                        execution, _ = run_fixture(example.execution, text)
                    result = EvaluationResult(
                        example_id=example.id,
                        task_type=example.task_type,
                        mode=mode,
                        generation=text,
                        exact_match=text.strip() == example.target.strip(),
                        fuzzy_score=fuzzy_match(text, example.target),
                        syntax_valid=syntax,
                        execution_passed=execution,
                        latency_seconds=generation.latency_seconds,
                        selected_skills=generation.selected_skills,
                        active_adapter_count=generation.active_adapter_count,
                        active_adapter_parameters=generation.active_adapter_parameters,
                        prompt_tokens=generation.prompt_tokens,
                        generated_tokens=generation.generated_tokens,
                        peak_memory_bytes=generation.peak_memory_bytes,
                        benchmark_group=example.group,
                    )
                except Exception as error:  # ponytail: keep the rest of the evaluation running.
                    result = EvaluationResult(
                        example_id=example.id,
                        task_type=example.task_type,
                        mode=mode,
                        generation="",
                        exact_match=False,
                        fuzzy_score=0,
                        syntax_valid=None,
                        execution_passed=None,
                        latency_seconds=0,
                        selected_skills=[],
                        active_adapter_count=0,
                        active_adapter_parameters=0,
                        error=str(error),
                        benchmark_group=example.group,
                    )
                row = result.to_dict()
                rows.append(row)
                handle.write(json.dumps(row) + "\n")
    summary = aggregate_results(rows)
    tasks = {
        task: aggregate_results([row for row in rows if row["task_type"] == task])
        for task in sorted({row["task_type"] for row in rows})
    }
    payload = {
        "hypothesis": None,
        "modes": summary,
        "tasks": tasks,
    }
    (output / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    (output / "report.md").write_text(_evaluation_report(skill, summary, tasks))
    return output / "summary.json"


def _evaluation_report(skill: str, summary: dict, tasks: dict) -> str:
    lines = [
        f"# SkillCortex Single Skill Evaluation: {skill}",
        "",
        "| Mode | Count | Fuzzy | Exact | Syntax | Execution | Active params |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for mode in ("base", "single-skill"):
        if mode not in summary:
            continue
        value = summary[mode]
        lines.append(
            f"| {mode} | {value['count']} | {value['fuzzy_score']:.3f} | "
            f"{value['exact_match_rate']:.3f} | {_format(value['syntax_valid_rate'])} | "
            f"{_format(value['execution_pass_rate'])} | {value['active_adapter_parameters']:.0f} |"
        )
    lines.extend(["", "## By task", ""])
    for task, modes in tasks.items():
        scores = ", ".join(
            f"{mode}={values['fuzzy_score']:.3f}" for mode, values in sorted(modes.items())
        )
        lines.append(f"- `{task}`: {scores}")
    return "\n".join(lines) + "\n"


def _build_readme(
    name: str,
    skill_id: str,
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
            f"- Skill ID: `{skill_id}`",
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


def _protected_input_paths(*, train_dataset: Path, eval_dataset: Path) -> list[Path]:
    paths = {
        train_dataset.resolve(),
        eval_dataset.resolve(),
        (CONFIG_DIR / "base.yaml").resolve(),
        (CONFIG_DIR / "training.yaml").resolve(),
        (CONFIG_DIR / "skill_registry.json").resolve(),
        (CONFIG_DIR / "skills.yaml").resolve(),
    }
    for path in sorted((ARTIFACT_DIR / "adapters").rglob("*")):
        if path.is_file():
            paths.add(path.resolve())
    benchmark_root = ROOT / "data" / "benchmarks"
    for path in sorted(benchmark_root.rglob("*")):
        if path.is_file():
            paths.add(path.resolve())
    return sorted(paths)


def _snapshot_files(paths: list[Path]) -> dict[str, str]:
    return {str(path): _sha256(path) for path in paths}


def _freeze_protected_inputs(before: dict[str, str]) -> dict:
    after = {path: _sha256(Path(path)) for path in before}
    files = {
        path: {
            "before_sha256": before[path],
            "after_sha256": after[path],
            "unchanged": before[path] == after[path],
        }
        for path in sorted(before)
    }
    changed = [path for path, snapshot in files.items() if not snapshot["unchanged"]]
    if changed:
        raise RuntimeError(f"protected input changed during packaging: {changed[0]}")
    return {"all_unchanged": True, "files": files}


def _package_checksums(root: Path) -> dict[str, str]:
    checksums = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if relative in CHECKSUM_EXCLUDES:
            continue
        checksums[relative] = _sha256(path)
    return checksums


def _validate_package_checksums(root: Path, checksums: dict[str, str]) -> None:
    for relative, expected in sorted(checksums.items()):
        path = root / relative
        if not path.exists():
            raise ValueError(f"checksummed file is missing: {relative}")
        if _sha256(path) != expected:
            raise ValueError(f"checksum mismatch for {relative}")


def _normalized_skill_id(skill_id: str) -> str:
    _nonempty("skill_id", skill_id)
    normalized = skill_id.strip().lower().replace("-", "_")
    if not all(char.isalnum() or char == "_" for char in normalized):
        raise ValueError("skill_id must contain only letters, numbers, dashes, or underscores")
    return normalized


def _nonempty(name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be non-empty")


def _adapter_weights_path(adapter_dir: Path) -> Path:
    path = adapter_dir / "adapters.safetensors"
    if not path.exists():
        raise FileNotFoundError(f"adapter weights not found: {path}")
    return path


def _sha256(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _line_count(path: Path | None) -> int:
    if path is None:
        return 0
    return sum(1 for line in path.read_text().splitlines() if line.strip())


def _load_json_if_exists(path: Path) -> dict:
    if not path.exists():
        return {}
    return _read_json(path)


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise ValueError(f"{path} contains invalid JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _read_yaml(path: Path) -> dict:
    value = yaml.safe_load(path.read_text()) or {}
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return value