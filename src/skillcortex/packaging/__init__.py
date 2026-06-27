import json
import shutil
import tempfile
from pathlib import Path

from ..datasets import ensure_datasets_are_trainable
from ..shared.io import load_json_if_exists as _load_json_if_exists, read_json as _read_json, read_yaml as _read_yaml
from .artifacts import (
    adapter_weights_path as _adapter_weights_path,
    freeze_protected_inputs as _freeze_protected_inputs,
    package_checksums as _package_checksums,
    protected_input_paths as _protected_input_paths,
    snapshot_files as _snapshot_files,
)
from .composition import validate_composition_metadata, normalized_skill_id as _normalized_skill_id
from .manifests import build_manifests as _build_manifests, write_package as _write_package
from .ops import (
    evaluate_generic_skill_adapter as _evaluate_generic_skill_adapter,
    evaluate_skill_adapter as _evaluate_skill_adapter,
    train_generic_skill_to_run_directory as _train_generic_skill_to_run_directory,
    train_skill_to_run_directory as _train_skill_to_run_directory,
)
from .validation import validate_skill_package


def train_skill_package(
    *,
    skill: str,
    mode: str = "preset",
    output: Path,
    train_dataset: Path,
    eval_dataset: Path,
    name: str | None = None,
    version: str = "0.1.0",
    description: str | None = None,
    examples: Path | None = None,
    composition: dict | None = None,
    seed: int | None = None,
    force: bool = False,
    dry_run: bool = False,
) -> dict:
    output = output.resolve()
    train_dataset = train_dataset.resolve()
    eval_dataset = eval_dataset.resolve()
    if examples is not None:
        examples = examples.resolve()
    validation = ensure_datasets_are_trainable(train_dataset, eval_dataset=eval_dataset)
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
            "dataset_validation": {
                "status": validation["status"],
                "warnings": validation["warnings"],
            },
        }
    if output.exists() and any(output.iterdir()) and not force:
        raise FileExistsError(f"{output} exists; pass --force to replace it")
    if run_directory.exists() and force:
        shutil.rmtree(run_directory)
    if output.exists() and force:
        shutil.rmtree(output)
    if mode == "generic":
        adapter_dir, metadata = _train_generic_skill_to_run_directory(
            skill_id=skill,
            train_dataset=train_dataset,
            run_directory=run_directory,
            seed=seed,
            force=force,
        )
        eval_summary = _evaluate_generic_skill_adapter(
            skill_id=skill,
            dataset=eval_dataset,
            output=run_directory / "evaluation",
            adapter_dir=adapter_dir,
        )
    else:
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
        composition=composition,
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
    composition: dict | None = None,
    force: bool = False,
    dry_run: bool = False,
    protected_inputs: dict | None = None,
    source_artifacts: dict | None = None,
    training_details: dict | None = None,
) -> dict:
    skill_id = _normalized_skill_id(skill_id)
    from .composition import nonempty

    nonempty("name", name)
    nonempty("version", version)
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
        composition=composition,
        protected_inputs=protected_inputs,
        source_artifacts=source_artifacts,
        training_details=training_details,
    )
    if dry_run:
        return {"status": "dry-run", "output": str(output), "skill_id": skill_id, "files": sorted(manifests)}
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
        metadata["protected_inputs"] = protected_inputs or _freeze_protected_inputs(protected_before)
        metadata["checksums"] = _package_checksums(staging)
        (staging / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
        validate_skill_package(staging)
        shutil.move(str(staging), str(output))
    return {"status": "complete", "output": str(output), "skill_id": skill_id, "version": version}
