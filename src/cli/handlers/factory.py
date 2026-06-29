from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from ...dataset_factory import generate_dataset_bundle
from ...datasets import validate_dataset_command
from ...packaging import package_slm, train_slm_package, validate_slm_package
from ...packaging.importers import import_lora
from ...shared.product import environment_diagnostics
from ..common import default_dataset_outputs, package_composition, resolve_train_slm


FACTORY_DEPENDENCY_GUARDED_COMMANDS = {"train-slm", "train-plasticity-lora"}


def ensure_factory_prerequisites(parsed) -> None:
    diagnostics = environment_diagnostics(
        workspace_root=Path(parsed.workspace) if getattr(parsed, "workspace", None) else None,
        product_mode="factory",
    )
    missing = [
        row["name"] for row in diagnostics.get("optional_factory_dependencies", []) if not row["available"]
    ]
    if not missing:
        return
    raise ValueError(
        "factory mode prerequisites missing for training workflows: "
        + ", ".join(missing)
        + ". Run 'slmcortex factory doctor' to inspect the environment and install the optional extras before retrying."
    )


def execute_factory_command(command: str, parsed) -> dict | None:
    if command == "generate-dataset":
        default_output, default_eval_output = default_dataset_outputs(parsed.slm_id)
        return generate_dataset_bundle(
            slm_id=parsed.slm_id,
            domain=parsed.domain,
            task_type=parsed.task_type,
            num_examples=parsed.num_examples,
            output=Path(parsed.output) if parsed.output else default_output,
            eval_output=Path(parsed.eval_output) if parsed.eval_output else default_eval_output,
            eval_size=parsed.eval_size,
            seed=parsed.seed,
            report_output=Path(parsed.report_output) if parsed.report_output else None,
        )
    if command == "validate-dataset":
        return validate_dataset_command(
            Path(parsed.dataset),
            eval_dataset=Path(parsed.eval_dataset) if parsed.eval_dataset else None,
            min_target_length=parsed.min_target_length,
            report_output=Path(parsed.report_output) if parsed.report_output else None,
        )
    if command == "train-slm":
        return _train_slm(parsed)
    if command == "train-plasticity-lora":
        return _train_plasticity_lora(parsed)
    if command == "import-lora":
        return import_lora(
            source=parsed.source,
            slm_id=parsed.slm_id,
            name=parsed.name,
            output=Path(parsed.output),
            train_dataset=Path(parsed.train_dataset),
            eval_dataset=Path(parsed.eval_dataset),
            version=parsed.version,
            description=parsed.description,
            cache_dir=Path(parsed.cache_dir) if parsed.cache_dir else None,
            max_download_bytes=parsed.max_download_bytes,
            force=parsed.force,
        )
    if command == "package-slm":
        return package_slm(
            slm_id=parsed.slm_id,
            name=parsed.name,
            adapter_dir=Path(parsed.adapter_dir),
            output=Path(parsed.output),
            train_dataset=Path(parsed.train_dataset),
            eval_dataset=Path(parsed.eval_dataset),
            eval_summary=Path(parsed.eval_summary),
            version=parsed.version,
            description=parsed.description,
            examples=Path(parsed.examples) if parsed.examples else None,
            composition=package_composition(parsed),
            force=parsed.force,
            dry_run=parsed.dry_run,
        )
    return validate_slm_package(Path(parsed.path)) if command == "validate-slm-package" else None


def _train_slm(parsed) -> dict:
    mode, slm_id, composition, defaults_applied = resolve_train_slm(parsed)
    result = train_slm_package(
        slm=slm_id,
        mode=mode,
        output=Path(parsed.output),
        train_dataset=Path(parsed.train_dataset),
        eval_dataset=Path(parsed.eval_dataset),
        name=parsed.name,
        version=parsed.version,
        description=parsed.description,
        examples=Path(parsed.examples) if parsed.examples else None,
        composition=composition,
        seed=parsed.seed,
        force=parsed.force,
        dry_run=parsed.dry_run,
    )
    if defaults_applied:
        result["defaults_applied"] = defaults_applied
        result["warnings"] = ["default composition metadata applied for arbitrary train-slm"]
    return result


def _train_plasticity_lora(parsed) -> dict:
    output = _plasticity_output(parsed)
    if parsed.dry_run:
        return {
            "status": "dry-run",
            "slm": parsed.slm_id,
            "output": str(output.resolve()),
            "publish_dir": str(output.parent.resolve()),
        }
    with tempfile.TemporaryDirectory(prefix=f"slmcortex-{parsed.slm_id}-publish-") as directory:
        staging = Path(directory) / parsed.slm_id
        result = train_slm_package(
            slm=parsed.slm_id,
            mode="generic",
            output=staging,
            train_dataset=Path(parsed.prompt_file),
            eval_dataset=Path(parsed.eval_dataset or parsed.prompt_file),
            name=parsed.name,
            version=parsed.version,
            description=parsed.description,
            composition={
                "capabilities": {"allowed_task_types": ["python_generation"]},
                "activation": {
                    "default_route_type": "adapter",
                    "scope": "task",
                    "semantic_families": [],
                },
                "compatibility": {"compatible_slms": [], "incompatible_slms": []},
                "routing": {"tasks": {}},
            },
            seed=parsed.seed,
            force=True,
            dry_run=False,
        )
        validate_slm_package(staging)
        if output.exists():
            if not parsed.force:
                raise FileExistsError(f"{output} exists; pass --force to replace it")
            shutil.rmtree(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(staging), str(output))
    result.update(
        {
            "output": str(output.resolve()),
            "publish_dir": str(output.parent.resolve()),
            "validation_status": "valid",
        }
    )
    return result


def _plasticity_output(parsed) -> Path:
    if bool(parsed.output) == bool(parsed.publish_dir):
        raise ValueError("train-plasticity-lora requires exactly one of --output or --publish-dir")
    if parsed.output:
        return Path(parsed.output)
    return Path(parsed.publish_dir) / parsed.slm_id
