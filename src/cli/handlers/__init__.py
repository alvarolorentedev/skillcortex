from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

from ...agent import run_agent
from ...catalog import compose_from_folder, compose_from_route, route_task
from ...composer import compose_slm_packages
from ...composer_app import run_composer_app, write_support_bundle
from ...packaging.importers import import_lora
from ...packaging import train_slm_package, validate_slm_package
from ...runtime import DynamicRuntime, SlmRuntime, serve_runtime, validate_runtime_bundle
from ...shared.product import environment_diagnostics
from ...shared.project import (
    configured_loras,
    init_project,
    load_project_config,
    project_cache_dir,
    project_dataset,
    project_slms_dir,
)
from ...shared.provisioning import provision_backend
from ..common import csv_paths, infer_payload
from .dynamic_agent import default_dynamic_runtime_path, run_dynamic_agent
from .factory import (
    FACTORY_DEPENDENCY_GUARDED_COMMANDS,
    ensure_factory_prerequisites,
    execute_factory_command,
)


TaskProviderFactory = Callable[[], object]


def execute_command(
    parsed: argparse.Namespace,
    *,
    collect_agent_tasks: Callable[[list[str] | None], list[str] | None],
    stream_agent_tasks: TaskProviderFactory,
) -> dict:
    command = _resolved_command(parsed)
    if parsed.command == "factory" and command in FACTORY_DEPENDENCY_GUARDED_COMMANDS:
        ensure_factory_prerequisites(parsed, environment_diagnostics_fn=environment_diagnostics)
    if command == "init":
        return init_project(Path(parsed.project))
    if command == "doctor":
        return _doctor(parsed)
    if command == "provision-backend":
        return provision_backend(
            backend=parsed.backend,
            workspace_root=Path(parsed.workspace) if parsed.workspace else None,
            dry_run=parsed.dry_run,
        )
    if command == "composer-app":
        return _composer_app(parsed)
    if command == "compose-folder":
        return compose_from_folder(
            folder=Path(parsed.folder),
            task=parsed.task,
            workspace_root=Path(parsed.workspace) if parsed.workspace else None,
            slms_dir=Path(parsed.slms_dir) if parsed.slms_dir else None,
            runtime_name=parsed.runtime_name,
            export_descriptor=Path(parsed.export_descriptor) if parsed.export_descriptor else None,
            allow_base=parsed.allow_base,
            overwrite=parsed.overwrite,
            product_mode=parsed.product_mode,
        )
    factory_result = execute_factory_command(
        command,
        parsed,
        environment_diagnostics_fn=environment_diagnostics,
        train_slm_package_fn=train_slm_package,
        validate_slm_package_fn=validate_slm_package,
    )
    if factory_result is not None:
        return factory_result
    if command == "compose-slms":
        return compose_slm_packages(
            slms=csv_paths(parsed.slms),
            strategy=parsed.strategy,
            output=Path(parsed.output),
            registry=Path(parsed.registry) if parsed.registry else None,
            force=parsed.force,
            dry_run=parsed.dry_run,
        )
    if command == "route":
        return route_task(
            slms_dir=Path(parsed.slms_dir),
            repo=Path(parsed.repo),
            task=parsed.task,
            explain=parsed.explain,
            current_base_model=parsed.base_model,
        )
    if command == "compose-from-route":
        return compose_from_route(
            slms_dir=Path(parsed.slms_dir),
            repo=Path(parsed.repo),
            task=parsed.task,
            runtime_out=Path(parsed.runtime_out),
            explain=parsed.explain,
            allow_base=parsed.allow_base,
            overwrite=parsed.overwrite,
        )
    if command == "validate-runtime":
        return validate_runtime_bundle(Path(parsed.runtime))
    if command == "infer":
        return _infer(parsed)
    if command == "loras":
        return _loras(parsed)
    if command == "serve":
        return _serve(parsed)
    if command == "agent":
        return _agent(parsed, collect_agent_tasks, stream_agent_tasks)
    raise ValueError(f"unknown command: {command}")


def _resolved_command(parsed: argparse.Namespace) -> str:
    if parsed.command == "factory":
        return parsed.factory_command
    return parsed.command


def _doctor(parsed) -> dict:
    diagnostics = environment_diagnostics(
        workspace_root=Path(parsed.workspace) if parsed.workspace else None,
        product_mode=parsed.product_mode,
        include_support_bundle=parsed.export_support_bundle,
    )
    if parsed.export_support_bundle:
        diagnostics["support_bundle"] = {
            "available": True,
            "path": write_support_bundle(
                workspace_root=Path(parsed.workspace) if parsed.workspace else None,
                runtime_name="doctor",
                compose_result=None,
                state=None,
                diagnostics=diagnostics,
                scan_summary=None,
                product_error=None,
                bundle_path=Path(parsed.support_bundle_path) if parsed.support_bundle_path else None,
            ),
        }
    return diagnostics


def _composer_app(parsed) -> dict:
    return run_composer_app(
        folder=Path(parsed.folder),
        workspace_root=Path(parsed.workspace) if parsed.workspace else None,
        slms_dir=Path(parsed.slms_dir) if parsed.slms_dir else None,
        task=parsed.task,
        runtime_name=parsed.runtime_name,
        outcome=parsed.outcome,
        run_target=parsed.run_target,
        prompt=parsed.prompt,
        export_descriptor=Path(parsed.export_descriptor) if parsed.export_descriptor else None,
        export_logs=parsed.export_logs,
        allow_base=parsed.allow_base,
        overwrite=parsed.overwrite,
        host=parsed.host,
        port=parsed.port,
        writes=parsed.writes,
        test_command=parsed.test_command,
        trace_out=Path(parsed.trace_out) if parsed.trace_out else None,
        dry_run=parsed.dry_run,
    )


def _infer(parsed) -> dict:
    _apply_project_runtime_defaults(parsed)
    if bool(parsed.runtime) == bool(parsed.slms_dir):
        raise ValueError("infer requires exactly one of --runtime or --slms-dir")
    payload = infer_payload(parsed)
    if parsed.slms_dir:
        return DynamicRuntime.load(
            Path(parsed.slms_dir),
            allow_remote_loras=parsed.allow_remote_loras,
            cache_dir=Path(parsed.lora_cache_dir) if parsed.lora_cache_dir else None,
        ).infer(
            messages=payload["messages"],
            max_tokens=payload.get("max_tokens"),
            temperature=payload.get("temperature"),
            dry_run=parsed.dry_run,
        )
    return SlmRuntime.load(Path(parsed.runtime)).infer(
        messages=payload["messages"],
        task_type=payload.get("task_type"),
        semantic_family=payload.get("semantic_family"),
        slm_override=payload.get("slm_override"),
        max_tokens=payload.get("max_tokens"),
        temperature=payload.get("temperature"),
        dry_run=parsed.dry_run,
    )


def _serve(parsed) -> dict:
    _apply_project_runtime_defaults(parsed)
    if bool(parsed.runtime) == bool(parsed.slms_dir):
        raise ValueError("serve requires exactly one of --runtime or --slms-dir")
    return serve_runtime(
        runtime_path=Path(parsed.runtime) if parsed.runtime else None,
        slms_dir=Path(parsed.slms_dir) if parsed.slms_dir else None,
        allow_remote_loras=parsed.allow_remote_loras,
        cache_dir=Path(parsed.lora_cache_dir) if parsed.lora_cache_dir else None,
        host=parsed.host,
        port=parsed.port,
        dry_run=parsed.dry_run,
    )


def _agent(parsed, collect_agent_tasks, stream_agent_tasks) -> dict:
    if parsed.agent_command != "run":
        raise ValueError(f"unknown agent command: {parsed.agent_command}")
    _apply_project_runtime_defaults(parsed)
    if not parsed.repo and parsed.slms_dir:
        parsed.repo = "."
    if bool(parsed.runtime) == bool(parsed.slms_dir):
        raise ValueError("agent run requires exactly one of --runtime or --slms-dir")
    if not parsed.repo:
        raise ValueError("agent run requires --repo unless .slmcortex.yaml provides project defaults")
    if parsed.slms_dir:
        return _dynamic_agent(parsed, collect_agent_tasks, stream_agent_tasks)
    tasks = collect_agent_tasks(parsed.task)
    return run_agent(
        runtime_path=Path(parsed.runtime),
        repo=Path(parsed.repo),
        task=tasks,
        writes=parsed.writes,
        test_command=parsed.test_command,
        trace_out=Path(parsed.trace_out) if parsed.trace_out else None,
        dry_run=parsed.dry_run,
        task_provider=None if tasks else stream_agent_tasks(),
    )


def _dynamic_agent(parsed, collect_agent_tasks, stream_agent_tasks) -> dict:
    tasks = collect_agent_tasks(parsed.task)
    task_provider = None
    if not tasks:
        task_provider = stream_agent_tasks()
        first_task = task_provider()
        if not first_task:
            raise ValueError("at least one task is required")
        tasks = [first_task]
    return run_dynamic_agent(
        slms_dir=Path(parsed.slms_dir),
        repo=Path(parsed.repo),
        task=tasks,
        runtime_out=Path(parsed.compose_runtime_out)
        if parsed.compose_runtime_out
        else default_dynamic_runtime_path(Path(parsed.repo), Path(parsed.slms_dir), tasks[0]),
        writes=parsed.writes,
        test_command=parsed.test_command,
        trace_out=Path(parsed.trace_out) if parsed.trace_out else None,
        dry_run=parsed.dry_run,
        task_provider=task_provider,
        overwrite=parsed.overwrite,
        compose_from_route_fn=compose_from_route,
        run_agent_fn=run_agent,
    )


def _loras(parsed) -> dict:
    if parsed.lora_command != "download":
        raise ValueError(f"unknown loras command: {parsed.lora_command}")
    return _download_loras(parsed)


def _download_loras(parsed) -> dict:
    config = load_project_config()
    items = list(parsed.items or [])
    if not items and not parsed.all:
        raise ValueError("choose LoRA names or use --all")
    if parsed.all and items:
        raise ValueError("use LoRA names or --all, not both")
    hf_items = [item for item in items if item.startswith("hf://")]
    if hf_items and (len(items) != 1 or not parsed.as_name):
        raise ValueError("one-off Hugging Face downloads require exactly one hf:// URL and --as")
    if parsed.as_name and not hf_items:
        raise ValueError("--as is only valid with a one-off hf:// URL")

    slms_dir = project_slms_dir(config) or Path(".slmcortex/slms")
    cache_dir = project_cache_dir(config)
    train_dataset = project_dataset(config, "train_dataset", "train.jsonl")
    eval_dataset = project_dataset(config, "eval_dataset", "eval.jsonl")
    entries = _download_entries(config, items, parsed.all, parsed.as_name)
    downloaded = []
    results = []
    for slm_id, entry in entries:
        source = entry.get("source")
        if not isinstance(source, str) or not source.startswith("hf://"):
            raise ValueError(f"{slm_id}: source must be hf://owner/repo[@revision]")
        result = import_lora(
            source=source,
            slm_id=slm_id,
            name=str(entry.get("name") or slm_id.replace("_", " ").title()),
            output=slms_dir / slm_id,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            description=entry.get("description"),
            cache_dir=cache_dir,
            force=parsed.force,
        )
        downloaded.append(slm_id)
        results.append(result)
    return {
        "status": "complete",
        "downloaded": downloaded,
        "slms_dir": str(slms_dir.resolve()),
        "lora_cache_dir": str(cache_dir.resolve()),
        "results": results,
    }


def _download_entries(config: dict, items: list[str], download_all: bool, as_name: str | None) -> list[tuple[str, dict]]:
    if items and items[0].startswith("hf://"):
        return [(as_name, {"source": items[0], "name": as_name})]
    loras = configured_loras(config)
    if download_all:
        if not loras:
            raise ValueError("no LoRAs configured in .slmcortex.yaml")
        names = sorted(loras)
    else:
        names = items
    missing = [name for name in names if name not in loras]
    if missing:
        raise ValueError(f"unknown LoRA(s): {', '.join(missing)}")
    return [(name, loras[name]) for name in names]


def _apply_project_runtime_defaults(parsed) -> None:
    if parsed.runtime or parsed.slms_dir:
        return
    slms_dir = project_slms_dir(load_project_config())
    if slms_dir is not None:
        parsed.slms_dir = str(slms_dir)
        if hasattr(parsed, "allow_remote_loras"):
            parsed.allow_remote_loras = True
