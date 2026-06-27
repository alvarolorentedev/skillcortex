from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

from ..agent import run_agent
from ..composer import compose_skill_packages
from ..dataset_factory import generate_dataset_bundle
from ..datasets import validate_dataset_command
from ..packaging import package_skill, train_skill_package, validate_skill_package
from ..runtime import SkillRuntime, serve_runtime, validate_runtime_bundle
from .common import csv_paths, default_dataset_outputs, infer_payload, package_composition, resolve_train_skill


TaskProviderFactory = Callable[[], object]


def execute_command(
    parsed: argparse.Namespace,
    *,
    collect_agent_tasks: Callable[[list[str] | None], list[str] | None],
    stream_agent_tasks: TaskProviderFactory,
) -> dict:
    if parsed.command == "generate-dataset":
        default_output, default_eval_output = default_dataset_outputs(parsed.skill_id)
        return generate_dataset_bundle(
            skill_id=parsed.skill_id,
            domain=parsed.domain,
            task_type=parsed.task_type,
            num_examples=parsed.num_examples,
            output=Path(parsed.output) if parsed.output else default_output,
            eval_output=Path(parsed.eval_output) if parsed.eval_output else default_eval_output,
            eval_size=parsed.eval_size,
            seed=parsed.seed,
            report_output=Path(parsed.report_output) if parsed.report_output else None,
        )
    if parsed.command == "validate-dataset":
        return validate_dataset_command(
            Path(parsed.dataset),
            eval_dataset=Path(parsed.eval_dataset) if parsed.eval_dataset else None,
            min_target_length=parsed.min_target_length,
            report_output=Path(parsed.report_output) if parsed.report_output else None,
        )
    if parsed.command == "train-skill":
        mode, skill_id, composition, defaults_applied = resolve_train_skill(parsed)
        result = train_skill_package(
            skill=skill_id,
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
            result["warnings"] = [
                "default composition metadata applied for arbitrary train-skill"
            ]
        return result
    if parsed.command == "package-skill":
        return package_skill(
            skill_id=parsed.skill_id,
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
    if parsed.command == "compose-skills":
        return compose_skill_packages(
            skills=csv_paths(parsed.skills),
            strategy=parsed.strategy,
            output=Path(parsed.output),
            registry=Path(parsed.registry) if parsed.registry else None,
            force=parsed.force,
            dry_run=parsed.dry_run,
        )
    if parsed.command == "validate-runtime":
        return validate_runtime_bundle(Path(parsed.runtime))
    if parsed.command == "infer":
        payload = infer_payload(parsed)
        return SkillRuntime.load(Path(parsed.runtime)).infer(
            messages=payload["messages"],
            task_type=payload.get("task_type"),
            semantic_family=payload.get("semantic_family"),
            skill_override=payload.get("skill_override"),
            max_tokens=payload.get("max_tokens"),
            temperature=payload.get("temperature"),
            dry_run=parsed.dry_run,
        )
    if parsed.command == "serve":
        return serve_runtime(
            runtime_path=Path(parsed.runtime),
            host=parsed.host,
            port=parsed.port,
            dry_run=parsed.dry_run,
        )
    if parsed.command == "agent":
        if parsed.agent_command != "run":
            raise ValueError(f"unknown agent command: {parsed.agent_command}")
        tasks = collect_agent_tasks(parsed.task)
        task_provider = None if tasks else stream_agent_tasks()
        return run_agent(
            runtime_path=Path(parsed.runtime),
            repo=Path(parsed.repo),
            task=tasks,
            writes=parsed.writes,
            test_command=parsed.test_command,
            trace_out=Path(parsed.trace_out) if parsed.trace_out else None,
            dry_run=parsed.dry_run,
            task_provider=task_provider,
        )
    return validate_skill_package(Path(parsed.path))
