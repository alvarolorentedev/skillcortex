from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Callable

from ..agent import run_agent
from ..catalog import compose_from_route, route_task
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
    if parsed.command == "route":
        return route_task(
            skills_dir=Path(parsed.skills_dir),
            repo=Path(parsed.repo),
            task=parsed.task,
            explain=parsed.explain,
            current_base_model=parsed.base_model,
        )
    if parsed.command == "compose-from-route":
        return compose_from_route(
            skills_dir=Path(parsed.skills_dir),
            repo=Path(parsed.repo),
            task=parsed.task,
            runtime_out=Path(parsed.runtime_out),
            explain=parsed.explain,
            allow_base=parsed.allow_base,
            overwrite=parsed.overwrite,
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
        if bool(parsed.runtime) == bool(parsed.skills_dir):
            raise ValueError("agent run requires exactly one of --runtime or --skills-dir")
        if parsed.skills_dir:
            tasks = collect_agent_tasks(parsed.task)
            if not tasks:
                raise ValueError("agent run --skills-dir --dry-run requires --task")
            if len(tasks) != 1:
                raise ValueError("agent run --skills-dir --dry-run accepts one --task")
            if parsed.writes == "on" or (not parsed.dry_run and parsed.writes != "confirm"):
                raise ValueError(
                    "dynamic skills-dir execution only supports --dry-run or --write-mode confirm"
                )
            return _run_dynamic_agent(
                skills_dir=Path(parsed.skills_dir),
                repo=Path(parsed.repo),
                task=tasks[0],
                runtime_out=Path(parsed.compose_runtime_out)
                if parsed.compose_runtime_out
                else _default_dynamic_runtime_path(
                    Path(parsed.repo), Path(parsed.skills_dir), tasks[0]
                ),
                writes=parsed.writes,
                test_command=parsed.test_command,
                trace_out=Path(parsed.trace_out) if parsed.trace_out else None,
                dry_run=parsed.dry_run,
                overwrite=parsed.overwrite,
            )
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


def _default_dynamic_runtime_path(repo: Path, skills_dir: Path, task: str) -> Path:
    repo_root = repo.resolve()
    key = f"{skills_dir.resolve()}|{repo_root}|{task}"
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
    return repo_root / ".skillcortex" / "runtimes" / digest


def _run_dynamic_agent(
    *,
    skills_dir: Path,
    repo: Path,
    task: str,
    runtime_out: Path,
    writes: str,
    test_command: str | None,
    trace_out: Path | None,
    dry_run: bool,
    overwrite: bool,
) -> dict:
    composition = compose_from_route(
        skills_dir=skills_dir,
        repo=repo,
        task=task,
        runtime_out=runtime_out,
        explain=True,
        overwrite=overwrite,
    )
    if composition["validation_status"] != "passed":
        raise ValueError(f"runtime validation failed: {composition['validation_status']}")
    agent_result = run_agent(
        runtime_path=runtime_out,
        repo=repo,
        task=[task],
        writes=writes,
        test_command=test_command,
        trace_out=None,
        dry_run=dry_run,
    )
    agent_status = (
        "dry_run_completed"
        if dry_run
        else "review_required"
        if agent_result.get("review_artifact_path")
        else "completed"
    )
    result = {
        "mode": "dynamic_agent",
        "routing_decision": composition["routing_decision"],
        "selected_skills": composition["selected_skills"],
        "runtime_out": composition["runtime_out"],
        "composition_strategy": composition["composition_strategy"],
        "composition_status": composition["composition_status"],
        "validation_status": composition["validation_status"],
        "agent_execution_status": agent_status,
        "write_mode": "dry_run" if dry_run else "confirm",
        "agent_result": agent_result,
        "trace_out": str(trace_out.resolve()) if trace_out is not None else None,
        "warnings": composition["warnings"],
        "errors": composition["errors"],
    }
    if trace_out is not None:
        trace_out = trace_out.resolve()
        trace_out.parent.mkdir(parents=True, exist_ok=True)
        trace_out.write_text(json.dumps(result, indent=2) + "\n")
    return result
