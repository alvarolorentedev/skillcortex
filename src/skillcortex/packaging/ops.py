from __future__ import annotations

import json
import shutil
import subprocess
import time
from pathlib import Path

from ..backends.legacy import (
    EvaluationResult,
    evaluator_backend,
    infer,
    load_jsonl,
    research_metadata,
    saved_parameter_count,
    select_for_skill,
    trainer_backend,
    training_config,
    write_mlx_dataset,
)
from ..training import evaluate_product_skill_adapter, train_product_skill_to_run_directory


aggregate_results = evaluator_backend.aggregate_results
extract_code = evaluator_backend.extract_code
fuzzy_match = evaluator_backend.fuzzy_match
python_syntax_valid = evaluator_backend.python_syntax_valid
run_fixture = evaluator_backend.run_fixture
build_skill_command = trainer_backend.build_skill_command


def train_generic_skill_to_run_directory(
    *,
    skill_id: str,
    train_dataset: Path,
    run_directory: Path,
    seed: int | None,
    force: bool,
) -> tuple[Path, dict]:
    return train_product_skill_to_run_directory(
        skill_id=skill_id,
        train_dataset=train_dataset,
        run_directory=run_directory,
        seed=seed,
        force=force,
    )


def evaluate_generic_skill_adapter(
    *,
    skill_id: str,
    dataset: Path,
    output: Path,
    adapter_dir: Path,
) -> Path:
    return evaluate_product_skill_adapter(
        skill_id=skill_id,
        dataset=dataset,
        output=output,
        adapter_dir=adapter_dir,
    )


def train_skill_to_run_directory(
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
    metadata["trainable_parameters"] = saved_parameter_count(adapter_directory)
    metadata["training_command"] = command
    (adapter_directory / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    return adapter_directory, metadata


def evaluate_skill_adapter(
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
                    syntax = python_syntax_valid(text) if example.task_type != "test_generation" else None
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
    tasks = {task: aggregate_results([row for row in rows if row["task_type"] == task]) for task in sorted({row["task_type"] for row in rows})}
    payload = {"hypothesis": None, "modes": summary, "tasks": tasks}
    (output / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    (output / "report.md").write_text(evaluation_report(skill, summary, tasks))
    return output / "summary.json"


def evaluation_report(skill: str, summary: dict, tasks: dict) -> str:
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


def _format(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"
