import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from .data import load_jsonl
from .inference import infer
from .metrics import (
    aggregate_results,
    classify_hypothesis,
    extract_code,
    fuzzy_match,
    paired_execution_comparison,
    python_syntax_valid,
)
from .schemas import EvaluationResult
from .utils import run_fixture, write_json

PRIMARY_SKILL = {
    "python_generation": "python_skill",
    "debugging": "debugging_skill",
    "test_generation": "test_generation_skill",
}
MODES = ("base", "generic", "single-skill", "lattice")


def evaluate(
    dataset: str | Path,
    *,
    output: str | Path | None = None,
    dry_run: bool = False,
) -> Path:
    examples = load_jsonl(dataset)
    output = Path(output) if output else _default_output()
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    raw_path = output / "results.jsonl"
    with raw_path.open("w") as handle:
        for index, example in enumerate(examples):
            modes = MODES[index % len(MODES) :] + MODES[: index % len(MODES)]
            for mode in modes:
                skill = (
                    PRIMARY_SKILL[example.task_type] if mode == "single-skill" else None
                )
                try:
                    generation = infer(
                        mode, example.prompt, skill=skill, dry_run=dry_run
                    )
                    text = (
                        example.target
                        if dry_run
                        else extract_code(generation.generation)
                    )
                    syntax = (
                        python_syntax_valid(text)
                        if example.task_type != "test_generation"
                        else None
                    )
                    execution = None
                    if example.execution and not dry_run:
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
                except (
                    Exception
                ) as error:  # ponytail: preserve the rest of a research run.
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
    task_summary = {
        task: aggregate_results(task_rows)
        for task, task_rows in _group_by_task(rows).items()
    }
    comparison = paired_execution_comparison(rows)
    hypothesis = "inconclusive" if dry_run else classify_hypothesis(summary, comparison)
    report = _report(summary, task_summary, comparison, hypothesis)
    write_json(
        output / "summary.json",
        {
            "hypothesis": hypothesis,
            "generic_vs_lattice_execution": comparison,
            "modes": summary,
            "tasks": task_summary,
        },
    )
    (output / "report.md").write_text(report)
    return output


def _default_output() -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path("artifacts") / "evaluations" / timestamp


def _group_by_task(rows: list[dict]) -> dict[str, list[dict]]:
    by_task: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_task[row["task_type"]].append(row)
    return by_task


def _report(
    summary: dict, task_summary: dict, comparison: dict, hypothesis: str
) -> str:
    lines = [
        "# SkillLatticeCoder Evaluation",
        "",
        f"**Hypothesis result:** {hypothesis}",
        "",
        "| Mode | Count | Fuzzy | Exact | Syntax | Execution | Active params |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for mode in MODES:
        if mode not in summary:
            continue
        value = summary[mode]
        lines.append(
            f"| {mode} | {value['count']} | {value['fuzzy_score']:.3f} | "
            f"{value['exact_match_rate']:.3f} | {_format(value['syntax_valid_rate'])} | "
            f"{_format(value['execution_pass_rate'])} | {value['active_adapter_parameters']:.0f} |"
        )
    lines.extend(["", "## Research questions", ""])
    base = summary.get("base", {}).get("execution_pass_rate", 0)
    generic = summary.get("generic", {}).get("execution_pass_rate", 0)
    lattice = summary.get("lattice", {}).get("execution_pass_rate", 0)
    single = summary.get("single-skill", {}).get("execution_pass_rate", 0)
    lines.extend(
        [
            f"1. Generic improves over base: **{_improves(generic, base)}**.",
            f"2. Single skill improves over generic: **{_improves(single, generic)}**.",
            f"3. Lattice improves over generic: **{_improves(lattice, generic)}**.",
            f"4. Paired lattice minus generic execution difference: "
            f"**{_format(comparison['difference'])}** "
            f"(95% bootstrap CI {_format(comparison['ci_low'])} to "
            f"{_format(comparison['ci_high'])}).",
            "",
            "## By task type",
            "",
        ]
    )
    for task, modes in sorted(task_summary.items()):
        scores = ", ".join(
            f"{mode}={values['fuzzy_score']:.3f}"
            for mode, values in sorted(modes.items())
        )
        lines.append(f"- `{task}`: {scores}")
    return "\n".join(lines) + "\n"


def _format(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _improves(candidate: float | None, baseline: float | None) -> str:
    return "n/a" if candidate is None or baseline is None else str(candidate > baseline)
