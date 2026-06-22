#!/usr/bin/env python3
"""Evaluate base and SkillCortex V1 on the FastAPI contract benchmark."""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from skill_lattice_coder.config import base_config
from skill_lattice_coder.inference import infer
from skill_lattice_coder.metrics import aggregate_results, extract_code, fuzzy_match
from skill_lattice_coder.schemas import ExecutionFixture
from skill_lattice_coder.utils import run_fixture, write_json


TASK_MAP = {
    "fastapi_contract_generation": "python_generation",
    "fastapi_contract_debugging": "debugging",
    "fastapi_contract_test_generation": "test_generation",
    "fastapi_contract_refactor": "debugging",
}
ROUTERS = ("base", "skillcortex_router_v1")
REQUIRED_ADAPTERS = (
    "python_skill",
    "debugging_skill",
    "test_generation_skill",
    "alternating_skill",
)


def _load_dataset(path):
    rows = []
    seen = set()
    for number, line in enumerate(Path(path).read_text().splitlines(), 1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:{number}: malformed JSON") from error
        if row.get("id") in seen:
            raise ValueError(f"{path}:{number}: duplicate id")
        if row.get("task_type") not in TASK_MAP:
            raise ValueError(f"{path}:{number}: unknown task type")
        if row.get("benchmark_family") != "fastapi_contract":
            raise ValueError(f"{path}:{number}: wrong benchmark family")
        seen.add(row["id"])
        rows.append(row)
    if not rows:
        raise ValueError(f"{path} is empty")
    return rows


def _adapters_ready(root):
    return all((Path(root) / name / "adapters.safetensors").exists() for name in REQUIRED_ADAPTERS)


def _evaluate_seed(examples, seed, output, routers, adapter_root, dry_run):
    output.mkdir(parents=True, exist_ok=True)
    model_cache = {}
    rows = []
    with (output / "results.jsonl").open("w") as handle:
        for example in examples:
            for router in routers:
                mapped_task = TASK_MAP[example["task_type"]]
                try:
                    generation = infer(
                        "base" if router == "base" else "lattice",
                        example["prompt"],
                        task_type=mapped_task,
                        semantic_family="fastapi_contract",
                        router_policy=(
                            "skillcortex_router_v1"
                            if router == "skillcortex_router_v1"
                            else None
                        ),
                        dry_run=dry_run,
                        adapter_root=adapter_root,
                        model_cache=model_cache,
                    )
                    text = example["target"] if dry_run else extract_code(generation.generation)
                    execution_passed = None
                    if not dry_run:
                        execution_passed, _ = run_fixture(
                            ExecutionFixture.from_dict(example["execution"]), text
                        )
                    row = {
                        "seed": seed,
                        "example_id": example["id"],
                        "benchmark_group": example["behavior_group"],
                        "domain": example["domain"],
                        "task_type": example["task_type"],
                        "mapped_router_task_type": mapped_task,
                        "mode": router,
                        "generation": text,
                        "exact_match": text.strip() == example["target"].strip(),
                        "fuzzy_score": fuzzy_match(text, example["target"]),
                        "syntax_valid": None,
                        "execution_passed": execution_passed,
                        "latency_seconds": generation.latency_seconds,
                        "selected_skills": generation.selected_skills,
                        "active_adapter_count": generation.active_adapter_count,
                        "active_adapter_parameters": generation.active_adapter_parameters,
                        "prompt_tokens": generation.prompt_tokens,
                        "generated_tokens": generation.generated_tokens,
                        "peak_memory_bytes": generation.peak_memory_bytes,
                        "error": None,
                    }
                except Exception as error:  # ponytail: preserve the remaining research run.
                    row = {
                        "seed": seed,
                        "example_id": example["id"],
                        "benchmark_group": example["behavior_group"],
                        "domain": example["domain"],
                        "task_type": example["task_type"],
                        "mapped_router_task_type": mapped_task,
                        "mode": router,
                        "generation": "",
                        "exact_match": False,
                        "fuzzy_score": 0.0,
                        "syntax_valid": None,
                        "execution_passed": None,
                        "latency_seconds": 0.0,
                        "selected_skills": [],
                        "active_adapter_count": 0,
                        "active_adapter_parameters": 0,
                        "prompt_tokens": None,
                        "generated_tokens": None,
                        "peak_memory_bytes": None,
                        "error": str(error),
                    }
                rows.append(row)
                handle.write(json.dumps(row) + "\n")
    return rows


def _summary(rows, seeds, dataset_sha256, dry_run):
    by_task = defaultdict(list)
    by_behavior = defaultdict(list)
    for row in rows:
        by_task[row["task_type"]].append(row)
        by_behavior[row["benchmark_group"]].append(row)
    return {
        "seeds": seeds,
        "benchmark_sha256": dataset_sha256,
        "validation": {
            "dry_run": dry_run,
            "training_performed": False,
            "candidate_activated": False,
            "router_modified": False,
            "benchmark_modified": False,
        },
        "modes": aggregate_results(rows),
        "tasks": {
            task: aggregate_results(task_rows)
            for task, task_rows in sorted(by_task.items())
        },
        "behavior_groups": {
            group: aggregate_results(group_rows)
            for group, group_rows in sorted(by_behavior.items())
        },
        "errors": [row for row in rows if row["error"]],
    }


def _markdown(summary):
    lines = [
        "# FastAPI Contract Baseline",
        "",
        f"- Seeds: `{', '.join(map(str, summary['seeds']))}`",
        f"- Dry run: **{str(summary['validation']['dry_run']).lower()}**",
        "- Training performed: **false**",
        "- Candidate activated: **false**",
        "",
        "| Router | Cases | Execution pass rate | Active parameters |",
        "|---|---:|---:|---:|",
    ]
    for mode, values in summary["modes"].items():
        rate = values["execution_pass_rate"]
        lines.append(
            f"| `{mode}` | {values['count']} | "
            f"{'n/a' if rate is None else f'{rate:.1%}'} | "
            f"{values['active_adapter_parameters']:.0f} |"
        )
    return "\n".join(lines) + "\n"


def run_baseline(
    dataset,
    *,
    output,
    seeds,
    routers=ROUTERS,
    adapter_experiment="artifacts/experiments/failure-born-skill/alternating_skill",
    dry_run=False,
):
    unknown = set(routers) - set(ROUTERS)
    if unknown:
        raise ValueError(f"unknown router: {sorted(unknown)[0]}")
    if base_config()["temperature"] != 0.0:
        raise ValueError("baseline evaluation requires temperature 0.0")
    dataset = Path(dataset)
    before = hashlib.sha256(dataset.read_bytes()).hexdigest()
    examples = _load_dataset(dataset)
    output = Path(output)
    all_rows = []
    for seed in seeds:
        adapter_root = Path(adapter_experiment) / f"seed-{seed}" / "adapters"
        if (
            not dry_run
            and "skillcortex_router_v1" in routers
            and not _adapters_ready(adapter_root)
        ):
            raise FileNotFoundError(f"missing promoted-router adapters: {adapter_root}")
        all_rows.extend(
            _evaluate_seed(
                examples,
                seed,
                output / f"seed-{seed}",
                tuple(routers),
                adapter_root,
                dry_run,
            )
        )
    if hashlib.sha256(dataset.read_bytes()).hexdigest() != before:
        raise RuntimeError("benchmark changed during baseline evaluation")
    summary = _summary(all_rows, list(seeds), before, dry_run)
    write_json(output / "summary.json", summary)
    (output / "summary.md").write_text(_markdown(summary))
    return output


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=(11, 22, 33, 44, 55))
    parser.add_argument(
        "--dataset",
        default="data/benchmarks/fastapi_contract/v1/benchmark.jsonl",
    )
    parser.add_argument(
        "--routers",
        nargs="+",
        default=ROUTERS,
        choices=ROUTERS,
    )
    parser.add_argument(
        "--adapter-experiment",
        default="artifacts/experiments/failure-born-skill/alternating_skill",
    )
    parser.add_argument(
        "--output", default="artifacts/experiments/fastapi-contract-baseline"
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    output = run_baseline(
        args.dataset,
        output=args.output,
        seeds=args.seeds,
        routers=args.routers,
        adapter_experiment=args.adapter_experiment,
        dry_run=args.dry_run,
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
