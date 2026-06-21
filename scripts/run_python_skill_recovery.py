#!/usr/bin/env python3
"""Prepare, run, and summarize the minimal Python-skill recovery experiment."""

import argparse
import json
import shutil
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean

from skill_lattice_coder.config import DATA_DIR, training_config
from skill_lattice_coder.data import load_jsonl, select_for_skill, write_mlx_dataset
from skill_lattice_coder.evaluation import evaluate
from skill_lattice_coder.train_skill import (
    _metadata,
    _saved_parameter_count,
    _training_command,
)
from skill_lattice_coder.utils import write_json

DEFAULT_SEEDS = (11, 22, 33, 44, 55)
SEVERE_FAMILIES = (
    "divide_or",
    "keys_for_value",
    "mask_address",
    "multiply_values",
    "trim_prefix",
    "without_none",
    "capped_total",
    "decimal_digit_sum",
    "substring_total",
)
VARIANTS = {
    "python_skill_low_lr": {"learning_rate": 5e-5, "iterations": 100, "preservation": False},
    "python_skill_fewer_steps": {"learning_rate": 1e-4, "iterations": 50, "preservation": False},
    "python_skill_preservation": {"learning_rate": 5e-5, "iterations": 50, "preservation": True},
}


def load_variant_examples(name, train_path, preservation_path):
    examples = select_for_skill(load_jsonl(train_path), "python_skill")
    if VARIANTS[name]["preservation"]:
        examples += load_jsonl(preservation_path)
    return examples


def prepare_variant(name, seed, seed_root, train_path, preservation_path):
    spec = VARIANTS[name]
    root = Path(seed_root) / name
    examples = load_variant_examples(name, train_path, preservation_path)
    data = write_mlx_dataset(examples, root / "training-data")
    command = _training_command(
        data,
        root / "adapters/python_skill",
        rank=8,
        seed=seed,
        iterations=spec["iterations"],
        learning_rate=spec["learning_rate"],
    )
    return {
        "name": name,
        "seed": seed,
        "examples": len(examples),
        "learning_rate": spec["learning_rate"],
        "iterations": spec["iterations"],
        "preservation": spec["preservation"],
        "command": command,
        "adapter_root": str(root / "adapters"),
        "output": str(root),
    }


def _link_existing_adapters(source, destination):
    destination.mkdir(parents=True, exist_ok=True)
    for name in ("generic", "debugging_skill", "test_generation_skill"):
        if not (source / name / "adapters.safetensors").exists():
            raise FileNotFoundError(f"baseline adapter not found: {source / name}")
        target = destination / name
        if not target.exists():
            target.symlink_to((source / name).resolve(), target_is_directory=True)


def _train(prepared, examples, *, force):
    output = Path(prepared["adapter_root"]) / "python_skill"
    metadata_path = output / "metadata.json"
    weights_path = output / "adapters.safetensors"
    if output.exists() and not force:
        if metadata_path.exists() and weights_path.exists():
            return json.loads(metadata_path.read_text())
        raise FileExistsError(f"{output} exists; pass --force to replace it")
    if output.exists():
        shutil.rmtree(output)
    start = time.perf_counter()
    subprocess.run(prepared["command"], check=True)
    metadata = _metadata(
        prepared["name"],
        examples,
        rank=8,
        elapsed=time.perf_counter() - start,
        seed=prepared["seed"],
        iterations=prepared["iterations"],
    )
    metadata["config"] = {
        **training_config(),
        "learning_rate": prepared["learning_rate"],
        "iterations": prepared["iterations"],
    }
    metadata["preservation_examples"] = (
        sum((example.group or "").startswith("preservation_training_only:") for example in examples)
        if prepared["preservation"]
        else 0
    )
    metadata["trainable_parameters"] = _saved_parameter_count(output)
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def _read_rows(root, seeds):
    rows = []
    for seed in seeds:
        path = Path(root) / f"seed-{seed}" / "results.jsonl"
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            row = json.loads(line)
            row["seed"] = seed
            rows.append(row)
    return rows


def _rate(rows):
    values = [bool(row["execution_passed"]) for row in rows if row.get("execution_passed") is not None]
    return mean(values) if values else None


def _mode_task(rows, mode, task=None):
    return [
        row for row in rows
        if row["mode"] == mode and (task is None or row["task_type"] == task)
    ]


def _transitions(rows):
    paired = defaultdict(dict)
    for row in rows:
        if row["task_type"] == "python_generation" and row["mode"] in {"base", "single-skill"}:
            paired[(row.get("seed"), row["example_id"])][row["mode"]] = row.get("execution_passed")
    complete = [pair for pair in paired.values() if pair.keys() >= {"base", "single-skill"}]
    if not complete:
        return {
            "python_pass_to_fail_vs_base": None,
            "python_fail_to_pass_vs_base": None,
        }
    return {
        "python_pass_to_fail_vs_base": sum(pair["base"] is True and pair["single-skill"] is False for pair in complete),
        "python_fail_to_pass_vs_base": sum(pair["base"] is False and pair["single-skill"] is True for pair in complete),
    }


def _families(rows):
    output = {}
    for family in SEVERE_FAMILIES:
        family_rows = [row for row in rows if row.get("benchmark_group") == family]
        output[family] = {
            "base_pass_rate": _rate(_mode_task(family_rows, "base", "python_generation")),
            "candidate_pass_rate": _rate(_mode_task(family_rows, "single-skill", "python_generation")),
        }
    return output


def _variant_summary(rows, metadata):
    lattice = _mode_task(rows, "lattice")
    values = {
        "python_generation_pass_rate": _rate(_mode_task(rows, "single-skill", "python_generation")),
        "overall_routed_lattice_pass_rate": _rate(lattice),
        "debugging_pass_rate": _rate(_mode_task(lattice, "lattice", "debugging")),
        "test_generation_pass_rate": _rate(_mode_task(lattice, "lattice", "test_generation")),
        "severe_regression_families": _families(rows),
        "active_adapter_parameters": (
            mean(row.get("active_adapter_parameters", 0) for row in lattice)
            if lattice else None
        ),
        "stored_adapter_parameters": metadata.get("stored_adapter_parameters"),
        "trainable_adapter_parameters": metadata.get("trainable_parameters"),
    }
    values.update(_transitions(rows))
    return values


def build_recovery_summary(baseline_rows, variant_rows, metadata, *, seeds):
    current = {}
    for label, mode in (("base", "base"), ("generic", "generic"), ("python_skill", "single-skill")):
        rows = _mode_task(baseline_rows, mode, "python_generation")
        current[label] = {
            "python_generation_pass_rate": _rate(rows),
            "active_adapter_parameters": (
                mean(row.get("active_adapter_parameters", 0) for row in rows) if rows else None
            ),
        }
    current_lattice = _mode_task(baseline_rows, "lattice")
    current["python_skill"].update(
        {
            "overall_routed_lattice_pass_rate": _rate(current_lattice),
            "debugging_pass_rate": _rate(
                _mode_task(current_lattice, "lattice", "debugging")
            ),
            "test_generation_pass_rate": _rate(
                _mode_task(current_lattice, "lattice", "test_generation")
            ),
            "severe_regression_families": _families(baseline_rows),
            **_transitions(baseline_rows),
        }
    )
    return {
        "seeds": seeds,
        "current": current,
        "variants": {
            name: _variant_summary(rows, metadata.get(name, {}))
            for name, rows in sorted(variant_rows.items())
        },
    }


def _report(summary):
    lines = [
        "# Python Skill Recovery",
        "",
        f"Status: **{summary['status']}**",
        "",
        "| Candidate | Python generation | Routed lattice | Debugging | Test generation | P→F | F→P |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, values in summary["current"].items():
        lines.append(
            f"| current {name} | {_percent(values['python_generation_pass_rate'])} | "
            f"{_percent(values.get('overall_routed_lattice_pass_rate'))} | "
            f"{_percent(values.get('debugging_pass_rate'))} | "
            f"{_percent(values.get('test_generation_pass_rate'))} | "
            f"{_value(values.get('python_pass_to_fail_vs_base'))} | "
            f"{_value(values.get('python_fail_to_pass_vs_base'))} |"
        )
    for name, values in summary["variants"].items():
        lines.append(
            f"| {name} | {_percent(values.get('python_generation_pass_rate'))} | "
            f"{_percent(values.get('overall_routed_lattice_pass_rate'))} | "
            f"{_percent(values.get('debugging_pass_rate'))} | "
            f"{_percent(values.get('test_generation_pass_rate'))} | "
            f"{_value(values.get('python_pass_to_fail_vs_base'))} | "
            f"{_value(values.get('python_fail_to_pass_vs_base'))} |"
        )
    lines.extend(
        [
            "",
            "## Adapter parameters",
            "",
            "| Candidate | Active | Stored | Trainable |",
            "|---|---:|---:|---:|",
        ]
    )
    for name, values in {**summary["current"], **summary["variants"]}.items():
        lines.append(
            f"| {name} | {_value(values.get('active_adapter_parameters'))} | "
            f"{_value(values.get('stored_adapter_parameters'))} | "
            f"{_value(values.get('trainable_adapter_parameters'))} |"
        )
    if summary["status"] == "complete":
        lines.extend(["", "## Severe-regression families", ""])
        for name, values in summary["variants"].items():
            lines.extend(
                [
                    f"### {name}",
                    "",
                    "| Family | Base | Candidate |",
                    "|---|---:|---:|",
                ]
            )
            for family, rates in values["severe_regression_families"].items():
                lines.append(
                    f"| {family} | {_percent(rates['base_pass_rate'])} | "
                    f"{_percent(rates['candidate_pass_rate'])} |"
                )
    lines.extend(["", "## Planned configurations", ""])
    for name, runs in summary["planned_runs"].items():
        sample = runs[0]
        lines.append(
            f"- `{name}`: lr={sample['learning_rate']}, steps={sample['iterations']}, "
            f"examples={sample['examples']}, preservation={sample['preservation']}"
        )
        lines.append(f"  - `{' '.join(sample['command'])}`")
    return "\n".join(lines) + "\n"


def _percent(value):
    return "n/a" if value is None else f"{value * 100:.1f}%"


def _value(value):
    return "n/a" if value is None else f"{value:.0f}"


def _current_metadata(root, seeds):
    values = defaultdict(list)
    for seed in seeds:
        adapter_root = Path(root) / f"seed-{seed}" / "adapters"
        metadata = {}
        for name in ("generic", "python_skill", "debugging_skill", "test_generation_skill"):
            path = adapter_root / name / "metadata.json"
            if path.exists():
                metadata[name] = json.loads(path.read_text()).get(
                    "trainable_parameters", 0
                )
        if "generic" in metadata:
            values["generic"].append(metadata["generic"])
        if "python_skill" in metadata:
            values["python_skill"].append(metadata["python_skill"])
            values["skill_pool"].append(
                sum(metadata.get(name, 0) for name in ("python_skill", "debugging_skill", "test_generation_skill"))
            )
    return {
        "base": {
            "stored_adapter_parameters": 0,
            "trainable_adapter_parameters": 0,
        },
        "generic": {
            "stored_adapter_parameters": mean(values["generic"]) if values["generic"] else None,
            "trainable_adapter_parameters": mean(values["generic"]) if values["generic"] else None,
        },
        "python_skill": {
            "stored_adapter_parameters": mean(values["skill_pool"]) if values["skill_pool"] else None,
            "trainable_adapter_parameters": mean(values["python_skill"]) if values["python_skill"] else None,
        },
    }


def _metadata_for(root, seeds):
    output = {}
    for name in VARIANTS:
        values = []
        for seed in seeds:
            adapter_root = Path(root) / f"seed-{seed}" / name / "adapters"
            path = adapter_root / "python_skill/metadata.json"
            if path.exists():
                python = json.loads(path.read_text())
                stored = 0
                for skill in ("python_skill", "debugging_skill", "test_generation_skill"):
                    skill_path = adapter_root / skill / "metadata.json"
                    if skill_path.exists():
                        stored += json.loads(skill_path.read_text()).get(
                            "trainable_parameters", 0
                        )
                values.append(
                    {
                        "trainable_parameters": python["trainable_parameters"],
                        "stored_adapter_parameters": stored,
                    }
                )
        if values:
            output[name] = {
                "trainable_parameters": mean(value["trainable_parameters"] for value in values),
                "stored_adapter_parameters": mean(
                    value["stored_adapter_parameters"] for value in values
                ),
            }
    return output


DEFAULT_PRESERVATION = DATA_DIR / "preservation/python_generation_preservation.jsonl"


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--output", default="artifacts/experiments/python-skill-recovery")
    parser.add_argument("--baseline-experiment", default="artifacts/experiments/five-seed")
    parser.add_argument("--dataset", default="data/eval.jsonl")
    parser.add_argument("--train-data", default="data/train.jsonl")
    parser.add_argument("--preservation-data", default=str(DEFAULT_PRESERVATION))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    root = Path(args.output)
    planned = defaultdict(list)
    for seed in args.seeds:
        seed_root = root / f"seed-{seed}"
        baseline_adapters = Path(args.baseline_experiment) / f"seed-{seed}" / "adapters"
        for name in VARIANTS:
            prepared = prepare_variant(name, seed, seed_root, args.train_data, args.preservation_data)
            planned[name].append(prepared)
            if args.dry_run:
                continue
            _link_existing_adapters(baseline_adapters, Path(prepared["adapter_root"]))
            examples = load_variant_examples(name, args.train_data, args.preservation_data)
            _train(prepared, examples, force=args.force)
            evaluate(args.dataset, output=prepared["output"], adapter_root=prepared["adapter_root"])
    baseline_rows = _read_rows(args.baseline_experiment, args.seeds)
    variant_rows = {name: [] for name in VARIANTS}
    # Results live one level below each seed rather than in seed-N directories.
    for name in VARIANTS:
        for seed in args.seeds:
            path = root / f"seed-{seed}" / name / "results.jsonl"
            if path.exists():
                for line in path.read_text().splitlines():
                    row = json.loads(line)
                    row["seed"] = seed
                    variant_rows[name].append(row)
    summary = build_recovery_summary(
        baseline_rows,
        variant_rows,
        _metadata_for(root, args.seeds),
        seeds=args.seeds,
    )
    for name, values in _current_metadata(args.baseline_experiment, args.seeds).items():
        summary["current"][name].update(values)
    summary["status"] = "dry-run" if args.dry_run else "complete"
    summary["planned_runs"] = dict(planned)
    write_json(root / "summary.json", summary)
    (root / "summary.md").write_text(_report(summary))
    print(json.dumps({"summary": str(root / "summary.json"), "report": str(root / "summary.md")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
