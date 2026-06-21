import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

from .schemas import MODES, SKILLS
from .utils import write_json


def load_experiment(experiment: str | Path) -> dict:
    root = Path(experiment)
    seed_paths = sorted(root.glob("seed-*/results.jsonl"))
    if not seed_paths:
        raise FileNotFoundError(f"no seed results found: {root}")
    rows, indexed, parameters = [], {}, {}
    for results_path in seed_paths:
        try:
            seed = int(results_path.parent.name.removeprefix("seed-"))
        except ValueError as error:
            raise ValueError(f"invalid seed directory: {results_path.parent}") from error
        for line in results_path.read_text().splitlines():
            row = json.loads(line)
            key = (seed, row["example_id"], row["mode"])
            if key in indexed:
                raise ValueError(f"duplicate result row: {key}")
            row["seed"] = seed
            row["selected_skills"] = sorted(row.get("selected_skills", []))
            indexed[key] = row
            rows.append(row)
        seed_parameters = {}
        for name in (*SKILLS, "generic"):
            path = results_path.parent / "adapters" / name / "metadata.json"
            if not path.exists():
                raise FileNotFoundError(f"adapter metadata not found: {path}")
            seed_parameters[name] = json.loads(path.read_text())[
                "trainable_parameters"
            ]
        if parameters and seed_parameters != parameters:
            raise ValueError("inconsistent adapter parameter metadata across seeds")
        parameters = seed_parameters
    examples = {(row["seed"], row["example_id"]) for row in rows}
    for seed, example_id in examples:
        missing = [
            mode for mode in MODES if (seed, example_id, mode) not in indexed
        ]
        if missing:
            raise ValueError(
                f"missing result modes for seed {seed} example {example_id}: {missing}"
            )
    return {
        "root": root,
        "rows": rows,
        "indexed": indexed,
        "seeds": sorted({row["seed"] for row in rows}),
        "adapter_parameters": parameters,
    }


def analyze_router(experiment: str | Path) -> dict:
    loaded = load_experiment(experiment)
    pairs = _pairs(loaded, "lattice", "oracle-lattice")
    data = {
        "experiment": str(loaded["root"]),
        "seeds": loaded["seeds"],
        "parameters": _parameters(loaded, ("lattice", "oracle-lattice")),
        "comparison": _comparison(pairs),
        "route_set_agreement_rate": mean(
            set(left["selected_skills"]) == set(right["selected_skills"])
            for left, right in pairs
        ),
        "by_task": _breakdown(pairs, "task_type"),
        "by_seed": _breakdown(pairs, "seed"),
        "by_route": _route_breakdown(pairs),
        "worst_families": _breakdown(pairs, "benchmark_group"),
    }
    data["worst_families"] = data["worst_families"][:10]
    _write_report(
        loaded["root"],
        "router_analysis",
        data,
        _router_markdown(data),
    )
    return data


def analyze_python_regression(experiment: str | Path) -> dict:
    loaded = load_experiment(experiment)
    modes = {}
    for mode in MODES[1:]:
        pairs = _pairs(loaded, "base", mode, task_type="python_generation")
        modes[mode] = _comparison(pairs)
        modes[mode]["syntax_valid_rate"] = _optional_rate(
            [candidate for _, candidate in pairs], "syntax_valid"
        )
    single_pairs = _pairs(
        loaded, "base", "single-skill", task_type="python_generation"
    )
    lattice_rows = [
        row
        for row in loaded["rows"]
        if row["task_type"] == "python_generation" and row["mode"] == "lattice"
    ]
    data = {
        "experiment": str(loaded["root"]),
        "seeds": loaded["seeds"],
        "parameters": _parameters(loaded, MODES),
        "modes": modes,
        "by_seed": {
            mode: _breakdown(
                _pairs(loaded, "base", mode, task_type="python_generation"), "seed"
            )
            for mode in MODES[1:]
        },
        "families_by_mode": {
            mode: _breakdown(
                _pairs(loaded, "base", mode, task_type="python_generation"),
                "benchmark_group",
                ascending=True,
            )[:10]
            for mode in MODES[1:]
        },
        "routed_skill_selections": _selection_counts(lattice_rows),
        "worst_families": _breakdown(
            single_pairs, "benchmark_group", ascending=True
        )[:10],
    }
    _write_report(
        loaded["root"],
        "python_regression_analysis",
        data,
        _python_markdown(data),
    )
    return data


def analyze_composition(experiment: str | Path) -> dict:
    loaded = load_experiment(experiment)
    lattice = _pairs(loaded, "single-skill", "lattice")
    oracle = _pairs(loaded, "single-skill", "oracle-lattice")
    lattice_two = [pair for pair in lattice if len(pair[1]["selected_skills"]) == 2]
    oracle_two = [pair for pair in oracle if len(pair[1]["selected_skills"]) == 2]
    data = {
        "experiment": str(loaded["root"]),
        "seeds": loaded["seeds"],
        "parameters": _parameters(
            loaded, ("single-skill", "lattice", "oracle-lattice")
        ),
        "comparisons": {
            "lattice_vs_single_skill": _comparison(lattice),
            "lattice_vs_single_skill_two_active": _comparison(lattice_two),
            "oracle_vs_single_skill": _comparison(oracle),
            "oracle_vs_single_skill_two_active": _comparison(oracle_two),
        },
        "by_task": {
            "lattice": _breakdown(lattice, "task_type"),
            "oracle": _breakdown(oracle, "task_type"),
        },
        "by_seed": {
            "lattice": _breakdown(lattice, "seed"),
            "oracle": _breakdown(oracle, "seed"),
        },
        "by_selection": _selection_breakdown(lattice),
        "semantic_families": {
            "lattice_gains": _breakdown(lattice, "benchmark_group")[:10],
            "lattice_harms": _breakdown(
                lattice, "benchmark_group", ascending=True
            )[:10],
            "oracle_gains": _breakdown(oracle, "benchmark_group")[:10],
            "oracle_harms": _breakdown(
                oracle, "benchmark_group", ascending=True
            )[:10],
        },
    }
    _write_report(
        loaded["root"],
        "composition_analysis",
        data,
        _composition_markdown(data),
    )
    return data


def _pairs(
    loaded: dict, baseline: str, candidate: str, task_type: str | None = None
) -> list[tuple[dict, dict]]:
    pairs = []
    for row in loaded["rows"]:
        if row["mode"] != baseline or (
            task_type is not None and row["task_type"] != task_type
        ):
            continue
        other = loaded["indexed"].get((row["seed"], row["example_id"], candidate))
        if other is None:
            raise ValueError(
                f"missing {candidate} result for seed {row['seed']} "
                f"example {row['example_id']}"
            )
        pairs.append((row, other))
    if not pairs:
        raise ValueError(f"no paired {baseline}/{candidate} results")
    return pairs


def _comparison(pairs: list[tuple[dict, dict]]) -> dict:
    if not pairs:
        return {
            "count": 0,
            "baseline_pass_rate": None,
            "candidate_pass_rate": None,
            "difference": None,
            "candidate_wins": 0,
            "baseline_wins": 0,
            "ties": 0,
        }
    baseline = [bool(left.get("execution_passed")) for left, _ in pairs]
    candidate = [bool(right.get("execution_passed")) for _, right in pairs]
    return {
        "count": len(pairs),
        "baseline_pass_rate": mean(baseline),
        "candidate_pass_rate": mean(candidate),
        "difference": mean(candidate) - mean(baseline),
        "candidate_wins": sum(not left and right for left, right in zip(baseline, candidate)),
        "baseline_wins": sum(left and not right for left, right in zip(baseline, candidate)),
        "ties": sum(left == right for left, right in zip(baseline, candidate)),
    }


def _breakdown(
    pairs: list[tuple[dict, dict]], key: str, *, ascending: bool = False
) -> list[dict]:
    grouped = defaultdict(list)
    for pair in pairs:
        grouped[pair[0].get(key) or pair[0]["example_id"]].append(pair)
    values = [{key: name, **_comparison(items)} for name, items in grouped.items()]
    return sorted(
        values,
        key=lambda value: (
            (value["difference"] or 0)
            if ascending
            else -(value["difference"] or 0),
            -value["count"],
            str(value[key]),
        ),
    )


def _route_breakdown(pairs: list[tuple[dict, dict]]) -> list[dict]:
    grouped = defaultdict(list)
    for left, right in pairs:
        grouped[
            (
                left["task_type"],
                tuple(left["selected_skills"]),
                tuple(right["selected_skills"]),
            )
        ].append((left, right))
    values = []
    for (task, routed, oracle), items in grouped.items():
        values.append(
            {
                "task_type": task,
                "routed_skills": list(routed),
                "oracle_skills": list(oracle),
                **_comparison(items),
            }
        )
    return sorted(
        values,
        key=lambda value: (
            -(value["candidate_wins"] - value["baseline_wins"]),
            -value["count"],
            value["task_type"],
            value["routed_skills"],
        ),
    )


def _selection_breakdown(pairs: list[tuple[dict, dict]]) -> list[dict]:
    grouped = defaultdict(list)
    for pair in pairs:
        grouped[tuple(pair[1]["selected_skills"])].append(pair)
    values = [
        {"selected_skills": list(skills), **_comparison(items)}
        for skills, items in grouped.items()
    ]
    return sorted(
        values,
        key=lambda value: (
            -(value["difference"] or 0),
            -value["count"],
            value["selected_skills"],
        ),
    )


def _selection_counts(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row["selected_skills"])].append(row)
    return sorted(
        (
            {
                "selected_skills": list(skills),
                "count": len(items),
                "execution_pass_rate": mean(
                    bool(item.get("execution_passed")) for item in items
                ),
            }
            for skills, items in grouped.items()
        ),
        key=lambda value: (-value["count"], value["selected_skills"]),
    )


def _parameters(loaded: dict, modes: tuple[str, ...]) -> dict:
    skill_pool = sum(loaded["adapter_parameters"][skill] for skill in SKILLS)
    output = {}
    for mode in modes:
        mode_rows = [row for row in loaded["rows"] if row["mode"] == mode]
        output[mode] = {
            "active_adapter_parameters": mean(
                row.get("active_adapter_parameters", 0) for row in mode_rows
            ),
            "stored_adapter_parameters": (
                loaded["adapter_parameters"]["generic"]
                if mode == "generic"
                else skill_pool if mode in {"single-skill", "lattice", "oracle-lattice"} else 0
            ),
        }
    return output


def _optional_rate(rows: list[dict], key: str) -> float | None:
    values = [row[key] for row in rows if row.get(key) is not None]
    return mean(values) if values else None


def _write_report(root: Path, name: str, data: dict, markdown: str) -> None:
    write_json(root / f"{name}.json", data)
    (root / f"{name}.md").write_text(markdown)


def _table(rows: list[dict], label: str) -> list[str]:
    lines = [
        f"| {label} | Count | Baseline | Candidate | Delta | Wins | Losses |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row[label]} | {row['count']} | "
            f"{_pct(row['baseline_pass_rate'])} | {_pct(row['candidate_pass_rate'])} | "
            f"{_signed(row['difference'])} | {row['candidate_wins']} | "
            f"{row['baseline_wins']} |"
        )
    return lines


def _router_markdown(data: dict) -> str:
    comparison = data["comparison"]
    worst = data["by_route"][0]
    omission = (
        " The largest gap occurs when routed selection omits `python_skill`."
        if "python_skill" not in worst["routed_skills"]
        and "python_skill" in worst["oracle_skills"]
        else ""
    )
    lines = [
        "# Router Analysis",
        "",
        f"Oracle exceeds routed execution by **{_signed(comparison['difference'])}** "
        f"({comparison['candidate_wins']} wins, {comparison['baseline_wins']} losses)."
        f"{omission}",
        "",
        f"Route-set agreement: **{_pct(data['route_set_agreement_rate'])}**.",
        "",
        "## By task type",
        "",
        *_table(data["by_task"], "task_type"),
        "",
        "## By routed and oracle skill selection",
        "",
        *_route_table(data["by_route"]),
        "",
        "## Largest semantic-family gaps",
        "",
        *_table(data["worst_families"], "benchmark_group"),
        "",
        "## Parameter footprint",
        "",
        *_parameter_table(data["parameters"]),
    ]
    return "\n".join(lines) + "\n"


def _python_markdown(data: dict) -> str:
    rows = [{"mode": mode, **values} for mode, values in data["modes"].items()]
    worst = data["worst_families"][0]
    return (
        "\n".join(
            [
                "# Python Generation Regression Analysis",
                "",
                "All trained modes are compared with the frozen base on existing "
                "Python-generation results. "
                f"The largest single-skill regression is `{worst['benchmark_group']}` "
                f"at **{_signed(worst['difference'])}**.",
                "",
                *_table(rows, "mode"),
                "",
                "## Largest semantic-family regressions",
                "",
                *_table(data["worst_families"], "benchmark_group"),
                "",
                "## Parameter footprint",
                "",
                *_parameter_table(data["parameters"]),
            ]
        )
        + "\n"
    )


def _composition_markdown(data: dict) -> str:
    rows = [
        {"comparison": name, **values}
        for name, values in data["comparisons"].items()
    ]
    return (
        "\n".join(
            [
                "# Composition Analysis",
                "",
                "Existing lattice and oracle results are compared with the "
                "task-specific single-skill mode.",
                "",
                *_table(rows, "comparison"),
                "",
                "## Routed composition by task",
                "",
                *_table(data["by_task"]["lattice"], "task_type"),
                "",
                "## Routed composition by skill selection",
                "",
                *_selection_table(data["by_selection"]),
                "",
                "## Parameter footprint",
                "",
                *_parameter_table(data["parameters"]),
            ]
        )
        + "\n"
    )


def _parameter_table(parameters: dict) -> list[str]:
    lines = [
        "| Mode | Active adapter parameters | Stored adapter parameters |",
        "|---|---:|---:|",
    ]
    for mode, values in parameters.items():
        lines.append(
            f"| {mode} | {values['active_adapter_parameters']:.0f} | "
            f"{values['stored_adapter_parameters']:.0f} |"
        )
    return lines


def _route_table(rows: list[dict]) -> list[str]:
    lines = [
        "| Task | Routed skills | Oracle skills | Count | Delta | Wins | Losses |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['task_type']} | {', '.join(row['routed_skills'])} | "
            f"{', '.join(row['oracle_skills'])} | {row['count']} | "
            f"{_signed(row['difference'])} | {row['candidate_wins']} | "
            f"{row['baseline_wins']} |"
        )
    return lines


def _selection_table(rows: list[dict]) -> list[str]:
    lines = [
        "| Selected skills | Count | Delta | Wins | Losses |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {', '.join(row['selected_skills'])} | {row['count']} | "
            f"{_signed(row['difference'])} | {row['candidate_wins']} | "
            f"{row['baseline_wins']} |"
        )
    return lines


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def _signed(value: float | None) -> str:
    return "n/a" if value is None else f"{value:+.1%}"
