#!/usr/bin/env python3
"""Rank protected-router failure clusters without training or inference."""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from skill_lattice_coder.data import load_jsonl
from skill_lattice_coder.utils import write_json

PROTECTED_MODE = "python_only_for_test_generation"
SOURCE_MODES = {
    "base": "base",
    "current_router": "lattice",
    "oracle_lattice": "oracle-lattice",
    "protected_skill_router": PROTECTED_MODE,
}
GOVERNED_MODES = {
    "base": "base",
    "protected_router_without_failure_born": "protected_skill_router_without_failure_born",
    "skillcortex_router_v1": "skillcortex_router_v1",
    "oracle_lattice": "oracle-lattice",
}


def _load_rows(root, seeds, examples):
    expected = {example.id: example for example in examples}
    rows = []
    for seed in seeds:
        path = Path(root) / f"seed-{seed}" / "results.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"missing validation results: {path}")
        seed_rows = []
        for number, line in enumerate(path.read_text().splitlines(), 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"malformed JSON at {path}:{number}") from error
            example = expected.get(row.get("example_id"))
            if example is None or row.get("task_type") != example.task_type:
                raise ValueError(f"unexpected result row at {path}:{number}")
            if row.get("mode") in SOURCE_MODES.values():
                row["seed"] = seed
                seed_rows.append(row)
        required = {
            (example.id, mode)
            for example in examples
            for mode in SOURCE_MODES.values()
        }
        counts = Counter((row["example_id"], row["mode"]) for row in seed_rows)
        if set(counts) != required or any(value != 1 for value in counts.values()):
            raise ValueError(f"incomplete or duplicate validation rows: {path}")
        rows.extend(seed_rows)
    return rows


def _load_governed_rows(promotion_root, validation_root, seeds, examples):
    expected = {example.id: example for example in examples}
    rows = []
    for seed in seeds:
        sources = (
            (
                Path(promotion_root) / f"seed-{seed}" / "fixed" / "results.jsonl",
                {
                    "protected_skill_router": GOVERNED_MODES[
                        "protected_router_without_failure_born"
                    ],
                    "protected_router_plus_alternating_skill": GOVERNED_MODES[
                        "skillcortex_router_v1"
                    ],
                },
            ),
            (
                Path(validation_root) / f"seed-{seed}" / "results.jsonl",
                {
                    "base": GOVERNED_MODES["base"],
                    "oracle-lattice": GOVERNED_MODES["oracle_lattice"],
                },
            ),
        )
        seed_rows = []
        for path, modes in sources:
            if not path.exists():
                raise FileNotFoundError(f"missing governed selection results: {path}")
            for number, line in enumerate(path.read_text().splitlines(), 1):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"malformed JSON at {path}:{number}") from error
                example = expected.get(row.get("example_id"))
                if example is None or row.get("task_type") != example.task_type:
                    raise ValueError(f"unexpected result row at {path}:{number}")
                if row.get("mode") in modes:
                    row["mode"] = modes[row["mode"]]
                    row["seed"] = seed
                    seed_rows.append(row)
        required = {
            (example.id, mode)
            for example in examples
            for mode in GOVERNED_MODES.values()
        }
        counts = Counter((row["example_id"], row["mode"]) for row in seed_rows)
        if set(counts) != required or any(value != 1 for value in counts.values()):
            raise ValueError(f"incomplete or duplicate governed rows for seed {seed}")
        rows.extend(seed_rows)
    return rows


def build_cluster_selection(rows, *, seeds, benchmark_sha256):
    indexed = {
        (row["seed"], row["example_id"], row["mode"]): row for row in rows
    }
    protected_rows = [
        row for row in rows if row["mode"] in {PROTECTED_MODE, "protected_skill_router"}
    ]
    if not protected_rows:
        raise ValueError("no protected-router rows")

    grouped = defaultdict(list)
    for row in protected_rows:
        grouped[(row["benchmark_group"], row["task_type"])].append(row)

    family_non_python_failures = Counter()
    for (family, task), items in grouped.items():
        if task != "python_generation":
            family_non_python_failures[family] += sum(
                not bool(row.get("execution_passed")) for row in items
            )

    clusters = []
    for (family, task), items in grouped.items():
        items = sorted(items, key=lambda row: row["seed"])
        failures = [row for row in items if not bool(row.get("execution_passed"))]
        mode_passes = {name: 0 for name in SOURCE_MODES}
        selected = Counter(tuple(row.get("selected_skills", [])) for row in items)
        for row in items:
            for name, mode in SOURCE_MODES.items():
                source = (
                    row
                    if name == "protected_skill_router"
                    else indexed.get((row["seed"], row["example_id"], mode))
                )
                if source is None:
                    raise ValueError(
                        f"missing {mode} row for seed {row['seed']} "
                        f"example {row['example_id']}"
                    )
                mode_passes[name] += bool(source.get("execution_passed"))
        failure_by_seed = {
            str(seed): sum(row["seed"] == seed for row in failures) for seed in seeds
        }
        existing_router_rescues = (
            mode_passes["current_router"] + mode_passes["oracle_lattice"]
        )
        eligible = task != "python_generation" and bool(failures)
        distinct_examples = len({row["example_id"] for row in items})
        likely_skill_specific = (
            eligible
            and len(failures) == len(items)
            and mode_passes["base"] == 0
            and existing_router_rescues == 0
        )
        clusters.append(
            {
                "semantic_family": family,
                "task_type": task,
                "total_examples": len(items),
                "pass_count": len(items) - len(failures),
                "fail_count": len(failures),
                "pass_rate": (len(items) - len(failures)) / len(items),
                "failure_count_by_seed": failure_by_seed,
                "selected_skill_tuple": [
                    {
                        "selected_skills": list(skills),
                        "count": count,
                    }
                    for skills, count in sorted(selected.items())
                ],
                "base_pass_count": mode_passes["base"],
                "base_passed": mode_passes["base"] > 0,
                "current_router_pass_count": mode_passes["current_router"],
                "current_router_passed": mode_passes["current_router"] > 0,
                "oracle_lattice_pass_count": mode_passes["oracle_lattice"],
                "oracle_lattice_passed": mode_passes["oracle_lattice"] > 0,
                "protected_router_failed": bool(failures),
                "distinct_benchmark_examples": distinct_examples,
                "family_non_python_fail_count": family_non_python_failures[family],
                "failure_seed_count": sum(value > 0 for value in failure_by_seed.values()),
                "existing_router_rescue_count": existing_router_rescues,
                "localized_semantic_pattern": True,
                "enough_repeated_failures_for_candidate_design": (
                    len(failures) >= 3
                    and sum(value > 0 for value in failure_by_seed.values()) >= 3
                ),
                "enough_independent_examples_for_promotion": distinct_examples >= 3,
                "likely_skill_specific": likely_skill_specific,
                "eligible_for_failure_born_skill": eligible,
                "exclusion_reason": (
                    "Python generation uses validated base fallback."
                    if task == "python_generation"
                    else None
                ),
            }
        )

    clusters.sort(
        key=lambda item: (
            not item["eligible_for_failure_born_skill"],
            -item["fail_count"],
            -item["family_non_python_fail_count"],
            -item["failure_seed_count"],
            item["base_pass_count"],
            item["existing_router_rescue_count"],
            item["semantic_family"],
            item["task_type"],
        )
    )
    primary = next(
        (cluster for cluster in clusters if cluster["eligible_for_failure_born_skill"]),
        None,
    )
    if primary is None:
        raise ValueError("no eligible failure cluster")
    recommendation = {
        "semantic_family": primary["semantic_family"],
        "task_type": primary["task_type"],
        "candidate_skill_name": f"{primary['semantic_family']}_skill",
    }
    return {
        "step": 1,
        "analysis_only": True,
        "requires_training": False,
        "requires_new_inference": False,
        "benchmark_sha256": benchmark_sha256,
        "seeds": seeds,
        "source": {
            "router": "protected_skill_router",
            "concrete_mode": PROTECTED_MODE,
        },
        "evaluation_leakage_warning": (
            "Seed repetitions are repeated generations of the same benchmark "
            "item, not independent examples. The selected benchmark item must "
            "remain evaluation-only; any candidate training data must be newly "
            "created training-only variants."
        ),
        "ranking_criteria": [
            "exclude Python-generation base-fallback failures",
            "higher protected-router failure count",
            "higher same-family non-Python failure count",
            "failure recurrence across more seeds",
            "fewer base-model rescues",
            "fewer current/oracle router rescues",
            "stable semantic-family and task-type tie-break",
        ],
        "recommended_primary_cluster": recommendation,
        "recommendation_reason": (
            f"`{primary['semantic_family']}` / `{primary['task_type']}` failed "
            f"{primary['fail_count']}/{primary['total_examples']} protected-router "
            f"runs across {primary['failure_seed_count']} seeds; its family has "
            f"{primary['family_non_python_fail_count']} non-Python failures and "
            f"{primary['base_pass_count']} base rescues."
        ),
        "clusters": clusters,
    }


def build_governed_selection(rows, *, registry, seeds, benchmark_sha256):
    promoted = [
        skill
        for skill in registry["skills"]
        if skill.get("origin") == "failure_born"
        and skill.get("status") == "promoted"
    ]
    promoted_families = sorted(
        {skill["semantic_family"] for skill in promoted}
    )
    budget = registry["capacity_budget"]
    rank8_parameters = next(
        (
            skill["trainable_parameters"]
            for skill in promoted
            if skill.get("rank") == 8
        ),
        311296,
    )
    current_parameters = budget["current_total_adapter_parameters"]
    remaining_parameters = budget["max_total_adapter_parameters"] - current_parameters
    one_more_fits = (
        remaining_parameters >= rank8_parameters
        and budget["current_promoted_failure_born_skills"] + 1
        <= budget["max_promoted_failure_born_skills"]
    )

    indexed = {
        (row["seed"], row["example_id"], row["mode"]): row for row in rows
    }
    current_rows = [
        row for row in rows if row["mode"] == GOVERNED_MODES["skillcortex_router_v1"]
    ]
    grouped = defaultdict(list)
    for row in current_rows:
        grouped[(row["benchmark_group"], row["task_type"])].append(row)

    family_total_failures = Counter()
    family_non_python_failures = Counter()
    for row in current_rows:
        if not bool(row.get("execution_passed")):
            family_total_failures[row["benchmark_group"]] += 1
            if row["task_type"] != "python_generation":
                family_non_python_failures[row["benchmark_group"]] += 1

    clusters = []
    for (family, task), items in grouped.items():
        if family in promoted_families:
            continue
        items = sorted(items, key=lambda row: row["seed"])
        failures = [row for row in items if not bool(row.get("execution_passed"))]
        passes = {}
        for name, mode in GOVERNED_MODES.items():
            passes[name] = sum(
                bool(indexed[(row["seed"], row["example_id"], mode)].get(
                    "execution_passed"
                ))
                for row in items
            )
        failure_seed_count = len({row["seed"] for row in failures})
        distinct_examples = len({row["example_id"] for row in items})
        existing_rescue = (
            passes["base"]
            + passes["protected_router_without_failure_born"]
            + passes["oracle_lattice"]
        )
        localized = bool(failures) and failure_seed_count >= 3
        eligible = (
            one_more_fits
            and task != "python_generation"
            and localized
        )
        selected = Counter(
            tuple(row.get("selected_skills", [])) for row in items
        )
        clusters.append(
            {
                "semantic_family": family,
                "task_type": task,
                "fail_count": len(failures),
                "pass_count": len(items) - len(failures),
                "pass_rate": (len(items) - len(failures)) / len(items),
                "failure_seed_count": failure_seed_count,
                "distinct_benchmark_examples": distinct_examples,
                "base_pass_count": passes["base"],
                "protected_router_without_failure_born_pass_count": passes[
                    "protected_router_without_failure_born"
                ],
                "skillcortex_router_v1_pass_count": passes[
                    "skillcortex_router_v1"
                ],
                "oracle_lattice_pass_count": passes["oracle_lattice"],
                "selected_skill_tuple": [
                    {"selected_skills": list(skills), "count": count}
                    for skills, count in sorted(selected.items())
                ],
                "family_total_fail_count": family_total_failures[family],
                "family_non_python_fail_count": family_non_python_failures[family],
                "whether_failure_remains_after_alternating_skill_promotion": bool(
                    failures
                ),
                "whether_existing_skills_partially_rescue_it": existing_rescue > 0,
                "whether_it_is_localized_enough_for_a_new_skill": localized,
                "whether_it_is_eligible_for_independent_synthetic_train_holdout_generation": (
                    task != "python_generation" and localized
                ),
                "estimated_capacity_impact": rank8_parameters,
                "registry_capacity_available": one_more_fits,
                "rollback_plan": (
                    "Keep the candidate quarantined and inactive; discard its "
                    "adapter and registry proposal if validation regresses."
                ),
                "recommended_action": (
                    "proceed_to_candidate_design" if eligible else "do_not_select"
                ),
                "eligible_for_failure_born_skill": eligible,
                "existing_rescue_count": existing_rescue,
            }
        )

    clusters.sort(
        key=lambda item: (
            not item["eligible_for_failure_born_skill"],
            -item["fail_count"],
            -item["failure_seed_count"],
            item["existing_rescue_count"],
            -item["family_non_python_fail_count"],
            item["semantic_family"],
            0 if item["task_type"] == "debugging" else 1,
        )
    )
    eligible = [cluster for cluster in clusters if cluster["eligible_for_failure_born_skill"]]
    primary = dict(eligible[0]) if eligible else None
    if primary:
        primary["candidate_skill_name"] = f"{primary['semantic_family']}_skill"
    backups = []
    used_families = {primary["semantic_family"]} if primary else set()
    for cluster in eligible[1:]:
        if cluster["semantic_family"] in used_families:
            continue
        backup = dict(cluster)
        backup["candidate_skill_name"] = f"{backup['semantic_family']}_skill"
        backups.append(backup)
        used_families.add(cluster["semantic_family"])
        if len(backups) == 2:
            break

    recommendation = (
        "proceed_to_candidate_design"
        if primary and one_more_fits
        else "do_not_create_new_skill_yet"
    )
    governance = {
        "promoted_failure_born_skills": [
            skill["skill_name"] for skill in promoted
        ],
        "promoted_semantic_families": promoted_families,
        "current_stored_adapter_parameters": current_parameters,
        "max_total_adapter_parameters": budget["max_total_adapter_parameters"],
        "remaining_adapter_parameter_budget": remaining_parameters,
        "current_promoted_failure_born_skills": budget[
            "current_promoted_failure_born_skills"
        ],
        "max_promoted_failure_born_skills": budget[
            "max_promoted_failure_born_skills"
        ],
        "rank8_skill_parameters": rank8_parameters,
        "whether_one_more_rank8_skill_fits": one_more_fits,
    }
    return {
        "phase": "2.3",
        "mode": "governed-phase-2-3",
        "analysis_only": True,
        "training_performed": False,
        "data_generated": False,
        "new_inference_performed": False,
        "router_modified": False,
        "benchmark_sha256": benchmark_sha256,
        "seeds": seeds,
        "source": {
            "router": registry["router"],
            "post_promotion_mode": GOVERNED_MODES["skillcortex_router_v1"],
        },
        "governance": governance,
        "primary_candidate": primary,
        "backup_candidates": backups,
        "clusters": clusters,
        "answers": {
            "best_next_candidate": (
                primary["candidate_skill_name"] if primary else None
            ),
            "why_better_than_alternatives": (
                "Highest-ranked repeated post-promotion non-Python failure with "
                "the least rescue from existing routes."
                if primary
                else "No candidate fits the current registry budget."
            ),
            "remains_unsolved_after_skillcortex_router_v1": bool(primary),
            "safe_under_capacity_budget": one_more_fits,
            "independent_data_without_benchmark_leakage": bool(primary),
            "candidate_skill_name": (
                primary["candidate_skill_name"] if primary else None
            ),
            "should_proceed_to_candidate_design": (
                recommendation == "proceed_to_candidate_design"
            ),
        },
        "recommendation": recommendation,
    }


def _pct(value):
    return f"{value:.1%}"


def _markdown(summary):
    primary = summary["recommended_primary_cluster"]
    lines = [
        "# Failure-Born Skill: Cluster Selection",
        "",
        "- Step: **1 — analysis only**",
        "- Training performed: **no**",
        "- New inference performed: **no**",
        f"- Source router: `{summary['source']['router']}`",
        "",
        "## Recommendation",
        "",
        f"Primary cluster: **`{primary['semantic_family']}` / "
        f"`{primary['task_type']}`**.",
        "",
        f"Provisional candidate name for the next decision: "
        f"`{primary['candidate_skill_name']}`.",
        "",
        summary["recommendation_reason"],
        "",
        "This is a selection recommendation only. No candidate skill, dataset, "
        "router, or training artifact has been created.",
        "",
        f"**Evaluation leakage warning:** {summary['evaluation_leakage_warning']}",
        "",
        "## Ranked clusters",
        "",
        "| Rank | Family | Task | Runs | Unique | Pass | Fail | Rate | Failure seeds | Skills | Base | Current | Oracle | Skill-specific | Eligible |",
        "|---:|---|---|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---|---|",
    ]
    for rank, cluster in enumerate(summary["clusters"], 1):
        skills = "; ".join(
            "+".join(item["selected_skills"]) or "base"
            for item in cluster["selected_skill_tuple"]
        )
        failure_seeds = ", ".join(
            seed
            for seed, count in cluster["failure_count_by_seed"].items()
            if count
        ) or "none"
        lines.append(
            f"| {rank} | {cluster['semantic_family']} | {cluster['task_type']} | "
            f"{cluster['total_examples']} | {cluster['distinct_benchmark_examples']} | "
            f"{cluster['pass_count']} | "
            f"{cluster['fail_count']} | {_pct(cluster['pass_rate'])} | "
            f"{failure_seeds} | {skills} | {cluster['base_pass_count']} | "
            f"{cluster['current_router_pass_count']} | "
            f"{cluster['oracle_lattice_pass_count']} | "
            f"{cluster['likely_skill_specific']} | "
            f"{cluster['eligible_for_failure_born_skill']} |"
        )
    lines.extend(
        [
            "",
            "## Ranking method",
            "",
            *[f"{index}. {criterion}." for index, criterion in enumerate(summary["ranking_criteria"], 1)],
        ]
    )
    return "\n".join(lines) + "\n"


def _governed_markdown(summary):
    governance = summary["governance"]
    primary = summary["primary_candidate"]
    lines = [
        "# Failure-Born Skill 2: Governed Cluster Selection",
        "",
        "- Analysis only: **true**",
        "- Training performed: **false**",
        "- Data generated: **false**",
        f"- Source router: `{summary['source']['router']}`",
        f"- Promoted families excluded: `{', '.join(governance['promoted_semantic_families'])}`",
        f"- Remaining adapter parameter budget: **{governance['remaining_adapter_parameter_budget']}**",
        f"- One more rank-8 skill fits: **{str(governance['whether_one_more_rank8_skill_fits']).lower()}**",
        "",
        "## Recommendation",
        "",
    ]
    if primary:
        lines.extend(
            [
                f"Primary: **`{primary['candidate_skill_name']}`** "
                f"(`{primary['semantic_family']}` / `{primary['task_type']}`).",
                "",
                f"Post-promotion failures: **{primary['fail_count']}** across "
                f"**{primary['failure_seed_count']}** seeds.",
            ]
        )
    else:
        lines.append("No candidate can be selected under the current capacity budget.")
    if summary["backup_candidates"]:
        lines.extend(
            [
                "",
                "Backups: "
                + ", ".join(
                    f"`{candidate['candidate_skill_name']}`"
                    for candidate in summary["backup_candidates"]
                )
                + ".",
            ]
        )
    lines.extend(
        [
            "",
            "## Ranked post-promotion clusters",
            "",
            "| Rank | Family | Task | Fail | Pass | Failure seeds | Base | Protected | SkillCortex | Oracle | Partial rescue | Eligible |",
            "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|",
        ]
    )
    for rank, cluster in enumerate(summary["clusters"], 1):
        lines.append(
            f"| {rank} | {cluster['semantic_family']} | {cluster['task_type']} | "
            f"{cluster['fail_count']} | {cluster['pass_count']} | "
            f"{cluster['failure_seed_count']} | {cluster['base_pass_count']} | "
            f"{cluster['protected_router_without_failure_born_pass_count']} | "
            f"{cluster['skillcortex_router_v1_pass_count']} | "
            f"{cluster['oracle_lattice_pass_count']} | "
            f"{cluster['whether_existing_skills_partially_rescue_it']} | "
            f"{cluster['eligible_for_failure_born_skill']} |"
        )
    lines.extend(["", f"Recommendation: {summary['recommendation']}", ""])
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("legacy", "governed-phase-2-3"),
        default="legacy",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=(11, 22, 33, 44, 55))
    parser.add_argument("--dataset", default="data/eval.jsonl")
    parser.add_argument("--registry", default="configs/skill_registry.json")
    parser.add_argument("--source-router", default="skillcortex_router_v1")
    parser.add_argument(
        "--promotion-experiment",
        default="artifacts/experiments/failure-born-skill/alternating_skill",
    )
    parser.add_argument(
        "--validation-experiment",
        default="artifacts/experiments/python-skill-gating-validation",
    )
    parser.add_argument("--output")
    args = parser.parse_args(argv)

    benchmark = Path(args.dataset)
    checksum = hashlib.sha256(benchmark.read_bytes()).hexdigest()
    examples = load_jsonl(benchmark)
    if args.mode == "governed-phase-2-3":
        registry = json.loads(Path(args.registry).read_text())
        if args.source_router != registry["router"]:
            raise ValueError("source router does not match skill registry")
        if checksum != registry["benchmark_sha256"]:
            raise ValueError("benchmark checksum does not match skill registry")
        rows = _load_governed_rows(
            args.promotion_experiment,
            args.validation_experiment,
            args.seeds,
            examples,
        )
        summary = build_governed_selection(
            rows,
            registry=registry,
            seeds=args.seeds,
            benchmark_sha256=checksum,
        )
        output = Path(
            args.output or "artifacts/experiments/failure-born-skill-2"
        )
        renderer = _governed_markdown
    else:
        rows = _load_rows(args.validation_experiment, args.seeds, examples)
        summary = build_cluster_selection(
            rows, seeds=args.seeds, benchmark_sha256=checksum
        )
        output = Path(args.output or "artifacts/experiments/failure-born-skill")
        renderer = _markdown
    if hashlib.sha256(benchmark.read_bytes()).hexdigest() != checksum:
        raise RuntimeError("benchmark changed during cluster analysis")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "cluster_selection.json", summary)
    (output / "cluster_selection.md").write_text(renderer(summary))
    print(
        json.dumps(
            {
                "json": str(output / "cluster_selection.json"),
                "markdown": str(output / "cluster_selection.md"),
                "recommended_primary_cluster": summary.get(
                    "recommended_primary_cluster", summary.get("primary_candidate")
                ),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
