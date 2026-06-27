"""Validate the SkillCortex registry and write its governance report."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

from skillcortex.contracts import KNOWN_SKILLS, TASK_TYPES
from skillcortex.runtime.router_rules import SkillCortexRouterV1


ROOT = Path(__file__).resolve().parents[1]


def validate_registry(registry: dict, router_report: dict) -> None:
    errors = []
    skills = registry.get("skills", [])
    by_name = {skill.get("skill_name"): skill for skill in skills}
    if None in by_name or len(by_name) != len(skills):
        errors.append("skill names must be present and unique")
    if set(by_name) != set(KNOWN_SKILLS):
        errors.append("registry skills do not match known skills")

    for task_type in TASK_TYPES:
        for semantic_family in (None, "other", "alternating"):
            for name in SkillCortexRouterV1().route(
                task_type, semantic_family
            ).selected_skills:
                skill = by_name.get(name)
                if not skill:
                    errors.append(f"router-referenced skill is missing: {name}")
                elif task_type not in skill.get("allowed_task_types", []):
                    errors.append(f"{name} is active outside its allowed task types")

    alternating = by_name.get("alternating_skill", {})
    if alternating:
        router = SkillCortexRouterV1()
        if alternating.get("activation_scope") != "strict_gate":
            errors.append("alternating_skill must use strict_gate activation")
        if any(
            "alternating_skill"
            in router.route(task_type, semantic_family).selected_skills
            for task_type in TASK_TYPES
            for semantic_family in (None, "other")
        ):
            errors.append("alternating_skill is active outside its strict gate")

    promoted = [skill for skill in skills if skill.get("status") == "promoted"]
    for skill in promoted:
        if not all(
            skill.get(field)
            for field in (
                "promotion_source_experiment",
                "promotion_status",
                "promotion_reason",
            )
        ):
            errors.append(f"{skill.get('skill_name')} lacks promotion evidence")
        if skill.get("origin") == "failure_born":
            quarantine = skill.get("historical_quarantine") or {}
            if not quarantine.get("quarantined"):
                errors.append(
                    f"{skill.get('skill_name')} lacks historical quarantine metadata"
                )
            if not skill.get("rollback_supported") or not skill.get(
                "rollback_router"
            ):
                errors.append(f"{skill.get('skill_name')} lacks rollback metadata")

    core = [skill for skill in skills if skill.get("status") == "core"]
    if any(skill.get("origin") != "seed_skill" for skill in core):
        errors.append("core skills must be seed skills")

    budget = registry.get("capacity_budget", {})
    total = sum(skill.get("trainable_parameters", 0) for skill in skills)
    failure_born_promoted = sum(
        skill.get("origin") == "failure_born" and skill.get("status") == "promoted"
        for skill in skills
    )
    if total != budget.get("current_total_adapter_parameters"):
        errors.append("current adapter parameter total is inconsistent")
    if failure_born_promoted != budget.get("current_promoted_failure_born_skills"):
        errors.append("current promoted failure-born skill count is inconsistent")
    if total > budget.get("max_total_adapter_parameters", -1):
        errors.append("capacity budget exceeded")
    if failure_born_promoted > budget.get(
        "max_promoted_failure_born_skills", -1
    ):
        errors.append("promoted failure-born skill budget exceeded")

    benchmark_sha256 = registry.get("benchmark_sha256")
    actual_sha256 = hashlib.sha256((ROOT / "data/eval.jsonl").read_bytes()).hexdigest()
    if not benchmark_sha256:
        errors.append("benchmark checksum is not recorded")
    elif benchmark_sha256 != actual_sha256:
        errors.append("benchmark checksum does not match data/eval.jsonl")
    if benchmark_sha256 != router_report.get("benchmark_sha256"):
        errors.append("benchmark checksum does not match router evidence")

    immune = registry.get("immune", {})
    required_immune = {
        "requires_quarantine_before_promotion": True,
        "requires_independent_holdout": True,
        "requires_non_target_regression_check": True,
        "auto_promotion_allowed": False,
        "rollback_on_regression": True,
    }
    if any(immune.get(key) is not value for key, value in required_immune.items()):
        errors.append("immune policy is incomplete")

    if errors:
        raise ValueError("; ".join(dict.fromkeys(errors)))


def build_report(registry: dict, router_report: dict) -> dict:
    skills = registry["skills"]
    budget = registry["capacity_budget"]
    promoted_count = budget["current_promoted_failure_born_skills"]
    total = budget["current_total_adapter_parameters"]
    next_skill_parameters = max(skill["trainable_parameters"] for skill in skills)
    within_capacity = (
        total <= budget["max_total_adapter_parameters"]
        and promoted_count <= budget["max_promoted_failure_born_skills"]
    )
    fixed = router_report["fixed_benchmark"]["routers"]["skillcortex_router_v1"]
    holdout = router_report["independent_alternating_holdout"]["routers"][
        "skillcortex_router_v1"
    ]
    alternating = next(
        skill for skill in skills if skill["skill_name"] == "alternating_skill"
    )
    return {
        "core_seed_skills": [
            skill["skill_name"] for skill in skills if skill["status"] == "core"
        ],
        "failure_born_skills": [
            skill["skill_name"]
            for skill in skills
            if skill["origin"] == "failure_born"
        ],
        "promoted_skills": [
            skill["skill_name"] for skill in skills if skill["status"] == "promoted"
        ],
        "quarantined_skills": [
            skill["skill_name"]
            for skill in skills
            if skill["status"] == "quarantined"
        ],
        "current_stored_adapter_parameters": total,
        "active_parameter_behavior": {
            "base_fallback": 0,
            "protected_route": 622592,
            "alternating_strict_gate": 933888,
            "fixed_benchmark_average": fixed["active_adapter_parameters"],
            "independent_holdout_average": holdout["active_adapter_parameters"],
        },
        "within_capacity_budget": within_capacity,
        "alternating_skill_rollback_supported": alternating["rollback_supported"],
        "rollback_router": alternating["rollback_router"],
        "ready_for_another_failure_born_skill_experiment": (
            within_capacity
            and promoted_count < budget["max_promoted_failure_born_skills"]
            and total + next_skill_parameters
            <= budget["max_total_adapter_parameters"]
        ),
        "benchmark_sha256": registry["benchmark_sha256"],
    }


def markdown(report: dict) -> str:
    active = report["active_parameter_behavior"]
    return "\n".join(
        [
            "# Skill Registry Governance Report",
            "",
            f"- Core seed skills: `{', '.join(report['core_seed_skills'])}`",
            f"- Failure-born skills: `{', '.join(report['failure_born_skills'])}`",
            f"- Promoted skills: `{', '.join(report['promoted_skills'])}`",
            f"- Quarantined skills: `{', '.join(report['quarantined_skills']) or 'none'}`",
            f"- Stored adapter parameters: **{report['current_stored_adapter_parameters']}**",
            f"- Fixed-benchmark average active parameters: **{active['fixed_benchmark_average']:.0f}**",
            f"- Base/protected/strict-gate active parameters: **{active['base_fallback']}/{active['protected_route']}/{active['alternating_strict_gate']}**",
            f"- Within capacity budget: **{str(report['within_capacity_budget']).lower()}**",
            f"- `alternating_skill` rollback supported: **{str(report['alternating_skill_rollback_supported']).lower()}**",
            f"- Ready for another controlled failure-born experiment: **{str(report['ready_for_another_failure_born_skill_experiment']).lower()}**",
            "",
        ]
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", default="configs/skill_registry.json")
    parser.add_argument(
        "--router-report",
        default="artifacts/experiments/skillcortex-router-v1/summary.json",
    )
    parser.add_argument(
        "--output", default="artifacts/experiments/skill-registry"
    )
    args = parser.parse_args(argv)
    try:
        registry = json.loads(Path(args.registry).read_text())
        router_report = json.loads(Path(args.router_report).read_text())
        validate_registry(registry, router_report)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        print(f"skill registry validation failed: {error}", file=sys.stderr)
        return 1

    report = build_report(registry, router_report)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    (output / "summary.md").write_text(markdown(report))
    print(f"skill registry valid: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
