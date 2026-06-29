"""Validate the SLMCortex registry and write its governance report."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

from slmcortex.contracts import KNOWN_SLMS, TASK_TYPES
from slmcortex.runtime.router_rules import SLMCortexRouterV1


ROOT = Path(__file__).resolve().parents[1]


def validate_registry(registry: dict, router_report: dict) -> None:
    errors = []
    slms = registry.get("slms", [])
    by_name = {slm.get("slm_name"): slm for slm in slms}
    if None in by_name or len(by_name) != len(slms):
        errors.append("slm names must be present and unique")
    if set(by_name) != set(KNOWN_SLMS):
        errors.append("registry slms do not match known slms")

    for task_type in TASK_TYPES:
        for semantic_family in (None, "other", "alternating"):
            for name in SLMCortexRouterV1().route(
                task_type, semantic_family
            ).selected_slms:
                slm = by_name.get(name)
                if not slm:
                    errors.append(f"router-referenced slm is missing: {name}")
                elif task_type not in slm.get("allowed_task_types", []):
                    errors.append(f"{name} is active outside its allowed task types")

    alternating = by_name.get("alternating_slm", {})
    if alternating:
        router = SLMCortexRouterV1()
        if alternating.get("activation_scope") != "strict_gate":
            errors.append("alternating_slm must use strict_gate activation")
        if any(
            "alternating_slm"
            in router.route(task_type, semantic_family).selected_slms
            for task_type in TASK_TYPES
            for semantic_family in (None, "other")
        ):
            errors.append("alternating_slm is active outside its strict gate")

    promoted = [slm for slm in slms if slm.get("status") == "promoted"]
    for slm in promoted:
        if not all(
            slm.get(field)
            for field in (
                "promotion_source_experiment",
                "promotion_status",
                "promotion_reason",
            )
        ):
            errors.append(f"{slm.get('slm_name')} lacks promotion evidence")
        if slm.get("origin") == "failure_born":
            quarantine = slm.get("historical_quarantine") or {}
            if not quarantine.get("quarantined"):
                errors.append(
                    f"{slm.get('slm_name')} lacks historical quarantine metadata"
                )
            if not slm.get("rollback_supported") or not slm.get(
                "rollback_router"
            ):
                errors.append(f"{slm.get('slm_name')} lacks rollback metadata")

    core = [slm for slm in slms if slm.get("status") == "core"]
    if any(slm.get("origin") != "seed_slm" for slm in core):
        errors.append("core slms must be seed slms")

    budget = registry.get("capacity_budget", {})
    total = sum(slm.get("trainable_parameters", 0) for slm in slms)
    failure_born_promoted = sum(
        slm.get("origin") == "failure_born" and slm.get("status") == "promoted"
        for slm in slms
    )
    if total != budget.get("current_total_adapter_parameters"):
        errors.append("current adapter parameter total is inconsistent")
    if failure_born_promoted != budget.get("current_promoted_failure_born_slms"):
        errors.append("current promoted failure-born slm count is inconsistent")
    if total > budget.get("max_total_adapter_parameters", -1):
        errors.append("capacity budget exceeded")
    if failure_born_promoted > budget.get(
        "max_promoted_failure_born_slms", -1
    ):
        errors.append("promoted failure-born slm budget exceeded")

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
    slms = registry["slms"]
    budget = registry["capacity_budget"]
    promoted_count = budget["current_promoted_failure_born_slms"]
    total = budget["current_total_adapter_parameters"]
    next_slm_parameters = max(slm["trainable_parameters"] for slm in slms)
    within_capacity = (
        total <= budget["max_total_adapter_parameters"]
        and promoted_count <= budget["max_promoted_failure_born_slms"]
    )
    fixed = router_report["fixed_benchmark"]["routers"]["slmcortex_router_v1"]
    holdout = router_report["independent_alternating_holdout"]["routers"][
        "slmcortex_router_v1"
    ]
    alternating = next(
        slm for slm in slms if slm["slm_name"] == "alternating_slm"
    )
    return {
        "core_seed_slms": [
            slm["slm_name"] for slm in slms if slm["status"] == "core"
        ],
        "failure_born_slms": [
            slm["slm_name"]
            for slm in slms
            if slm["origin"] == "failure_born"
        ],
        "promoted_slms": [
            slm["slm_name"] for slm in slms if slm["status"] == "promoted"
        ],
        "quarantined_slms": [
            slm["slm_name"]
            for slm in slms
            if slm["status"] == "quarantined"
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
        "alternating_slm_rollback_supported": alternating["rollback_supported"],
        "rollback_router": alternating["rollback_router"],
        "ready_for_another_failure_born_slm_experiment": (
            within_capacity
            and promoted_count < budget["max_promoted_failure_born_slms"]
            and total + next_slm_parameters
            <= budget["max_total_adapter_parameters"]
        ),
        "benchmark_sha256": registry["benchmark_sha256"],
    }


def markdown(report: dict) -> str:
    active = report["active_parameter_behavior"]
    return "\n".join(
        [
            "# Slm Registry Governance Report",
            "",
            f"- Core seed slms: `{', '.join(report['core_seed_slms'])}`",
            f"- Failure-born slms: `{', '.join(report['failure_born_slms'])}`",
            f"- Promoted slms: `{', '.join(report['promoted_slms'])}`",
            f"- Quarantined slms: `{', '.join(report['quarantined_slms']) or 'none'}`",
            f"- Stored adapter parameters: **{report['current_stored_adapter_parameters']}**",
            f"- Fixed-benchmark average active parameters: **{active['fixed_benchmark_average']:.0f}**",
            f"- Base/protected/strict-gate active parameters: **{active['base_fallback']}/{active['protected_route']}/{active['alternating_strict_gate']}**",
            f"- Within capacity budget: **{str(report['within_capacity_budget']).lower()}**",
            f"- `alternating_slm` rollback supported: **{str(report['alternating_slm_rollback_supported']).lower()}**",
            f"- Ready for another controlled failure-born experiment: **{str(report['ready_for_another_failure_born_slm_experiment']).lower()}**",
            "",
        ]
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", default="src/slmcortex_resources/configs/slm_registry.json")
    parser.add_argument(
        "--router-report",
        default="artifacts/governance-fixtures/slmcortex-router-v1/summary.json",
    )
    parser.add_argument(
        "--output", default="artifacts/governance/slm-registry"
    )
    args = parser.parse_args(argv)
    try:
        registry = json.loads(Path(args.registry).read_text())
        router_report = json.loads(Path(args.router_report).read_text())
        validate_registry(registry, router_report)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        print(f"slm registry validation failed: {error}", file=sys.stderr)
        return 1

    report = build_report(registry, router_report)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    (output / "summary.md").write_text(markdown(report))
    print(f"slm registry valid: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
