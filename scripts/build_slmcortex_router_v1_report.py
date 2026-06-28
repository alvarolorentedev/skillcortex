"""Build the promotion integration report from validated experiment artifacts."""

import argparse
import json
from pathlib import Path


BASELINE = "protected_slm_router_without_failure_born"
ROUTER = "slmcortex_router_v1"


def build_summary(source: dict) -> dict:
    def section(name: str) -> dict:
        data = source[name]
        return {
            "routers": {
                BASELINE: data["modes"]["protected_slm_router"],
                ROUTER: data["modes"]["protected_router_plus_alternating_slm"],
            },
            "pass_fail_vs_previous_protected_router": data[
                "pass_fail_vs_protected"
            ],
            "non_target_regressions": data["non_target_regressions"],
        }

    return {
        "router": ROUTER,
        "promoted_slm": "alternating_slm",
        "benchmark_sha256": source["benchmark_sha256"],
        "validation": {
            "uses_existing_artifacts": True,
            "new_training": False,
            "new_inference": False,
            "integration_validation_only": True,
        },
        "fixed_benchmark": section("fixed_benchmark"),
        "independent_alternating_holdout": section("independent_holdout"),
        "historical_quarantine": {
            "quarantined": True,
            **source["quarantine"],
            "promotion_status": source["promotion_decision"]["status"],
        },
    }


def _pct(value):
    return "n/a" if value is None else f"{value:.1%}"


def markdown(summary: dict) -> str:
    lines = [
        "# SlmCortex Router V1 Promotion Integration",
        "",
        "- Uses existing artifacts: **true**",
        "- New training: **false**",
        "- New inference: **false**",
    ]
    for key, title in (
        ("fixed_benchmark", "Fixed benchmark"),
        ("independent_alternating_holdout", "Independent alternating holdout"),
    ):
        lines.extend(["", f"## {title}", ""])
        for name, values in summary[key]["routers"].items():
            lines.append(
                f"- `{name}`: overall {_pct(values['overall_execution_pass_rate'])}; "
                f"Python {_pct(values['python_generation_pass_rate'])}; "
                f"debugging {_pct(values['debugging_pass_rate'])}; "
                f"test generation {_pct(values['test_generation_pass_rate'])}; "
                f"alternating/debugging {_pct(values['alternating_debugging_pass_rate'])}; "
                f"alternating/test generation {_pct(values['alternating_test_generation_pass_rate'])}; "
                f"target {_pct(values['target_cluster_pass_rate'])}; "
                f"non-target {_pct(values['non_target_pass_rate'])}; "
                f"active {values['active_adapter_parameters']:.0f}; "
                f"stored {values['stored_adapter_parameters']}; "
                f"trainable {values['trainable_adapter_parameters']}.")
            lines.append(
                f"  Selected slm tuples: `{json.dumps(values['selected_slm_tuple_distribution'], separators=(',', ':'))}`"
            )
        changes = summary[key]["pass_fail_vs_previous_protected_router"]
        lines.append(
            f"- Fail-to-pass/pass-to-fail: {changes['fail_to_pass']}/{changes['pass_to_fail']}."
        )
        lines.append(
            f"- Non-target regressions: {summary[key]['non_target_regressions']}."
        )
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        default="artifacts/governance-fixtures/alternating_slm/summary.json",
    )
    parser.add_argument(
        "--output", default="artifacts/governance/slmcortex-router-v1"
    )
    args = parser.parse_args(argv)
    summary = build_summary(json.loads(Path(args.source).read_text()))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output / "summary.md").write_text(markdown(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
