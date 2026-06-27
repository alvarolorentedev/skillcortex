def evaluation_report(skill_id: str, summary: dict, tasks: dict) -> str:
    lines = [
        f"# SkillCortex Single Skill Evaluation: {skill_id}",
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
            f"{value['exact_match_rate']:.3f} | {format_metric(value['syntax_valid_rate'])} | "
            f"{format_metric(value['execution_pass_rate'])} | {value['active_adapter_parameters']:.0f} |"
        )
    lines.extend(["", "## By task", ""])
    for task, modes in tasks.items():
        scores = ", ".join(f"{mode}={values['fuzzy_score']:.3f}" for mode, values in sorted(modes.items()))
        lines.append(f"- `{task}`: {scores}")
    return "\n".join(lines) + "\n"


def format_metric(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}"
