from __future__ import annotations

import ast
import re
from collections import defaultdict
from difflib import SequenceMatcher
from math import sqrt
from random import Random
from statistics import mean, pstdev


def extract_code(text: str) -> str:
    match = re.search(r"```(?:python)?\s*\n?(.*?)```", text, re.DOTALL | re.IGNORECASE)
    return (match.group(1) if match else text).strip()


def fuzzy_match(generation: str, target: str) -> float:
    return SequenceMatcher(None, generation.strip(), target.strip()).ratio()


def python_syntax_valid(text: str) -> bool:
    try:
        ast.parse(extract_code(text))
        return True
    except SyntaxError:
        return False


def aggregate_results(rows: list[dict]) -> dict[str, dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["mode"]].append(row)
    summary = {}
    for mode, mode_rows in grouped.items():
        fuzzy_values = [row["fuzzy_score"] for row in mode_rows]
        fuzzy = mean(fuzzy_values)
        margin = 1.96 * pstdev(fuzzy_values) / sqrt(len(fuzzy_values))
        execution_values = [
            row["execution_passed"]
            for row in mode_rows
            if row.get("execution_passed") is not None
        ]
        execution = mean(execution_values) if execution_values else None
        execution_margin = (
            1.96 * pstdev(execution_values) / sqrt(len(execution_values))
            if execution_values
            else None
        )
        parameters = mean(row.get("active_adapter_parameters", 0) for row in mode_rows)
        summary[mode] = {
            "count": len(mode_rows),
            "fuzzy_score": fuzzy,
            "fuzzy_ci_low": max(0.0, fuzzy - margin),
            "fuzzy_ci_high": min(1.0, fuzzy + margin),
            "exact_match_rate": mean(bool(row.get("exact_match")) for row in mode_rows),
            "syntax_valid_rate": _optional_rate(mode_rows, "syntax_valid"),
            "execution_pass_rate": execution,
            "execution_ci_low": (
                max(0.0, execution - execution_margin)
                if execution is not None
                else None
            ),
            "execution_ci_high": (
                min(1.0, execution + execution_margin)
                if execution is not None
                else None
            ),
            "mean_latency_seconds": mean(
                row.get("latency_seconds", 0) for row in mode_rows
            ),
            "active_adapter_parameters": parameters,
            "score_per_million_active_parameters": (
                fuzzy / (parameters / 1_000_000) if parameters else None
            ),
            "execution_per_million_active_parameters": (
                execution / (parameters / 1_000_000)
                if parameters and execution is not None
                else None
            ),
        }
    return summary


def _optional_rate(rows: list[dict], key: str) -> float | None:
    values = [row[key] for row in rows if row.get(key) is not None]
    return mean(values) if values else None


def paired_execution_comparison(
    rows: list[dict],
    *,
    candidate: str = "lattice",
    baseline: str = "generic",
    samples: int = 10_000,
    seed: int = 42,
) -> dict:
    by_group: dict[str, dict[str, list[bool]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row.get("execution_passed") is not None:
            group = row.get("benchmark_group") or row["example_id"]
            by_group[group][row["mode"]].append(row["execution_passed"])
    differences = [
        mean(modes[candidate]) - mean(modes[baseline])
        for modes in by_group.values()
        if baseline in modes and candidate in modes
    ]
    if not differences:
        return {"count": 0, "difference": None, "ci_low": None, "ci_high": None}
    random = Random(seed)
    bootstraps = sorted(
        mean(random.choices(differences, k=len(differences))) for _ in range(samples)
    )
    return {
        "count": len(differences),
        "difference": mean(differences),
        "ci_low": bootstraps[int(samples * 0.025)],
        "ci_high": bootstraps[int(samples * 0.975)],
    }


def classify_hypothesis(
    summary: dict[str, dict], comparison: dict | None = None
) -> str:
    generic = summary.get("generic")
    lattice = summary.get("lattice")
    if not generic or not lattice:
        return "inconclusive"
    if (
        generic["execution_pass_rate"] is not None
        and lattice["execution_pass_rate"] is not None
    ):
        if comparison and comparison["difference"] is not None:
            if comparison["difference"] < 0.05:
                return "falsified"
            if comparison["ci_low"] <= 0:
                return "inconclusive"
            return (
                "supported"
                if lattice["execution_per_million_active_parameters"]
                > generic["execution_per_million_active_parameters"]
                else "inconclusive"
            )
        if lattice["execution_pass_rate"] < generic["execution_pass_rate"] + 0.05:
            return "falsified"
        if lattice["execution_ci_low"] <= generic["execution_ci_high"]:
            return "inconclusive"
        return (
            "supported"
            if lattice["execution_per_million_active_parameters"]
            > generic["execution_per_million_active_parameters"]
            else "inconclusive"
        )
    if lattice["fuzzy_score"] <= generic["fuzzy_score"]:
        return "falsified"
    if lattice["fuzzy_ci_low"] <= generic["fuzzy_ci_high"]:
        return "inconclusive"
    lattice_efficiency = lattice.get("score_per_million_active_parameters")
    generic_efficiency = generic.get("score_per_million_active_parameters")
    if lattice_efficiency is None or generic_efficiency is None:
        return "inconclusive"
    return "supported" if lattice_efficiency > generic_efficiency else "inconclusive"
