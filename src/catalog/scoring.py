from __future__ import annotations

import re
from typing import Any

from .types import SlmRecord


ROUTE_THRESHOLD = 0.25


def candidate(
    slm: SlmRecord,
    task: str,
    repo_context: dict[str, Any],
    *,
    current_base_model: str | None,
) -> dict[str, Any]:
    task_text = task.lower()
    repo_signals = set(repo_context["language_signals"]) | set(repo_context["framework_signals"])
    matched: set[str] = set()
    negative: set[str] = set()
    breakdown = {
        "task": 0.0,
        "repo": 0.0,
        "positive_examples": 0.0,
        "negative": 0.0,
        "eval": 0.0,
        "task_type_hint": 0.0,
        "base_model": 0.0,
    }
    positive_terms = [slm.description, *slm.capabilities, *slm.activation_cues]
    for term in positive_terms:
        signal = _matching_signal(term, task_text)
        if signal:
            matched.add(signal)
            breakdown["task"] += 0.12
        repo_signal = _repo_signal(term, repo_signals)
        if repo_signal:
            matched.add(repo_signal)
            breakdown["repo"] += 0.08
    for example in slm.routing_card.positive_examples:
        signal = _matching_signal(example, task_text)
        if signal:
            matched.add(signal)
            breakdown["positive_examples"] += 0.08
    for term in [*slm.avoid_when, *slm.routing_card.negative_examples]:
        signal = _matching_signal(term, task_text)
        if signal:
            negative.add(signal)
            breakdown["negative"] -= 0.18
    if slm.task_type_hint and _matching_signal(slm.task_type_hint.replace("_", " "), task_text):
        matched.add(slm.task_type_hint)
        breakdown["task_type_hint"] = 0.04
    breakdown["eval"] = _eval_bonus(slm.eval_summary)
    compatible = True
    if current_base_model and slm.base_model and current_base_model != slm.base_model:
        compatible = False
        negative.add("base_model")
        breakdown["base_model"] = -1.0
    score = max(0.0, min(1.0, sum(breakdown.values())))
    if not compatible:
        score = 0.0
    return {
        "slm_id": slm.slm_id,
        "score": round(score, 4),
        "selected": False,
        "compatible": compatible,
        "matched_signals": sorted(matched),
        "negative_signals": sorted(negative),
        "score_breakdown": {key: round(value, 4) for key, value in breakdown.items() if value},
        "reason": _reason(score, compatible, matched, negative),
    }


def _matching_signal(term: str, text: str) -> str | None:
    words = _words(term)
    if not words:
        return None
    if len(words) > 1 and " ".join(words) in text:
        return " ".join(words)
    for word in words:
        if len(word) >= 3 and re.search(rf"\b{re.escape(word)}\b", text):
            return word
    return None


def _repo_signal(term: str, repo_signals: set[str]) -> str | None:
    for word in _words(term):
        if word in repo_signals:
            return word
    return None


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _eval_bonus(summary: dict[str, Any]) -> float:
    values = []
    for mode in (summary.get("modes") or {}).values():
        if isinstance(mode, dict):
            for key in ("validation_pass_rate", "execution_pass_rate", "fuzzy_score"):
                value = mode.get(key)
                if isinstance(value, int | float):
                    values.append(float(value))
    return min(values or [0.0]) * 0.05


def _reason(score: float, compatible: bool, matched: set[str], negative: set[str]) -> str:
    if not compatible:
        return "Slm base model is incompatible with the current base model."
    if score >= ROUTE_THRESHOLD and matched:
        return "Matched capability signals: " + ", ".join(sorted(matched)[:6]) + "."
    if negative:
        return "Negative routing signals outweighed capability matches."
    return "Insufficient capability evidence for this task."
