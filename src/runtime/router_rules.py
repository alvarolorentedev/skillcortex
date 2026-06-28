from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any

from ..contracts import KNOWN_SLMS, PRESET_SLMS, TASK_TYPES


RULES = {
    "debugging_slm": ("traceback", "error", "exception", "failing", "fix", "bug"),
    "test_generation_slm": (
        "pytest",
        "unit test",
        "tests for",
        "test cases",
        "generate tests",
    ),
    "python_slm": (
        "python",
        "function",
        "def ",
        "class ",
        "list",
        "dictionary",
        "algorithm",
    ),
}
PRIORITY = ("debugging_slm", "test_generation_slm", "python_slm")


def _nonempty(name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be non-empty")


def _known_slms(slms: list[str]) -> None:
    unknown = set(slms) - set(KNOWN_SLMS)
    if unknown:
        raise ValueError(f"unknown slm: {sorted(unknown)[0]}")


@dataclass(slots=True)
class RouteDecision:
    selected_slms: list[str]
    confidence: float
    reason: str
    route_type: str = "adapter"

    def __post_init__(self) -> None:
        _known_slms(self.selected_slms)
        if len(self.selected_slms) > 3:
            raise ValueError("at most three slms may be selected")
        if not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")
        _nonempty("reason", self.reason)
        _nonempty("route_type", self.route_type)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RuleRouter:
    def route(self, prompt: str, tags: list[str] | None = None) -> RouteDecision:
        if tags:
            selected = [slm for slm in tags if slm in PRESET_SLMS][:2]
            if not selected:
                raise ValueError("tags contain no known slms")
            return RouteDecision(
                selected, 1.0, "Explicit dataset tags override prompt rules."
            )

        text = prompt.lower()
        scores = Counter(
            {
                slm: sum(marker in text for marker in markers)
                for slm, markers in RULES.items()
            }
        )
        ranked = [slm for slm in PRIORITY if scores[slm] > 0]
        if not ranked:
            return RouteDecision(
                ["python_slm"],
                0.35,
                "No strong marker; defaulted to Python generation.",
            )
        selected = ranked[:2]
        total = sum(scores.values())
        confidence = min(0.95, 0.55 + 0.1 * total)
        markers = ", ".join(f"{slm}={scores[slm]}" for slm in selected)
        return RouteDecision(selected, confidence, f"Matched prompt rules: {markers}.")


class PythonOnlyForTestGenerationRouter:
    def route(self, task_type: str) -> RouteDecision:
        if task_type not in TASK_TYPES:
            raise ValueError(f"unknown task_type: {task_type}")
        if task_type == "python_generation":
            return RouteDecision(
                [],
                1.0,
                "Pure Python generation uses the frozen base model.",
                route_type="base_fallback",
            )
        selected = (
            ["debugging_slm", "python_slm"]
            if task_type == "debugging"
            else ["python_slm", "test_generation_slm"]
        )
        return RouteDecision(selected, 1.0, f"Protected route for {task_type}.")


ProtectedSlmRouter = PythonOnlyForTestGenerationRouter
ProtectedSlmRouterWithoutFailureBorn = PythonOnlyForTestGenerationRouter


class SLMCortexRouterV1:
    def route(self, task_type: str, semantic_family: str | None) -> RouteDecision:
        protected = ProtectedSlmRouterWithoutFailureBorn().route(task_type)
        if semantic_family != "alternating" or task_type == "python_generation":
            return protected
        selected = (
            ["debugging_slm", "python_slm", "alternating_slm"]
            if task_type == "debugging"
            else ["python_slm", "test_generation_slm", "alternating_slm"]
        )
        return RouteDecision(selected, 1.0, "Promoted alternating route.")


class ProtectedRouterPlusAlternatingSlm:
    quarantined = True

    def route(self, task_type: str, semantic_family: str | None) -> RouteDecision:
        protected = ProtectedSlmRouter().route(task_type)
        if semantic_family != "alternating" or task_type == "python_generation":
            return protected
        selected = (
            ["debugging_slm", "python_slm", "alternating_slm"]
            if task_type == "debugging"
            else ["python_slm", "test_generation_slm", "alternating_slm"]
        )
        return RouteDecision(
            selected,
            1.0,
            "Quarantined alternating candidate route.",
            route_type="quarantined_candidate",
        )


def route_text(text: str) -> list[str]:
    return list(RuleRouter().route(text).selected_slms)

# Note: `SLMCortexRouterV1` is the canonical router class name.
