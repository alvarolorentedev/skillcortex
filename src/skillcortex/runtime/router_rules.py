from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any

from ..contracts import KNOWN_SKILLS, SKILLS, TASK_TYPES


RULES = {
    "debugging_skill": ("traceback", "error", "exception", "failing", "fix", "bug"),
    "test_generation_skill": (
        "pytest",
        "unit test",
        "tests for",
        "test cases",
        "generate tests",
    ),
    "python_skill": (
        "python",
        "function",
        "def ",
        "class ",
        "list",
        "dictionary",
        "algorithm",
    ),
}
PRIORITY = ("debugging_skill", "test_generation_skill", "python_skill")


def _nonempty(name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be non-empty")


def _known_skills(skills: list[str]) -> None:
    unknown = set(skills) - set(KNOWN_SKILLS)
    if unknown:
        raise ValueError(f"unknown skill: {sorted(unknown)[0]}")


@dataclass(slots=True)
class RouteDecision:
    selected_skills: list[str]
    confidence: float
    reason: str
    route_type: str = "adapter"

    def __post_init__(self) -> None:
        _known_skills(self.selected_skills)
        if len(self.selected_skills) > 3:
            raise ValueError("at most three skills may be selected")
        if not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")
        _nonempty("reason", self.reason)
        _nonempty("route_type", self.route_type)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RuleRouter:
    def route(self, prompt: str, tags: list[str] | None = None) -> RouteDecision:
        if tags:
            selected = [skill for skill in tags if skill in SKILLS][:2]
            if not selected:
                raise ValueError("tags contain no known skills")
            return RouteDecision(
                selected, 1.0, "Explicit dataset tags override prompt rules."
            )

        text = prompt.lower()
        scores = Counter(
            {
                skill: sum(marker in text for marker in markers)
                for skill, markers in RULES.items()
            }
        )
        ranked = [skill for skill in PRIORITY if scores[skill] > 0]
        if not ranked:
            return RouteDecision(
                ["python_skill"],
                0.35,
                "No strong marker; defaulted to Python generation.",
            )
        selected = ranked[:2]
        total = sum(scores.values())
        confidence = min(0.95, 0.55 + 0.1 * total)
        markers = ", ".join(f"{skill}={scores[skill]}" for skill in selected)
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
            ["debugging_skill", "python_skill"]
            if task_type == "debugging"
            else ["python_skill", "test_generation_skill"]
        )
        return RouteDecision(selected, 1.0, f"Protected route for {task_type}.")


ProtectedSkillRouter = PythonOnlyForTestGenerationRouter
ProtectedSkillRouterWithoutFailureBorn = PythonOnlyForTestGenerationRouter


class SkillCortexRouterV1:
    def route(self, task_type: str, semantic_family: str | None) -> RouteDecision:
        protected = ProtectedSkillRouterWithoutFailureBorn().route(task_type)
        if semantic_family != "alternating" or task_type == "python_generation":
            return protected
        selected = (
            ["debugging_skill", "python_skill", "alternating_skill"]
            if task_type == "debugging"
            else ["python_skill", "test_generation_skill", "alternating_skill"]
        )
        return RouteDecision(selected, 1.0, "Promoted alternating route.")


class ProtectedRouterPlusAlternatingSkill:
    quarantined = True

    def route(self, task_type: str, semantic_family: str | None) -> RouteDecision:
        protected = ProtectedSkillRouter().route(task_type)
        if semantic_family != "alternating" or task_type == "python_generation":
            return protected
        selected = (
            ["debugging_skill", "python_skill", "alternating_skill"]
            if task_type == "debugging"
            else ["python_skill", "test_generation_skill", "alternating_skill"]
        )
        return RouteDecision(
            selected,
            1.0,
            "Quarantined alternating candidate route.",
            route_type="quarantined_candidate",
        )


def route_text(text: str) -> list[str]:
    return list(RuleRouter().route(text).selected_skills)
