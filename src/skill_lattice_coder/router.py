from collections import Counter

from .schemas import RouteDecision, SKILLS

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
