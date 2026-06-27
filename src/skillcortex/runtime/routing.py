from __future__ import annotations

from typing import Any, Callable

from ..contracts import TASK_TYPES
from .models import RuntimeRouteDecision


def infer_task_type(messages: list[dict[str, str]], route_text: Callable[[str], list[str]]) -> str:
    user_text = "\n".join(message["content"] for message in messages if message["role"] == "user")
    selected = route_text(user_text)
    if "debugging_skill" in selected:
        return "debugging"
    if "test_generation_skill" in selected:
        return "test_generation"
    return "python_generation"


def select_route(
    routes: list[dict[str, Any]],
    task_type: str,
    semantic_family: str | None,
) -> dict[str, Any]:
    for route in routes:
        if route["task_type"] == task_type and route.get("semantic_family") == semantic_family:
            return route
    for route in routes:
        if route["task_type"] == task_type and route.get("semantic_family") is None:
            return route
    raise ValueError(f"runtime bundle is missing a default route for {task_type}")


def build_route_decision(
    routes: list[dict[str, Any]],
    messages: list[dict[str, str]],
    *,
    task_type: str | None,
    semantic_family: str | None,
    skill_override: str | None,
    available_skills: set[str],
    route_text: Callable[[str], list[str]],
) -> RuntimeRouteDecision:
    if skill_override is not None:
        if skill_override not in available_skills:
            raise ValueError(f"unknown runtime skill override: {skill_override}")
        return RuntimeRouteDecision(
            [skill_override],
            1.0,
            f"explicit skill override selected task_type={task_type or 'python_generation'}",
        )

    resolved_task_type = task_type or infer_task_type(messages, route_text)
    if resolved_task_type not in TASK_TYPES:
        raise ValueError(f"unknown task_type: {resolved_task_type}")
    route = select_route(routes, resolved_task_type, semantic_family)
    return RuntimeRouteDecision(
        list(route["selected_skills"]),
        1.0,
        f"runtime bundle route {route['route_id']} selected task_type={resolved_task_type}",
        route_type=route["route_type"],
    )
