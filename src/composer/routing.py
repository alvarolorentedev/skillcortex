from __future__ import annotations

from ..contracts import TASK_TYPES


def build_routes(loaded: list[dict]) -> list[dict]:
    task_scoped = [item for item in loaded if item["composition"]["activation"]["scope"] == "task"]
    semantic_scoped = [item for item in loaded if item not in task_scoped]
    available = {item["slm_id"] for item in loaded}
    routes = []
    for task_type in TASK_TYPES:
        selected = route_selection(task_scoped, task_type, available)
        routes.append(
            {
                "route_id": f"{task_type}.default",
                "task_type": task_type,
                "semantic_family": None,
                "route_type": "adapter" if selected else "base_fallback",
                "selected_slms": selected,
            }
        )
    semantic_families = sorted(
        {
            family
            for item in semantic_scoped
            for family in item["composition"]["activation"].get("semantic_families", [])
        }
    )
    for semantic_family in semantic_families:
        for task_type in TASK_TYPES:
            selected = route_selection(task_scoped, task_type, available)
            selected.extend(
                route_selection(
                    [
                        item
                        for item in semantic_scoped
                        if semantic_family in item["composition"]["activation"].get("semantic_families", [])
                    ],
                    task_type,
                    available,
                )
            )
            default = next(
                route
                for route in routes
                if route["task_type"] == task_type and route["semantic_family"] is None
            )
            if selected == default["selected_slms"]:
                continue
            routes.append(
                {
                    "route_id": f"{task_type}.{semantic_family}",
                    "task_type": task_type,
                    "semantic_family": semantic_family,
                    "route_type": "adapter" if selected else "base_fallback",
                    "selected_slms": selected,
                }
            )
    return routes


def route_selection(items: list[dict], task_type: str, available: set[str]) -> list[str]:
    selected = []
    candidates = []
    for item in items:
        allowed = item["composition"]["capabilities"].get("allowed_task_types") or []
        if task_type not in allowed:
            continue
        rule = ((item["composition"].get("routing") or {}).get("tasks") or {}).get(task_type, {})
        requires_all = set(rule.get("requires_all_of") or [])
        requires_any = set(rule.get("requires_any_of") or [])
        if not requires_all.issubset(available):
            continue
        if requires_any and not (requires_any & available):
            continue
        candidates.append((int(rule.get("order", 100)), item["slm_id"]))
    for _, slm_id in sorted(candidates, key=lambda value: (value[0], value[1])):
        selected.append(slm_id)
    return selected
