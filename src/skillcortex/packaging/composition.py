from __future__ import annotations

import json


DEFAULT_COMPOSITION = {
    "python_skill": {
        "capabilities": {
            "allowed_task_types": ["debugging", "test_generation"],
        },
        "activation": {
            "default_route_type": "adapter",
            "scope": "task",
            "semantic_families": [],
        },
        "compatibility": {
            "compatible_skills": [],
            "incompatible_skills": [],
        },
        "routing": {
            "tasks": {
                "debugging": {
                    "order": 20,
                    "requires_any_of": ["debugging_skill"],
                },
                "test_generation": {
                    "order": 10,
                    "requires_any_of": ["test_generation_skill"],
                },
            }
        },
    },
    "debugging_skill": {
        "capabilities": {
            "allowed_task_types": ["debugging"],
        },
        "activation": {
            "default_route_type": "adapter",
            "scope": "task",
            "semantic_families": [],
        },
        "compatibility": {
            "compatible_skills": [],
            "incompatible_skills": [],
        },
        "routing": {
            "tasks": {
                "debugging": {
                    "order": 10,
                    "requires_all_of": ["python_skill"],
                }
            }
        },
    },
    "test_generation_skill": {
        "capabilities": {
            "allowed_task_types": ["test_generation"],
        },
        "activation": {
            "default_route_type": "adapter",
            "scope": "task",
            "semantic_families": [],
        },
        "compatibility": {
            "compatible_skills": [],
            "incompatible_skills": [],
        },
        "routing": {
            "tasks": {
                "test_generation": {
                    "order": 20,
                    "requires_all_of": ["python_skill"],
                }
            }
        },
    },
    "alternating_skill": {
        "capabilities": {
            "allowed_task_types": ["debugging", "test_generation"],
        },
        "activation": {
            "default_route_type": "adapter",
            "scope": "semantic_family",
            "semantic_families": ["alternating"],
        },
        "compatibility": {
            "compatible_skills": [],
            "incompatible_skills": [],
        },
        "routing": {
            "tasks": {
                "debugging": {
                    "order": 30,
                    "requires_all_of": ["debugging_skill", "python_skill"],
                },
                "test_generation": {
                    "order": 30,
                    "requires_all_of": ["python_skill", "test_generation_skill"],
                },
            }
        },
    },
}

COMPOSITION_SCOPES = {"task", "semantic_family"}
COMPOSITION_ROUTE_TYPES = {"adapter", "base_fallback"}
KNOWN_TASK_TYPES = {"python_generation", "debugging", "test_generation"}


def normalized_skill_id(skill_id: str) -> str:
    nonempty("skill_id", skill_id)
    normalized = skill_id.strip().lower().replace("-", "_")
    if not all(char.isalnum() or char == "_" for char in normalized):
        raise ValueError("skill_id must contain only letters, numbers, dashes, or underscores")
    return normalized


def default_composition(skill_id: str) -> dict | None:
    composition = DEFAULT_COMPOSITION.get(skill_id)
    if composition is None:
        return None
    return json.loads(json.dumps(composition, sort_keys=True))


def validate_composition_metadata(composition: dict) -> None:
    if not isinstance(composition, dict):
        raise ValueError("composition metadata must be a mapping")
    capabilities = composition.get("capabilities") or {}
    activation = composition.get("activation") or {}
    compatibility = composition.get("compatibility") or {}
    routing = composition.get("routing") or {}
    allowed_task_types = capabilities.get("allowed_task_types") or []
    if not allowed_task_types:
        raise ValueError("composition.capabilities.allowed_task_types must be non-empty")
    unknown_tasks = set(allowed_task_types) - KNOWN_TASK_TYPES
    if unknown_tasks:
        raise ValueError(f"unknown composition allowed_task_type: {sorted(unknown_tasks)[0]}")
    route_type = activation.get("default_route_type")
    if route_type not in COMPOSITION_ROUTE_TYPES:
        raise ValueError(
            "composition.activation.default_route_type must be 'adapter' or 'base_fallback'"
        )
    scope = activation.get("scope")
    if scope not in COMPOSITION_SCOPES:
        raise ValueError("composition.activation.scope must be 'task' or 'semantic_family'")
    semantic_families = activation.get("semantic_families") or []
    if scope == "semantic_family" and not semantic_families:
        raise ValueError(
            "composition.activation.semantic_families must be non-empty for semantic_family scope"
        )
    for key in ("compatible_skills", "incompatible_skills"):
        value = compatibility.get(key) or []
        if len(value) != len(set(value)):
            raise ValueError(f"composition.compatibility.{key} must not contain duplicates")
    task_routing = routing.get("tasks") or {}
    for task_type, task_rules in sorted(task_routing.items()):
        if task_type not in KNOWN_TASK_TYPES:
            raise ValueError(f"unknown composition routing task: {task_type}")
        if task_type not in allowed_task_types:
            raise ValueError(
                f"composition.routing.tasks.{task_type} requires task to be in allowed_task_types"
            )
        if not isinstance(task_rules, dict):
            raise ValueError(f"composition.routing.tasks.{task_type} must be a mapping")
        order = task_rules.get("order")
        if order is not None and (not isinstance(order, int) or order < 0):
            raise ValueError(
                f"composition.routing.tasks.{task_type}.order must be a non-negative integer"
            )
        for key in ("requires_all_of", "requires_any_of"):
            values = task_rules.get(key) or []
            if len(values) != len(set(values)):
                raise ValueError(
                    f"composition.routing.tasks.{task_type}.{key} must not contain duplicates"
                )


def nonempty(name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be non-empty")
