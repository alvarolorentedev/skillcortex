from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


REQUIRED_RUNTIME_FILES = (
    "composition.yaml",
    "router_config.json",
    "active_skills.json",
    "compatibility_report.json",
    "budget_report.json",
    "checksums.json",
)


@dataclass(slots=True)
class RuntimeSkill:
    skill_id: str
    name: str
    version: str
    package_path: Path
    adapter_path: Path
    fingerprint: str
    allowed_task_types: list[str]
    activation: dict[str, Any]
    trainable_parameters: int


@dataclass(slots=True)
class RuntimeBundle:
    path: Path
    name: str
    runtime_model: str
    source_model: str
    quantization: str
    strategy: str
    routes: list[dict[str, Any]]
    skills: dict[str, RuntimeSkill]
    compatibility_report: dict[str, Any]
    budget_report: dict[str, Any]
    checksums: dict[str, Any]


@dataclass(slots=True)
class RuntimeRouteDecision:
    selected_skills: list[str]
    confidence: float
    reason: str
    route_type: str = "adapter"

    def __post_init__(self) -> None:
        if any(not isinstance(skill_id, str) or not skill_id.strip() for skill_id in self.selected_skills):
            raise ValueError("selected_skills must contain non-empty skill ids")
        if not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError("reason must be non-empty")
        if not isinstance(self.route_type, str) or not self.route_type.strip():
            raise ValueError("route_type must be non-empty")
