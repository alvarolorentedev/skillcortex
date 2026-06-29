from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class RoutingCard:
    summary: str = ""
    embedding_text: str = ""
    positive_examples: list[str] = field(default_factory=list)
    negative_examples: list[str] = field(default_factory=list)
    observed_success_contexts: list[dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class SlmRecord:
    slm_id: str
    name: str
    path: Path
    description: str = ""
    capabilities: list[str] = field(default_factory=list)
    activation_cues: list[str] = field(default_factory=list)
    avoid_when: list[str] = field(default_factory=list)
    task_type_hint: str | None = None
    base_model: str | None = None
    adapter_path: Path | None = None
    routing_card: RoutingCard = field(default_factory=RoutingCard)
    eval_summary: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class CatalogResult:
    slms: list[SlmRecord]
    errors: list[str]
    warnings: list[str]
