from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from ..registry import ResolvedAdapter


@dataclass(slots=True)
class DynamicRouteDecision:
    base_model: str
    selected_slms: list[str]
    remote_loras: list[str]
    task_type: str | None
    semantic_family: str | None
    train_new_lora: bool
    reason: str


Router = Callable[[list[dict[str, str]], list[ResolvedAdapter]], DynamicRouteDecision]
