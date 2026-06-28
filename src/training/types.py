from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from ..contracts import KNOWN_SLMS, TASK_TYPES


def _nonempty(name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be non-empty")


def _known_slms(slms: list[str]) -> None:
    unknown = set(slms) - set(KNOWN_SLMS)
    if unknown:
        raise ValueError(f"unknown slm: {sorted(unknown)[0]}")


@dataclass(slots=True)
class ExecutionFixture:
    files: dict[str, str]
    command: list[str]
    timeout_seconds: int = 10

    def __post_init__(self) -> None:
        if not self.files or any(
            not name or name.startswith("/") or ".." in name.split("/")
            for name in self.files
        ):
            raise ValueError("files must contain safe relative paths")
        if not self.command or not all(
            isinstance(part, str) and part for part in self.command
        ):
            raise ValueError("command must be non-empty")
        if not 1 <= self.timeout_seconds <= 60:
            raise ValueError("timeout_seconds must be between 1 and 60")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ExecutionFixture":
        return cls(**value)


@dataclass(slots=True)
class EvaluationResult:
    example_id: str
    task_type: str
    mode: str
    generation: str
    exact_match: bool
    fuzzy_score: float
    syntax_valid: bool | None
    execution_passed: bool | None
    latency_seconds: float
    selected_slms: list[str]
    active_adapter_count: int
    active_adapter_parameters: int
    prompt_tokens: int | None = None
    generated_tokens: int | None = None
    peak_memory_bytes: int | None = None
    qualitative_score: float | None = None
    error: str | None = None
    benchmark_group: str | None = None

    def __post_init__(self) -> None:
        _nonempty("example_id", self.example_id)
        if self.task_type not in TASK_TYPES:
            raise ValueError(f"unknown task_type: {self.task_type}")
        _known_slms(self.selected_slms)
        if not 0 <= self.fuzzy_score <= 1:
            raise ValueError("fuzzy_score must be between 0 and 1")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
