from dataclasses import asdict, dataclass, field
from typing import Any

SKILLS = ("python_skill", "debugging_skill", "test_generation_skill")
MODES = ("base", "generic", "single-skill", "lattice")
TASK_TYPES = ("python_generation", "debugging", "test_generation")


def _nonempty(name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be non-empty")


def _valid_skills(skills: list[str]) -> None:
    _known_skills(skills)
    if not skills:
        raise ValueError("skills must be non-empty")


def _known_skills(skills: list[str]) -> None:
    unknown = set(skills) - set(SKILLS)
    if unknown:
        raise ValueError(f"unknown skill: {sorted(unknown)[0]}")


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
class DatasetExample:
    id: str
    task_type: str
    skills: list[str]
    prompt: str
    target: str
    execution: ExecutionFixture | None = None
    group: str | None = None

    def __post_init__(self) -> None:
        _nonempty("id", self.id)
        _nonempty("prompt", self.prompt)
        _nonempty("target", self.target)
        if self.task_type not in TASK_TYPES:
            raise ValueError(f"unknown task_type: {self.task_type}")
        _valid_skills(self.skills)
        if isinstance(self.execution, dict):
            self.execution = ExecutionFixture.from_dict(self.execution)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "DatasetExample":
        return cls(**value)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RouteDecision:
    selected_skills: list[str]
    confidence: float
    reason: str

    def __post_init__(self) -> None:
        _valid_skills(self.selected_skills)
        if len(self.selected_skills) > 2:
            raise ValueError("at most two skills may be selected")
        if not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")
        _nonempty("reason", self.reason)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class GenerationResult:
    mode: str
    generation: str
    selected_skills: list[str] = field(default_factory=list)
    route: RouteDecision | None = None
    latency_seconds: float = 0.0
    prompt_tokens: int | None = None
    generated_tokens: int | None = None
    peak_memory_bytes: int | None = None
    active_adapter_count: int = 0
    active_adapter_parameters: int = 0
    error: str | None = None

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"unknown mode: {self.mode}")
        _known_skills(self.selected_skills)
        if self.active_adapter_count < 0 or self.active_adapter_parameters < 0:
            raise ValueError("adapter statistics must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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
    selected_skills: list[str]
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
        if self.mode not in MODES:
            raise ValueError(f"unknown mode: {self.mode}")
        if not 0 <= self.fuzzy_score <= 1:
            raise ValueError("fuzzy_score must be between 0 and 1")
        _known_skills(self.selected_skills)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
