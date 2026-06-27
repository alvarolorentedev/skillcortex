import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from ..contracts import KNOWN_SKILLS, TASK_TYPES
from .types import ExecutionFixture


@dataclass(slots=True)
class ProductTrainingExample:
    id: str
    task_type: str
    prompt: str
    target: str
    execution: ExecutionFixture | None = None
    group: str | None = None
    metadata: dict | None = None
    semantic_family: str | None = None
    skills: list[str] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id.strip():
            raise ValueError("id must be non-empty")
        if not isinstance(self.prompt, str) or not self.prompt.strip():
            raise ValueError("prompt must be non-empty")
        if not isinstance(self.target, str) or not self.target.strip():
            raise ValueError("target must be non-empty")
        if self.task_type not in TASK_TYPES:
            raise ValueError(f"unknown task_type: {self.task_type}")
        if isinstance(self.execution, dict):
            self.execution = ExecutionFixture.from_dict(self.execution)
        if self.metadata is not None and not isinstance(self.metadata, dict):
            raise ValueError("metadata must be a mapping")
        if self.semantic_family is not None and not isinstance(self.semantic_family, str):
            raise ValueError("semantic_family must be a string")
        if self.skills is not None:
            if not isinstance(self.skills, list) or any(not isinstance(item, str) or not item.strip() for item in self.skills):
                raise ValueError("skills must be a list of non-empty strings")

    @classmethod
    def from_dict(cls, value: dict) -> "ProductTrainingExample":
        return cls(**value)

    def to_dict(self) -> dict:
        return asdict(self)


def load_product_jsonl(path: str | Path) -> list[ProductTrainingExample]:
    examples: list[ProductTrainingExample] = []
    seen: set[str] = set()
    candidate = Path(path)
    with candidate.open() as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                example = ProductTrainingExample.from_dict(json.loads(line))
            except (TypeError, ValueError, json.JSONDecodeError) as error:
                raise ValueError(f"{candidate}:{line_number}: {error}") from error
            if example.id in seen:
                raise ValueError(f"{candidate}:{line_number}: duplicate id {example.id}")
            seen.add(example.id)
            examples.append(example)
    if not examples:
        raise ValueError(f"{candidate} is empty")
    return examples


def load_jsonl(path: str | Path) -> list[ProductTrainingExample]:
    return load_product_jsonl(path)


def select_for_skill(
    examples: Iterable[ProductTrainingExample], skill: str
) -> list[ProductTrainingExample]:
    if skill not in KNOWN_SKILLS:
        raise ValueError(f"unknown skill: {skill}")
    return [example for example in examples if skill in (example.skills or [])]


def dataset_hash(examples: Iterable[ProductTrainingExample]) -> str:
    import hashlib

    payload = "\n".join(
        json.dumps(example.to_dict(), sort_keys=True, separators=(",", ":"))
        for example in examples
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def write_product_mlx_dataset(examples: list[ProductTrainingExample], directory: str | Path) -> Path:
    candidate = Path(directory)
    candidate.mkdir(parents=True, exist_ok=True)
    valid = examples[::10] or examples[:1]
    train = [example for index, example in enumerate(examples) if index % 10] or list(examples)
    for name, rows in (("train.jsonl", train), ("valid.jsonl", valid)):
        (candidate / name).write_text(
            "".join(
                json.dumps({"messages": [{"role": "user", "content": example.prompt}, {"role": "assistant", "content": example.target}]}) + "\n"
                for example in rows
            )
        )
    return candidate


def write_mlx_dataset(examples: list[ProductTrainingExample], directory: str | Path) -> Path:
    return write_product_mlx_dataset(examples, directory)
