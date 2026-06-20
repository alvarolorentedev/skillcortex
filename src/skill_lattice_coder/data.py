import hashlib
import json
from pathlib import Path
from typing import Iterable

from .schemas import DatasetExample, SKILLS


def load_jsonl(path: str | Path) -> list[DatasetExample]:
    examples: list[DatasetExample] = []
    seen: set[str] = set()
    with Path(path).open() as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                example = DatasetExample.from_dict(json.loads(line))
            except (TypeError, ValueError, json.JSONDecodeError) as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error
            if example.id in seen:
                raise ValueError(f"{path}:{line_number}: duplicate id {example.id}")
            seen.add(example.id)
            examples.append(example)
    if not examples:
        raise ValueError(f"{path} is empty")
    return examples


def select_for_skill(
    examples: Iterable[DatasetExample], skill: str
) -> list[DatasetExample]:
    if skill not in SKILLS:
        raise ValueError(f"unknown skill: {skill}")
    return [example for example in examples if skill in example.skills]


def dataset_hash(examples: Iterable[DatasetExample]) -> str:
    payload = "\n".join(
        json.dumps(example.to_dict(), sort_keys=True, separators=(",", ":"))
        for example in examples
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def to_mlx_chat(example: DatasetExample) -> dict:
    return {
        "messages": [
            {"role": "user", "content": example.prompt},
            {"role": "assistant", "content": example.target},
        ]
    }


def write_mlx_dataset(examples: list[DatasetExample], directory: str | Path) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    valid = examples[::10]
    train = [example for index, example in enumerate(examples) if index % 10]
    for name, rows in (("train.jsonl", train), ("valid.jsonl", valid)):
        (directory / name).write_text(
            "".join(json.dumps(to_mlx_chat(example)) + "\n" for example in rows)
        )
    return directory
