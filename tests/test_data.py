import json

import pytest

from skill_lattice_coder.data import (
    dataset_hash,
    load_jsonl,
    select_for_skill,
    to_mlx_chat,
)


def test_load_jsonl_rejects_duplicate_ids(tmp_path):
    path = tmp_path / "data.jsonl"
    row = {
        "id": "same",
        "task_type": "python_generation",
        "skills": ["python_skill"],
        "prompt": "Write Python",
        "target": "print('ok')",
    }
    path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="duplicate"):
        load_jsonl(path)


def test_filter_hash_and_mlx_conversion_are_deterministic(tmp_path):
    path = tmp_path / "data.jsonl"
    rows = [
        {
            "id": "a",
            "task_type": "debugging",
            "skills": ["python_skill", "debugging_skill"],
            "prompt": "Fix it",
            "target": "fixed",
        },
        {
            "id": "b",
            "task_type": "test_generation",
            "skills": ["python_skill", "test_generation_skill"],
            "prompt": "Write tests",
            "target": "def test_x(): pass",
        },
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    examples = load_jsonl(path)
    assert [x.id for x in select_for_skill(examples, "debugging_skill")] == ["a"]
    assert dataset_hash(examples) == dataset_hash(examples)
    assert to_mlx_chat(examples[0])["messages"][-1] == {
        "role": "assistant",
        "content": "fixed",
    }


def test_checked_in_benchmark_is_balanced_executable_and_held_out():
    train = load_jsonl("data/train.jsonl")
    evaluation = load_jsonl("data/eval.jsonl")
    assert len(train) == 60
    assert len(evaluation) == 150
    assert {
        task: sum(row.task_type == task for row in train)
        for task in {"python_generation", "debugging", "test_generation"}
    } == {
        "python_generation": 20,
        "debugging": 20,
        "test_generation": 20,
    }
    assert {
        task: sum(row.task_type == task for row in evaluation)
        for task in {"python_generation", "debugging", "test_generation"}
    } == {
        "python_generation": 50,
        "debugging": 50,
        "test_generation": 50,
    }
    assert not ({row.id for row in train} & {row.id for row in evaluation})
    assert all(row.execution is not None for row in evaluation)
    assert len({row.prompt for row in evaluation}) == len(evaluation)
    assert len({row.group for row in evaluation}) == 50
    assert all(
        sum(other.group == row.group for other in evaluation) == 3 for row in evaluation
    )
