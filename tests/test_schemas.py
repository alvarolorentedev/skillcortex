import pytest

from skill_lattice_coder.schemas import (
    DatasetExample,
    ExecutionFixture,
    RouteDecision,
)


def test_dataset_example_validates_required_fields_and_skills():
    example = DatasetExample(
        id="example-1",
        task_type="debugging",
        skills=["python_skill", "debugging_skill"],
        prompt="Fix this Python error",
        target="def fixed(): pass",
    )
    assert example.id == "example-1"

    with pytest.raises(ValueError, match="unknown skill"):
        DatasetExample("bad", "debugging", ["magic_skill"], "prompt", "target")
    with pytest.raises(ValueError, match="prompt"):
        DatasetExample("bad", "debugging", ["debugging_skill"], " ", "target")


def test_execution_fixture_requires_files_and_command():
    fixture = ExecutionFixture(
        files={"test_solution.py": "assert True"}, command=["pytest", "-q"]
    )
    assert fixture.timeout_seconds == 10
    with pytest.raises(ValueError, match="files"):
        ExecutionFixture(files={}, command=["pytest"])
    with pytest.raises(ValueError, match="command"):
        ExecutionFixture(files={"x.py": ""}, command=[])


def test_route_decision_validates_confidence():
    assert RouteDecision(["python_skill"], 0.8, "Python marker").confidence == 0.8
    with pytest.raises(ValueError, match="confidence"):
        RouteDecision(["python_skill"], 1.1, "bad")
