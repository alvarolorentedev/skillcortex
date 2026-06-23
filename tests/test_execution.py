import subprocess
from pathlib import Path

import skill_lattice_coder.utils as utils
from skill_lattice_coder.schemas import ExecutionFixture
from skill_lattice_coder.utils import run_fixture


def test_run_fixture_handles_solution_and_generated_tests():
    function_fixture = ExecutionFixture(
        files={
            "test_solution.py": "from solution import answer\n\ndef test_answer(): assert answer() == 42\n"
        },
        command=["python", "-m", "pytest", "-q"],
    )
    assert run_fixture(function_fixture, "def answer(): return 42")[0]

    test_fixture = ExecutionFixture(
        files={"solution.py": "def answer(): return 42\n"},
        command=["python", "-m", "pytest", "-q", "test_generated.py"],
    )
    assert run_fixture(
        test_fixture,
        "from solution import answer\n\ndef test_answer(): assert answer() == 42",
    )[0]


def test_run_fixture_reports_failure_and_timeout():
    failure = ExecutionFixture(
        files={"test_solution.py": "def test_failure(): assert False\n"},
        command=["python", "-m", "pytest", "-q"],
    )
    assert not run_fixture(failure, "")[0]

    timeout = ExecutionFixture(
        files={"wait.py": "import time; time.sleep(2)"},
        command=["python", "wait.py"],
        timeout_seconds=1,
    )
    passed, output = run_fixture(timeout, "")
    assert not passed
    assert output == "execution timed out"


def test_run_fixture_uses_uv_for_pytest_when_module_missing(monkeypatch):
    fixture = ExecutionFixture(
        files={"test_solution.py": "def test_answer(): assert True\n"},
        command=["python", "-m", "pytest", "-q"],
    )
    recorded = {}

    def fake_find_spec(name):
        return None if name == "pytest" else object()

    monkeypatch.setattr(utils.importlib.util, "find_spec", fake_find_spec)
    monkeypatch.setattr(utils.shutil, "which", lambda name: "/usr/bin/uv")

    def fake_run(command, **kwargs):
        recorded["command"] = command
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(utils.subprocess, "run", fake_run)

    assert run_fixture(fixture, "")[0]
    assert recorded["command"][:6] == [
        "/usr/bin/uv",
        "run",
        "--project",
        str(Path(utils.__file__).resolve().parents[2]),
        "--extra",
        "test",
    ]
    assert recorded["command"][6:] == ["python", "-m", "pytest", "-q"]
