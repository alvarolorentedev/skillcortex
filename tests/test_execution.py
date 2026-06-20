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
