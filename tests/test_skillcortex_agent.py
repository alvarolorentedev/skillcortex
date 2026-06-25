import json

from skillcortex.cli import main


class FakeRuntime:
    def __init__(self):
        self.calls = []

    def validate(self):
        return {"status": "valid"}

    def infer(self, *, messages, task_type=None, semantic_family=None, skill_override=None, max_tokens=None, temperature=None, dry_run=False):
        prompt = messages[0]["content"]
        self.calls.append(
            {
                "task_type": task_type,
                "semantic_family": semantic_family,
                "skill_override": skill_override,
                "dry_run": dry_run,
                "prompt": prompt,
            }
        )
        if dry_run:
            return {
                "status": "dry-run",
                "selected_skills": ["debugging_skill", "python_skill"] if task_type == "debugging" else [],
                "route_type": "adapter" if task_type == "debugging" else "base_fallback",
                "reason": f"runtime bundle route {task_type or 'python_generation'}.default selected task_type={task_type or 'python_generation'}",
            }
        if task_type == "debugging":
            return {
                "status": "complete",
                "selected_skills": ["debugging_skill", "python_skill"],
                "route_type": "adapter",
                "reason": "runtime bundle route debugging.default selected task_type=debugging",
                "generation": json.dumps(
                    {
                        "kind": "proposed_diff",
                        "summary": "Debug the failing test.",
                        "diff": "--- a/app.py\n+++ b/app.py\n@@\n-return 0\n+return 42\n",
                    }
                ),
            }
        if "Create a short execution plan" in prompt:
            generation = "1. Inspect files\n2. Propose a patch\n3. Run validation"
        else:
            generation = json.dumps(
                {
                    "kind": "file_replace",
                    "path": "app.py",
                    "summary": "Replace the implementation.",
                    "content": "def answer():\n    return 42\n",
                }
            )
        return {
            "status": "complete",
            "selected_skills": [],
            "route_type": "base_fallback",
            "reason": f"runtime bundle route python_generation.default selected task_type={task_type}",
            "generation": generation,
        }


def _toy_repo(tmp_path):
    repo = tmp_path / "toy-repo"
    repo.mkdir()
    (repo / "app.py").write_text("def answer():\n    return 0\n")
    (repo / "test_app.py").write_text(
        "from app import answer\n\n\ndef test_answer():\n    assert answer() == 42\n"
    )
    return repo


def test_agent_run_records_dynamic_skill_switch_and_trace(tmp_path, monkeypatch, capsys):
    repo = _toy_repo(tmp_path)
    trace = tmp_path / "trace.json"
    fake_runtime = FakeRuntime()

    monkeypatch.setattr("skillcortex.agent.SkillRuntime.load", lambda path: fake_runtime)
    monkeypatch.setattr(
        "skillcortex.agent._run_validation_command",
        lambda command, repo: {
            "status": "failed",
            "command": command,
            "exit_code": 1,
            "stdout": "",
            "stderr": "AssertionError: expected 42",
        },
    )

    assert (
        main(
            [
                "agent",
                "run",
                "--runtime",
                str(tmp_path / "runtime"),
                "--repo",
                str(repo),
                "--task",
                "Fix the failing answer implementation.",
                "--test-command",
                "pytest -q",
                "--trace-out",
                str(trace),
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert result["writes_mode"] == "confirm"
    assert repo.joinpath("app.py").read_text() == "def answer():\n    return 0\n"
    assert result["validation"]["status"] == "failed"
    assert result["steps"][1]["selected_skills"] == []
    assert result["steps"][2]["write_status"] == "approval_required"
    assert result["steps"][3]["status"] == "failed"
    assert result["steps"][4]["selected_skills"] == ["debugging_skill", "python_skill"]
    trace_payload = json.loads(trace.read_text())
    assert trace_payload["steps"][4]["selected_skills"] == ["debugging_skill", "python_skill"]
    assert fake_runtime.calls[-1]["task_type"] == "debugging"


def test_agent_run_can_apply_file_replace_when_writes_on(tmp_path, monkeypatch, capsys):
    repo = _toy_repo(tmp_path)
    fake_runtime = FakeRuntime()

    monkeypatch.setattr("skillcortex.agent.SkillRuntime.load", lambda path: fake_runtime)
    monkeypatch.setattr(
        "skillcortex.agent._run_validation_command",
        lambda command, repo: {
            "status": "passed",
            "command": command,
            "exit_code": 0,
            "stdout": "1 passed\n",
            "stderr": "",
        },
    )

    assert (
        main(
            [
                "agent",
                "run",
                "--runtime",
                str(tmp_path / "runtime"),
                "--repo",
                str(repo),
                "--task",
                "Fix the failing answer implementation.",
                "--writes",
                "on",
                "--test-command",
                "pytest -q",
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert repo.joinpath("app.py").read_text() == "def answer():\n    return 42\n"
    assert result["steps"][2]["write_status"] == "applied"
    assert result["validation"]["status"] == "passed"


def test_agent_run_supports_dry_run_without_materializing_changes(tmp_path, monkeypatch, capsys):
    repo = _toy_repo(tmp_path)
    fake_runtime = FakeRuntime()

    monkeypatch.setattr("skillcortex.agent.SkillRuntime.load", lambda path: fake_runtime)

    assert (
        main(
            [
                "agent",
                "run",
                "--runtime",
                str(tmp_path / "runtime"),
                "--repo",
                str(repo),
                "--task",
                "Fix the failing answer implementation.",
                "--dry-run",
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert result["steps"][1]["status"] == "dry-run"
    assert result["steps"][2]["write_status"] == "dry-run"
    assert repo.joinpath("app.py").read_text() == "def answer():\n    return 0\n"