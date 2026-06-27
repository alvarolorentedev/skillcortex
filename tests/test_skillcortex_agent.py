import json
from pathlib import Path

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


class FakeDiffRuntime(FakeRuntime):
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
                "selected_skills": [],
                "route_type": "base_fallback",
                "reason": f"runtime bundle route {task_type or 'python_generation'}.default selected task_type={task_type or 'python_generation'}",
            }
        if "Create a short execution plan" in prompt:
            return {
                "status": "complete",
                "selected_skills": [],
                "route_type": "base_fallback",
                "reason": f"runtime bundle route python_generation.default selected task_type={task_type}",
                "generation": "1. Inspect files\n2. Propose a patch\n3. Run validation",
            }
        return {
            "status": "complete",
            "selected_skills": ["python_skill"],
            "route_type": "adapter",
            "reason": f"runtime bundle route python_generation.default selected task_type={task_type}",
            "generation": json.dumps(
                {
                    "kind": "proposed_diff",
                    "summary": "Patch the implementation.",
                    "diff": "--- a/app.py\n+++ b/app.py\n@@ -1,2 +1,2 @@\n def answer():\n-    return 0\n+    return 42\n",
                }
            ),
        }


class FakeRawCodeRuntime(FakeRuntime):
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
                "selected_skills": [],
                "route_type": "base_fallback",
                "reason": f"runtime bundle route {task_type or 'python_generation'}.default selected task_type={task_type or 'python_generation'}",
            }
        if "Create a short execution plan" in prompt:
            return {
                "status": "complete",
                "selected_skills": [],
                "route_type": "base_fallback",
                "reason": f"runtime bundle route python_generation.default selected task_type={task_type}",
                "generation": "1. Inspect files\n2. Update app.py\n3. Run validation",
            }
        return {
            "status": "complete",
            "selected_skills": ["python_skill"],
            "route_type": "adapter",
            "reason": f"runtime bundle route python_generation.default selected task_type={task_type}",
            "generation": "def answer():\n    return 42\n",
        }


class FakeMultiActionRuntime(FakeRuntime):
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
                "selected_skills": [],
                "route_type": "base_fallback",
                "reason": f"runtime bundle route {task_type or 'python_generation'}.default selected task_type={task_type or 'python_generation'}",
            }
        if "Create a short execution plan" in prompt:
            return {
                "status": "complete",
                "selected_skills": [],
                "route_type": "base_fallback",
                "reason": f"runtime bundle route python_generation.default selected task_type={task_type}",
                "generation": "1. Add endpoint\n2. Add schema\n3. Run validation",
            }
        return {
            "status": "complete",
            "selected_skills": ["python_skill", "debugging_skill"],
            "route_type": "adapter",
            "reason": f"runtime bundle route python_generation.default selected task_type={task_type}",
            "generation": json.dumps(
                {
                    "actions": [
                        {
                            "kind": "file_replace",
                            "path": "app.py",
                            "summary": "Add FastAPI endpoint.",
                            "content": "from fastapi import FastAPI\nfrom schemas import UserCreate\n\napp = FastAPI()\n\n@app.post('/users')\ndef create_user(user: UserCreate):\n    return user.model_dump()\n",
                        },
                        {
                            "kind": "file_replace",
                            "path": "schemas.py",
                            "summary": "Add request schema.",
                            "content": "from pydantic import BaseModel, EmailStr\n\n\nclass UserCreate(BaseModel):\n    email: EmailStr\n    name: str\n",
                        },
                    ]
                }
            ),
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
    assert result["status"] == "validation_failed"
    assert result["validation"]["status"] == "failed"
    assert result["steps"][1]["selected_skills"] == []
    assert result["steps"][2]["write_status"] == "review_required"
    assert result["review_artifact_path"] is not None
    assert Path(result["review_artifact_path"]).exists()
    assert result["steps"][3]["status"] == "failed"
    assert result["steps"][4]["selected_skills"] == ["debugging_skill", "python_skill"]
    trace_payload = json.loads(trace.read_text())
    assert trace_payload["status"] == "validation_failed"
    assert trace_payload["generated_patch"]
    assert trace_payload["review_artifact_path"]
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
                "--write-mode",
                "on",
                "--test-command",
                "pytest -q",
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert repo.joinpath("app.py").read_text() == "def answer():\n    return 42\n"
    assert result["status"] == "applied"
    assert result["steps"][2]["write_status"] == "applied"
    assert result["validation"]["status"] == "passed"
    assert "actions' array" in fake_runtime.calls[1]["prompt"]


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
    assert result["status"] == "dry-run"
    assert result["execution_mode"] == "dry-run-route-plan-only"
    assert result["steps"][1]["status"] == "dry-run"
    assert result["steps"][1]["mode_label"] == "route/plan only"
    assert result["steps"][2]["write_status"] == "dry-run"
    assert result["validation"]["status"] == "skipped"
    assert repo.joinpath("app.py").read_text() == "def answer():\n    return 0\n"


def test_agent_run_dry_run_skips_generation_apply_and_validation(tmp_path, monkeypatch, capsys):
    repo = _toy_repo(tmp_path)
    fake_runtime = FakeRuntime()

    monkeypatch.setattr("skillcortex.agent.SkillRuntime.load", lambda path: fake_runtime)

    def fail_validation(command, repo):
        raise AssertionError("validation should not run in dry-run mode")

    monkeypatch.setattr("skillcortex.agent._run_validation_command", fail_validation)

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
                "--test-command",
                "pytest -q",
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert all(call["dry_run"] is True for call in fake_runtime.calls)
    assert len(fake_runtime.calls) == 2
    assert result["review_artifact_path"] is None
    assert result["validation"]["status"] == "skipped"
    assert "route/plan only" in result["final_summary"]
    assert repo.joinpath("app.py").read_text() == "def answer():\n    return 0\n"


def test_agent_run_confirm_mode_writes_review_artifact(tmp_path, monkeypatch, capsys):
    repo = _toy_repo(tmp_path)
    trace = tmp_path / "trace.json"
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
                "--trace-out",
                str(trace),
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    review_artifact = Path(result["review_artifact_path"])
    assert result["status"] == "review_required"
    assert review_artifact.exists()
    assert "+++ b/app.py" in review_artifact.read_text()
    trace_payload = json.loads(trace.read_text())
    assert trace_payload["review_artifact_path"] == str(review_artifact)
    assert trace_payload["generated_patch"] == result["last_proposed_diff"]
    assert trace_payload["steps"][2]["review_artifact_path"] == str(review_artifact)


def test_agent_run_can_apply_proposed_diff_when_writes_on(tmp_path, monkeypatch, capsys):
    repo = _toy_repo(tmp_path)
    fake_runtime = FakeDiffRuntime()

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
                "--write-mode",
                "on",
                "--test-command",
                "pytest -q",
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert repo.joinpath("app.py").read_text() == "def answer():\n    return 42\n"
    assert result["steps"][2]["selected_skills"] == ["python_skill"]
    assert result["steps"][2]["write_status"] == "applied"
    assert result["status"] == "applied"


def test_agent_run_can_apply_raw_code_generation_when_writes_on(tmp_path, monkeypatch, capsys):
    repo = _toy_repo(tmp_path)
    fake_runtime = FakeRawCodeRuntime()

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
                "--write-mode",
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
    assert result["status"] == "applied"


def test_agent_run_can_apply_multiple_explicit_actions_when_writes_on(tmp_path, monkeypatch, capsys):
    repo = _toy_repo(tmp_path)
    fake_runtime = FakeMultiActionRuntime()

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
                "Create a FastAPI endpoint for creating a user with Pydantic validation.",
                "--write-mode",
                "on",
                "--test-command",
                "pytest -q",
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert "actions' array" in fake_runtime.calls[1]["prompt"]
    assert repo.joinpath("app.py").read_text().startswith("from fastapi import FastAPI")
    assert repo.joinpath("schemas.py").read_text().startswith("from pydantic import BaseModel")
    assert result["status"] == "applied"
    assert result["generated_actions"][0]["path"] == "app.py"
    assert result["generated_actions"][1]["path"] == "schemas.py"
    assert set(result["steps"][2]["files_changed"]) == {"app.py", "schemas.py"}