import json
from pathlib import Path

import pytest

from skillcortex.cli import main
from skillcortex.packaging import package_skill
from skillcortex.runtime.dynamic import DynamicRuntime, DynamicRouteDecision


def _skill(tmp_path, skill_id, *, description, capabilities=()):
    root = tmp_path / "skills" / skill_id
    eval_summary = tmp_path / f"{skill_id}-eval.json"
    eval_summary.write_text(json.dumps({"modes": {}, "tasks": {}}) + "\n")
    package_skill(
        skill_id=skill_id,
        name=skill_id.replace("_", " ").title(),
        adapter_dir=Path("artifacts/adapters/python_skill"),
        output=root,
        train_dataset=Path("data/train.jsonl"),
        eval_dataset=Path("data/eval.jsonl"),
        eval_summary=eval_summary,
        version="0.1.0",
        description=description,
        composition={
            "capabilities": {"allowed_task_types": ["python_generation"]},
            "activation": {
                "default_route_type": "adapter",
                "scope": "task",
                "semantic_families": list(capabilities),
            },
            "compatibility": {"compatible_skills": [], "incompatible_skills": []},
            "routing": {"tasks": {}},
        },
        force=True,
    )
    return root


def test_dynamic_infer_dry_run_selects_matching_lora(tmp_path, capsys):
    _skill(tmp_path, "fastapi_skill", description="FastAPI endpoint validation", capabilities=["fastapi"])
    _skill(tmp_path, "sql_skill", description="SQL query tuning", capabilities=["sql"])

    assert (
        main(
            [
                "infer",
                "--skills-dir",
                str(tmp_path / "skills"),
                "--prompt",
                "Fix a FastAPI validation bug",
                "--dry-run",
            ]
        )
        == 0
    )

    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "dry-run"
    assert output["selected_skills"] == ["fastapi_skill"]


def test_dynamic_infer_dry_run_falls_back_to_base(tmp_path, capsys):
    _skill(tmp_path, "sql_skill", description="SQL query tuning", capabilities=["sql"])

    assert (
        main(
            [
                "infer",
                "--skills-dir",
                str(tmp_path / "skills"),
                "--prompt",
                "Write a README",
                "--dry-run",
            ]
        )
        == 0
    )

    output = json.loads(capsys.readouterr().out)
    assert output["selected_skills"] == []
    assert output["reason"] == "base fallback"


def test_dynamic_router_rejects_unknown_skill(tmp_path):
    _skill(tmp_path, "fastapi_skill", description="FastAPI endpoint validation", capabilities=["fastapi"])
    runtime = DynamicRuntime.load(tmp_path / "skills")

    with pytest.raises(ValueError, match="unknown dynamic skill"):
        runtime.route(
            [{"role": "user", "content": "Fix FastAPI"}],
            router=lambda _messages, _skills: DynamicRouteDecision(
                base_model="mlx-test-base",
                selected_skills=["missing_skill"],
                remote_loras=[],
                task_type="python_generation",
                semantic_family=None,
                train_new_lora=False,
                reason="bad router",
            ),
        )


def test_dynamic_router_allows_unknown_skill_when_remote_lora_is_available(tmp_path, monkeypatch):
    _skill(tmp_path, "fastapi_skill", description="FastAPI endpoint validation", capabilities=["fastapi"])
    runtime = DynamicRuntime.load(tmp_path / "skills", allow_remote_loras=True)

    def fake_resolve(source, skill_id, name=None):
        return runtime.registry.local["fastapi_skill"]

    monkeypatch.setattr(runtime.registry, "resolve_remote", fake_resolve)
    decision = runtime.route(
        [{"role": "user", "content": "Fix FastAPI"}],
        router=lambda _messages, _skills: DynamicRouteDecision(
            base_model="mlx-test-base",
            selected_skills=["remote_skill"],
            remote_loras=["hf://owner/repo"],
            task_type="python_generation",
            semantic_family=None,
            train_new_lora=False,
            reason="remote router",
        ),
    )

    assert decision.selected_skills == ["fastapi_skill"]


def test_dynamic_runtime_cache_key_includes_base_model_and_loras(tmp_path, monkeypatch):
    _skill(tmp_path, "fastapi_skill", description="FastAPI endpoint validation", capabilities=["fastapi"])
    runtime = DynamicRuntime.load(tmp_path / "skills")
    calls = []

    def fake_load_model(adapter=None, model_name=None):
        calls.append((model_name, str(adapter) if adapter else None))
        return f"model:{model_name}", "tokenizer"

    monkeypatch.setattr("skillcortex.runtime.dynamic.load_model", fake_load_model)

    first = runtime._get_model("base-a", ("fastapi_skill",))
    second = runtime._get_model("base-b", ("fastapi_skill",))

    assert first[0] == "model:base-a"
    assert second[0] == "model:base-b"
    assert len(calls) == 2


def test_dynamic_router_malformed_json_falls_back_to_base(tmp_path, monkeypatch):
    _skill(tmp_path, "fastapi_skill", description="FastAPI endpoint validation", capabilities=["fastapi"])
    runtime = DynamicRuntime.load(tmp_path / "skills")

    monkeypatch.setattr("skillcortex.runtime.dynamic.load_model", lambda model_name=None, adapter=None: ("m", "t"))
    monkeypatch.setattr("skillcortex.runtime.dynamic.generate_text", lambda *args, **kwargs: ("not json", 0, 0))

    decision = runtime.route([{"role": "user", "content": "Fix FastAPI"}], router=runtime._router_model)

    assert decision.selected_skills == []
    assert decision.reason == "router fallback"
