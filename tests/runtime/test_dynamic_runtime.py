import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from slmcortex.cli import main
from slmcortex.packaging import package_skill
from slmcortex.runtime.dynamic import DynamicRuntime, DynamicRouteDecision


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


def test_dynamic_infer_dry_run_selects_remote_catalog_match(tmp_path, monkeypatch, capsys):
    calls = []

    monkeypatch.setattr(
        "slmcortex.runtime.dynamic.base_config",
        lambda: {
            "model": "mlx-test-base",
            "default_runtime_model": "mlx-test-base",
            "remote_lora_catalog": [
                {"skill_id": "fastapi_remote", "source": "hf://owner/fastapi", "cues": ["fastapi"]}
            ],
        },
    )
    runtime = DynamicRuntime.load(tmp_path / "skills", allow_remote_loras=True)

    def fake_resolve(source, skill_id, name=None):
        calls.append((source, skill_id, name))
        _skill(tmp_path, skill_id, description="FastAPI remote adapter", capabilities=["fastapi"])
        runtime.registry.reload()
        return runtime.registry.local[skill_id]

    monkeypatch.setattr(runtime.registry, "resolve_remote", fake_resolve)
    monkeypatch.setattr("slmcortex.runtime.dynamic.DynamicRuntime.load", lambda *args, **kwargs: runtime)

    assert (
        main(
            [
                "infer",
                "--skills-dir",
                str(tmp_path / "skills"),
                "--prompt",
                "Fix a FastAPI validation bug",
                "--allow-remote-loras",
                "--dry-run",
            ]
        )
        == 0
    )

    output = json.loads(capsys.readouterr().out)
    assert calls == [("hf://owner/fastapi", "fastapi_remote", None)]
    assert output["selected_skills"] == ["fastapi_remote"]
    assert output["remote_loras"] == ["hf://owner/fastapi"]
    assert output["route_branch"] == "remote_lora"
    assert output["route_trace"]["final_selected_skills"] == ["fastapi_remote"]
    assert output["adaptation_summary"]["branch"] == "remote_lora"
    assert output["adaptation_summary"]["fetched_sources"] == ["hf://owner/fastapi"]


def test_dynamic_infer_dry_run_matches_richer_remote_catalog_fields(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        "slmcortex.runtime.dynamic.base_config",
        lambda: {
            "model": "mlx-test-base",
            "default_runtime_model": "mlx-test-base",
            "remote_lora_catalog": [
                {
                    "skill_id": "fastapi_remote",
                    "source": "hf://owner/fastapi",
                    "name": "FastAPI Contract Remote",
                    "description": "Pydantic response model validation",
                    "task_types": ["python_generation"],
                    "semantic_families": ["fastapi_contract"],
                }
            ],
        },
    )
    runtime = DynamicRuntime.load(tmp_path / "skills", allow_remote_loras=True)

    def fake_resolve(source, skill_id, name=None):
        _skill(tmp_path, skill_id, description="FastAPI remote adapter", capabilities=["fastapi"])
        runtime.registry.reload()
        return runtime.registry.local[skill_id]

    monkeypatch.setattr(runtime.registry, "resolve_remote", fake_resolve)
    monkeypatch.setattr("slmcortex.runtime.dynamic.DynamicRuntime.load", lambda *args, **kwargs: runtime)

    assert main([
        "infer",
        "--skills-dir",
        str(tmp_path / "skills"),
        "--prompt",
        "Fix Pydantic response model validation",
        "--allow-remote-loras",
        "--dry-run",
    ]) == 0

    output = json.loads(capsys.readouterr().out)
    assert output["selected_skills"] == ["fastapi_remote"]
    assert output["route_branch"] == "remote_lora"


def test_dynamic_infer_dry_run_selects_fetched_remote_catalog_match(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        "slmcortex.runtime.dynamic.base_config",
        lambda: {
            "model": "mlx-test-base",
            "default_runtime_model": "mlx-test-base",
            "remote_lora_catalog_url": "https://example.invalid/catalog.json",
        },
    )
    monkeypatch.setattr(
        "slmcortex.runtime.dynamic._fetch_remote_lora_catalog",
        lambda url: [
            {
                "skill_id": "fastapi_remote",
                "source": "hf://owner/fastapi",
                "name": "FastAPI Remote",
                "description": "Pydantic response model validation",
                "semantic_families": ["fastapi_contract"],
            }
        ],
    )
    runtime = DynamicRuntime.load(tmp_path / "skills", allow_remote_loras=True)

    def fake_resolve(source, skill_id, name=None):
        _skill(tmp_path, skill_id, description="FastAPI remote adapter", capabilities=["fastapi"])
        runtime.registry.reload()
        return runtime.registry.local[skill_id]

    monkeypatch.setattr(runtime.registry, "resolve_remote", fake_resolve)
    monkeypatch.setattr("slmcortex.runtime.dynamic.DynamicRuntime.load", lambda *args, **kwargs: runtime)

    assert main([
        "infer",
        "--skills-dir",
        str(tmp_path / "skills"),
        "--prompt",
        "Fix Pydantic response model validation",
        "--allow-remote-loras",
        "--dry-run",
    ]) == 0

    output = json.loads(capsys.readouterr().out)
    assert output["selected_skills"] == ["fastapi_remote"]
    assert output["remote_loras"] == ["hf://owner/fastapi"]
    assert output["route_branch"] == "remote_lora"


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


def test_dynamic_router_rejects_training_when_disabled(tmp_path, monkeypatch):
    runtime = DynamicRuntime.load(tmp_path / "skills")
    monkeypatch.setattr(
        "slmcortex.runtime.dynamic.base_config",
        lambda: {
            "model": "mlx-test-base",
            "default_runtime_model": "mlx-test-base",
            "training_enabled": False,
        },
    )

    with pytest.raises(ValueError, match="dynamic plasticity training is disabled"):
        runtime.route(
            [{"role": "user", "content": "Fix FastAPI"}],
            router=lambda _messages, _skills: DynamicRouteDecision(
                base_model="mlx-test-base",
                selected_skills=[],
                remote_loras=[],
                task_type="python_generation",
                semantic_family=None,
                train_new_lora=True,
                reason="needs training",
            ),
        )


def test_dynamic_router_rejects_ambiguous_train_and_remote(tmp_path, monkeypatch):
    runtime = DynamicRuntime.load(tmp_path / "skills")
    monkeypatch.setattr(
        "slmcortex.runtime.dynamic.base_config",
        lambda: {
            "model": "mlx-test-base",
            "default_runtime_model": "mlx-test-base",
            "training_enabled": True,
            "plasticity_train_dataset": "data/train.jsonl",
            "plasticity_publish_dir": str(tmp_path / "skills"),
        },
    )

    with pytest.raises(ValueError, match="ambiguous dynamic route"):
        runtime.route(
            [{"role": "user", "content": "Fix FastAPI"}],
            router=lambda _messages, _skills: DynamicRouteDecision(
                base_model="mlx-test-base",
                selected_skills=["remote_skill"],
                remote_loras=["hf://owner/repo"],
                task_type="python_generation",
                semantic_family=None,
                train_new_lora=True,
                reason="bad router",
            ),
        )


def test_dynamic_router_trains_plasticity_lora_when_enabled(tmp_path, monkeypatch):
    runtime = DynamicRuntime.load(tmp_path / "skills")
    prompt = "Fix FastAPI validation"
    expected_skill = "plasticity_" + hashlib.sha256(prompt.encode()).hexdigest()[:8]
    calls = []

    monkeypatch.setattr(
        "slmcortex.runtime.dynamic.base_config",
        lambda: {
            "model": "mlx-test-base",
            "default_runtime_model": "mlx-test-base",
            "training_enabled": True,
            "plasticity_train_dataset": "data/train.jsonl",
            "plasticity_eval_dataset": "data/eval.jsonl",
            "plasticity_publish_dir": str(tmp_path / "skills"),
        },
    )

    def fake_train_skill_package(**kwargs):
        calls.append(kwargs)
        calls.append({"rows": [json.loads(line) for line in kwargs["train_dataset"].read_text().splitlines()]})
        eval_summary = tmp_path / "plasticity-eval.json"
        eval_summary.write_text(json.dumps({"modes": {}, "tasks": {}}) + "\n")
        package_skill(
            skill_id=kwargs["skill"],
            name=kwargs["name"],
            adapter_dir=Path("artifacts/adapters/python_skill"),
            output=kwargs["output"],
            train_dataset=kwargs["train_dataset"],
            eval_dataset=kwargs["eval_dataset"],
            eval_summary=eval_summary,
            version=kwargs["version"],
            description=kwargs["description"],
            composition=kwargs["composition"],
            force=True,
        )
        return {"status": "complete", "skill_id": kwargs["skill"]}

    monkeypatch.setattr("slmcortex.runtime.dynamic.train_skill_package", fake_train_skill_package)

    decision = runtime.route(
        [{"role": "user", "content": prompt}],
        router=lambda _messages, _skills: DynamicRouteDecision(
            base_model="mlx-test-base",
            selected_skills=[],
            remote_loras=[],
            task_type="python_generation",
            semantic_family="fastapi",
            train_new_lora=True,
            reason="needs training",
        ),
    )

    assert decision.selected_skills == [expected_skill]
    assert calls[0]["skill"] == expected_skill
    assert calls[0]["mode"] == "generic"
    assert calls[1]["rows"] == [
        {
            "id": expected_skill,
            "task_type": "python_generation",
            "prompt": prompt,
            "target": "Adapt to this task.",
            "semantic_family": "fastapi",
            "metadata": {"source": "dynamic_plasticity"},
        }
    ]
    assert (tmp_path / "skills" / expected_skill / "skill.yaml").exists()


def test_dynamic_router_uses_live_source_for_plasticity_training(tmp_path, monkeypatch):
    runtime = DynamicRuntime.load(tmp_path / "skills")
    prompt = ""
    expected_skill = "plasticity_" + hashlib.sha256(prompt.encode()).hexdigest()[:8]
    calls = []

    def live_source_handler(**kwargs):
        assert kwargs["skill_id"] == expected_skill
        assert kwargs["decision"].semantic_family == "fastapi"
        return [
            {
                "id": expected_skill,
                "task_type": "python_generation",
                "prompt": "Live FastAPI trace",
                "target": "Adapt to live signal.",
                "semantic_family": "fastapi",
                "metadata": {"source": "live_source"},
            }
        ]

    monkeypatch.setattr(
        "slmcortex.runtime.dynamic.base_config",
        lambda: {
            "model": "mlx-test-base",
            "default_runtime_model": "mlx-test-base",
            "training_enabled": True,
            "plasticity_live_source_handler": live_source_handler,
            "plasticity_eval_dataset": "data/eval.jsonl",
            "plasticity_publish_dir": str(tmp_path / "skills"),
        },
    )

    def fake_train_skill_package(**kwargs):
        calls.append(kwargs)
        calls.append({"rows": [json.loads(line) for line in kwargs["train_dataset"].read_text().splitlines()]})
        eval_summary = tmp_path / "plasticity-live-eval.json"
        eval_summary.write_text(json.dumps({"modes": {}, "tasks": {}}) + "\n")
        package_skill(
            skill_id=kwargs["skill"],
            name=kwargs["name"],
            adapter_dir=Path("artifacts/adapters/python_skill"),
            output=kwargs["output"],
            train_dataset=kwargs["train_dataset"],
            eval_dataset=kwargs["eval_dataset"],
            eval_summary=eval_summary,
            version=kwargs["version"],
            description=kwargs["description"],
            composition=kwargs["composition"],
            force=True,
        )
        return {"status": "complete", "skill_id": kwargs["skill"]}

    monkeypatch.setattr("slmcortex.runtime.dynamic.train_skill_package", fake_train_skill_package)

    decision = runtime.route(
        [{"role": "user", "content": prompt}],
        router=lambda _messages, _skills: DynamicRouteDecision(
            base_model="mlx-test-base",
            selected_skills=[],
            remote_loras=[],
            task_type="python_generation",
            semantic_family="fastapi",
            train_new_lora=True,
            reason="needs training",
        ),
    )

    assert decision.selected_skills == [expected_skill]
    assert calls[0]["skill"] == expected_skill
    assert calls[1]["rows"] == [
        {
            "id": expected_skill,
            "task_type": "python_generation",
            "prompt": "Live FastAPI trace",
            "target": "Adapt to live signal.",
            "semantic_family": "fastapi",
            "metadata": {"source": "live_source"},
        }
    ]
    assert (tmp_path / "skills" / expected_skill / "skill.yaml").exists()


def test_dynamic_router_reuses_existing_plasticity_lora(tmp_path, monkeypatch):
    prompt = "Fix FastAPI validation"
    expected_skill = "plasticity_" + hashlib.sha256(prompt.encode()).hexdigest()[:8]
    _skill(tmp_path, expected_skill, description="Existing plasticity adapter")
    runtime = DynamicRuntime.load(tmp_path / "skills")
    monkeypatch.setattr(
        "slmcortex.runtime.dynamic.base_config",
        lambda: {
            "model": "mlx-test-base",
            "default_runtime_model": "mlx-test-base",
            "training_enabled": True,
            "plasticity_train_dataset": "data/train.jsonl",
            "plasticity_publish_dir": str(tmp_path / "skills"),
        },
    )
    monkeypatch.setattr(
        "slmcortex.runtime.dynamic.train_skill_package",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("should reuse existing skill")),
    )

    decision = runtime.route(
        [{"role": "user", "content": prompt}],
        router=lambda _messages, _skills: DynamicRouteDecision(
            base_model="mlx-test-base",
            selected_skills=[],
            remote_loras=[],
            task_type="python_generation",
            semantic_family=None,
            train_new_lora=True,
            reason="needs training",
        ),
    )

    assert decision.selected_skills == [expected_skill]


def test_dynamic_router_does_not_publish_invalid_plasticity_lora(tmp_path, monkeypatch):
    runtime = DynamicRuntime.load(tmp_path / "skills")
    prompt = "Fix FastAPI validation"
    expected_skill = "plasticity_" + hashlib.sha256(prompt.encode()).hexdigest()[:8]
    monkeypatch.setattr(
        "slmcortex.runtime.dynamic.base_config",
        lambda: {
            "model": "mlx-test-base",
            "default_runtime_model": "mlx-test-base",
            "training_enabled": True,
            "plasticity_train_dataset": "data/train.jsonl",
            "plasticity_publish_dir": str(tmp_path / "skills"),
        },
    )

    def fake_train_skill_package(**kwargs):
        kwargs["output"].mkdir(parents=True)
        (kwargs["output"] / "skill.yaml").write_text("skill_id: broken\n")
        return {"status": "complete", "skill_id": kwargs["skill"]}

    monkeypatch.setattr("slmcortex.runtime.dynamic.train_skill_package", fake_train_skill_package)
    monkeypatch.setattr(
        "slmcortex.runtime.dynamic.validate_skill_package",
        lambda path: (_ for _ in ()).throw(ValueError("invalid package")),
    )

    with pytest.raises(ValueError, match="invalid package"):
        runtime.route(
            [{"role": "user", "content": prompt}],
            router=lambda _messages, _skills: DynamicRouteDecision(
                base_model="mlx-test-base",
                selected_skills=[],
                remote_loras=[],
                task_type="python_generation",
                semantic_family=None,
                train_new_lora=True,
                reason="needs training",
            ),
        )

    assert not (tmp_path / "skills" / expected_skill).exists()


def test_dynamic_router_respects_plasticity_skill_cap(tmp_path, monkeypatch):
    _skill(tmp_path, "plasticity_existing", description="Existing plasticity adapter")
    runtime = DynamicRuntime.load(tmp_path / "skills")
    monkeypatch.setattr(
        "slmcortex.runtime.dynamic.base_config",
        lambda: {
            "model": "mlx-test-base",
            "default_runtime_model": "mlx-test-base",
            "training_enabled": True,
            "plasticity_train_dataset": "data/train.jsonl",
            "plasticity_publish_dir": str(tmp_path / "skills"),
            "max_plasticity_loras": 1,
        },
    )
    monkeypatch.setattr(
        "slmcortex.runtime.dynamic.train_skill_package",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("should refuse before training")),
    )

    with pytest.raises(ValueError, match="plasticity skill cap reached"):
        runtime.route(
            [{"role": "user", "content": "A new unmatched task"}],
            router=lambda _messages, _skills: DynamicRouteDecision(
                base_model="mlx-test-base",
                selected_skills=[],
                remote_loras=[],
                task_type="python_generation",
                semantic_family=None,
                train_new_lora=True,
                reason="needs training",
            ),
        )


def test_dynamic_infer_falls_back_to_base_when_adaptation_fails(tmp_path, monkeypatch):
    runtime = DynamicRuntime.load(tmp_path / "skills", allow_remote_loras=True)
    runtime._router_model = lambda _messages, _skills: DynamicRouteDecision(
        base_model="mlx-test-base",
        selected_skills=["remote_skill"],
        remote_loras=["hf://owner/repo"],
        task_type="python_generation",
        semantic_family=None,
        train_new_lora=False,
        reason="remote router",
    )
    monkeypatch.setattr(
        runtime.registry,
        "resolve_remote",
        lambda source, skill_id, name=None: (_ for _ in ()).throw(ValueError("fetch failed")),
    )
    monkeypatch.setattr("slmcortex.runtime.dynamic.load_model", lambda model_name=None, adapter=None: ("m", "t"))
    monkeypatch.setattr("slmcortex.runtime.dynamic.generate_text", lambda *args, **kwargs: ("base answer", 1, 2))

    result = runtime.infer(prompt="Fix FastAPI")

    assert result["status"] == "complete"
    assert result["selected_skills"] == []
    assert result["route_branch"] == "base_fallback"
    assert result["adaptation_error"] == "fetch failed"
    assert result["adaptation_summary"]["fallback_error"] == "fetch failed"
    assert result["generation"] == "base answer"


def test_dynamic_router_gguf_multi_adapter_falls_back_to_first_skill(tmp_path):
    runtime = DynamicRuntime.load(tmp_path / "skills")
    runtime.skills = {
        "debugging_skill": SimpleNamespace(skill_id="debugging_skill", adapter_format="gguf-lora"),
        "python_skill": SimpleNamespace(skill_id="python_skill", adapter_format="gguf-lora"),
    }

    decision = runtime.route(
        [{"role": "user", "content": "Fix FastAPI"}],
        router=lambda _messages, _skills: DynamicRouteDecision(
            base_model="model.gguf",
            selected_skills=["debugging_skill", "python_skill"],
            remote_loras=[],
            task_type="python_generation",
            semantic_family=None,
            train_new_lora=False,
            reason="matched both",
        ),
    )

    assert decision.selected_skills == ["debugging_skill"]
    assert "gguf single-adapter fallback selected debugging_skill" in decision.reason


def test_dynamic_infer_trains_reloads_and_reuses_plasticity_lora(tmp_path, monkeypatch):
    runtime = DynamicRuntime.load(tmp_path / "skills")
    prompt = "Fix FastAPI validation"
    expected_skill = "plasticity_" + hashlib.sha256(prompt.encode()).hexdigest()[:8]
    calls = []
    loaded = []
    monkeypatch.setattr(
        "slmcortex.runtime.dynamic.base_config",
        lambda: {
            "model": "mlx-test-base",
            "default_runtime_model": "mlx-test-base",
            "training_enabled": True,
            "plasticity_train_dataset": "data/train.jsonl",
            "plasticity_eval_dataset": "data/eval.jsonl",
            "plasticity_publish_dir": str(tmp_path / "skills"),
            "max_plasticity_loras": 8,
        },
    )

    def fake_train_skill_package(**kwargs):
        calls.append(kwargs)
        eval_summary = tmp_path / "plasticity-loop-eval.json"
        eval_summary.write_text(json.dumps({"modes": {}, "tasks": {}}) + "\n")
        package_skill(
            skill_id=kwargs["skill"],
            name=kwargs["name"],
            adapter_dir=Path("artifacts/adapters/python_skill"),
            output=kwargs["output"],
            train_dataset=kwargs["train_dataset"],
            eval_dataset=kwargs["eval_dataset"],
            eval_summary=eval_summary,
            version=kwargs["version"],
            description=kwargs["description"],
            composition=kwargs["composition"],
            force=True,
        )
        return {"status": "complete", "skill_id": kwargs["skill"]}

    runtime._router_model = lambda _messages, _skills: DynamicRouteDecision(
        base_model="mlx-test-base",
        selected_skills=[],
        remote_loras=[],
        task_type="python_generation",
        semantic_family="fastapi",
        train_new_lora=True,
        reason="needs training",
    )
    monkeypatch.setattr("slmcortex.runtime.dynamic.train_skill_package", fake_train_skill_package)
    monkeypatch.setattr(
        "slmcortex.runtime.dynamic.load_model",
        lambda model_name=None, adapter=None: (loaded.append(adapter) or "m", "t"),
    )
    monkeypatch.setattr("slmcortex.runtime.dynamic.generate_text", lambda *args, **kwargs: ("adapter answer", 1, 2))

    first = runtime.infer(prompt=prompt)
    second = runtime.infer(prompt=prompt)

    assert first["generation"] == "adapter answer"
    assert first["route_branch"] == "plasticity_train"
    assert first["selected_skills"] == [expected_skill]
    assert first["adaptation_summary"]["trained_skill"] == expected_skill
    assert second["selected_skills"] == [expected_skill]
    assert len(calls) == 1
    assert len(loaded) == 1
    assert str(loaded[0]).endswith("adapter")


def test_dynamic_acceptance_flow_local_remote_and_plasticity(tmp_path, monkeypatch):
    _skill(tmp_path, "fastapi_skill", description="FastAPI endpoint validation", capabilities=["fastapi"])
    runtime = DynamicRuntime.load(tmp_path / "skills", allow_remote_loras=True)
    train_calls = []
    remote_calls = []
    loaded = []
    monkeypatch.setattr(
        "slmcortex.runtime.dynamic.base_config",
        lambda: {
            "model": "mlx-test-base",
            "default_runtime_model": "mlx-test-base",
            "training_enabled": True,
            "plasticity_train_dataset": "data/train.jsonl",
            "plasticity_eval_dataset": "data/eval.jsonl",
            "plasticity_publish_dir": str(tmp_path / "skills"),
            "max_plasticity_loras": 8,
            "remote_lora_catalog": [
                {"skill_id": "sql_remote", "source": "hf://owner/sql", "cues": ["sql"]}
            ],
        },
    )

    def fake_resolve(source, skill_id, name=None):
        remote_calls.append((source, skill_id))
        _skill(tmp_path, skill_id, description="SQL remote adapter", capabilities=["sql"])
        runtime.registry.reload()
        return runtime.registry.local[skill_id]

    def fake_train_skill_package(**kwargs):
        train_calls.append(kwargs["skill"])
        eval_summary = tmp_path / "acceptance-eval.json"
        eval_summary.write_text(json.dumps({"modes": {}, "tasks": {}}) + "\n")
        package_skill(
            skill_id=kwargs["skill"],
            name=kwargs["name"],
            adapter_dir=Path("artifacts/adapters/python_skill"),
            output=kwargs["output"],
            train_dataset=kwargs["train_dataset"],
            eval_dataset=kwargs["eval_dataset"],
            eval_summary=eval_summary,
            version=kwargs["version"],
            description=kwargs["description"],
            composition=kwargs["composition"],
            force=True,
        )
        return {"status": "complete", "skill_id": kwargs["skill"]}

    def router(messages, skills):
        text = messages[-1]["content"]
        if "train" in text:
            return DynamicRouteDecision(
                base_model="mlx-test-base",
                selected_skills=[],
                remote_loras=[],
                task_type="python_generation",
                semantic_family="custom",
                train_new_lora=True,
                reason="needs training",
            )
        return runtime._rule_router(messages, skills)

    runtime._router_model = router
    monkeypatch.setattr(runtime.registry, "resolve_remote", fake_resolve)
    monkeypatch.setattr("slmcortex.runtime.dynamic.train_skill_package", fake_train_skill_package)
    monkeypatch.setattr(
        "slmcortex.runtime.dynamic.load_model",
        lambda model_name=None, adapter=None: (loaded.append(adapter) or "m", "t"),
    )
    monkeypatch.setattr("slmcortex.runtime.dynamic.generate_text", lambda *args, **kwargs: ("answer", 1, 2))

    local = runtime.infer(prompt="Fix a FastAPI validation bug")
    remote = runtime.infer(prompt="Tune a SQL query")
    trained = runtime.infer(prompt="train custom adapter")

    assert local["route_branch"] == "local_lora"
    assert local["selected_skills"] == ["fastapi_skill"]
    assert remote["route_branch"] == "remote_lora"
    assert remote["selected_skills"] == ["sql_remote"]
    assert remote_calls == [("hf://owner/sql", "sql_remote")]
    assert trained["route_branch"] == "plasticity_train"
    assert trained["selected_skills"] == train_calls
    assert len(loaded) == 3


def test_dynamic_runtime_cache_key_includes_base_model_and_loras(tmp_path, monkeypatch):
    _skill(tmp_path, "fastapi_skill", description="FastAPI endpoint validation", capabilities=["fastapi"])
    runtime = DynamicRuntime.load(tmp_path / "skills")
    calls = []

    def fake_load_model(adapter=None, model_name=None):
        calls.append((model_name, str(adapter) if adapter else None))
        return f"model:{model_name}", "tokenizer"

    monkeypatch.setattr("slmcortex.runtime.dynamic.load_model", fake_load_model)

    first = runtime._get_model("base-a", ("fastapi_skill",))
    second = runtime._get_model("base-b", ("fastapi_skill",))

    assert first[0] == "model:base-a"
    assert second[0] == "model:base-b"
    assert len(calls) == 2


def test_dynamic_router_malformed_json_falls_back_to_base(tmp_path, monkeypatch):
    _skill(tmp_path, "fastapi_skill", description="FastAPI endpoint validation", capabilities=["fastapi"])
    runtime = DynamicRuntime.load(tmp_path / "skills")

    monkeypatch.setattr("slmcortex.runtime.dynamic.load_model", lambda model_name=None, adapter=None: ("m", "t"))
    monkeypatch.setattr("slmcortex.runtime.dynamic.generate_text", lambda *args, **kwargs: ("not json", 0, 0))

    decision = runtime.route([{"role": "user", "content": "Fix FastAPI"}], router=runtime._router_model)

    assert decision.selected_skills == []
    assert decision.reason == "router fallback"
