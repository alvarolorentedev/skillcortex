import json
from pathlib import Path

import pytest

from slmcortex.packaging import package_skill
from slmcortex.runtime.registry import AdapterRegistry


def _package(tmp_path, skill_id):
    root = tmp_path / "skills" / skill_id
    eval_summary = tmp_path / f"{skill_id}-eval.json"
    eval_summary.write_text(json.dumps({"modes": {}, "tasks": {}}) + "\n")
    package_skill(
        skill_id=skill_id,
        name=skill_id,
        adapter_dir=Path("artifacts/adapters/python_skill"),
        output=root,
        train_dataset=Path("data/train.jsonl"),
        eval_dataset=Path("data/eval.jsonl"),
        eval_summary=eval_summary,
        version="0.1.0",
        composition={
            "capabilities": {"allowed_task_types": ["python_generation"]},
            "activation": {"default_route_type": "adapter", "scope": "task", "semantic_families": []},
            "compatibility": {"compatible_skills": [], "incompatible_skills": []},
            "routing": {"tasks": {}},
        },
        force=True,
    )
    return root


def test_registry_discovers_valid_local_package_without_network(tmp_path, monkeypatch):
    root = _package(tmp_path, "local_skill")
    monkeypatch.setattr("slmcortex.runtime.registry.import_lora", lambda **kwargs: pytest.fail("network import should not run"))

    registry = AdapterRegistry.load(tmp_path / "skills")

    assert registry.local["local_skill"].package_path == root.resolve()


def test_registry_skips_invalid_local_package(tmp_path):
    broken = tmp_path / "skills" / "broken"
    broken.mkdir(parents=True)
    (broken / "skill.yaml").write_text("skill_id: broken\n")

    registry = AdapterRegistry.load(tmp_path / "skills")

    assert "broken" not in registry.local


def test_registry_resolves_remote_lora_when_allowed(tmp_path, monkeypatch):
    imported = _package(tmp_path, "remote_skill")

    def fake_import_lora(**kwargs):
        return {"status": "complete", "output": str(imported), "skill_id": kwargs["skill_id"]}

    monkeypatch.setattr("slmcortex.runtime.registry.import_lora", fake_import_lora)
    registry = AdapterRegistry.load(tmp_path / "skills", allow_remote=True)

    resolved = registry.resolve_remote("hf://owner/repo", "remote_skill")

    assert resolved.skill_id == "remote_skill"
    assert resolved.package_path == imported.resolve()


def test_registry_uses_configured_remote_import_datasets(tmp_path, monkeypatch):
    imported = _package(tmp_path, "remote_skill")
    calls = []
    monkeypatch.setattr(
        "slmcortex.runtime.registry.base_config",
        lambda: {
            "remote_lora_train_dataset": "data/import-train.jsonl",
            "remote_lora_eval_dataset": "data/import-eval.jsonl",
        },
    )

    def fake_import_lora(**kwargs):
        calls.append(kwargs)
        return {"status": "complete", "output": str(imported), "skill_id": kwargs["skill_id"]}

    monkeypatch.setattr("slmcortex.runtime.registry.import_lora", fake_import_lora)
    registry = AdapterRegistry.load(tmp_path / "skills", allow_remote=True)

    registry.resolve_remote("hf://owner/repo", "remote_skill")

    assert calls[0]["train_dataset"] == Path("data/import-train.jsonl")
    assert calls[0]["eval_dataset"] == Path("data/import-eval.jsonl")


def test_registry_blocks_remote_lora_by_default(tmp_path):
    registry = AdapterRegistry.load(tmp_path / "skills")

    with pytest.raises(ValueError, match="remote LoRA downloads are disabled"):
        registry.resolve_remote("hf://owner/repo", "remote_skill")
