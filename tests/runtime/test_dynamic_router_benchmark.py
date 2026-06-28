import json
from pathlib import Path

from scripts.benchmark_dynamic_router import run_benchmark
from slmcortex.packaging import package_skill


def _skill(tmp_path, skill_id, description, capabilities=()):
    eval_summary = tmp_path / f"{skill_id}-eval.json"
    eval_summary.write_text(json.dumps({"modes": {}, "tasks": {}}) + "\n")
    package_skill(
        skill_id=skill_id,
        name=skill_id.replace("_", " ").title(),
        adapter_dir=Path("artifacts/adapters/python_skill"),
        output=tmp_path / "skills" / skill_id,
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


def test_dynamic_router_benchmark_counts_expected_branches(tmp_path, monkeypatch):
    _skill(tmp_path, "fastapi_skill", "FastAPI endpoint validation", ["fastapi"])
    monkeypatch.setattr(
        "slmcortex.runtime.dynamic.base_config",
        lambda: {
            "model": "mlx-test-base",
            "default_runtime_model": "mlx-test-base",
            "remote_lora_catalog": [
                {
                    "skill_id": "code_remote",
                    "source": "hf://owner/code",
                    "description": "Code implementation and refactor",
                }
            ],
        },
    )

    result = run_benchmark(tmp_path / "skills")

    assert result["total"] == 8
    assert result["accuracy"] >= 0.75
    assert result["expected"] == {
        "base_fallback": 2,
        "local_lora": 2,
        "plasticity_train": 2,
        "remote_lora": 2,
    }
