import json
from pathlib import Path

from slmcortex.packaging import package_slm
from slmcortex.runtime.registry import AdapterRegistry
from slmcortex.shared.hashing import sha256
from slmcortex.shared.io import read_json, read_yaml


def _slm(tmp_path, slm_id):
    root = tmp_path / slm_id
    eval_summary = tmp_path / f"{slm_id}-eval.json"
    eval_summary.write_text(json.dumps({"modes": {}, "tasks": {}}) + "\n")
    package_slm(
        slm_id=slm_id,
        name=slm_id,
        adapter_dir=Path("artifacts/adapters/python_slm"),
        output=root,
        train_dataset=Path("data/train.jsonl"),
        eval_dataset=Path("data/eval.jsonl"),
        eval_summary=eval_summary,
        version="0.1.0",
        description=slm_id,
        composition={
            "capabilities": {"allowed_task_types": ["python_generation"]},
            "activation": {"default_route_type": "adapter", "scope": "task", "semantic_families": []},
            "compatibility": {"compatible_slms": [], "incompatible_slms": []},
            "routing": {"tasks": {}},
        },
        force=True,
    )
    return root


def _rewrite_backend(package, backend, adapter_format):
    slm_yaml = read_yaml(package / "slm.yaml")
    metadata = read_json(package / "metadata.json")
    slm_yaml["base"]["backend"] = backend
    metadata["base"]["backend"] = backend
    slm_yaml["adapter"]["format"] = adapter_format
    metadata["adapter"]["format"] = adapter_format
    (package / "slm.yaml").write_text(__import__("yaml").safe_dump(slm_yaml, sort_keys=False))
    metadata["checksums"] = {
        path.relative_to(package).as_posix(): sha256(path)
        for path in sorted(package.rglob("*"))
        if path.is_file() and path.name != "metadata.json"
    }
    (package / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")


def test_registry_skips_packages_for_other_backend(tmp_path, monkeypatch):
    mlx = _slm(tmp_path, "mlx_slm")
    gguf = _slm(tmp_path, "gguf_slm")
    _rewrite_backend(mlx, "mlx", "mlx-lora")
    _rewrite_backend(gguf, "gguf", "gguf-lora")
    monkeypatch.setattr("slmcortex.runtime.registry.base_config", lambda: {"backend": "gguf"})

    registry = AdapterRegistry.load(tmp_path)

    assert sorted(registry.local) == ["gguf_slm"]
