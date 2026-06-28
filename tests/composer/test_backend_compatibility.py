import pytest

from slmcortex.composer.compatibility import build_compatibility_report


def _loaded(slm_id, backend, adapter_format, runtime_model):
    return {
        "slm_id": slm_id,
        "manifest": {
            "base": {
                "source_model": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
                "runtime_model": runtime_model,
                "quantization": "4bit",
                "backend": backend,
            }
        },
        "metadata": {
            "base": {
                "runtime_model": runtime_model,
                "quantization": "4bit",
                "backend": backend,
            },
            "adapter": {
                "format": adapter_format,
                "target_modules": ["q_proj"],
            },
        },
        "adapter_config": {
            "fine_tune_type": "lora",
            "num_layers": 1,
            "lora_parameters": {"scale": 20, "dropout": 0, "keys": ["q_proj"], "rank": 8},
        },
        "composition": {"compatibility": {"incompatible_slms": []}},
    }


def test_composition_rejects_mixed_backends():
    report = build_compatibility_report(
        [
            _loaded("mlx_slm", "mlx", "mlx-lora", "mlx-community/model"),
            _loaded("gguf_slm", "gguf", "gguf-lora", "models/base.gguf"),
        ],
        {"enabled": False, "matched_slms": []},
    )

    assert "incompatible package base backend" in report["errors"]


def test_composition_rejects_backend_adapter_format_mismatch():
    report = build_compatibility_report(
        [_loaded("bad_slm", "gguf", "mlx-lora", "models/base.gguf")],
        {"enabled": False, "matched_slms": []},
    )

    assert "adapter format mlx-lora is incompatible with backend gguf" in report["errors"]
