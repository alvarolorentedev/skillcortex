import json

import mlx.core as mx
import numpy as np
import pytest

from skill_lattice_coder.compose import (
    compose_adapter_directories,
    compose_lora_arrays,
    validate_adapter_metadata,
)


def test_rank_concatenation_equals_weighted_delta_sum():
    a1 = np.array([[1.0, 2.0], [3.0, 4.0]])
    b1 = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])
    a2 = np.array([[2.0, 0.0], [0.0, 2.0]])
    b2 = np.array([[1.0, 1.0, 0.0], [1.0, 0.0, 1.0]])
    a, b = compose_lora_arrays([(a1, b1), (a2, b2)], [0.25, 0.75])
    expected = 0.25 * (a1 @ b1) + 0.75 * (a2 @ b2)
    assert np.allclose(a @ b, expected)


def test_metadata_must_match():
    first = {"base_model": "base", "target_modules": ["q_proj"], "quantization": "4bit"}
    validate_adapter_metadata([first, dict(first)])
    with pytest.raises(ValueError, match="base_model"):
        validate_adapter_metadata([first, {**first, "base_model": "other"}])


def test_composes_real_mlx_adapter_directories(tmp_path):
    metadata = {
        "base_model": "base",
        "target_modules": ["q_proj"],
        "quantization": "4bit",
    }
    config = {
        "fine_tune_type": "lora",
        "num_layers": 1,
        "lora_parameters": {"rank": 1, "scale": 20.0, "dropout": 0.0},
    }
    paths = []
    for index, value in enumerate((1.0, 2.0)):
        path = tmp_path / f"adapter-{index}"
        path.mkdir()
        (path / "metadata.json").write_text(json.dumps(metadata))
        (path / "adapter_config.json").write_text(json.dumps(config))
        mx.save_safetensors(
            str(path / "adapters.safetensors"),
            {
                "model.layers.0.q_proj.lora_a": mx.array([[value], [value]]),
                "model.layers.0.q_proj.lora_b": mx.array([[value, value]]),
            },
        )
        paths.append(path)

    output = compose_adapter_directories(paths, tmp_path / "composed")
    arrays = mx.load(str(output / "adapters.safetensors"))
    composed_config = json.loads((output / "adapter_config.json").read_text())
    assert arrays["model.layers.0.q_proj.lora_a"].shape == (2, 2)
    assert arrays["model.layers.0.q_proj.lora_b"].shape == (2, 2)
    assert composed_config["lora_parameters"]["rank"] == 2


def test_rejects_different_adapter_scales(tmp_path):
    base = {
        "fine_tune_type": "lora",
        "num_layers": 1,
        "lora_parameters": {"rank": 1, "scale": 20.0, "dropout": 0.0},
    }
    from skill_lattice_coder.compose import validate_adapter_configs

    with pytest.raises(ValueError, match="scale"):
        validate_adapter_configs(
            [
                base,
                {**base, "lora_parameters": {**base["lora_parameters"], "scale": 10.0}},
            ]
        )
