from __future__ import annotations

import json
import math
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Sequence


def compose_lora_arrays(
    pairs: Sequence[tuple[object, object]], weights: Sequence[float]
) -> tuple[object, object]:
    if not pairs or len(pairs) != len(weights):
        raise ValueError("pairs and weights must have the same non-zero length")
    if any(weight <= 0 for weight in weights):
        raise ValueError("weights must be positive")
    total = sum(weights)
    normalized = [weight / total for weight in weights]
    module = _array_module(pairs[0][0])
    scaled_a = [a * math.sqrt(weight) for (a, _), weight in zip(pairs, normalized)]
    scaled_b = [b * math.sqrt(weight) for (_, b), weight in zip(pairs, normalized)]
    _validate_pair_shapes(pairs)
    return module.concatenate(scaled_a, axis=1), module.concatenate(scaled_b, axis=0)


def _array_module(array: object):
    module_name = type(array).__module__.split(".")[0]
    if module_name == "numpy":
        import numpy

        return numpy
    import mlx.core as mx

    return mx


def _validate_pair_shapes(pairs: Sequence[tuple[object, object]]) -> None:
    in_features = pairs[0][0].shape[0]
    out_features = pairs[0][1].shape[1]
    for a, b in pairs:
        if len(a.shape) != 2 or len(b.shape) != 2 or a.shape[1] != b.shape[0]:
            raise ValueError("invalid LoRA A/B shapes")
        if a.shape[0] != in_features or b.shape[1] != out_features:
            raise ValueError("incompatible LoRA non-rank dimensions")


def validate_adapter_metadata(metadata: Sequence[dict]) -> None:
    if not metadata:
        raise ValueError("metadata is required")
    for key in ("base_model", "target_modules", "quantization"):
        expected = metadata[0].get(key)
        if any(item.get(key) != expected for item in metadata[1:]):
            raise ValueError(f"incompatible adapter {key}")


def validate_adapter_configs(configs: Sequence[dict]) -> None:
    if not configs:
        raise ValueError("adapter configs are required")
    first = configs[0]
    for key in ("fine_tune_type", "num_layers"):
        if any(config.get(key) != first.get(key) for config in configs[1:]):
            raise ValueError(f"incompatible adapter {key}")
    first_lora = first.get("lora_parameters", {})
    for key in ("scale", "dropout", "keys"):
        if any(
            config.get("lora_parameters", {}).get(key) != first_lora.get(key)
            for config in configs[1:]
        ):
            raise ValueError(f"incompatible adapter {key}")
    if any(not config.get("lora_parameters", {}).get("rank") for config in configs):
        raise ValueError("adapter rank is required")


def compose_adapter_directories(
    adapter_directories: Sequence[str | Path],
    output_directory: str | Path,
    weights: Sequence[float] | None = None,
) -> Path:
    import mlx.core as mx

    directories = [Path(path) for path in adapter_directories]
    weights = list(weights or [1.0] * len(directories))
    configs = [
        json.loads((path / "adapter_config.json").read_text()) for path in directories
    ]
    metadata = [
        json.loads((path / "metadata.json").read_text()) for path in directories
    ]
    validate_adapter_metadata(metadata)
    validate_adapter_configs(configs)
    arrays = [mx.load(str(path / "adapters.safetensors")) for path in directories]
    keys = set(arrays[0])
    if any(set(item) != keys for item in arrays[1:]):
        raise ValueError("adapter tensor keys do not match")
    output = {}
    for key in sorted(keys):
        if key.endswith("lora_a"):
            b_key = key[:-1] + "b"
            if b_key not in keys:
                raise ValueError(f"missing tensor pair for {key}")
            output[key], output[b_key] = compose_lora_arrays(
                [(item[key], item[b_key]) for item in arrays], weights
            )
        elif not key.endswith("lora_b"):
            first = arrays[0][key]
            if any(item[key].shape != first.shape for item in arrays[1:]):
                raise ValueError(f"incompatible tensor shape for {key}")
            output[key] = first
    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(destination / "adapters.safetensors"), output)
    config = dict(configs[0])
    config["lora_parameters"] = dict(config.get("lora_parameters", {}))
    config["lora_parameters"]["rank"] = sum(
        item.get("lora_parameters", {}).get("rank", 0) for item in configs
    )
    (destination / "adapter_config.json").write_text(
        json.dumps(config, indent=2) + "\n"
    )
    (destination / "metadata.json").write_text(
        json.dumps(
            {
                **metadata[0],
                "composed_from": [path.name for path in directories],
                "weights": [weight / sum(weights) for weight in weights],
            },
            indent=2,
        )
        + "\n"
    )
    return destination


@contextmanager
def temporary_composed_adapter(
    paths: Sequence[Path], weights: Sequence[float] | None = None
) -> Iterator[Path]:
    with tempfile.TemporaryDirectory(prefix="skillcortex-adapter-") as directory:
        yield compose_adapter_directories(paths, directory, weights)
