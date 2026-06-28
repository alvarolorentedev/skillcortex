from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..packaging.validation import validate_slm_package
from ..packaging.importers import import_lora
from ..shared.config import base_config
from ..shared.config import adapter_format_for_backend, resolve_backend
from ..shared.hashing import package_fingerprint, sha256
from ..shared.io import read_json, read_yaml


@dataclass(slots=True)
class ResolvedAdapter:
    slm_id: str
    package_path: Path
    adapter_path: Path
    source: str | None
    sha256: str
    name: str
    description: str
    capabilities: list[str]
    activation_cues: list[str]
    base_model: str | None
    fingerprint: str
    adapter_format: str


class AdapterRegistry:
    def __init__(
        self,
        slms_dir: Path,
        *,
        allow_remote: bool = False,
        cache_dir: Path | None = None,
    ):
        self.slms_dir = slms_dir.resolve()
        self.allow_remote = allow_remote
        self.cache_dir = cache_dir or Path(base_config().get("lora_cache_dir") or ".slmcortex/lora-cache")
        self.local = self._discover()

    @classmethod
    def load(
        cls,
        slms_dir: Path,
        *,
        allow_remote: bool = False,
        cache_dir: Path | None = None,
    ) -> "AdapterRegistry":
        return cls(slms_dir, allow_remote=allow_remote, cache_dir=cache_dir)

    def reload(self) -> None:
        self.local = self._discover()

    def resolve_remote(self, source: str, slm_id: str, name: str | None = None) -> ResolvedAdapter:
        config = base_config()
        if not (self.allow_remote or bool(config.get("allow_remote_lora_downloads"))):
            raise ValueError("remote LoRA downloads are disabled")
        output = self.slms_dir / slm_id
        import_lora(
            source=source,
            slm_id=slm_id,
            name=name or slm_id.replace("_", " ").title(),
            output=output,
            train_dataset=Path(config.get("remote_lora_train_dataset") or "data/train.jsonl"),
            eval_dataset=Path(config.get("remote_lora_eval_dataset") or "data/eval.jsonl"),
            cache_dir=self.cache_dir,
        )
        self.reload()
        if slm_id not in self.local:
            raise ValueError(f"resolved remote LoRA did not produce a valid package: {slm_id}")
        return self.local[slm_id]

    def _discover(self) -> dict[str, ResolvedAdapter]:
        if not self.slms_dir.exists():
            return {}
        found: dict[str, ResolvedAdapter] = {}
        for package in sorted(path for path in self.slms_dir.iterdir() if path.is_dir()):
            try:
                adapter = _resolved_adapter(package)
            except (FileNotFoundError, KeyError, TypeError, ValueError):
                continue
            if adapter.adapter_format != adapter_format_for_backend(resolve_backend(base_config())):
                continue
            found[adapter.slm_id] = adapter
        return found


def _resolved_adapter(package: Path) -> ResolvedAdapter:
    validate_slm_package(package)
    manifest = read_yaml(package / "slm.yaml")
    metadata = read_json(package / "metadata.json")
    adapter = manifest.get("adapter") or {}
    adapter_path = package / (adapter.get("path") or "adapter/adapters.safetensors")
    composition = manifest.get("composition") or {}
    capabilities = composition.get("capabilities") or {}
    activation = composition.get("activation") or {}
    checksums = metadata.get("checksums") or {}
    weight_key = "adapter/adapters.safetensors"
    return ResolvedAdapter(
        slm_id=manifest["slm_id"],
        package_path=package.resolve(),
        adapter_path=adapter_path.resolve(),
        source=(metadata.get("source_artifacts") or {}).get("source"),
        sha256=checksums.get(weight_key) or sha256(adapter_path),
        name=manifest.get("name") or manifest["slm_id"],
        description=" ".join(
            item
            for item in (
                manifest["slm_id"].replace("_", " "),
                str(manifest.get("name") or ""),
                str(manifest.get("description") or ""),
            )
            if item
        ),
        capabilities=[
            *_text_list(manifest.get("capabilities")),
            *_text_list(capabilities.get("allowed_task_types")),
            *_text_list(activation.get("semantic_families")),
        ],
        activation_cues=_text_list(manifest.get("activation_cues")),
        base_model=(metadata.get("base") or {}).get("runtime_model"),
        fingerprint=package_fingerprint(manifest, metadata),
        adapter_format=adapter.get("format") or (metadata.get("adapter") or {}).get("format") or "mlx-lora",
    )


def _text_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item.strip() for item in value if isinstance(item, str) and item.strip()]
