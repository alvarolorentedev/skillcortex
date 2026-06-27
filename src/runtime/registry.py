from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..packaging import validate_skill_package
from ..packaging.importers import import_lora
from ..shared.config import base_config
from ..shared.hashing import package_fingerprint, sha256
from ..shared.io import read_json, read_yaml


@dataclass(slots=True)
class ResolvedAdapter:
    skill_id: str
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


class AdapterRegistry:
    def __init__(
        self,
        skills_dir: Path,
        *,
        allow_remote: bool = False,
        cache_dir: Path | None = None,
    ):
        self.skills_dir = skills_dir.resolve()
        self.allow_remote = allow_remote
        self.cache_dir = cache_dir or Path(base_config().get("lora_cache_dir") or ".skillcortex/lora-cache")
        self.local = self._discover()

    @classmethod
    def load(
        cls,
        skills_dir: Path,
        *,
        allow_remote: bool = False,
        cache_dir: Path | None = None,
    ) -> "AdapterRegistry":
        return cls(skills_dir, allow_remote=allow_remote, cache_dir=cache_dir)

    def reload(self) -> None:
        self.local = self._discover()

    def resolve_remote(self, source: str, skill_id: str, name: str | None = None) -> ResolvedAdapter:
        config = base_config()
        if not (self.allow_remote or bool(config.get("allow_remote_lora_downloads"))):
            raise ValueError("remote LoRA downloads are disabled")
        output = self.skills_dir / skill_id
        import_lora(
            source=source,
            skill_id=skill_id,
            name=name or skill_id.replace("_", " ").title(),
            output=output,
            train_dataset=Path("data/train.jsonl"),
            eval_dataset=Path("data/eval.jsonl"),
            cache_dir=self.cache_dir,
        )
        self.reload()
        if skill_id not in self.local:
            raise ValueError(f"resolved remote LoRA did not produce a valid package: {skill_id}")
        return self.local[skill_id]

    def _discover(self) -> dict[str, ResolvedAdapter]:
        if not self.skills_dir.exists():
            return {}
        found: dict[str, ResolvedAdapter] = {}
        for package in sorted(path for path in self.skills_dir.iterdir() if path.is_dir()):
            try:
                adapter = _resolved_adapter(package)
            except (FileNotFoundError, KeyError, TypeError, ValueError):
                continue
            found[adapter.skill_id] = adapter
        return found


def _resolved_adapter(package: Path) -> ResolvedAdapter:
    validate_skill_package(package)
    manifest = read_yaml(package / "skill.yaml")
    metadata = read_json(package / "metadata.json")
    adapter = manifest.get("adapter") or {}
    adapter_path = package / (adapter.get("path") or "adapter/adapters.safetensors")
    composition = manifest.get("composition") or {}
    capabilities = composition.get("capabilities") or {}
    activation = composition.get("activation") or {}
    checksums = metadata.get("checksums") or {}
    weight_key = "adapter/adapters.safetensors"
    return ResolvedAdapter(
        skill_id=manifest["skill_id"],
        package_path=package.resolve(),
        adapter_path=adapter_path.resolve(),
        source=(metadata.get("source_artifacts") or {}).get("source"),
        sha256=checksums.get(weight_key) or sha256(adapter_path),
        name=manifest.get("name") or manifest["skill_id"],
        description=" ".join(
            item
            for item in (
                manifest["skill_id"].replace("_", " "),
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
    )


def _text_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item.strip() for item in value if isinstance(item, str) and item.strip()]
