from __future__ import annotations

from pathlib import Path
from typing import Any

from ..shared.io import read_json, read_yaml
from .types import CatalogResult, RoutingCard, SlmRecord


class SlmCatalog:
    @staticmethod
    def discover(slms_dir: Path) -> CatalogResult:
        root = slms_dir.resolve()
        if not root.exists() or not root.is_dir():
            raise FileNotFoundError(f"slms directory not found: {root}")
        slms: list[SlmRecord] = []
        errors: list[str] = []
        warnings: list[str] = []
        for package in sorted(path for path in root.iterdir() if path.is_dir()):
            manifest_path = package / "slm.yaml"
            if not manifest_path.exists():
                errors.append(f"{package.name}: missing slm.yaml")
                continue
            try:
                manifest = read_yaml(manifest_path)
                slms.append(_slm_from_manifest(package, manifest, warnings))
            except ValueError as error:
                errors.append(f"{package.name}: {error}")
        return CatalogResult(slms=slms, errors=errors, warnings=warnings)


def _slm_from_manifest(package: Path, manifest: dict[str, Any], warnings: list[str]) -> SlmRecord:
    slm_id = _required_text(manifest, "slm_id")
    name = _required_text(manifest, "name")
    base = manifest.get("base") or {}
    adapter = manifest.get("adapter") or {}
    task_type_hint = manifest.get("task_type_hint") or manifest.get("task_type")
    composition = manifest.get("composition") or {}
    allowed_task_types = ((composition.get("capabilities") or {}).get("allowed_task_types") or [])
    if not task_type_hint and len(allowed_task_types) == 1:
        task_type_hint = allowed_task_types[0]
    routing_card = _load_routing_card(package / "routing_card.json", warnings)
    eval_summary = _load_optional_json(package / "eval_summary.json", warnings)
    return SlmRecord(
        slm_id=slm_id,
        name=name,
        path=package.resolve(),
        description=str(manifest.get("description") or ""),
        capabilities=_text_list(manifest.get("capabilities"), f"{slm_id}: capabilities", warnings),
        activation_cues=_text_list(manifest.get("activation_cues"), f"{slm_id}: activation_cues", warnings),
        avoid_when=_text_list(manifest.get("avoid_when"), f"{slm_id}: avoid_when", warnings),
        task_type_hint=str(task_type_hint) if task_type_hint else None,
        base_model=manifest.get("base_model") or base.get("runtime_model") or base.get("source_model"),
        adapter_path=package / (manifest.get("adapter_path") or adapter.get("path") or "adapter"),
        routing_card=routing_card,
        eval_summary=eval_summary,
    )


def _required_text(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value.strip()


def _text_list(value: Any, label: str, warnings: list[str]) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        warnings.append(f"{label} must be a list of strings; ignoring")
        return []
    return [item.strip() for item in value if item.strip()]


def _load_routing_card(path: Path, warnings: list[str]) -> RoutingCard:
    payload = _load_optional_json(path, warnings)
    return RoutingCard(
        summary=str(payload.get("summary") or ""),
        embedding_text=str(payload.get("embedding_text") or ""),
        positive_examples=_text_list(payload.get("positive_examples"), f"{path}: positive_examples", warnings),
        negative_examples=_text_list(payload.get("negative_examples"), f"{path}: negative_examples", warnings),
        observed_success_contexts=list(payload.get("observed_success_contexts") or []),
    )


def _load_optional_json(path: Path, warnings: list[str]) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return read_json(path)
    except ValueError as error:
        warnings.append(f"{path.name}: {error}")
        return {}
