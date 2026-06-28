from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import snapshot_download

from ..shared.config import base_config
from ..shared.config import resolve_backend
from ..shared.hashing import sha256


ADAPTER_FILES = ("adapters.safetensors", "adapter_model.safetensors")


def parse_hf_source(source: str) -> tuple[str, str, str]:
    if not source.startswith("hf://"):
        raise ValueError("only hf://owner/repo sources are supported")
    value = source.removeprefix("hf://").strip("/")
    repo, _, revision = value.partition("@")
    if repo.count("/") != 1:
        raise ValueError("hf source must be hf://owner/repo[@revision]")
    owner, name = repo.split("/", 1)
    return owner, name, revision or "main"


def resolve_hf_lora_cache(
    *,
    source: str,
    cache_dir: Path,
    force: bool = False,
    max_download_bytes: int | None = None,
) -> Path:
    config = base_config()
    owner, repo_name, revision = parse_hf_source(source)
    allowed = list(config.get("allowed_hf_publishers") or [])
    if allowed and owner not in allowed:
        raise ValueError(f"publisher is not allowed: {owner}")

    target = cache_dir.resolve() / "hf" / owner / repo_name / revision
    if (target / "source.json").exists() and not force:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(
        tempfile.mkdtemp(prefix=f".{repo_name}-{revision}-", dir=target.parent)
    )
    try:
        snapshot_download(
            f"{owner}/{repo_name}",
            revision=revision,
            local_dir=staging_root,
            allow_patterns=["adapter_config.json", *ADAPTER_FILES],
        )
        adapter = _adapter_file(staging_root)
        limit = int(max_download_bytes or config.get("max_download_bytes") or 0)
        if limit and adapter.stat().st_size > limit and not force:
            raise ValueError("download exceeds max_download_bytes")
        _write_source_json(staging_root, source, owner=owner, repo_name=repo_name, revision=revision)
    except Exception:
        shutil.rmtree(staging_root)
        raise
    if target.exists():
        shutil.rmtree(target)
    shutil.move(str(staging_root), str(target))
    return target


def import_lora(
    *,
    source: str,
    slm_id: str,
    name: str,
    output: Path,
    train_dataset: Path,
    eval_dataset: Path,
    version: str = "0.1.0",
    description: str | None = None,
    cache_dir: Path | None = None,
    max_download_bytes: int | None = None,
    force: bool = False,
) -> dict:
    from . import package_slm

    cache_root = cache_dir or Path(base_config().get("lora_cache_dir") or ".slmcortex/lora-cache")
    cached = resolve_hf_lora_cache(
        source=source,
        cache_dir=cache_root,
        force=force,
        max_download_bytes=max_download_bytes,
    )
    with tempfile.TemporaryDirectory(prefix=f"slmcortex-import-{slm_id}-") as directory:
        root = Path(directory)
        adapter = root / "adapter"
        adapter.mkdir()
        config = base_config()
        backend = resolve_backend(config)
        if backend == "gguf":
            peft = root / "peft"
            peft.mkdir()
            shutil.copy2(cached / "adapter_config.json", peft / "adapter_config.json")
            shutil.copy2(_adapter_file(cached), peft / "adapter_model.safetensors")
            converter = config.get("gguf_converter")
            if not converter:
                raise ValueError("GGUF import requires gguf_converter in base config")
            subprocess.run(
                ["python", str(converter), str(peft), "--outfile", str(adapter / "adapter.gguf")],
                check=True,
            )
        else:
            shutil.copy2(cached / "adapter_config.json", adapter / "adapter_config.json")
            shutil.copy2(_adapter_file(cached), adapter / "adapters.safetensors")
        (adapter / "metadata.json").write_text(
            json.dumps(
                {
                    "adapter": slm_id,
                    "source_model": config.get("source_model"),
                    "base_model": config.get("default_runtime_model") or config.get("model"),
                    "quantization": "unknown",
                    "backend": backend,
                    "format": "gguf-lora" if backend == "gguf" else "mlx-lora",
                    "rank": 0,
                    "trainable_parameters": 0,
                    "source": source,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        eval_summary = root / "eval-summary.json"
        eval_summary.write_text(json.dumps({"modes": {}, "tasks": {}}) + "\n")
        result = package_slm(
            slm_id=slm_id,
            name=name,
            adapter_dir=adapter,
            output=output,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            eval_summary=eval_summary,
            version=version,
            description=description or f"Imported LoRA package from {source}.",
            force=force,
            source_artifacts={"source": source, "cache_path": str(cached)},
        )
    result["source"] = source
    result["cache_path"] = str(cached)
    return result


def _adapter_file(root: Path) -> Path:
    for name in ADAPTER_FILES:
        path = root / name
        if path.exists():
            return path
    raise FileNotFoundError(f"adapter weights not found in cache: {root}")


def _write_source_json(root: Path, source: str, *, owner: str, repo_name: str, revision: str) -> None:
    files = {}
    for path in sorted(root.iterdir()):
        if path.is_file() and path.name != "source.json":
            files[path.name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    (root / "source.json").write_text(
        json.dumps(
            {
                "source": source,
                "repo_id": f"{owner}/{repo_name}",
                "revision": revision,
                "cached_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "files": files,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
