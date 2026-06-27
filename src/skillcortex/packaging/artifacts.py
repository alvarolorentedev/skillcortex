from __future__ import annotations

from pathlib import Path

from ..shared.config import ARTIFACT_DIR, CONFIG_DIR, ROOT
from ..shared.hashing import sha256


REQUIRED_PACKAGE_FILES = (
    "skill.yaml",
    "metadata.json",
    "training_config.json",
    "eval.json",
    "README.md",
    "adapter/adapters.safetensors",
)

CHECKSUM_EXCLUDES = {"metadata.json"}


def protected_input_paths(*, train_dataset: Path, eval_dataset: Path) -> list[Path]:
    paths = {
        train_dataset.resolve(),
        eval_dataset.resolve(),
        (CONFIG_DIR / "base.yaml").resolve(),
        (CONFIG_DIR / "training.yaml").resolve(),
        (CONFIG_DIR / "skill_registry.json").resolve(),
        (CONFIG_DIR / "skills.yaml").resolve(),
    }
    for path in sorted((ARTIFACT_DIR / "adapters").rglob("*")):
        if path.is_file():
            paths.add(path.resolve())
    benchmark_root = ROOT / "data" / "benchmarks"
    for path in sorted(benchmark_root.rglob("*")):
        if path.is_file():
            paths.add(path.resolve())
    return sorted(paths)


def snapshot_files(paths: list[Path]) -> dict[str, str]:
    return {str(path): sha256(path) for path in paths}


def freeze_protected_inputs(before: dict[str, str]) -> dict:
    after = {path: sha256(Path(path)) for path in before}
    files = {
        path: {
            "before_sha256": before[path],
            "after_sha256": after[path],
            "unchanged": before[path] == after[path],
        }
        for path in sorted(before)
    }
    changed = [path for path, snapshot in files.items() if not snapshot["unchanged"]]
    if changed:
        raise RuntimeError(f"protected input changed during packaging: {changed[0]}")
    return {"all_unchanged": True, "files": files}


def package_checksums(root: Path) -> dict[str, str]:
    checksums = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if relative in CHECKSUM_EXCLUDES:
            continue
        checksums[relative] = sha256(path)
    return checksums


def validate_package_checksums(root: Path, checksums: dict[str, str]) -> None:
    for relative, expected in sorted(checksums.items()):
        path = root / relative
        if not path.exists():
            raise ValueError(f"checksummed file is missing: {relative}")
        if sha256(path) != expected:
            raise ValueError(f"checksum mismatch for {relative}")


def adapter_weights_path(adapter_dir: Path) -> Path:
    path = adapter_dir / "adapters.safetensors"
    if not path.exists():
        raise FileNotFoundError(f"adapter weights not found: {path}")
    return path


def line_count(path: Path | None) -> int:
    if path is None:
        return 0
    return sum(1 for line in path.read_text().splitlines() if line.strip())
