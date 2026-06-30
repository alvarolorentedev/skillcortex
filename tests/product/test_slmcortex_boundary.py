from pathlib import Path
import tomllib


def _repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("repository root not found")


ROOT = _repo_root()


def test_src_tree_is_flat_product_layout():
    entries = {
        path.name
        for path in (ROOT / "src").iterdir()
        if path.name != "__pycache__" and not path.name.endswith(".egg-info")
    }
    assert entries == {
        "agent",
        "catalog",
        "cli",
        "composer",
        "composer_app",
        "contracts.py",
        "dataset_factory",
        "datasets",
        "packaging",
        "runtime",
        "shared",
        "slmcortex.py",
        "slmcortex_resources",
        "training",
    }


def test_console_scripts_expose_product_entrypoints():
    payload = tomllib.loads((ROOT / "pyproject.toml").read_text())
    scripts = payload["project"]["scripts"]
    assert scripts == {
        "slmcortex": "slmcortex:main",
        "slmcortex-composer": "slmcortex:composer_main",
    }


def test_packaged_configs_match_repo_configs():
    for path in sorted((ROOT / "src" / "slmcortex_resources" / "configs").iterdir()):
        if path.suffix not in {".yaml", ".json"}:
            continue
        packaged = ROOT / "src" / "slmcortex_resources" / "configs" / path.name
        assert packaged.read_bytes() == path.read_bytes(), path.name
