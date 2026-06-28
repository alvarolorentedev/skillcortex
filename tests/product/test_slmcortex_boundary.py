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
        if path.name != "__pycache__"
    }
    assert entries == {
        "agent",
        "catalog.py",
        "cli",
        "composer",
        "contracts.py",
        "dataset_factory",
        "datasets",
        "packaging",
        "runtime",
        "shared",
        "slmcortex.py",
        "training",
    }


def test_console_scripts_expose_only_slmcortex_product_entrypoint():
    payload = tomllib.loads((ROOT / "pyproject.toml").read_text())
    scripts = payload["project"]["scripts"]
    assert scripts == {"slmcortex": "slmcortex:main"}
