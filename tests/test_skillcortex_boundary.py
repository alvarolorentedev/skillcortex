from pathlib import Path
import tomllib


ROOT = Path(__file__).resolve().parents[1]
SKILLCORTEX_ROOT = ROOT / "src" / "skillcortex"


def _product_python_files() -> list[Path]:
    files = []
    for path in sorted(SKILLCORTEX_ROOT.rglob("*.py")):
        if any(parent.name == "__pycache__" for parent in path.parents):
            continue
        files.append(path)
    return files


def test_src_tree_contains_only_skillcortex_product_package():
    packages = {
        path.name
        for path in (ROOT / "src").iterdir()
        if path.is_dir() and path.name != "__pycache__"
    }
    assert packages == {"skillcortex"}


def test_console_scripts_expose_only_skillcortex_product_entrypoint():
    payload = tomllib.loads((ROOT / "pyproject.toml").read_text())
    scripts = payload["project"]["scripts"]
    assert scripts == {"skillcortex": "skillcortex.cli:main"}
