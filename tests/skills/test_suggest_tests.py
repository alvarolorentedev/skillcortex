from pathlib import Path

from repo_brain.indexer import index_repository
from repo_brain.skills.suggest_tests import suggest_tests


def test_suggests_focused_python_test(tmp_path: Path) -> None:
    (tmp_path / "api.py").write_text("def login():\n    return True\n")
    (tmp_path / "test_api.py").write_text(
        "from api import login\n\ndef test_login():\n    assert login()\n"
    )
    index_repository(tmp_path)
    suggestions = suggest_tests(tmp_path, "fix login")
    assert suggestions
    assert suggestions[0]["command"].startswith("pytest test_api.py::")
    assert suggestions[0]["reason"]

