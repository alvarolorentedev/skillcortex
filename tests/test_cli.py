import json
from pathlib import Path

from repo_brain.cli import main


def test_index_json_matches_human_counts(tmp_path: Path, capsys: object) -> None:
    (tmp_path / "a.py").write_text("x = 1\n")
    assert main(["--repo", str(tmp_path), "--json", "index"]) == 0
    output = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]
    assert output["schema_version"] == 1
    assert output["command"] == "index"
    assert output["data"]["added"] == 1


def test_malformed_config_returns_storage_error(tmp_path: Path) -> None:
    state = tmp_path / ".repo-brain"
    state.mkdir()
    (state / "config.json").write_text("{")
    assert main(["--repo", str(tmp_path), "index"]) == 3


def test_map_command_returns_repository_summary(tmp_path: Path, capsys: object) -> None:
    (tmp_path / "a.py").write_text("def run():\n    pass\n")
    assert main(["--repo", str(tmp_path), "index"]) == 0
    capsys.readouterr()  # type: ignore[attr-defined]
    assert main(["--repo", str(tmp_path), "--json", "map"]) == 0
    output = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]
    assert output["data"]["symbol_counts"]["function"] == 1


def test_localize_and_evidence_commands(tmp_path: Path, capsys: object) -> None:
    (tmp_path / "auth.py").write_text("def authenticate(token):\n    return bool(token)\n")
    main(["--repo", str(tmp_path), "index"])
    capsys.readouterr()  # type: ignore[attr-defined]
    assert main(["--repo", str(tmp_path), "--json", "localize", "authenticate token"]) == 0
    localized = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]
    assert localized["data"][0]["path"] == "auth.py"
    assert main(["--repo", str(tmp_path), "evidence", "authenticate token"]) == 0
    assert "auth.py:" in capsys.readouterr().out  # type: ignore[attr-defined]


def test_validation_and_failure_commands(tmp_path: Path, capsys: object) -> None:
    (tmp_path / "app.py").write_text("value = 1\n")
    patch = tmp_path / "change.diff"
    patch.write_text(
        "diff --git a/app.py b/app.py\n--- a/app.py\n+++ b/app.py\n"
        "@@ -1 +1 @@\n-value = 1\n+value = 2\n"
    )
    assert main(["--repo", str(tmp_path), "--json", "validate", "--patch", str(patch)]) == 0
    report = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]
    assert report["data"]["passed"]
    log = tmp_path / "test.log"
    log.write_text("src/a.ts(4,2): error TS2322: invalid")
    assert main(["--repo", str(tmp_path), "--json", "explain-failure", "--log", str(log)]) == 0
    explanation = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]
    assert explanation["data"]["category"] == "typescript"
