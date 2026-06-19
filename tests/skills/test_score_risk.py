from pathlib import Path

from repo_brain.indexer import index_repository
from repo_brain.skills.score_risk import score_risk


def test_security_config_patch_scores_above_test_patch(tmp_path: Path) -> None:
    (tmp_path / "auth.py").write_text("def authorize(): return True\n")
    (tmp_path / "test_auth.py").write_text("def test_authorize(): pass\n")
    index_repository(tmp_path)
    risky = score_risk(
        tmp_path,
        "diff --git a/auth.py b/auth.py\n--- a/auth.py\n+++ b/auth.py\n"
        "@@ -1 +1 @@\n-def authorize(): return True\n+def authorize(): return False\n",
    )
    safe = score_risk(
        tmp_path,
        "diff --git a/test_auth.py b/test_auth.py\n--- a/test_auth.py\n+++ b/test_auth.py\n"
        "@@ -1 +1 @@\n-def test_authorize(): pass\n+def test_authorize(): assert True\n",
    )
    assert risky.score > safe.score
    assert risky.findings
    assert 0 <= risky.score <= 100


def test_risk_score_caps_at_100(tmp_path: Path) -> None:
    names = ["auth.py", "crypto.py", "migration.py", "billing.py", "secrets.py", "token.py"]
    files = "\n".join(
        f"diff --git a/{name} b/{name}\n--- a/{name}\n+++ b/{name}\n"
        "@@ -0,0 +1 @@\n+secret = 'x'\n"
        for name in names
    )
    report = score_risk(tmp_path, files)
    assert report.score == 100
    assert report.band == "critical"
