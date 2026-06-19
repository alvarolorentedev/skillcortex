from pathlib import Path

from repo_brain.skills.validate_patch import validate_patch

PATCH = """diff --git a/app.py b/app.py
--- a/app.py
+++ b/app.py
@@ -1 +1 @@
-value = 1
+value = 2
"""


def test_validates_patch_without_mutating_repository(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("value = 1\n")
    patch = tmp_path / "change.diff"
    patch.write_text(PATCH)
    report = validate_patch(tmp_path, patch)
    assert report.passed
    assert (tmp_path / "app.py").read_text() == "value = 1\n"
    assert report.affected_files == ("app.py",)


def test_rejects_path_traversal_before_execution(tmp_path: Path) -> None:
    patch = tmp_path / "bad.diff"
    patch.write_text("--- a/../../secret\n+++ b/../../secret\n@@ -0,0 +1 @@\n+x\n")
    report = validate_patch(tmp_path, patch)
    assert not report.applicable
    assert not report.passed

