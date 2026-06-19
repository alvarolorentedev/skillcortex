from pathlib import Path

from repo_brain.skills.explain_failure import explain_failure


def test_explains_python_traceback() -> None:
    log = """Traceback (most recent call last):
  File "app.py", line 7, in run
    raise ValueError("bad token")
ValueError: bad token
"""
    explanation = explain_failure(Path("."), log)
    assert explanation.category == "python_exception"
    assert explanation.primary_message == "ValueError: bad token"
    assert explanation.frames[0]["path"] == "app.py"


def test_explains_typescript_diagnostic() -> None:
    explanation = explain_failure(Path("."), "src/a.ts(4,2): error TS2322: Type 'x' is invalid")
    assert explanation.category == "typescript"
    assert explanation.frames[0]["line"] == 4
