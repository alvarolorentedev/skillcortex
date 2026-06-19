from repo_brain.tracing.redaction import redact


def test_redacts_tokens_and_home_paths() -> None:
    value = redact("TOKEN=super-secret /Users/alice/project api_key: abcdef")
    assert "super-secret" not in value
    assert "abcdef" not in value
    assert "/Users/alice" not in value

