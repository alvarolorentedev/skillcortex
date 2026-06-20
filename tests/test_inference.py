from contextlib import contextmanager
from pathlib import Path

import pytest

import skill_lattice_coder.inference as inference


def test_all_inference_modes_use_expected_adapters(monkeypatch):
    loaded = []

    monkeypatch.setattr(inference, "require_adapter", lambda name: Path(name))
    monkeypatch.setattr(
        inference,
        "adapter_metadata",
        lambda name: {"trainable_parameters": 10},
    )

    @contextmanager
    def composed(paths):
        assert [path.name for path in paths] == ["debugging_skill", "python_skill"]
        yield Path("composed")

    monkeypatch.setattr(inference, "temporary_composed_adapter", composed)
    monkeypatch.setattr(
        inference,
        "load_model",
        lambda adapter: loaded.append(adapter) or ("model", "tokenizer"),
    )
    monkeypatch.setattr(inference, "generate_text", lambda *args: ("generated", 4, 2))

    assert inference.infer("base", "Write code").active_adapter_count == 0
    assert inference.infer("generic", "Write code").active_adapter_count == 1
    assert inference.infer(
        "single-skill", "Fix code", skill="debugging_skill"
    ).selected_skills == ["debugging_skill"]
    lattice = inference.infer("lattice", "Fix this Python traceback")
    assert lattice.selected_skills == ["debugging_skill", "python_skill"]
    assert [None, Path("generic"), Path("debugging_skill"), Path("composed")] == loaded


def test_single_skill_requires_skill_name():
    with pytest.raises(ValueError, match="required"):
        inference.infer("single-skill", "Fix code", dry_run=True)
