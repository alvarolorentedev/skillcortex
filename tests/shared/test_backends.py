import pytest

from slmcortex.shared.config import resolve_backend, validate_runtime_model


def test_auto_backend_uses_mlx_only_on_apple_silicon(monkeypatch):
    monkeypatch.setattr("platform.system", lambda: "Darwin")
    monkeypatch.setattr("platform.machine", lambda: "arm64")

    assert resolve_backend({"backend": "auto"}) == "mlx"


def test_auto_backend_uses_gguf_on_linux_windows_and_macos_intel(monkeypatch):
    for system, machine in [("Linux", "x86_64"), ("Windows", "AMD64"), ("Darwin", "x86_64")]:
        monkeypatch.setattr("platform.system", lambda value=system: value)
        monkeypatch.setattr("platform.machine", lambda value=machine: value)

        assert resolve_backend({"backend": "auto"}) == "gguf"


def test_mlx_backend_is_rejected_outside_apple_silicon(monkeypatch):
    monkeypatch.setattr("platform.system", lambda: "Linux")
    monkeypatch.setattr("platform.machine", lambda: "x86_64")

    with pytest.raises(ValueError, match="MLX backend requires macOS arm64"):
        resolve_backend({"backend": "mlx"})


def test_gguf_runtime_model_must_be_gguf_file():
    with pytest.raises(ValueError, match="GGUF backend requires a .gguf runtime model"):
        validate_runtime_model({"backend": "gguf", "model": "mlx-community/model"})

