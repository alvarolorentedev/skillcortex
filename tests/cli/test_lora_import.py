import json
from pathlib import Path

from slmcortex.cli import main
from slmcortex.shared.hashing import sha256


def test_import_lora_from_huggingface_source_uses_cache_and_provenance(tmp_path, monkeypatch, capsys):
    calls = []

    def fake_snapshot_download(repo_id, *, revision, local_dir, allow_patterns):
        calls.append((repo_id, revision, Path(local_dir)))
        root = Path(local_dir)
        root.mkdir(parents=True, exist_ok=True)
        (root / "adapter_config.json").write_text("{}")
        (root / "adapters.safetensors").write_text("weights")
        return str(root)

    monkeypatch.setattr("slmcortex.packaging.importers.snapshot_download", fake_snapshot_download)

    assert (
        main(
            [
                "import-lora",
                "--source",
                "hf://owner/repo",
                "--skill-id",
                "fastapi_skill",
                "--name",
                "FastAPI Skill",
                "--output",
                str(tmp_path / "fastapi_skill"),
                "--train-dataset",
                "data/train.jsonl",
                "--eval-dataset",
                "data/eval.jsonl",
                "--cache-dir",
                str(tmp_path / "cache"),
                "--force",
            ]
        )
        == 0
    )

    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "complete"
    assert calls[0][0:2] == ("owner/repo", "main")
    assert calls[0][2].parent == tmp_path / "cache" / "hf" / "owner" / "repo"
    source = json.loads((tmp_path / "cache" / "hf" / "owner" / "repo" / "main" / "source.json").read_text())
    assert source["source"] == "hf://owner/repo"
    assert source["revision"] == "main"
    assert source["files"]["adapters.safetensors"]["sha256"] == sha256(tmp_path / "cache" / "hf" / "owner" / "repo" / "main" / "adapters.safetensors")
    assert (tmp_path / "fastapi_skill" / "skill.yaml").exists()
    assert (tmp_path / "fastapi_skill" / "adapter" / "adapters.safetensors").exists()
    metadata = json.loads((tmp_path / "fastapi_skill" / "metadata.json").read_text())
    assert metadata["source_artifacts"]["source"] == "hf://owner/repo"
    assert metadata["source_artifacts"]["cache_path"].endswith("/cache/hf/owner/repo/main")
    assert metadata["datasets"]["train"]["role"] == "packaging_reference"


def test_import_lora_reuses_cache_without_force(tmp_path, monkeypatch, capsys):
    cache = tmp_path / "cache" / "hf" / "owner" / "repo" / "main"
    cache.mkdir(parents=True)
    (cache / "adapter_config.json").write_text("{}")
    (cache / "adapters.safetensors").write_text("weights")
    (cache / "source.json").write_text(json.dumps({"source": "hf://owner/repo", "files": {}}))

    monkeypatch.setattr("slmcortex.packaging.importers.snapshot_download", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should reuse cache")))

    assert main([
        "import-lora",
        "--source", "hf://owner/repo",
        "--skill-id", "fastapi_skill",
        "--name", "FastAPI Skill",
        "--output", str(tmp_path / "fastapi_skill"),
        "--train-dataset", "data/train.jsonl",
        "--eval-dataset", "data/eval.jsonl",
        "--cache-dir", str(tmp_path / "cache"),
    ]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "complete"


def test_import_lora_rejects_disallowed_publisher(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("slmcortex.packaging.importers.base_config", lambda: {"allowed_hf_publishers": ["trusted"]})

    assert main([
        "import-lora",
        "--source", "hf://owner/repo",
        "--skill-id", "fastapi_skill",
        "--name", "FastAPI Skill",
        "--output", str(tmp_path / "fastapi_skill"),
        "--train-dataset", "data/train.jsonl",
        "--eval-dataset", "data/eval.jsonl",
        "--cache-dir", str(tmp_path / "cache"),
    ]) == 2
    assert "publisher is not allowed" in capsys.readouterr().err


def test_import_lora_rejects_oversized_adapter(tmp_path, monkeypatch, capsys):
    def fake_snapshot_download(repo_id, *, revision, local_dir, allow_patterns):
        root = Path(local_dir)
        root.mkdir(parents=True, exist_ok=True)
        (root / "adapter_config.json").write_text("{}")
        (root / "adapters.safetensors").write_text("too large")
        return str(root)

    monkeypatch.setattr("slmcortex.packaging.importers.snapshot_download", fake_snapshot_download)

    assert main([
        "import-lora",
        "--source", "hf://owner/repo",
        "--skill-id", "fastapi_skill",
        "--name", "FastAPI Skill",
        "--output", str(tmp_path / "fastapi_skill"),
        "--train-dataset", "data/train.jsonl",
        "--eval-dataset", "data/eval.jsonl",
        "--cache-dir", str(tmp_path / "cache"),
        "--max-download-bytes", "3",
    ]) == 2
    assert "download exceeds max_download_bytes" in capsys.readouterr().err
