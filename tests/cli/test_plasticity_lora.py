import json

from slmcortex.cli import main


def test_train_plasticity_lora_dry_run_does_not_train(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("slmcortex.cli.handlers.train_skill_package", lambda **kwargs: (_ for _ in ()).throw(AssertionError("should not train")))

    assert main([
        "train-plasticity-lora",
        "--skill-id", "local_fix",
        "--name", "Local Fix",
        "--prompt-file", "data/train.jsonl",
        "--eval-dataset", "data/eval.jsonl",
        "--publish-dir", str(tmp_path / "skills"),
        "--dry-run",
    ]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "dry-run"
    assert output["output"].endswith("/skills/local_fix")


def test_train_plasticity_lora_publishes_atomically(tmp_path, monkeypatch, capsys):
    def fake_train_skill_package(**kwargs):
        output = kwargs["output"]
        output.mkdir(parents=True)
        (output / "skill.yaml").write_text("skill_id: local_fix\n")
        return {"status": "complete", "output": str(output), "skill_id": "local_fix"}

    monkeypatch.setattr("slmcortex.cli.handlers.train_skill_package", fake_train_skill_package)
    monkeypatch.setattr("slmcortex.cli.handlers.validate_skill_package", lambda path: None)

    assert main([
        "train-plasticity-lora",
        "--skill-id", "local_fix",
        "--name", "Local Fix",
        "--prompt-file", "data/train.jsonl",
        "--eval-dataset", "data/eval.jsonl",
        "--publish-dir", str(tmp_path / "skills"),
    ]) == 0

    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "complete"
    assert output["validation_status"] == "valid"
    assert (tmp_path / "skills" / "local_fix" / "skill.yaml").exists()
