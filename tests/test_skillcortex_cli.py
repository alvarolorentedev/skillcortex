import json
import hashlib
from pathlib import Path

import pytest

from skillcortex.cli import main


def test_skillcortex_cli_alias_supports_dry_run():
    assert main(["train-skill", "python_skill", "--dry-run"]) == 0


def test_package_skill_and_validate_package(tmp_path):
    output = tmp_path / "python_skill"
    eval_summary = tmp_path / "eval-summary.json"
    eval_summary.write_text(
        json.dumps(
            {
                "hypothesis": "inconclusive",
                "modes": {"single-skill": {"count": 1, "fuzzy_score": 1.0}},
                "tasks": {"python_generation": {"single-skill": {"count": 1}}},
            }
        )
        + "\n"
    )
    examples = tmp_path / "examples.jsonl"
    examples.write_text(
        json.dumps({"prompt": "Write a function", "target": "def answer():\n    return 42"})
        + "\n"
    )

    assert (
        main(
            [
                "package-skill",
                "--skill-id",
                "python_skill",
                "--name",
                "Python Skill",
                "--adapter-dir",
                "artifacts/adapters/python_skill",
                "--output",
                str(output),
                "--train-dataset",
                "data/train.jsonl",
                "--eval-dataset",
                "data/eval.jsonl",
                "--eval-summary",
                str(eval_summary),
                "--examples",
                str(examples),
                "--description",
                "General Python generation skill.",
            ]
        )
        == 0
    )
    assert (output / "skill.yaml").exists()
    assert (output / "metadata.json").exists()
    assert (output / "training_config.json").exists()
    assert (output / "eval.json").exists()
    assert (output / "README.md").exists()
    assert (output / "examples.jsonl").exists()
    assert (output / "adapter" / "adapters.safetensors").exists()
    metadata = json.loads((output / "metadata.json").read_text())
    assert metadata["checksums"]["README.md"]
    assert metadata["protected_inputs"]["all_unchanged"] is True

    assert main(["validate-skill-package", "--path", str(output)]) == 0


def test_package_skill_dry_run_does_not_write_output(tmp_path):
    output = tmp_path / "python_skill"
    eval_summary = tmp_path / "eval-summary.json"
    eval_summary.write_text(json.dumps({"modes": {}}) + "\n")

    assert (
        main(
            [
                "package-skill",
                "--skill-id",
                "python_skill",
                "--name",
                "Python Skill",
                "--adapter-dir",
                "artifacts/adapters/python_skill",
                "--output",
                str(output),
                "--train-dataset",
                "data/train.jsonl",
                "--eval-dataset",
                "data/eval.jsonl",
                "--eval-summary",
                str(eval_summary),
                "--dry-run",
            ]
        )
        == 0
    )
    assert not output.exists()


def test_validate_package_rejects_checksum_tamper(tmp_path):
    output = tmp_path / "python_skill"
    eval_summary = tmp_path / "eval-summary.json"
    eval_summary.write_text(json.dumps({"modes": {}, "tasks": {}}) + "\n")

    assert (
        main(
            [
                "package-skill",
                "--skill-id",
                "python_skill",
                "--name",
                "Python Skill",
                "--adapter-dir",
                "artifacts/adapters/python_skill",
                "--output",
                str(output),
                "--train-dataset",
                "data/train.jsonl",
                "--eval-dataset",
                "data/eval.jsonl",
                "--eval-summary",
                str(eval_summary),
            ]
        )
        == 0
    )
    (output / "README.md").write_text("tampered\n")

    assert main(["validate-skill-package", "--path", str(output)]) == 2


def test_product_train_skill_creates_isolated_run_and_package(monkeypatch, tmp_path):
    protected_adapter = Path("artifacts/adapters/python_skill/adapters.safetensors")
    before = hashlib.sha256(protected_adapter.read_bytes()).hexdigest()

    import skillcortex.packaging as packaging

    def fake_train(*, skill, train_dataset, run_directory, seed, force):
        adapter_dir = run_directory / "adapters" / skill
        adapter_dir.mkdir(parents=True, exist_ok=True)
        shutil_source = Path("artifacts/adapters/python_skill/adapters.safetensors")
        adapter_config = Path("artifacts/adapters/python_skill/adapter_config.json")
        adapter_metadata = json.loads(Path("artifacts/adapters/python_skill/metadata.json").read_text())
        adapter_dir.joinpath("adapters.safetensors").write_bytes(shutil_source.read_bytes())
        adapter_dir.joinpath("adapter_config.json").write_text(adapter_config.read_text())
        adapter_metadata["training_command"] = ["python", "-m", "mlx_lm", "lora"]
        adapter_dir.joinpath("metadata.json").write_text(json.dumps(adapter_metadata, indent=2) + "\n")
        return adapter_dir, adapter_metadata

    def fake_eval(*, skill, dataset, output, adapter_root):
        output.mkdir(parents=True, exist_ok=True)
        summary = {
            "hypothesis": None,
            "modes": {
                "base": {"count": 1, "fuzzy_score": 0.0},
                "single-skill": {"count": 1, "fuzzy_score": 1.0},
            },
            "tasks": {"python_generation": {"single-skill": {"count": 1, "fuzzy_score": 1.0}}},
        }
        path = output / "summary.json"
        path.write_text(json.dumps(summary) + "\n")
        return path

    monkeypatch.setattr(packaging, "_train_skill_to_run_directory", fake_train)
    monkeypatch.setattr(packaging, "_evaluate_skill_adapter", fake_eval)

    output = tmp_path / "product-python-skill"
    assert (
        main(
            [
                "train-skill",
                "python_skill",
                "--output",
                str(output),
                "--force",
            ]
        )
        == 0
    )
    assert (output / "skill.yaml").exists()
    assert (output / "adapter" / "adapters.safetensors").exists()
    assert (output.parent / f".{output.name}.run").exists()
    after = hashlib.sha256(protected_adapter.read_bytes()).hexdigest()
    assert before == after
    assert main(["validate-skill-package", "--path", str(output)]) == 0
