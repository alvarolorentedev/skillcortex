import hashlib
import json
from pathlib import Path

import yaml

from skill_lattice_coder.schemas import DatasetExample
from scripts.run_python_skill_recovery import (
    VARIANTS,
    build_recovery_summary,
    load_variant_examples,
    main,
    prepare_variant,
)
from skill_lattice_coder.data import load_jsonl


ROOT = Path(__file__).resolve().parents[1]
PRESERVATION = ROOT / "data/preservation/python_generation_preservation.jsonl"


def test_preservation_data_is_training_only_python_generation():
    examples = load_jsonl(PRESERVATION)
    assert len(examples) == 9
    assert all(example.task_type == "python_generation" for example in examples)
    assert all(example.skills == ["python_skill"] for example in examples)
    assert all(
        example.group.startswith("preservation_training_only:")
        for example in examples
    )
    assert {
        example.group.removeprefix("preservation_training_only:")
        for example in examples
    } == {
        "divide_or",
        "keys_for_value",
        "mask_address",
        "multiply_values",
        "trim_prefix",
        "without_none",
        "capped_total",
        "decimal_digit_sum",
        "substring_total",
    }


def test_variant_configs_are_deterministic_and_only_preservation_adds_data(tmp_path):
    train = ROOT / "data/train.jsonl"
    base_count = len(load_variant_examples("python_skill_low_lr", train, PRESERVATION))
    assert len(load_variant_examples("python_skill_fewer_steps", train, PRESERVATION)) == base_count
    assert len(load_variant_examples("python_skill_preservation", train, PRESERVATION)) == base_count + 9

    for name, expected in VARIANTS.items():
        first = prepare_variant(name, 11, tmp_path, train, PRESERVATION)
        first_yaml = Path(first["command"][-1]).read_text()
        second = prepare_variant(name, 11, tmp_path, train, PRESERVATION)
        assert Path(second["command"][-1]).read_text() == first_yaml
        config = yaml.safe_load(first_yaml)
        assert config["learning_rate"] == expected["learning_rate"]
        assert config["iters"] == expected["iterations"]
        assert Path(config["adapter_path"]).name == "python_skill"
        assert "debugging_skill" not in first_yaml
        assert "test_generation_skill" not in first_yaml


def test_summary_compares_base_current_skill_and_variant():
    baseline = [
        _row("x", "python_generation", "base", True, "divide_or", 0),
        _row("x", "python_generation", "generic", False, "divide_or", 30),
        _row("x", "python_generation", "single-skill", False, "divide_or", 10),
    ]
    recovered = [
        _row("x", "python_generation", "base", True, "divide_or", 0),
        _row("x", "python_generation", "single-skill", True, "divide_or", 10),
        _row("d", "debugging", "lattice", True, "debug", 20),
        _row("t", "test_generation", "lattice", False, "test", 20),
        _row("x", "python_generation", "lattice", True, "divide_or", 10),
    ]
    summary = build_recovery_summary(
        baseline,
        {"python_skill_low_lr": recovered},
        {"python_skill_low_lr": {"trainable_parameters": 10}},
        seeds=[11],
    )
    assert summary["current"]["base"]["python_generation_pass_rate"] == 1.0
    assert summary["current"]["generic"]["python_generation_pass_rate"] == 0.0
    assert summary["current"]["python_skill"]["python_generation_pass_rate"] == 0.0
    variant = summary["variants"]["python_skill_low_lr"]
    assert variant["python_generation_pass_rate"] == 1.0
    assert variant["python_pass_to_fail_vs_base"] == 0
    assert variant["python_fail_to_pass_vs_base"] == 0
    assert variant["severe_regression_families"]["divide_or"]["candidate_pass_rate"] == 1.0
    assert variant["debugging_pass_rate"] == 1.0
    assert variant["test_generation_pass_rate"] == 0.0
    assert variant["trainable_adapter_parameters"] == 10


def test_train_counts_missing_preservation_groups_as_zero(tmp_path, monkeypatch):
    prepared = {
        "name": "python_skill_preservation",
        "seed": 11,
        "learning_rate": 5e-5,
        "iterations": 50,
        "preservation": True,
        "adapter_root": str(tmp_path / "adapters"),
        "command": ["true"],
    }
    examples = [
        DatasetExample(
            id="example-1",
            task_type="python_generation",
            skills=["python_skill"],
            prompt="p",
            target="t",
        )
    ]
    def fake_run(*args, **kwargs):
        Path(prepared["adapter_root"], "python_skill").mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr("scripts.run_python_skill_recovery.subprocess.run", fake_run)
    monkeypatch.setattr("scripts.run_python_skill_recovery._saved_parameter_count", lambda output: 0)

    metadata = __import__("scripts.run_python_skill_recovery", fromlist=["_train"])._train(prepared, examples, force=True)

    assert metadata["preservation_examples"] == 0
    assert json.loads((Path(prepared["adapter_root"]) / "python_skill" / "metadata.json").read_text())[
        "preservation_examples"
    ] == 0


def test_train_reuses_existing_complete_output_without_force(tmp_path, monkeypatch):
    prepared = {
        "name": "python_skill_low_lr",
        "seed": 11,
        "learning_rate": 5e-5,
        "iterations": 50,
        "preservation": False,
        "adapter_root": str(tmp_path / "adapters"),
        "command": ["true"],
    }
    output = Path(prepared["adapter_root"]) / "python_skill"
    output.mkdir(parents=True)
    (output / "adapters.safetensors").write_bytes(b"x")
    (output / "metadata.json").write_text(
        json.dumps({"adapter": "python_skill_low_lr", "preservation_examples": 0})
    )
    def fake_run(*args, **kwargs):
        raise AssertionError("should not train")

    monkeypatch.setattr("scripts.run_python_skill_recovery.subprocess.run", fake_run)

    metadata = __import__("scripts.run_python_skill_recovery", fromlist=["_train"])._train(prepared, [], force=False)

    assert metadata["adapter"] == "python_skill_low_lr"
    assert metadata["preservation_examples"] == 0


def test_dry_run_writes_reports_without_changing_benchmark(tmp_path):
    benchmark = ROOT / "data/eval.jsonl"
    before = hashlib.sha256(benchmark.read_bytes()).hexdigest()
    output = tmp_path / "recovery"
    assert main(
        [
            "--seeds",
            "11",
            "--output",
            str(output),
            "--baseline-experiment",
            str(ROOT / "artifacts/experiments/five-seed"),
            "--dry-run",
        ]
    ) == 0
    assert hashlib.sha256(benchmark.read_bytes()).hexdigest() == before
    summary = json.loads((output / "summary.json").read_text())
    assert summary["status"] == "dry-run"
    assert set(summary["variants"]) == set(VARIANTS)
    assert (output / "summary.md").exists()
    assert not list(output.rglob("adapters.safetensors"))


def _row(example, task, mode, passed, family, parameters):
    return {
        "example_id": example,
        "task_type": task,
        "mode": mode,
        "execution_passed": passed,
        "syntax_valid": True,
        "active_adapter_parameters": parameters,
        "benchmark_group": family,
    }
