import json
import hashlib
from pathlib import Path

import pytest
import yaml

from skillcortex.cli import main
from skillcortex.dataset_factory import REQUIRED_FASTAPI_FEATURES


def test_skillcortex_cli_alias_supports_dry_run():
    assert main(["train-skill", "python_skill", "--output", "/tmp/skillcortex-dry-run", "--dry-run"]) == 0


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
    skill_manifest = yaml.safe_load((output / "skill.yaml").read_text())
    assert metadata["checksums"]["README.md"]
    assert metadata["protected_inputs"]["all_unchanged"] is True
    assert skill_manifest["composition"]["capabilities"]["allowed_task_types"] == [
        "debugging",
        "test_generation",
    ]

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


def test_product_train_skill_accepts_arbitrary_skill_id_and_composes(monkeypatch, tmp_path):
    import skillcortex.packaging as packaging

    train_dataset = tmp_path / "train.jsonl"
    eval_dataset = tmp_path / "eval.jsonl"
    train_dataset.write_text(
        json.dumps(
            {
                "id": "train-1",
                "task_type": "python_generation",
                "prompt": "Write a FastAPI route.",
                "target": "def build_route():\n    return 42\n",
                "semantic_family": "fastapi_contract",
            }
        )
        + "\n"
    )
    eval_dataset.write_text(
        json.dumps(
            {
                "id": "eval-1",
                "task_type": "debugging",
                "prompt": "Fix the FastAPI route.",
                "target": "def build_route():\n    return 42\n",
                "semantic_family": "fastapi_contract",
            }
        )
        + "\n"
    )

    def fake_train(*, skill_id, train_dataset, run_directory, seed, force):
        adapter_dir = run_directory / "adapters" / skill_id
        adapter_dir.mkdir(parents=True, exist_ok=True)
        shutil_source = Path("artifacts/adapters/python_skill/adapters.safetensors")
        adapter_config = Path("artifacts/adapters/python_skill/adapter_config.json")
        adapter_metadata = json.loads(Path("artifacts/adapters/python_skill/metadata.json").read_text())
        adapter_metadata["adapter"] = skill_id
        adapter_metadata["training_command"] = ["python", "-m", "mlx_lm", "lora"]
        adapter_dir.joinpath("adapters.safetensors").write_bytes(shutil_source.read_bytes())
        adapter_dir.joinpath("adapter_config.json").write_text(adapter_config.read_text())
        adapter_dir.joinpath("metadata.json").write_text(json.dumps(adapter_metadata, indent=2) + "\n")
        return adapter_dir, adapter_metadata

    def fake_eval(*, skill_id, dataset, output, adapter_dir):
        output.mkdir(parents=True, exist_ok=True)
        summary = {
            "hypothesis": None,
            "modes": {
                "base": {"count": 1, "fuzzy_score": 0.25},
                "single-skill": {"count": 1, "fuzzy_score": 1.0},
            },
            "tasks": {
                "debugging": {
                    "base": {"count": 1, "fuzzy_score": 0.25},
                    "single-skill": {"count": 1, "fuzzy_score": 1.0},
                }
            },
        }
        path = output / "summary.json"
        path.write_text(json.dumps(summary) + "\n")
        return path

    monkeypatch.setattr(packaging, "_train_generic_skill_to_run_directory", fake_train)
    monkeypatch.setattr(packaging, "_evaluate_generic_skill_adapter", fake_eval)

    output = tmp_path / "fastapi_contract"
    assert (
        main(
            [
                "train-skill",
                "--skill-id",
                "fastapi_contract",
                "--name",
                "FastAPI Contract Skill",
                "--train-dataset",
                str(train_dataset),
                "--eval-dataset",
                str(eval_dataset),
                "--output",
                str(output),
                "--allowed-task-types",
                "python_generation",
                "debugging",
                "--activation-scope",
                "task",
            ]
        )
        == 0
    )

    skill_manifest = yaml.safe_load((output / "skill.yaml").read_text())
    assert skill_manifest["skill_id"] == "fastapi_contract"
    assert skill_manifest["composition"]["capabilities"]["allowed_task_types"] == [
        "python_generation",
        "debugging",
    ]
    assert main(["validate-skill-package", "--path", str(output)]) == 0

    runtime = tmp_path / "runtime"
    assert (
        main(
            [
                "compose-skills",
                "--skills",
                str(output),
                "--strategy",
                "routed",
                "--output",
                str(runtime),
            ]
        )
        == 0
    )
    assert main(["validate-runtime", "--runtime", str(runtime)]) == 0


def test_product_train_skill_defaults_routing_metadata_for_arbitrary_skill(
    monkeypatch, tmp_path, capsys
):
    import skillcortex.packaging as packaging

    train_dataset = tmp_path / "train.jsonl"
    eval_dataset = tmp_path / "eval.jsonl"
    train_dataset.write_text(
        json.dumps(
            {
                "id": "train-1",
                "task_type": "python_generation",
                "prompt": "Write a FastAPI route.",
                "target": "def build_route():\n    return 42\n",
            }
        )
        + "\n"
    )
    eval_dataset.write_text(
        json.dumps(
            {
                "id": "eval-1",
                "task_type": "python_generation",
                "prompt": "Write a second FastAPI route.",
                "target": "def build_second_route():\n    return 42\n",
            }
        )
        + "\n"
    )

    def fake_train(*, skill_id, train_dataset, run_directory, seed, force):
        adapter_dir = run_directory / "adapters" / skill_id
        adapter_dir.mkdir(parents=True, exist_ok=True)
        shutil_source = Path("artifacts/adapters/python_skill/adapters.safetensors")
        adapter_config = Path("artifacts/adapters/python_skill/adapter_config.json")
        adapter_metadata = json.loads(Path("artifacts/adapters/python_skill/metadata.json").read_text())
        adapter_metadata["adapter"] = skill_id
        adapter_metadata["training_command"] = ["python", "-m", "mlx_lm", "lora"]
        adapter_dir.joinpath("adapters.safetensors").write_bytes(shutil_source.read_bytes())
        adapter_dir.joinpath("adapter_config.json").write_text(adapter_config.read_text())
        adapter_dir.joinpath("metadata.json").write_text(json.dumps(adapter_metadata, indent=2) + "\n")
        return adapter_dir, adapter_metadata

    def fake_eval(*, skill_id, dataset, output, adapter_dir):
        output.mkdir(parents=True, exist_ok=True)
        path = output / "summary.json"
        path.write_text(json.dumps({"hypothesis": None, "modes": {}, "tasks": {}}) + "\n")
        return path

    monkeypatch.setattr(packaging, "_train_generic_skill_to_run_directory", fake_train)
    monkeypatch.setattr(packaging, "_evaluate_generic_skill_adapter", fake_eval)

    output = tmp_path / "fastapi_contract"
    assert (
        main(
            [
                "train-skill",
                "--skill-id",
                "fastapi_contract",
                "--name",
                "FastAPI Contract Skill",
                "--train-dataset",
                str(train_dataset),
                "--eval-dataset",
                str(eval_dataset),
                "--output",
                str(output),
            ]
        )
        == 0
    )

    result = json.loads(capsys.readouterr().out)
    assert result["defaults_applied"] == {
        "allowed_task_types": ["python_generation"],
        "activation_scope": "task",
    }
    assert "default composition metadata applied" in result["warnings"][0]

    skill_manifest = yaml.safe_load((output / "skill.yaml").read_text())
    assert skill_manifest["composition"]["capabilities"]["allowed_task_types"] == [
        "python_generation"
    ]
    assert skill_manifest["composition"]["activation"]["scope"] == "task"


def test_product_train_skill_explicit_routing_metadata_overrides_defaults(
    monkeypatch, tmp_path, capsys
):
    import skillcortex.packaging as packaging

    train_dataset = tmp_path / "train.jsonl"
    eval_dataset = tmp_path / "eval.jsonl"
    train_dataset.write_text(
        json.dumps(
            {
                "id": "train-1",
                "task_type": "debugging",
                "prompt": "Fix a FastAPI route.",
                "target": "def build_route():\n    return 42\n",
            }
        )
        + "\n"
    )
    eval_dataset.write_text(
        json.dumps(
            {
                "id": "eval-1",
                "task_type": "debugging",
                "prompt": "Fix another FastAPI route.",
                "target": "def build_second_route():\n    return 42\n",
            }
        )
        + "\n"
    )

    def fake_train(*, skill_id, train_dataset, run_directory, seed, force):
        adapter_dir = run_directory / "adapters" / skill_id
        adapter_dir.mkdir(parents=True, exist_ok=True)
        shutil_source = Path("artifacts/adapters/python_skill/adapters.safetensors")
        adapter_config = Path("artifacts/adapters/python_skill/adapter_config.json")
        adapter_metadata = json.loads(Path("artifacts/adapters/python_skill/metadata.json").read_text())
        adapter_metadata["adapter"] = skill_id
        adapter_metadata["training_command"] = ["python", "-m", "mlx_lm", "lora"]
        adapter_dir.joinpath("adapters.safetensors").write_bytes(shutil_source.read_bytes())
        adapter_dir.joinpath("adapter_config.json").write_text(adapter_config.read_text())
        adapter_dir.joinpath("metadata.json").write_text(json.dumps(adapter_metadata, indent=2) + "\n")
        return adapter_dir, adapter_metadata

    def fake_eval(*, skill_id, dataset, output, adapter_dir):
        output.mkdir(parents=True, exist_ok=True)
        path = output / "summary.json"
        path.write_text(json.dumps({"hypothesis": None, "modes": {}, "tasks": {}}) + "\n")
        return path

    monkeypatch.setattr(packaging, "_train_generic_skill_to_run_directory", fake_train)
    monkeypatch.setattr(packaging, "_evaluate_generic_skill_adapter", fake_eval)

    output = tmp_path / "fastapi_contract"
    assert (
        main(
            [
                "train-skill",
                "--skill-id",
                "fastapi_contract",
                "--name",
                "FastAPI Contract Skill",
                "--train-dataset",
                str(train_dataset),
                "--eval-dataset",
                str(eval_dataset),
                "--output",
                str(output),
                "--allowed-task-types",
                "debugging",
                "--activation-scope",
                "semantic_family",
                "--semantic-families",
                "fastapi_contract",
            ]
        )
        == 0
    )

    result = json.loads(capsys.readouterr().out)
    assert "defaults_applied" not in result
    skill_manifest = yaml.safe_load((output / "skill.yaml").read_text())
    assert skill_manifest["composition"]["capabilities"]["allowed_task_types"] == [
        "debugging"
    ]
    assert skill_manifest["composition"]["activation"]["scope"] == "semantic_family"
    assert skill_manifest["composition"]["activation"]["semantic_families"] == [
        "fastapi_contract"
    ]


def test_product_train_skill_unknown_positional_skill_has_actionable_message(capsys, tmp_path):
    output = tmp_path / "fastapi_contract"
    assert main(["train-skill", "fastapi_contract", "--output", str(output)]) == 2
    assert "use --skill-id for arbitrary skills" in capsys.readouterr().err


def test_product_train_skill_rejects_invalid_dataset_before_training(
    monkeypatch, tmp_path, capsys
):
    import skillcortex.packaging as packaging

    train_dataset = tmp_path / "train.jsonl"
    eval_dataset = tmp_path / "eval.jsonl"
    train_dataset.write_text(
        json.dumps(
            {
                "id": "train-1",
                "task_type": "python_generation",
                "prompt": "Write a FastAPI route.",
                "target": "!!!!!!!!!!!!!!!!!",
            }
        )
        + "\n"
    )
    eval_dataset.write_text(
        json.dumps(
            {
                "id": "eval-1",
                "task_type": "python_generation",
                "prompt": "Write another FastAPI route.",
                "target": "def build_route():\n    return 42\n" + "x" * 120,
            }
        )
        + "\n"
    )

    called = {"train": False}

    def fake_train(**kwargs):
        called["train"] = True
        raise AssertionError("training should not start")

    monkeypatch.setattr(packaging, "_train_generic_skill_to_run_directory", fake_train)

    output = tmp_path / "fastapi_contract"
    assert (
        main(
            [
                "train-skill",
                "--skill-id",
                "fastapi_contract",
                "--name",
                "FastAPI Contract Skill",
                "--train-dataset",
                str(train_dataset),
                "--eval-dataset",
                str(eval_dataset),
                "--output",
                str(output),
            ]
        )
        == 2
    )
    assert called["train"] is False
    assert "dataset validation failed" in capsys.readouterr().err


def test_generate_dataset_creates_deterministic_fastapi_files(tmp_path):
    output = tmp_path / "datasets" / "fastapi_contract" / "train.jsonl"
    eval_output = tmp_path / "datasets" / "fastapi_contract" / "eval.jsonl"

    assert (
        main(
            [
                "generate-dataset",
                "--skill-id",
                "fastapi_contract",
                "--domain",
                "fastapi",
                "--task-type",
                "python_generation",
                "--num-examples",
                "12",
                "--output",
                str(output),
                "--eval-output",
                str(eval_output),
                "--seed",
                "7",
            ]
        )
        == 0
    )

    mirror_output = tmp_path / "mirror" / "train.jsonl"
    mirror_eval_output = tmp_path / "mirror" / "eval.jsonl"
    assert (
        main(
            [
                "generate-dataset",
                "--skill-id",
                "fastapi_contract",
                "--domain",
                "fastapi",
                "--task-type",
                "python_generation",
                "--num-examples",
                "12",
                "--output",
                str(mirror_output),
                "--eval-output",
                str(mirror_eval_output),
                "--seed",
                "7",
            ]
        )
        == 0
    )

    assert output.read_text() == mirror_output.read_text()
    assert eval_output.read_text() == mirror_eval_output.read_text()

    report = json.loads((output.parent / "dataset-report.json").read_text())
    assert report["status"] == "ok"
    assert report["train"]["counts"]["valid"] == 12
    assert report["eval"]["counts"]["valid"] >= 1
    assert report["coverage"]["missing_features"] == []
    assert sorted(report["coverage"]["required_features"]) == sorted(REQUIRED_FASTAPI_FEATURES)


def test_generate_dataset_uses_beginner_defaults(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    assert (
        main(
            [
                "generate-dataset",
                "--skill-id",
                "fastapi_contract",
                "--domain",
                "fastapi",
            ]
        )
        == 0
    )

    output = tmp_path / "datasets" / "fastapi_contract" / "train.jsonl"
    eval_output = tmp_path / "datasets" / "fastapi_contract" / "eval.jsonl"
    report_output = tmp_path / "datasets" / "fastapi_contract" / "dataset-report.json"
    assert output.exists()
    assert eval_output.exists()
    assert report_output.exists()

    report = json.loads(report_output.read_text())
    assert report["generation"]["seed"] == 42
    assert report["generation"]["task_type"] == "python_generation"
    assert report["generation"]["train_examples"] == 100
    assert report["generation"]["eval_examples"] == 20


def test_generate_dataset_explicit_overrides_are_honored(tmp_path):
    output = tmp_path / "custom" / "train.jsonl"
    eval_output = tmp_path / "custom" / "eval.jsonl"
    report_output = tmp_path / "custom" / "report.json"

    assert (
        main(
            [
                "generate-dataset",
                "--skill-id",
                "fastapi_contract",
                "--domain",
                "fastapi",
                "--task-type",
                "python_generation",
                "--num-examples",
                "12",
                "--output",
                str(output),
                "--eval-output",
                str(eval_output),
                "--seed",
                "99",
                "--report-output",
                str(report_output),
            ]
        )
        == 0
    )

    report = json.loads(report_output.read_text())
    assert report["generation"]["seed"] == 99
    assert report["generation"]["train_examples"] == 12
    assert report["generation"]["eval_examples"] >= 1
    assert output.exists()
    assert eval_output.exists()


def test_validate_dataset_detects_leakage_and_writes_report(tmp_path):
    train_dataset = tmp_path / "train.jsonl"
    eval_dataset = tmp_path / "eval.jsonl"
    row = {
        "id": "train-1",
        "task_type": "python_generation",
        "prompt": "Write FastAPI code for a GET endpoint that returns a response model.",
        "target": "from fastapi import APIRouter\nfrom pydantic import BaseModel\n\nrouter = APIRouter()\n\nclass DemoResponse(BaseModel):\n    status: str\n\n@router.get('/demo', response_model=DemoResponse)\ndef get_demo() -> DemoResponse:\n    return DemoResponse(status='ok')\n",
    }
    train_dataset.write_text(json.dumps(row) + "\n")
    leaked = dict(row)
    leaked["id"] = "eval-1"
    eval_dataset.write_text(json.dumps(leaked) + "\n")
    report_output = tmp_path / "validation-report.json"

    assert (
        main(
            [
                "validate-dataset",
                str(train_dataset),
                "--eval-dataset",
                str(eval_dataset),
                "--report-output",
                str(report_output),
            ]
        )
        == 2
    )
    report = json.loads(report_output.read_text())
    assert report["cross_split"]["leakage_count"] == 1
    assert report["status"] == "invalid"


def test_validate_dataset_accepts_generated_fastapi_dataset(tmp_path):
    output = tmp_path / "datasets" / "fastapi_contract" / "train.jsonl"
    eval_output = tmp_path / "datasets" / "fastapi_contract" / "eval.jsonl"

    assert (
        main(
            [
                "generate-dataset",
                "--skill-id",
                "fastapi_contract",
                "--domain",
                "fastapi_contract",
                "--task-type",
                "python_generation",
                "--num-examples",
                "10",
                "--output",
                str(output),
                "--eval-output",
                str(eval_output),
                "--seed",
                "3",
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "validate-dataset",
                str(output),
                "--eval-dataset",
                str(eval_output),
            ]
        )
        == 0
    )


def test_package_skill_can_record_custom_composition_metadata(tmp_path):
    output = tmp_path / "external_skill"
    eval_summary = tmp_path / "eval-summary.json"
    eval_summary.write_text(json.dumps({"modes": {}, "tasks": {}}) + "\n")

    assert (
        main(
            [
                "package-skill",
                "--skill-id",
                "external_skill",
                "--name",
                "External Skill",
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
                "--allowed-task-types",
                "debugging",
                "--activation-scope",
                "task",
            ]
        )
        == 0
    )
    skill_manifest = yaml.safe_load((output / "skill.yaml").read_text())
    assert skill_manifest["composition"]["capabilities"]["allowed_task_types"] == [
        "debugging"
    ]
    assert skill_manifest["composition"]["routing"] == {"tasks": {}}


def test_compose_skills_writes_runtime_bundle(tmp_path):
    python_output = tmp_path / "python_skill"
    debugging_output = tmp_path / "debugging_skill"
    eval_summary = tmp_path / "eval-summary.json"
    eval_summary.write_text(json.dumps({"modes": {}, "tasks": {}}) + "\n")

    for skill_id, name, output in (
        ("python_skill", "Python Skill", python_output),
        ("debugging_skill", "Debugging Skill", debugging_output),
    ):
        assert (
            main(
                [
                    "package-skill",
                    "--skill-id",
                    skill_id,
                    "--name",
                    name,
                    "--adapter-dir",
                    f"artifacts/adapters/{skill_id}",
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

    runtime = tmp_path / "runtime"
    assert (
        main(
            [
                "compose-skills",
                "--skills",
                f"{python_output},{debugging_output}",
                "--strategy",
                "routed",
                "--output",
                str(runtime),
            ]
        )
        == 0
    )
    assert (runtime / "composition.yaml").exists()
    assert (runtime / "router_config.json").exists()
    assert (runtime / "active_skills.json").exists()
    assert (runtime / "compatibility_report.json").exists()
    assert (runtime / "budget_report.json").exists()
    assert (runtime / "checksums.json").exists()
    router = json.loads((runtime / "router_config.json").read_text())
    debugging_route = next(
        route for route in router["routes"] if route["route_id"] == "debugging.default"
    )
    assert debugging_route["selected_skills"] == ["debugging_skill", "python_skill"]
    python_route = next(
        route for route in router["routes"] if route["route_id"] == "python_generation.default"
    )
    assert python_route["route_type"] == "base_fallback"
    assert python_route["selected_skills"] == []


def test_compose_skills_defaults_strategy_to_routed(tmp_path):
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

    runtime = tmp_path / "runtime"
    assert (
        main(
            [
                "compose-skills",
                "--skills",
                str(output),
                "--output",
                str(runtime),
            ]
        )
        == 0
    )
    composition = yaml.safe_load((runtime / "composition.yaml").read_text())
    assert composition["strategy"] == "routed"


def test_official_composer_routes_match_validated_alternating_behavior(tmp_path):
    packages = {}
    eval_summary = tmp_path / "eval-summary.json"
    eval_summary.write_text(json.dumps({"modes": {}, "tasks": {}}) + "\n")

    specs = (
        ("python_skill", "Python Skill", "artifacts/adapters/python_skill"),
        ("debugging_skill", "Debugging Skill", "artifacts/adapters/debugging_skill"),
        ("test_generation_skill", "Test Generation Skill", "artifacts/adapters/test_generation_skill"),
        (
            "alternating_skill",
            "Alternating Skill",
            "artifacts/experiments/failure-born-skill/alternating_skill/seed-11/adapters/alternating_skill",
        ),
    )
    for skill_id, name, adapter_dir in specs:
        output = tmp_path / skill_id
        packages[skill_id] = output
        assert (
            main(
                [
                    "package-skill",
                    "--skill-id",
                    skill_id,
                    "--name",
                    name,
                    "--adapter-dir",
                    adapter_dir,
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

    runtime = tmp_path / "runtime-alternating"
    assert (
        main(
            [
                "compose-skills",
                "--skills",
                ",".join(str(packages[skill_id]) for skill_id in specs and packages),
                "--strategy",
                "routed",
                "--output",
                str(runtime),
            ]
        )
        == 0
    )
    router = json.loads((runtime / "router_config.json").read_text())
    debugging_default = next(
        route for route in router["routes"] if route["route_id"] == "debugging.default"
    )
    test_default = next(
        route for route in router["routes"] if route["route_id"] == "test_generation.default"
    )
    debugging_alternating = next(
        route for route in router["routes"] if route["route_id"] == "debugging.alternating"
    )
    test_alternating = next(
        route for route in router["routes"] if route["route_id"] == "test_generation.alternating"
    )
    assert debugging_default["selected_skills"] == ["debugging_skill", "python_skill"]
    assert test_default["selected_skills"] == ["python_skill", "test_generation_skill"]
    assert debugging_alternating["selected_skills"] == [
        "debugging_skill",
        "python_skill",
        "alternating_skill",
    ]
    assert test_alternating["selected_skills"] == [
        "python_skill",
        "test_generation_skill",
        "alternating_skill",
    ]


def test_compose_skills_can_attach_optional_registry_enrichment_without_override(tmp_path):
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
    registry = tmp_path / "registry.json"
    registry.write_text(
        json.dumps(
            {
                "skills": [
                    {
                        "skill_name": "python_skill",
                        "status": "core",
                        "origin": "seed_skill",
                        "router": "skillcortex_router_v1",
                        "activation_scope": "protected_router",
                        "allowed_task_types": ["debugging", "test_generation", "python_generation"],
                    }
                ]
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    runtime = tmp_path / "runtime"
    assert (
        main(
            [
                "compose-skills",
                "--skills",
                str(output),
                "--strategy",
                "routed",
                "--registry",
                str(registry),
                "--output",
                str(runtime),
            ]
        )
        == 0
    )
    compatibility = json.loads((runtime / "compatibility_report.json").read_text())
    composition = yaml.safe_load((runtime / "composition.yaml").read_text())
    assert compatibility["optional_enrichment_used"] is True
    assert compatibility["registry_enrichment"]["source_of_truth"] == "package"
    assert compatibility["registry_enrichment"]["override_applied"] is False
    assert compatibility["warnings"] == [
        "registry enrichment differs from package metadata for python_skill allowed_task_types"
    ]
    python_skill = next(
        item for item in composition["skills"] if item["skill_id"] == "python_skill"
    )
    assert python_skill["composition"]["capabilities"]["allowed_task_types"] == [
        "debugging",
        "test_generation",
    ]


def test_compose_skills_rejects_legacy_package_without_composition_metadata(tmp_path):
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
    skill_manifest = yaml.safe_load((output / "skill.yaml").read_text())
    metadata = json.loads((output / "metadata.json").read_text())
    skill_manifest.pop("composition")
    metadata.pop("composition")
    (output / "skill.yaml").write_text(yaml.safe_dump(skill_manifest, sort_keys=False))
    metadata["checksums"]["skill.yaml"] = hashlib.sha256(
        (output / "skill.yaml").read_bytes()
    ).hexdigest()
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")

    runtime = tmp_path / "runtime"
    assert (
        main(
            [
                "compose-skills",
                "--skills",
                str(output),
                "--strategy",
                "routed",
                "--output",
                str(runtime),
            ]
        )
        == 2
    )
