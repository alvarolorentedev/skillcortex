import json
import hashlib
from pathlib import Path

import pytest
import yaml

from slmcortex.cli import main
from slmcortex.dataset_factory import REQUIRED_FASTAPI_FEATURES


def test_slmcortex_cli_alias_supports_dry_run():
    assert main(["train-slm", "python_slm", "--output", "/tmp/slmcortex-dry-run", "--dry-run"]) == 0


def test_package_slm_and_validate_package(tmp_path):
    output = tmp_path / "python_slm"
    eval_summary = tmp_path / "eval-summary.json"
    eval_summary.write_text(
        json.dumps(
            {
                "hypothesis": "inconclusive",
                "modes": {"single-slm": {"count": 1, "fuzzy_score": 1.0}},
                "tasks": {"python_generation": {"single-slm": {"count": 1}}},
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
                "package-slm",
                "--slm-id",
                "python_slm",
                "--name",
                "Python Slm",
                "--adapter-dir",
                "artifacts/adapters/python_slm",
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
                "General Python generation slm.",
            ]
        )
        == 0
    )
    assert (output / "slm.yaml").exists()
    assert (output / "metadata.json").exists()
    assert (output / "training_config.json").exists()
    assert (output / "eval.json").exists()
    assert (output / "README.md").exists()
    assert (output / "examples.jsonl").exists()
    assert (output / "adapter" / "adapters.safetensors").exists()
    metadata = json.loads((output / "metadata.json").read_text())
    slm_manifest = yaml.safe_load((output / "slm.yaml").read_text())
    assert metadata["checksums"]["README.md"]
    assert metadata["protected_inputs"]["all_unchanged"] is True
    assert slm_manifest["composition"]["capabilities"]["allowed_task_types"] == [
        "debugging",
        "test_generation",
    ]

    assert main(["validate-slm-package", "--path", str(output)]) == 0


def test_package_slm_dry_run_does_not_write_output(tmp_path):
    output = tmp_path / "python_slm"
    eval_summary = tmp_path / "eval-summary.json"
    eval_summary.write_text(json.dumps({"modes": {}}) + "\n")

    assert (
        main(
            [
                "package-slm",
                "--slm-id",
                "python_slm",
                "--name",
                "Python Slm",
                "--adapter-dir",
                "artifacts/adapters/python_slm",
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
    output = tmp_path / "python_slm"
    eval_summary = tmp_path / "eval-summary.json"
    eval_summary.write_text(json.dumps({"modes": {}, "tasks": {}}) + "\n")

    assert (
        main(
            [
                "package-slm",
                "--slm-id",
                "python_slm",
                "--name",
                "Python Slm",
                "--adapter-dir",
                "artifacts/adapters/python_slm",
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

    assert main(["validate-slm-package", "--path", str(output)]) == 2


def test_product_train_slm_creates_isolated_run_and_package(monkeypatch, tmp_path):
    protected_adapter = Path("artifacts/adapters/python_slm/adapters.safetensors")
    before = hashlib.sha256(protected_adapter.read_bytes()).hexdigest()

    import slmcortex.packaging as packaging

    def fake_train(*, slm, train_dataset, run_directory, seed, force):
        adapter_dir = run_directory / "adapters" / slm
        adapter_dir.mkdir(parents=True, exist_ok=True)
        shutil_source = Path("artifacts/adapters/python_slm/adapters.safetensors")
        adapter_config = Path("artifacts/adapters/python_slm/adapter_config.json")
        adapter_metadata = json.loads(Path("artifacts/adapters/python_slm/metadata.json").read_text())
        adapter_dir.joinpath("adapters.safetensors").write_bytes(shutil_source.read_bytes())
        adapter_dir.joinpath("adapter_config.json").write_text(adapter_config.read_text())
        adapter_metadata["training_command"] = ["python", "-m", "mlx_lm", "lora"]
        adapter_dir.joinpath("metadata.json").write_text(json.dumps(adapter_metadata, indent=2) + "\n")
        return adapter_dir, adapter_metadata

    def fake_eval(*, slm, dataset, output, adapter_root):
        output.mkdir(parents=True, exist_ok=True)
        summary = {
            "hypothesis": None,
            "modes": {
                "base": {"count": 1, "fuzzy_score": 0.0},
                "single-slm": {"count": 1, "fuzzy_score": 1.0},
            },
            "tasks": {"python_generation": {"single-slm": {"count": 1, "fuzzy_score": 1.0}}},
        }
        path = output / "summary.json"
        path.write_text(json.dumps(summary) + "\n")
        return path

    monkeypatch.setattr(packaging, "_train_slm_to_run_directory", fake_train)
    monkeypatch.setattr(packaging, "_evaluate_slm_adapter", fake_eval)

    output = tmp_path / "product-python-slm"
    assert (
        main(
            [
                "train-slm",
                "python_slm",
                "--output",
                str(output),
                "--force",
            ]
        )
        == 0
    )
    assert (output / "slm.yaml").exists()
    assert (output / "adapter" / "adapters.safetensors").exists()
    assert (output.parent / f".{output.name}.run").exists()
    after = hashlib.sha256(protected_adapter.read_bytes()).hexdigest()
    assert before == after
    assert main(["validate-slm-package", "--path", str(output)]) == 0


def test_product_train_slm_accepts_arbitrary_slm_id_and_composes(monkeypatch, tmp_path):
    import slmcortex.packaging as packaging

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

    def fake_train(*, slm_id, train_dataset, run_directory, seed, force):
        adapter_dir = run_directory / "adapters" / slm_id
        adapter_dir.mkdir(parents=True, exist_ok=True)
        shutil_source = Path("artifacts/adapters/python_slm/adapters.safetensors")
        adapter_config = Path("artifacts/adapters/python_slm/adapter_config.json")
        adapter_metadata = json.loads(Path("artifacts/adapters/python_slm/metadata.json").read_text())
        adapter_metadata["adapter"] = slm_id
        adapter_metadata["training_command"] = ["python", "-m", "mlx_lm", "lora"]
        adapter_dir.joinpath("adapters.safetensors").write_bytes(shutil_source.read_bytes())
        adapter_dir.joinpath("adapter_config.json").write_text(adapter_config.read_text())
        adapter_dir.joinpath("metadata.json").write_text(json.dumps(adapter_metadata, indent=2) + "\n")
        return adapter_dir, adapter_metadata

    def fake_eval(*, slm_id, dataset, output, adapter_dir):
        output.mkdir(parents=True, exist_ok=True)
        summary = {
            "hypothesis": None,
            "modes": {
                "base": {"count": 1, "fuzzy_score": 0.25},
                "single-slm": {"count": 1, "fuzzy_score": 1.0},
            },
            "tasks": {
                "debugging": {
                    "base": {"count": 1, "fuzzy_score": 0.25},
                    "single-slm": {"count": 1, "fuzzy_score": 1.0},
                }
            },
        }
        path = output / "summary.json"
        path.write_text(json.dumps(summary) + "\n")
        return path

    monkeypatch.setattr(packaging, "_train_generic_slm_to_run_directory", fake_train)
    monkeypatch.setattr(packaging, "_evaluate_generic_slm_adapter", fake_eval)

    output = tmp_path / "fastapi_contract"
    assert (
        main(
            [
                "train-slm",
                "--slm-id",
                "fastapi_contract",
                "--name",
                "FastAPI Contract Slm",
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

    slm_manifest = yaml.safe_load((output / "slm.yaml").read_text())
    assert slm_manifest["slm_id"] == "fastapi_contract"
    assert slm_manifest["composition"]["capabilities"]["allowed_task_types"] == [
        "python_generation",
        "debugging",
    ]
    assert main(["validate-slm-package", "--path", str(output)]) == 0

    runtime = tmp_path / "runtime"
    assert (
        main(
            [
                "compose-slms",
                "--slms",
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


def test_product_train_slm_defaults_routing_metadata_for_arbitrary_slm(
    monkeypatch, tmp_path, capsys
):
    import slmcortex.packaging as packaging

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

    def fake_train(*, slm_id, train_dataset, run_directory, seed, force):
        adapter_dir = run_directory / "adapters" / slm_id
        adapter_dir.mkdir(parents=True, exist_ok=True)
        shutil_source = Path("artifacts/adapters/python_slm/adapters.safetensors")
        adapter_config = Path("artifacts/adapters/python_slm/adapter_config.json")
        adapter_metadata = json.loads(Path("artifacts/adapters/python_slm/metadata.json").read_text())
        adapter_metadata["adapter"] = slm_id
        adapter_metadata["training_command"] = ["python", "-m", "mlx_lm", "lora"]
        adapter_dir.joinpath("adapters.safetensors").write_bytes(shutil_source.read_bytes())
        adapter_dir.joinpath("adapter_config.json").write_text(adapter_config.read_text())
        adapter_dir.joinpath("metadata.json").write_text(json.dumps(adapter_metadata, indent=2) + "\n")
        return adapter_dir, adapter_metadata

    def fake_eval(*, slm_id, dataset, output, adapter_dir):
        output.mkdir(parents=True, exist_ok=True)
        path = output / "summary.json"
        path.write_text(json.dumps({"hypothesis": None, "modes": {}, "tasks": {}}) + "\n")
        return path

    monkeypatch.setattr(packaging, "_train_generic_slm_to_run_directory", fake_train)
    monkeypatch.setattr(packaging, "_evaluate_generic_slm_adapter", fake_eval)

    output = tmp_path / "fastapi_contract"
    assert (
        main(
            [
                "train-slm",
                "--slm-id",
                "fastapi_contract",
                "--name",
                "FastAPI Contract Slm",
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

    slm_manifest = yaml.safe_load((output / "slm.yaml").read_text())
    assert slm_manifest["composition"]["capabilities"]["allowed_task_types"] == [
        "python_generation"
    ]
    assert slm_manifest["composition"]["activation"]["scope"] == "task"


def test_product_train_slm_explicit_routing_metadata_overrides_defaults(
    monkeypatch, tmp_path, capsys
):
    import slmcortex.packaging as packaging

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

    def fake_train(*, slm_id, train_dataset, run_directory, seed, force):
        adapter_dir = run_directory / "adapters" / slm_id
        adapter_dir.mkdir(parents=True, exist_ok=True)
        shutil_source = Path("artifacts/adapters/python_slm/adapters.safetensors")
        adapter_config = Path("artifacts/adapters/python_slm/adapter_config.json")
        adapter_metadata = json.loads(Path("artifacts/adapters/python_slm/metadata.json").read_text())
        adapter_metadata["adapter"] = slm_id
        adapter_metadata["training_command"] = ["python", "-m", "mlx_lm", "lora"]
        adapter_dir.joinpath("adapters.safetensors").write_bytes(shutil_source.read_bytes())
        adapter_dir.joinpath("adapter_config.json").write_text(adapter_config.read_text())
        adapter_dir.joinpath("metadata.json").write_text(json.dumps(adapter_metadata, indent=2) + "\n")
        return adapter_dir, adapter_metadata

    def fake_eval(*, slm_id, dataset, output, adapter_dir):
        output.mkdir(parents=True, exist_ok=True)
        path = output / "summary.json"
        path.write_text(json.dumps({"hypothesis": None, "modes": {}, "tasks": {}}) + "\n")
        return path

    monkeypatch.setattr(packaging, "_train_generic_slm_to_run_directory", fake_train)
    monkeypatch.setattr(packaging, "_evaluate_generic_slm_adapter", fake_eval)

    output = tmp_path / "fastapi_contract"
    assert (
        main(
            [
                "train-slm",
                "--slm-id",
                "fastapi_contract",
                "--name",
                "FastAPI Contract Slm",
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
    slm_manifest = yaml.safe_load((output / "slm.yaml").read_text())
    assert slm_manifest["composition"]["capabilities"]["allowed_task_types"] == [
        "debugging"
    ]
    assert slm_manifest["composition"]["activation"]["scope"] == "semantic_family"
    assert slm_manifest["composition"]["activation"]["semantic_families"] == [
        "fastapi_contract"
    ]


def test_product_train_slm_unknown_positional_slm_has_actionable_message(capsys, tmp_path):
    output = tmp_path / "fastapi_contract"
    assert main(["train-slm", "fastapi_contract", "--output", str(output)]) == 2
    assert "use --slm-id for arbitrary slms" in capsys.readouterr().err


def test_product_train_slm_rejects_invalid_dataset_before_training(
    monkeypatch, tmp_path, capsys
):
    import slmcortex.packaging as packaging

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

    monkeypatch.setattr(packaging, "_train_generic_slm_to_run_directory", fake_train)

    output = tmp_path / "fastapi_contract"
    assert (
        main(
            [
                "train-slm",
                "--slm-id",
                "fastapi_contract",
                "--name",
                "FastAPI Contract Slm",
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
                "--slm-id",
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
                "--slm-id",
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
                "--slm-id",
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
                "--slm-id",
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
                "--slm-id",
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


def test_package_slm_can_record_custom_composition_metadata(tmp_path):
    output = tmp_path / "external_slm"
    eval_summary = tmp_path / "eval-summary.json"
    eval_summary.write_text(json.dumps({"modes": {}, "tasks": {}}) + "\n")

    assert (
        main(
            [
                "package-slm",
                "--slm-id",
                "external_slm",
                "--name",
                "External Slm",
                "--adapter-dir",
                "artifacts/adapters/python_slm",
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
    slm_manifest = yaml.safe_load((output / "slm.yaml").read_text())
    assert slm_manifest["composition"]["capabilities"]["allowed_task_types"] == [
        "debugging"
    ]
    assert slm_manifest["composition"]["routing"] == {"tasks": {}}


def test_compose_slms_writes_runtime_bundle(tmp_path):
    python_output = tmp_path / "python_slm"
    debugging_output = tmp_path / "debugging_slm"
    eval_summary = tmp_path / "eval-summary.json"
    eval_summary.write_text(json.dumps({"modes": {}, "tasks": {}}) + "\n")

    for slm_id, name, output in (
        ("python_slm", "Python Slm", python_output),
        ("debugging_slm", "Debugging Slm", debugging_output),
    ):
        assert (
            main(
                [
                    "package-slm",
                    "--slm-id",
                    slm_id,
                    "--name",
                    name,
                    "--adapter-dir",
                    f"artifacts/adapters/{slm_id}",
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
                "compose-slms",
                "--slms",
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
    assert (runtime / "active_slms.json").exists()
    assert (runtime / "compatibility_report.json").exists()
    assert (runtime / "budget_report.json").exists()
    assert (runtime / "checksums.json").exists()
    router = json.loads((runtime / "router_config.json").read_text())
    debugging_route = next(
        route for route in router["routes"] if route["route_id"] == "debugging.default"
    )
    assert debugging_route["selected_slms"] == ["debugging_slm", "python_slm"]
    python_route = next(
        route for route in router["routes"] if route["route_id"] == "python_generation.default"
    )
    assert python_route["route_type"] == "base_fallback"
    assert python_route["selected_slms"] == []


def test_compose_slms_defaults_strategy_to_routed(tmp_path):
    output = tmp_path / "python_slm"
    eval_summary = tmp_path / "eval-summary.json"
    eval_summary.write_text(json.dumps({"modes": {}, "tasks": {}}) + "\n")

    assert (
        main(
            [
                "package-slm",
                "--slm-id",
                "python_slm",
                "--name",
                "Python Slm",
                "--adapter-dir",
                "artifacts/adapters/python_slm",
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
                "compose-slms",
                "--slms",
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
        ("python_slm", "Python Slm", "artifacts/adapters/python_slm"),
        ("debugging_slm", "Debugging Slm", "artifacts/adapters/debugging_slm"),
        ("test_generation_slm", "Test Generation Slm", "artifacts/adapters/test_generation_slm"),
        (
            "alternating_slm",
            "Alternating Slm",
            "artifacts/governance-fixtures/alternating_slm/seed-11/adapters/alternating_slm",
        ),
    )
    for slm_id, name, adapter_dir in specs:
        output = tmp_path / slm_id
        packages[slm_id] = output
        assert (
            main(
                [
                    "package-slm",
                    "--slm-id",
                    slm_id,
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
                "compose-slms",
                "--slms",
                ",".join(str(packages[slm_id]) for slm_id in specs and packages),
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
    assert debugging_default["selected_slms"] == ["debugging_slm", "python_slm"]
    assert test_default["selected_slms"] == ["python_slm", "test_generation_slm"]
    assert debugging_alternating["selected_slms"] == [
        "debugging_slm",
        "python_slm",
        "alternating_slm",
    ]
    assert test_alternating["selected_slms"] == [
        "python_slm",
        "test_generation_slm",
        "alternating_slm",
    ]


def test_compose_slms_can_attach_optional_registry_enrichment_without_override(tmp_path):
    output = tmp_path / "python_slm"
    eval_summary = tmp_path / "eval-summary.json"
    eval_summary.write_text(json.dumps({"modes": {}, "tasks": {}}) + "\n")

    assert (
        main(
            [
                "package-slm",
                "--slm-id",
                "python_slm",
                "--name",
                "Python Slm",
                "--adapter-dir",
                "artifacts/adapters/python_slm",
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
                "slms": [
                    {
                        "slm_name": "python_slm",
                        "status": "core",
                        "origin": "seed_slm",
                        "router": "slmcortex_router_v1",
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
                "compose-slms",
                "--slms",
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
        "registry enrichment differs from package metadata for python_slm allowed_task_types"
    ]
    python_slm = next(
        item for item in composition["slms"] if item["slm_id"] == "python_slm"
    )
    assert python_slm["composition"]["capabilities"]["allowed_task_types"] == [
        "debugging",
        "test_generation",
    ]


def test_compose_slms_rejects_legacy_package_without_composition_metadata(tmp_path):
    output = tmp_path / "python_slm"
    eval_summary = tmp_path / "eval-summary.json"
    eval_summary.write_text(json.dumps({"modes": {}, "tasks": {}}) + "\n")

    assert (
        main(
            [
                "package-slm",
                "--slm-id",
                "python_slm",
                "--name",
                "Python Slm",
                "--adapter-dir",
                "artifacts/adapters/python_slm",
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
    slm_manifest = yaml.safe_load((output / "slm.yaml").read_text())
    metadata = json.loads((output / "metadata.json").read_text())
    slm_manifest.pop("composition")
    metadata.pop("composition")
    (output / "slm.yaml").write_text(yaml.safe_dump(slm_manifest, sort_keys=False))
    metadata["checksums"]["slm.yaml"] = hashlib.sha256(
        (output / "slm.yaml").read_bytes()
    ).hexdigest()
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")

    runtime = tmp_path / "runtime"
    assert (
        main(
            [
                "compose-slms",
                "--slms",
                str(output),
                "--strategy",
                "routed",
                "--output",
                str(runtime),
            ]
        )
        == 2
    )
