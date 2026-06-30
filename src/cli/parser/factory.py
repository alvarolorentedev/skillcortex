from __future__ import annotations

import argparse

from ...contracts import TASK_TYPES
from ...dataset_factory import DEFAULT_DATASET_SEED
from ...datasets import DEFAULT_MIN_TARGET_LENGTH
from ..common import COMPOSITION_SCOPES, parser_kwargs
from .descriptions import FACTORY_COMMAND_DESCRIPTION


def factory_parser_kwargs(description: str, examples: str | None = None, *, hidden: bool = False) -> dict:
    kwargs = parser_kwargs(description, examples)
    kwargs["help"] = argparse.SUPPRESS if hidden else description.splitlines()[0]
    return kwargs


def add_factory_parser(commands) -> None:
    factory = commands.add_parser(
        "factory",
        **parser_kwargs(
            FACTORY_COMMAND_DESCRIPTION,
            "slmcortex factory doctor\n"
            "slmcortex factory generate-dataset --slm-id fastapi_contract --domain fastapi\n"
            "slmcortex factory train-slm --slm-id fastapi_contract --name \"FastAPI Contract Slm\" --train-dataset datasets/fastapi_contract/train.jsonl --eval-dataset datasets/fastapi_contract/eval.jsonl --output slms/fastapi_contract",
            summary="Advanced Factory: enter dataset, training, packaging, and import workflows explicitly.",
        ),
    )
    factory.set_defaults(product_mode="factory")
    factory_commands = factory.add_subparsers(dest="factory_command", required=True, title="advanced factory commands")
    from .composer import add_doctor_parser

    add_doctor_parser(factory_commands)
    add_generate_dataset_parser(factory_commands)
    add_validate_dataset_parser(factory_commands)
    add_train_slm_parser(factory_commands)
    add_train_plasticity_lora_parser(factory_commands)
    add_import_lora_parser(factory_commands)
    add_package_slm_parser(factory_commands)
    add_validate_slm_package_parser(factory_commands)


def add_generate_dataset_parser(commands, *, hidden: bool = False) -> None:
    generate = commands.add_parser(
        "generate-dataset",
        **factory_parser_kwargs(
            "Generate a deterministic train/eval JSONL dataset for product train-slm.",
            "slmcortex generate-dataset --slm-id fastapi_contract --domain fastapi\n"
            "slmcortex generate-dataset --slm-id fastapi_contract --domain fastapi --task-type python_generation --num-examples 120 --output custom/train.jsonl --eval-output custom/eval.jsonl --seed 99",
            hidden=hidden,
        ),
    )
    generate.add_argument("--slm-id", required=True)
    generate.add_argument("--domain", required=True)
    generate.add_argument("--task-type", default="python_generation", choices=TASK_TYPES)
    generate.add_argument("--num-examples", default=100, type=int)
    generate.add_argument("--output")
    generate.add_argument("--eval-output")
    generate.add_argument("--eval-size", type=int)
    generate.add_argument("--seed", type=int, default=DEFAULT_DATASET_SEED)
    generate.add_argument("--report-output")


def add_validate_dataset_parser(commands, *, hidden: bool = False) -> None:
    validate_dataset = commands.add_parser(
        "validate-dataset",
        **factory_parser_kwargs(
            "Validate product training datasets and emit a machine-readable report.",
            "slmcortex validate-dataset datasets/fastapi_contract/train.jsonl --eval-dataset datasets/fastapi_contract/eval.jsonl",
            hidden=hidden,
        ),
    )
    validate_dataset.add_argument("dataset")
    validate_dataset.add_argument("--eval-dataset")
    validate_dataset.add_argument("--min-target-length", type=int, default=DEFAULT_MIN_TARGET_LENGTH)
    validate_dataset.add_argument("--report-output")


def add_train_slm_parser(commands, *, hidden: bool = False) -> None:
    train = commands.add_parser(
        "train-slm",
        **factory_parser_kwargs(
            "Train a LoRA slm from datasets and package it as a Slm Cortex artifact.",
            "slmcortex train-slm --slm-id fastapi_contract --name \"FastAPI Contract Slm\" --train-dataset datasets/fastapi_contract/train.jsonl --eval-dataset datasets/fastapi_contract/eval.jsonl --output slms/fastapi_contract\n"
            "slmcortex train-slm python_slm --output slms/python_slm_run --force",
            hidden=hidden,
        ),
    )
    train.add_argument("slm", nargs="?")
    train.add_argument("--slm-id")
    train.add_argument("--output", required=True)
    train.add_argument("--train-dataset", default="data/train.jsonl")
    train.add_argument("--eval-dataset", default="data/eval.jsonl")
    train.add_argument("--name")
    train.add_argument("--version", default="0.1.1")
    train.add_argument("--description")
    train.add_argument("--examples")
    train.add_argument("--allowed-task-types", nargs="+", choices=TASK_TYPES)
    train.add_argument("--activation-scope", choices=COMPOSITION_SCOPES)
    train.add_argument("--semantic-families", nargs="+")
    train.add_argument("--compatible-slms", nargs="+")
    train.add_argument("--incompatible-slms", nargs="+")
    train.add_argument("--seed", type=int)
    train.add_argument("--force", action="store_true")
    train.add_argument("--dry-run", action="store_true")


def add_train_plasticity_lora_parser(commands, *, hidden: bool = False) -> None:
    train = commands.add_parser(
        "train-plasticity-lora",
        **factory_parser_kwargs(
            "Train an explicit on-demand LoRA from a JSONL prompt/target dataset.",
            "slmcortex train-plasticity-lora --slm-id local_fix --name \"Local Fix\" --prompt-file data/train.jsonl --output slms/local_fix --dry-run",
            hidden=hidden,
        ),
    )
    train.add_argument("--slm-id", required=True)
    train.add_argument("--name", required=True)
    train.add_argument("--prompt-file", required=True)
    train.add_argument("--eval-dataset")
    train.add_argument("--output")
    train.add_argument("--publish-dir")
    train.add_argument("--version", default="0.1.1")
    train.add_argument("--description")
    train.add_argument("--seed", type=int)
    train.add_argument("--force", action="store_true")
    train.add_argument("--dry-run", action="store_true")


def add_import_lora_parser(commands, *, hidden: bool = False) -> None:
    import_lora = commands.add_parser(
        "import-lora",
        **factory_parser_kwargs(
            "Import a public Hugging Face LoRA into a local SlmCortex package.",
            "slmcortex import-lora --source hf://owner/repo --slm-id fastapi_slm --name \"FastAPI Slm\" --output slms/fastapi_slm --train-dataset data/train.jsonl --eval-dataset data/eval.jsonl",
            hidden=hidden,
        ),
    )
    import_lora.add_argument("--source", required=True)
    import_lora.add_argument("--slm-id", required=True)
    import_lora.add_argument("--name", required=True)
    import_lora.add_argument("--output", required=True)
    import_lora.add_argument("--train-dataset", required=True)
    import_lora.add_argument("--eval-dataset", required=True)
    import_lora.add_argument("--cache-dir")
    import_lora.add_argument("--max-download-bytes", type=int)
    import_lora.add_argument("--version", default="0.1.1")
    import_lora.add_argument("--description")
    import_lora.add_argument("--force", action="store_true")


def add_package_slm_parser(commands, *, hidden: bool = False) -> None:
    package = commands.add_parser(
        "package-slm",
        **factory_parser_kwargs(
            "Package an existing LoRA adapter into a self-describing slm artifact.",
            "slmcortex package-slm --slm-id python_slm --name \"Python Slm\" --adapter-dir artifacts/adapters/python_slm --train-dataset tests/fixtures/slmcortex_demo/train.jsonl --eval-dataset tests/fixtures/slmcortex_demo/eval.jsonl --eval-summary tests/fixtures/slmcortex_demo/eval-summary.json --output /tmp/slmcortex-demo/python_slm\n"
            "slmcortex package-slm --slm-id debugging_slm --name \"Debugging Slm\" --adapter-dir artifacts/adapters/debugging_slm --train-dataset tests/fixtures/slmcortex_demo/train.jsonl --eval-dataset tests/fixtures/slmcortex_demo/eval.jsonl --eval-summary tests/fixtures/slmcortex_demo/eval-summary.json --output /tmp/slmcortex-demo/debugging_slm",
            hidden=hidden,
        ),
    )
    package.add_argument("--slm-id", required=True)
    package.add_argument("--name", required=True)
    package.add_argument("--adapter-dir", required=True)
    package.add_argument("--output", required=True)
    package.add_argument("--train-dataset", required=True)
    package.add_argument("--eval-dataset", required=True)
    package.add_argument("--eval-summary", required=True)
    package.add_argument("--version", default="0.1.1")
    package.add_argument("--description")
    package.add_argument("--examples")
    package.add_argument("--allowed-task-types", nargs="+", choices=TASK_TYPES)
    package.add_argument("--activation-scope", choices=COMPOSITION_SCOPES)
    package.add_argument("--semantic-families", nargs="+")
    package.add_argument("--compatible-slms", nargs="+")
    package.add_argument("--incompatible-slms", nargs="+")
    package.add_argument("--force", action="store_true")
    package.add_argument("--dry-run", action="store_true")


def add_validate_slm_package_parser(commands, *, hidden: bool = False) -> None:
    validate = commands.add_parser(
        "validate-slm-package",
        **factory_parser_kwargs(
            "Validate a packaged slm artifact and its recorded fingerprints.",
            "slmcortex validate-slm-package --path /tmp/slmcortex-demo/python_slm",
            hidden=hidden,
        ),
    )
    validate.add_argument("--path", required=True)
