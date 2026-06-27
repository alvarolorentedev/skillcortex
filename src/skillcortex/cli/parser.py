from __future__ import annotations

import argparse
from textwrap import dedent

from ..agent import WRITE_MODES
from ..contracts import TASK_TYPES
from ..dataset_factory import DEFAULT_DATASET_SEED
from ..datasets import DEFAULT_MIN_TARGET_LENGTH
from .common import COMPOSITION_SCOPES, parser_kwargs


def build_parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        prog="skillcortex",
        **parser_kwargs(
            "Package, compose, validate, and run Skill Cortex runtime bundles.",
            dedent(
                """
                skillcortex package-skill --skill-id python_skill --name \"Python Skill\" --adapter-dir artifacts/adapters/python_skill --train-dataset tests/fixtures/skillcortex_demo/train.jsonl --eval-dataset tests/fixtures/skillcortex_demo/eval.jsonl --eval-summary tests/fixtures/skillcortex_demo/eval-summary.json --output /tmp/skillcortex-demo/python_skill
                skillcortex compose-skills --skills /tmp/skillcortex-demo/python_skill,/tmp/skillcortex-demo/debugging_skill --strategy routed --output /tmp/skillcortex-demo/runtime
                skillcortex validate-runtime --runtime /tmp/skillcortex-demo/runtime
                skillcortex infer --runtime /tmp/skillcortex-demo/runtime --prompt \"Fix this Python traceback\" --dry-run
                skillcortex agent run --runtime /tmp/skillcortex-demo/runtime --repo /tmp/skillcortex-demo/toy-repo --task \"Fix the failing answer implementation.\" --dry-run
                """
            ).strip(),
        ),
    )
    commands = root.add_subparsers(dest="command", required=True, title="product commands")
    _add_generate_dataset_parser(commands)
    _add_validate_dataset_parser(commands)
    _add_train_skill_parser(commands)
    _add_package_skill_parser(commands)
    _add_validate_skill_package_parser(commands)
    _add_validate_runtime_parser(commands)
    _add_compose_skills_parser(commands)
    _add_infer_parser(commands)
    _add_serve_parser(commands)
    _add_agent_parser(commands)
    return root


def _add_generate_dataset_parser(commands) -> None:
    generate = commands.add_parser(
        "generate-dataset",
        **parser_kwargs(
            "Generate a deterministic train/eval JSONL dataset for product train-skill.",
            "skillcortex generate-dataset --skill-id fastapi_contract --domain fastapi\n"
            "skillcortex generate-dataset --skill-id fastapi_contract --domain fastapi --task-type python_generation --num-examples 120 --output custom/train.jsonl --eval-output custom/eval.jsonl --seed 99",
        ),
    )
    generate.add_argument("--skill-id", required=True)
    generate.add_argument("--domain", required=True)
    generate.add_argument("--task-type", default="python_generation", choices=TASK_TYPES)
    generate.add_argument("--num-examples", default=100, type=int)
    generate.add_argument("--output")
    generate.add_argument("--eval-output")
    generate.add_argument("--eval-size", type=int)
    generate.add_argument("--seed", type=int, default=DEFAULT_DATASET_SEED)
    generate.add_argument("--report-output")


def _add_validate_dataset_parser(commands) -> None:
    validate_dataset = commands.add_parser(
        "validate-dataset",
        **parser_kwargs(
            "Validate product training datasets and emit a machine-readable report.",
            "skillcortex validate-dataset datasets/fastapi_contract/train.jsonl --eval-dataset datasets/fastapi_contract/eval.jsonl",
        ),
    )
    validate_dataset.add_argument("dataset")
    validate_dataset.add_argument("--eval-dataset")
    validate_dataset.add_argument("--min-target-length", type=int, default=DEFAULT_MIN_TARGET_LENGTH)
    validate_dataset.add_argument("--report-output")


def _add_train_skill_parser(commands) -> None:
    train = commands.add_parser(
        "train-skill",
        **parser_kwargs(
            "Train a LoRA skill from datasets and package it as a Skill Cortex artifact.",
            "skillcortex train-skill --skill-id fastapi_contract --name \"FastAPI Contract Skill\" --train-dataset datasets/fastapi_contract/train.jsonl --eval-dataset datasets/fastapi_contract/eval.jsonl --output skills/fastapi_contract\n"
            "skillcortex train-skill python_skill --output skills/python_skill_run --force",
        ),
    )
    train.add_argument("skill", nargs="?")
    train.add_argument("--skill-id")
    train.add_argument("--output", required=True)
    train.add_argument("--train-dataset", default="data/train.jsonl")
    train.add_argument("--eval-dataset", default="data/eval.jsonl")
    train.add_argument("--name")
    train.add_argument("--version", default="0.1.0")
    train.add_argument("--description")
    train.add_argument("--examples")
    train.add_argument("--allowed-task-types", nargs="+", choices=TASK_TYPES)
    train.add_argument("--activation-scope", choices=COMPOSITION_SCOPES)
    train.add_argument("--semantic-families", nargs="+")
    train.add_argument("--compatible-skills", nargs="+")
    train.add_argument("--incompatible-skills", nargs="+")
    train.add_argument("--seed", type=int)
    train.add_argument("--force", action="store_true")
    train.add_argument("--dry-run", action="store_true")


def _add_package_skill_parser(commands) -> None:
    package = commands.add_parser(
        "package-skill",
        **parser_kwargs(
            "Package an existing LoRA adapter into a self-describing skill artifact.",
            "skillcortex package-skill --skill-id python_skill --name \"Python Skill\" --adapter-dir artifacts/adapters/python_skill --train-dataset tests/fixtures/skillcortex_demo/train.jsonl --eval-dataset tests/fixtures/skillcortex_demo/eval.jsonl --eval-summary tests/fixtures/skillcortex_demo/eval-summary.json --output /tmp/skillcortex-demo/python_skill\n"
            "skillcortex package-skill --skill-id debugging_skill --name \"Debugging Skill\" --adapter-dir artifacts/adapters/debugging_skill --train-dataset tests/fixtures/skillcortex_demo/train.jsonl --eval-dataset tests/fixtures/skillcortex_demo/eval.jsonl --eval-summary tests/fixtures/skillcortex_demo/eval-summary.json --output /tmp/skillcortex-demo/debugging_skill",
        ),
    )
    package.add_argument("--skill-id", required=True)
    package.add_argument("--name", required=True)
    package.add_argument("--adapter-dir", required=True)
    package.add_argument("--output", required=True)
    package.add_argument("--train-dataset", required=True)
    package.add_argument("--eval-dataset", required=True)
    package.add_argument("--eval-summary", required=True)
    package.add_argument("--version", default="0.1.0")
    package.add_argument("--description")
    package.add_argument("--examples")
    package.add_argument("--allowed-task-types", nargs="+", choices=TASK_TYPES)
    package.add_argument("--activation-scope", choices=COMPOSITION_SCOPES)
    package.add_argument("--semantic-families", nargs="+")
    package.add_argument("--compatible-skills", nargs="+")
    package.add_argument("--incompatible-skills", nargs="+")
    package.add_argument("--force", action="store_true")
    package.add_argument("--dry-run", action="store_true")


def _add_validate_skill_package_parser(commands) -> None:
    validate = commands.add_parser(
        "validate-skill-package",
        **parser_kwargs(
            "Validate a packaged skill artifact and its recorded fingerprints.",
            "skillcortex validate-skill-package --path /tmp/skillcortex-demo/python_skill",
        ),
    )
    validate.add_argument("--path", required=True)


def _add_validate_runtime_parser(commands) -> None:
    validate_runtime = commands.add_parser(
        "validate-runtime",
        **parser_kwargs(
            "Validate a composed runtime bundle before inference or serving.",
            "skillcortex validate-runtime --runtime /tmp/skillcortex-demo/runtime",
        ),
    )
    validate_runtime.add_argument("--runtime", required=True)


def _add_compose_skills_parser(commands) -> None:
    compose = commands.add_parser(
        "compose-skills",
        **parser_kwargs(
            "Compose validated skill packages into a deterministic runtime bundle.",
            "skillcortex compose-skills --skills /tmp/skillcortex-demo/python_skill,/tmp/skillcortex-demo/debugging_skill --output /tmp/skillcortex-demo/runtime",
        ),
    )
    compose.add_argument("--skills", required=True)
    compose.add_argument("--strategy", choices=("routed",), default="routed")
    compose.add_argument("--output", required=True)
    compose.add_argument("--registry")
    compose.add_argument("--force", action="store_true")
    compose.add_argument("--dry-run", action="store_true")


def _add_infer_parser(commands) -> None:
    infer = commands.add_parser(
        "infer",
        **parser_kwargs(
            "Run local inference against a Skill Cortex runtime bundle.",
            "skillcortex infer --runtime /tmp/skillcortex-demo/runtime --prompt \"Fix this Python traceback\" --dry-run\n"
            "skillcortex infer --runtime /tmp/skillcortex-demo/runtime --request-file tests/fixtures/skillcortex_demo/request.json --dry-run",
        ),
    )
    infer.add_argument("--runtime", required=True)
    infer.add_argument("--prompt")
    infer.add_argument("--request-file")
    infer.add_argument("--system")
    infer.add_argument("--task-type", choices=TASK_TYPES)
    infer.add_argument("--semantic-family")
    infer.add_argument("--skill-override")
    infer.add_argument("--max-tokens", type=int)
    infer.add_argument("--temperature", type=float)
    infer.add_argument("--dry-run", action="store_true")


def _add_serve_parser(commands) -> None:
    serve = commands.add_parser(
        "serve",
        **parser_kwargs(
            "Start the minimal OpenAI-compatible server for a runtime bundle.",
            "skillcortex serve --runtime /tmp/skillcortex-demo/runtime --host 127.0.0.1 --port 8000\n"
            "skillcortex serve --runtime /tmp/skillcortex-demo/runtime --dry-run",
        ),
    )
    serve.add_argument("--runtime", required=True)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--dry-run", action="store_true")


def _add_agent_parser(commands) -> None:
    agent = commands.add_parser(
        "agent",
        **parser_kwargs(
            "Run the bounded local agent on top of a runtime bundle.",
            "skillcortex agent run --runtime /tmp/skillcortex-demo/runtime --repo /tmp/skillcortex-demo/toy-repo --task \"Fix the failing answer implementation.\" --dry-run\n"
            "skillcortex agent run --runtime /tmp/skillcortex-demo/runtime --repo /tmp/skillcortex-demo/toy-repo --task \"Fix the failing answer implementation.\" --write-mode on --test-command \"pytest -q\"",
        ),
    )
    agent_commands = agent.add_subparsers(dest="agent_command", required=True)
    agent_run = agent_commands.add_parser(
        "run",
        **parser_kwargs(
            "Inspect a local repository, propose a change, and optionally validate it.",
            "skillcortex agent run --runtime /tmp/skillcortex-demo/runtime --repo /tmp/skillcortex-demo/toy-repo --task \"Fix the failing answer implementation.\" --dry-run\n"
            "skillcortex agent run --runtime /tmp/skillcortex-demo/runtime --repo /tmp/skillcortex-demo/toy-repo --task \"Fix the failing answer implementation.\" --write-mode confirm --trace-out /tmp/skillcortex-demo/agent-trace.json",
        ),
    )
    agent_run.add_argument("--runtime", required=True)
    agent_run.add_argument("--repo", required=True)
    agent_run.add_argument(
        "--task",
        action="append",
        help="Optional. Repeat to preload tasks. If omitted, tasks are read from stdin or prompted interactively.",
    )
    agent_run.add_argument("--writes", "--write-mode", dest="writes", choices=WRITE_MODES, default="confirm")
    agent_run.add_argument("--test-command")
    agent_run.add_argument("--trace-out")
    agent_run.add_argument("--dry-run", action="store_true")
