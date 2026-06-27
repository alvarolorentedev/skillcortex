import argparse
import json
import sys
from pathlib import Path
from textwrap import dedent

from skill_lattice_coder.cli import main as _main
from skill_lattice_coder.schemas import SKILLS, TASK_TYPES

from .composer import compose_skill_packages
from .packaging import package_skill, train_skill_package, validate_skill_package
from .agent import WRITE_MODES, run_agent
from .runtime import SkillRuntime, load_chat_request, serve_runtime, validate_runtime_bundle


COMPOSITION_SCOPES = ("task", "semantic_family")


def _parser_kwargs(description: str, examples: str | None = None) -> dict:
    kwargs = {
        "description": description,
        "formatter_class": argparse.RawDescriptionHelpFormatter,
    }
    if examples:
        kwargs["epilog"] = f"Examples:\n{examples}"
    return kwargs


def _csv_paths(value: str) -> list[Path]:
    return [Path(item.strip()) for item in value.split(",") if item.strip()]


def _infer_payload(parsed: argparse.Namespace) -> dict:
    if bool(parsed.prompt) == bool(parsed.request_file):
        raise ValueError("exactly one of --prompt or --request-file is required")
    if parsed.request_file:
        return load_chat_request(Path(parsed.request_file))
    payload = {
        "messages": ([{"role": "system", "content": parsed.system}] if parsed.system else [])
        + [{"role": "user", "content": parsed.prompt}],
        "task_type": parsed.task_type,
        "semantic_family": parsed.semantic_family,
        "skill_override": parsed.skill_override,
        "max_tokens": parsed.max_tokens,
        "temperature": parsed.temperature,
    }
    return load_chat_request_payload(payload)


def load_chat_request_payload(payload: dict) -> dict:
    from .runtime import normalize_chat_request

    return normalize_chat_request(payload)


def _package_composition(parsed: argparse.Namespace) -> dict | None:
    if not any(
        (
            parsed.allowed_task_types,
            parsed.activation_scope,
            parsed.semantic_families,
            parsed.compatible_skills,
            parsed.incompatible_skills,
        )
    ):
        return None
    if not parsed.allowed_task_types:
        raise ValueError("--allowed-task-types is required when composition metadata is provided")
    if not parsed.activation_scope:
        raise ValueError("--activation-scope is required when composition metadata is provided")
    return {
        "capabilities": {
            "allowed_task_types": list(parsed.allowed_task_types),
        },
        "activation": {
            "default_route_type": "adapter",
            "scope": parsed.activation_scope,
            "semantic_families": list(parsed.semantic_families or []),
        },
        "compatibility": {
            "compatible_skills": list(parsed.compatible_skills or []),
            "incompatible_skills": list(parsed.incompatible_skills or []),
        },
        "routing": {
            "tasks": {},
        },
    }


def _train_skill_composition(parsed: argparse.Namespace) -> tuple[dict | None, dict[str, object]]:
    if not parsed.skill_id:
        return _package_composition(parsed), {}

    defaults_applied: dict[str, object] = {}
    allowed_task_types = list(parsed.allowed_task_types or [])
    activation_scope = parsed.activation_scope
    if not allowed_task_types:
        allowed_task_types = ["python_generation"]
        defaults_applied["allowed_task_types"] = list(allowed_task_types)
    if activation_scope is None:
        activation_scope = "task"
        defaults_applied["activation_scope"] = activation_scope
    return {
        "capabilities": {
            "allowed_task_types": allowed_task_types,
        },
        "activation": {
            "default_route_type": "adapter",
            "scope": activation_scope,
            "semantic_families": list(parsed.semantic_families or []),
        },
        "compatibility": {
            "compatible_skills": list(parsed.compatible_skills or []),
            "incompatible_skills": list(parsed.incompatible_skills or []),
        },
        "routing": {
            "tasks": {},
        },
    }, defaults_applied


def _resolve_train_skill(parsed: argparse.Namespace) -> tuple[str, str, dict | None, dict[str, object]]:
    preset_skill = parsed.skill
    explicit_skill_id = parsed.skill_id
    if explicit_skill_id and preset_skill and explicit_skill_id != preset_skill:
        raise ValueError("provide either a preset positional skill or a matching --skill-id")
    if explicit_skill_id:
        composition, defaults_applied = _train_skill_composition(parsed)
        return "generic", explicit_skill_id, composition, defaults_applied
    if preset_skill is None:
        raise ValueError("train-skill requires either a preset skill or --skill-id")
    if preset_skill not in SKILLS:
        raise ValueError(
            f"unknown built-in preset: {preset_skill}; use --skill-id for arbitrary skills"
        )
    return "preset", preset_skill, None, {}


def _parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        prog="skillcortex",
        **_parser_kwargs(
            "Package, compose, validate, and run Skill Cortex runtime bundles.",
            dedent(
                """
                skillcortex package-skill --skill-id python_skill --name "Python Skill" --adapter-dir artifacts/adapters/python_skill --train-dataset tests/fixtures/skillcortex_demo/train.jsonl --eval-dataset tests/fixtures/skillcortex_demo/eval.jsonl --eval-summary tests/fixtures/skillcortex_demo/eval-summary.json --output /tmp/skillcortex-demo/python_skill
                skillcortex compose-skills --skills /tmp/skillcortex-demo/python_skill,/tmp/skillcortex-demo/debugging_skill --strategy routed --output /tmp/skillcortex-demo/runtime
                skillcortex validate-runtime --runtime /tmp/skillcortex-demo/runtime
                skillcortex infer --runtime /tmp/skillcortex-demo/runtime --prompt "Fix this Python traceback" --dry-run
                skillcortex agent run --runtime /tmp/skillcortex-demo/runtime --repo /tmp/skillcortex-demo/toy-repo --task "Fix the failing answer implementation." --dry-run
                """
            ).strip(),
        ),
    )
    commands = root.add_subparsers(dest="command", required=True, title="product commands")

    train = commands.add_parser(
        "train-skill",
        **_parser_kwargs(
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

    package = commands.add_parser(
        "package-skill",
        **_parser_kwargs(
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

    validate = commands.add_parser(
        "validate-skill-package",
        **_parser_kwargs(
            "Validate a packaged skill artifact and its recorded fingerprints.",
            "skillcortex validate-skill-package --path /tmp/skillcortex-demo/python_skill",
        ),
    )
    validate.add_argument("--path", required=True)

    validate_runtime = commands.add_parser(
        "validate-runtime",
        **_parser_kwargs(
            "Validate a composed runtime bundle before inference or serving.",
            "skillcortex validate-runtime --runtime /tmp/skillcortex-demo/runtime",
        ),
    )
    validate_runtime.add_argument("--runtime", required=True)

    compose = commands.add_parser(
        "compose-skills",
        **_parser_kwargs(
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

    infer = commands.add_parser(
        "infer",
        **_parser_kwargs(
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

    serve = commands.add_parser(
        "serve",
        **_parser_kwargs(
            "Start the minimal OpenAI-compatible server for a runtime bundle.",
            "skillcortex serve --runtime /tmp/skillcortex-demo/runtime --host 127.0.0.1 --port 8000\n"
            "skillcortex serve --runtime /tmp/skillcortex-demo/runtime --dry-run",
        ),
    )
    serve.add_argument("--runtime", required=True)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--dry-run", action="store_true")

    agent = commands.add_parser(
        "agent",
        **_parser_kwargs(
            "Run the bounded local agent on top of a runtime bundle.",
            "skillcortex agent run --runtime /tmp/skillcortex-demo/runtime --repo /tmp/skillcortex-demo/toy-repo --task \"Fix the failing answer implementation.\" --dry-run\n"
            "skillcortex agent run --runtime /tmp/skillcortex-demo/runtime --repo /tmp/skillcortex-demo/toy-repo --task \"Fix the failing answer implementation.\" --writes on --test-command \"pytest -q\"",
        ),
    )
    agent_commands = agent.add_subparsers(dest="agent_command", required=True)
    agent_run = agent_commands.add_parser(
        "run",
        **_parser_kwargs(
            "Inspect a local repository, propose a change, and optionally validate it.",
            "skillcortex agent run --runtime /tmp/skillcortex-demo/runtime --repo /tmp/skillcortex-demo/toy-repo --task \"Fix the failing answer implementation.\" --dry-run\n"
            "skillcortex agent run --runtime /tmp/skillcortex-demo/runtime --repo /tmp/skillcortex-demo/toy-repo --task \"Fix the failing answer implementation.\" --writes confirm --trace-out /tmp/skillcortex-demo/agent-trace.json",
        ),
    )
    agent_run.add_argument("--runtime", required=True)
    agent_run.add_argument("--repo", required=True)
    agent_run.add_argument("--task", required=True)
    agent_run.add_argument("--writes", choices=WRITE_MODES, default="confirm")
    agent_run.add_argument("--test-command")
    agent_run.add_argument("--trace-out")
    agent_run.add_argument("--dry-run", action="store_true")
    return root


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    product_commands = {
        "package-skill",
        "validate-skill-package",
        "validate-runtime",
        "compose-skills",
        "infer",
        "serve",
        "agent",
    }
    is_product_train = bool(
        arguments
        and arguments[0] == "train-skill"
        and ("--output" in arguments or "-h" in arguments or "--help" in arguments)
    )
    if not arguments or arguments[0] in {"-h", "--help"}:
        try:
            _parser().parse_args(arguments or ["--help"])
        except SystemExit as error:
            return int(error.code or 0)
        return 0
    if arguments[0] not in product_commands and not is_product_train:
        return _main(argv, prog="skillcortex")

    parsed = _parser().parse_args(arguments)
    try:
        if parsed.command == "train-skill":
            mode, skill_id, composition, defaults_applied = _resolve_train_skill(parsed)
            result = train_skill_package(
                skill=skill_id,
                mode=mode,
                output=Path(parsed.output),
                train_dataset=Path(parsed.train_dataset),
                eval_dataset=Path(parsed.eval_dataset),
                name=parsed.name,
                version=parsed.version,
                description=parsed.description,
                examples=Path(parsed.examples) if parsed.examples else None,
                composition=composition,
                seed=parsed.seed,
                force=parsed.force,
                dry_run=parsed.dry_run,
            )
            if defaults_applied:
                result["defaults_applied"] = defaults_applied
                result["warnings"] = [
                    "default composition metadata applied for arbitrary train-skill"
                ]
        elif parsed.command == "package-skill":
            result = package_skill(
                skill_id=parsed.skill_id,
                name=parsed.name,
                adapter_dir=Path(parsed.adapter_dir),
                output=Path(parsed.output),
                train_dataset=Path(parsed.train_dataset),
                eval_dataset=Path(parsed.eval_dataset),
                eval_summary=Path(parsed.eval_summary),
                version=parsed.version,
                description=parsed.description,
                examples=Path(parsed.examples) if parsed.examples else None,
                composition=_package_composition(parsed),
                force=parsed.force,
                dry_run=parsed.dry_run,
            )
        elif parsed.command == "compose-skills":
            result = compose_skill_packages(
                skills=_csv_paths(parsed.skills),
                strategy=parsed.strategy,
                output=Path(parsed.output),
                registry=Path(parsed.registry) if parsed.registry else None,
                force=parsed.force,
                dry_run=parsed.dry_run,
            )
        elif parsed.command == "validate-runtime":
            result = validate_runtime_bundle(Path(parsed.runtime))
        elif parsed.command == "infer":
            payload = _infer_payload(parsed)
            result = SkillRuntime.load(Path(parsed.runtime)).infer(
                messages=payload["messages"],
                task_type=payload.get("task_type"),
                semantic_family=payload.get("semantic_family"),
                skill_override=payload.get("skill_override"),
                max_tokens=payload.get("max_tokens"),
                temperature=payload.get("temperature"),
                dry_run=parsed.dry_run,
            )
        elif parsed.command == "serve":
            result = serve_runtime(
                runtime_path=Path(parsed.runtime),
                host=parsed.host,
                port=parsed.port,
                dry_run=parsed.dry_run,
            )
        elif parsed.command == "agent":
            if parsed.agent_command != "run":
                raise ValueError(f"unknown agent command: {parsed.agent_command}")
            result = run_agent(
                runtime_path=Path(parsed.runtime),
                repo=Path(parsed.repo),
                task=parsed.task,
                writes=parsed.writes,
                test_command=parsed.test_command,
                trace_out=Path(parsed.trace_out) if parsed.trace_out else None,
                dry_run=parsed.dry_run,
            )
        else:
            result = validate_skill_package(Path(parsed.path))
        print(json.dumps(result, indent=2))
        return 0
    except (FileNotFoundError, FileExistsError, ValueError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
