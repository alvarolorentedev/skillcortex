import argparse
import json
import sys
from pathlib import Path

from skill_lattice_coder.cli import main as _main
from skill_lattice_coder.schemas import SKILLS, TASK_TYPES

from .composer import compose_skill_packages
from .packaging import package_skill, train_skill_package, validate_skill_package
from .agent import WRITE_MODES, run_agent
from .runtime import SkillRuntime, load_chat_request, serve_runtime, validate_runtime_bundle


COMPOSITION_SCOPES = ("task", "semantic_family")


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


def _parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="skillcortex")
    commands = root.add_subparsers(dest="command", required=True)

    train = commands.add_parser("train-skill")
    train.add_argument("skill", choices=SKILLS)
    train.add_argument("--output", required=True)
    train.add_argument("--train-dataset", default="data/train.jsonl")
    train.add_argument("--eval-dataset", default="data/eval.jsonl")
    train.add_argument("--name")
    train.add_argument("--version", default="0.1.0")
    train.add_argument("--description")
    train.add_argument("--examples")
    train.add_argument("--seed", type=int)
    train.add_argument("--force", action="store_true")
    train.add_argument("--dry-run", action="store_true")

    package = commands.add_parser("package-skill")
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

    validate = commands.add_parser("validate-skill-package")
    validate.add_argument("--path", required=True)

    validate_runtime = commands.add_parser("validate-runtime")
    validate_runtime.add_argument("--runtime", required=True)

    compose = commands.add_parser("compose-skills")
    compose.add_argument("--skills", required=True)
    compose.add_argument("--strategy", choices=("routed",), required=True)
    compose.add_argument("--output", required=True)
    compose.add_argument("--registry")
    compose.add_argument("--force", action="store_true")
    compose.add_argument("--dry-run", action="store_true")

    infer = commands.add_parser("infer")
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

    serve = commands.add_parser("serve")
    serve.add_argument("--runtime", required=True)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--dry-run", action="store_true")

    agent = commands.add_parser("agent")
    agent_commands = agent.add_subparsers(dest="agent_command", required=True)
    agent_run = agent_commands.add_parser("run")
    agent_run.add_argument("--runtime", required=True)
    agent_run.add_argument("--repo", required=True)
    agent_run.add_argument("--task", required=True)
    agent_run.add_argument("--writes", choices=WRITE_MODES, default="confirm")
    agent_run.add_argument("--test-command")
    agent_run.add_argument("--trace-out")
    agent_run.add_argument("--dry-run", action="store_true")
    return root


def main(argv: list[str] | None = None) -> int:
    arguments = list(argv or [])
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
        arguments and arguments[0] == "train-skill" and "--output" in arguments
    )
    if not arguments or (arguments[0] not in product_commands and not is_product_train):
        return _main(argv, prog="skillcortex")

    parsed = _parser().parse_args(arguments)
    try:
        if parsed.command == "train-skill":
            result = train_skill_package(
                skill=parsed.skill,
                output=Path(parsed.output),
                train_dataset=Path(parsed.train_dataset),
                eval_dataset=Path(parsed.eval_dataset),
                name=parsed.name,
                version=parsed.version,
                description=parsed.description,
                examples=Path(parsed.examples) if parsed.examples else None,
                seed=parsed.seed,
                force=parsed.force,
                dry_run=parsed.dry_run,
            )
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
