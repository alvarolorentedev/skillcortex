import argparse
import json
import sys
from pathlib import Path

from skill_lattice_coder.cli import main as _main
from skill_lattice_coder.schemas import SKILLS

from .packaging import package_skill, train_skill_package, validate_skill_package


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
    package.add_argument("--force", action="store_true")
    package.add_argument("--dry-run", action="store_true")

    validate = commands.add_parser("validate-skill-package")
    validate.add_argument("--path", required=True)
    return root


def main(argv: list[str] | None = None) -> int:
    arguments = list(argv or [])
    product_commands = {"package-skill", "validate-skill-package"}
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
                force=parsed.force,
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
