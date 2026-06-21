import argparse
import json
import sys
from pathlib import Path

from .analysis import (
    analyze_composition,
    analyze_python_regression,
    analyze_router,
)
from .evaluation import evaluate
from .inference import infer
from .schemas import MODES, SKILLS
from .train_generic import train_generic
from .train_skill import train_skill


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="skill-lattice")
    commands = root.add_subparsers(dest="command", required=True)

    skill = commands.add_parser("train-skill")
    skill.add_argument("skill", choices=SKILLS)
    skill.add_argument("--dry-run", action="store_true")
    skill.add_argument("--force", action="store_true")
    skill.add_argument("--seed", type=int)
    skill.add_argument("--adapter-root")

    generic = commands.add_parser("train-generic")
    generic.add_argument("--dry-run", action="store_true")
    generic.add_argument("--force", action="store_true")
    generic.add_argument("--seed", type=int)
    generic.add_argument("--adapter-root")

    inference = commands.add_parser("infer")
    inference.add_argument("--mode", choices=MODES, required=True)
    inference.add_argument("--skill", choices=SKILLS)
    inference.add_argument("--skills", nargs="+", choices=SKILLS)
    inference.add_argument("--prompt", required=True)
    inference.add_argument("--dry-run", action="store_true")
    inference.add_argument("--adapter-root")

    evaluation = commands.add_parser("eval")
    evaluation.add_argument("--dataset", required=True)
    evaluation.add_argument("--output")
    evaluation.add_argument("--dry-run", action="store_true")
    evaluation.add_argument("--adapter-root")

    for name in (
        "analyze-router",
        "analyze-python-regression",
        "analyze-composition",
    ):
        analysis = commands.add_parser(name)
        analysis.add_argument("--experiment", required=True)
    return root


def main(argv: list[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    try:
        if arguments.command == "train-skill":
            result = train_skill(
                arguments.skill,
                dry_run=arguments.dry_run,
                force=arguments.force,
                seed=arguments.seed,
                adapter_root=arguments.adapter_root,
            )
            print(json.dumps(result, indent=2))
        elif arguments.command == "train-generic":
            print(
                json.dumps(
                    train_generic(
                        dry_run=arguments.dry_run,
                        force=arguments.force,
                        seed=arguments.seed,
                        adapter_root=arguments.adapter_root,
                    ),
                    indent=2,
                )
            )
        elif arguments.command == "infer":
            print(
                json.dumps(
                    infer(
                        arguments.mode,
                        arguments.prompt,
                        skill=arguments.skill,
                        skills=arguments.skills,
                        dry_run=arguments.dry_run,
                        adapter_root=arguments.adapter_root,
                    ).to_dict(),
                    indent=2,
                )
            )
        elif arguments.command == "eval":
            output = evaluate(
                arguments.dataset,
                output=arguments.output,
                dry_run=arguments.dry_run,
                adapter_root=arguments.adapter_root,
            )
            print(json.dumps({"output": str(output)}, indent=2))
        else:
            function, stem = {
                "analyze-router": (analyze_router, "router_analysis"),
                "analyze-python-regression": (
                    analyze_python_regression,
                    "python_regression_analysis",
                ),
                "analyze-composition": (
                    analyze_composition,
                    "composition_analysis",
                ),
            }[arguments.command]
            function(arguments.experiment)
            root = Path(arguments.experiment)
            print(
                json.dumps(
                    {
                        "json": str(root / f"{stem}.json"),
                        "markdown": str(root / f"{stem}.md"),
                    },
                    indent=2,
                )
            )
        return 0
    except (FileNotFoundError, FileExistsError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
