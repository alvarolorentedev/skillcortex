import json
import sys

from .handlers import execute_command
from .parser import build_parser


def _collect_agent_tasks(task_args: list[str] | None) -> list[str]:
    if task_args:
        tasks = [item.strip() for item in task_args if item and item.strip()]
        if tasks:
            return tasks


def _stream_agent_tasks() -> object:
    if not sys.stdin.isatty():
        def stdin_provider() -> str | None:
            while True:
                line = sys.stdin.readline()
                if line == "":
                    return None
                value = line.strip()
                if value:
                    return value

        return stdin_provider

    print("Enter agent tasks, one per line. Submit an empty line to start execution.", file=sys.stderr)
    task_count = 0

    def prompt_provider() -> str | None:
        nonlocal task_count
        while True:
            try:
                line = input(f"task {task_count + 1}> ")
            except EOFError as error:
                if task_count:
                    return None
                raise ValueError("at least one task is required") from error
            value = line.strip()
            if not value:
                if task_count:
                    return None
                continue
            task_count += 1
            return value

    return prompt_provider


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments or arguments[0] in {"-h", "--help"}:
        try:
            build_parser().parse_args(arguments or ["--help"])
        except SystemExit as error:
            return int(error.code or 0)
        return 0

    parsed = build_parser().parse_args(arguments)
    try:
        result = execute_command(
            parsed,
            collect_agent_tasks=_collect_agent_tasks,
            stream_agent_tasks=_stream_agent_tasks,
        )
        print(json.dumps(result, indent=2))
        return int(result.get("exit_code", 0))
    except (FileNotFoundError, FileExistsError, ValueError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
