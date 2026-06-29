from __future__ import annotations

from ...agent import WRITE_MODES
from ..common import parser_kwargs


def add_agent_parser(commands) -> None:
    agent = commands.add_parser(
        "agent",
        **parser_kwargs(
            "Run the bounded local agent on top of a runtime bundle.",
            "slmcortex agent run --runtime /tmp/slmcortex-demo/runtime --repo /tmp/slmcortex-demo/toy-repo --task \"Fix the failing answer implementation.\" --dry-run\n"
            "slmcortex agent run --runtime /tmp/slmcortex-demo/runtime --repo /tmp/slmcortex-demo/toy-repo --task \"Fix the failing answer implementation.\" --write-mode on --test-command \"pytest -q\"",
            summary="Composer: run the bounded local agent on a composed runtime.",
        ),
    )
    agent_commands = agent.add_subparsers(dest="agent_command", required=True)
    agent_run = agent_commands.add_parser(
        "run",
        **parser_kwargs(
            "Inspect a local repository, propose a change, and optionally validate it.",
            "slmcortex agent run --runtime /tmp/slmcortex-demo/runtime --repo /tmp/slmcortex-demo/toy-repo --task \"Fix the failing answer implementation.\" --dry-run\n"
            "slmcortex agent run --runtime /tmp/slmcortex-demo/runtime --repo /tmp/slmcortex-demo/toy-repo --task \"Fix the failing answer implementation.\" --write-mode confirm --trace-out /tmp/slmcortex-demo/agent-trace.json",
        ),
    )
    agent_run.add_argument("--runtime")
    agent_run.add_argument("--slms-dir", dest="slms_dir")
    agent_run.add_argument("--repo", required=True)
    agent_run.add_argument(
        "--task",
        action="append",
        help="Optional. Repeat to preload tasks. If omitted, tasks are read from stdin or prompted interactively.",
    )
    agent_run.add_argument("--writes", "--write-mode", dest="writes", choices=WRITE_MODES, default="confirm")
    agent_run.add_argument("--test-command")
    agent_run.add_argument("--trace-out")
    agent_run.add_argument("--compose-runtime-out")
    agent_run.add_argument("--overwrite", action="store_true")
    agent_run.add_argument("--dry-run", action="store_true")
