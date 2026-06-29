from __future__ import annotations

from ...agent import WRITE_MODES
from ...contracts import TASK_TYPES
from ..common import parser_kwargs


def add_doctor_parser(commands) -> None:
    doctor = commands.add_parser(
        "doctor",
        **parser_kwargs(
            "Report platform, backend, and workspace diagnostics for the Composer-first product path.",
            "slmcortex doctor\nslmcortex doctor --workspace /tmp/slmcortex-app",
            summary="Composer: inspect packaged-app readiness and workspace layout.",
        ),
    )
    doctor.add_argument("--workspace")
    doctor.add_argument("--export-support-bundle", action="store_true")
    doctor.add_argument("--support-bundle-path")


def add_provision_backend_parser(commands) -> None:
    provision = commands.add_parser(
        "provision-backend",
        **parser_kwargs(
            "Install optional runtime backend dependencies without changing the base Composer install.",
            "slmcortex provision-backend --backend mlx --dry-run\nslmcortex provision-backend --backend gguf",
            summary="Composer: install optional backend dependencies on demand.",
        ),
    )
    provision.add_argument("--backend", required=True, choices=("mlx", "gguf"))
    provision.add_argument("--workspace")
    provision.add_argument("--dry-run", action="store_true")


def add_composer_app_parser(commands) -> None:
    app = commands.add_parser(
        "composer-app",
        **parser_kwargs(
            "Run the guided Composer App workflow with onboarding, folder scan, compose, and run or export outcomes.",
            "slmcortex composer-app --folder . --task \"Create a FastAPI endpoint with request validation\"\n"
            "slmcortex composer-app --workspace /tmp/slmcortex-app --folder . --outcome export_bundle --export-logs",
            summary="Composer: run the guided app workflow for the common folder-to-runtime path.",
        ),
    )
    app.add_argument("--folder", required=True)
    app.add_argument("--workspace")
    app.add_argument("--slms-dir", dest="slms_dir")
    app.add_argument("--task")
    app.add_argument("--runtime-name")
    app.add_argument("--outcome", choices=("local_run", "export_bundle"), default="local_run")
    app.add_argument("--run-target", choices=("compatibility_server", "inference", "agent_flow"), default="compatibility_server")
    app.add_argument("--prompt")
    app.add_argument("--export-descriptor")
    app.add_argument("--export-logs", action="store_true")
    app.add_argument("--allow-base", action="store_true")
    app.add_argument("--overwrite", action="store_true")
    app.add_argument("--host", default="127.0.0.1")
    app.add_argument("--port", type=int, default=8000)
    app.add_argument("--writes", "--write-mode", dest="writes", choices=WRITE_MODES, default="confirm")
    app.add_argument("--test-command")
    app.add_argument("--trace-out")
    app.add_argument("--dry-run", action="store_true")


def add_compose_from_folder_parser(commands) -> None:
    compose = commands.add_parser(
        "compose-folder",
        **parser_kwargs(
            "Compose a runtime from a local folder using the packaged app workspace contract.",
            "slmcortex compose-folder --folder . --task \"Create a FastAPI endpoint with request validation\"\n"
            "slmcortex compose-folder --folder . --workspace /tmp/slmcortex-app --task \"Fix the failing Python test\" --export-descriptor /tmp/slmcortex-app/exports/repo.json",
            summary="Composer: scan a folder, select packages, compose, and validate.",
        ),
    )
    compose.add_argument("--folder", required=True)
    compose.add_argument("--task", required=True)
    compose.add_argument("--workspace")
    compose.add_argument("--slms-dir", dest="slms_dir")
    compose.add_argument("--runtime-name")
    compose.add_argument("--export-descriptor")
    compose.add_argument("--allow-base", action="store_true")
    compose.add_argument("--overwrite", action="store_true")


def add_validate_runtime_parser(commands) -> None:
    validate_runtime = commands.add_parser(
        "validate-runtime",
        **parser_kwargs(
            "Validate a composed runtime bundle before inference or serving.",
            "slmcortex validate-runtime --runtime /tmp/slmcortex-demo/runtime",
            summary="Composer: verify a runtime bundle before use.",
        ),
    )
    validate_runtime.add_argument("--runtime", required=True)


def add_compose_slms_parser(commands) -> None:
    compose = commands.add_parser(
        "compose-slms",
        **parser_kwargs(
            "Compose validated slm packages into a deterministic runtime bundle.",
            "slmcortex compose-slms --slms /tmp/slmcortex-demo/python_slm,/tmp/slmcortex-demo/debugging_slm --output /tmp/slmcortex-demo/runtime",
            summary="Composer: compose selected packages into one runtime bundle.",
        ),
    )
    compose.add_argument("--slms", required=True)
    compose.add_argument("--strategy", choices=("routed",), default="routed")
    compose.add_argument("--output", required=True)
    compose.add_argument("--registry")
    compose.add_argument("--force", action="store_true")
    compose.add_argument("--dry-run", action="store_true")


def add_route_parser(commands) -> None:
    route = commands.add_parser(
        "route",
        **parser_kwargs(
            "Route a task against discovered slm packages without loading adapters.",
            "slmcortex route --slms-dir slms --repo . --task \"Create a FastAPI endpoint\" --explain",
            summary="Composer: preview which packages match a folder and task.",
        ),
    )
    route.add_argument("--slms-dir", required=True, dest="slms_dir")
    route.add_argument("--repo", required=True)
    route.add_argument("--task", required=True)
    route.add_argument("--base-model")
    route.add_argument("--explain", action="store_true")


def add_compose_from_route_parser(commands) -> None:
    compose = commands.add_parser(
        "compose-from-route",
        **parser_kwargs(
            "Route a task and compose selected slm packages into a runtime bundle.",
            "slmcortex compose-from-route --slms-dir slms --repo . --task \"Create a FastAPI endpoint\" --runtime-out runtime/generated",
            summary="Composer: route a task and write a runtime bundle.",
        ),
    )
    compose.add_argument("--slms-dir", required=True, dest="slms_dir")
    compose.add_argument("--repo", required=True)
    compose.add_argument("--task", required=True)
    compose.add_argument("--runtime-out", required=True)
    compose.add_argument("--explain", action="store_true")
    compose.add_argument("--allow-base", action="store_true")
    compose.add_argument("--overwrite", action="store_true")


def add_infer_parser(commands) -> None:
    infer = commands.add_parser(
        "infer",
        **parser_kwargs(
            "Run local inference against a Slm Cortex runtime bundle.",
            "slmcortex infer --runtime /tmp/slmcortex-demo/runtime --prompt \"Fix this Python traceback\" --dry-run\n"
            "slmcortex infer --runtime /tmp/slmcortex-demo/runtime --request-file tests/fixtures/slmcortex_demo/request.json --dry-run",
            summary="Composer: run or dry-run inference against a runtime.",
        ),
    )
    infer.add_argument("--runtime")
    infer.add_argument("--slms-dir", dest="slms_dir")
    infer.add_argument("--allow-remote-loras", action="store_true")
    infer.add_argument("--lora-cache-dir")
    infer.add_argument("--prompt")
    infer.add_argument("--request-file")
    infer.add_argument("--system")
    infer.add_argument("--task-type", choices=TASK_TYPES)
    infer.add_argument("--semantic-family")
    infer.add_argument("--slm-override")
    infer.add_argument("--max-tokens", type=int)
    infer.add_argument("--temperature", type=float)
    infer.add_argument("--dry-run", action="store_true")


def add_serve_parser(commands) -> None:
    serve = commands.add_parser(
        "serve",
        **parser_kwargs(
            "Start the minimal OpenAI-compatible server for a runtime bundle or SLM directory.",
            "slmcortex serve --runtime /tmp/slmcortex-demo/runtime --host 127.0.0.1 --port 8000\n"
            "slmcortex serve --slms-dir slms --host 127.0.0.1 --port 8000 --dry-run\n"
            "slmcortex serve --runtime /tmp/slmcortex-demo/runtime --dry-run",
            summary="Composer: expose a runtime through the local compatibility API.",
        ),
    )
    serve.add_argument("--runtime")
    serve.add_argument("--slms-dir", dest="slms_dir")
    serve.add_argument("--allow-remote-loras", action="store_true")
    serve.add_argument("--lora-cache-dir")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--dry-run", action="store_true")
