from __future__ import annotations

import argparse

from ...shared.product import PRODUCT_MODES
from ..common import parser_kwargs
from .agent import add_agent_parser
from .composer import (
    add_compose_from_folder_parser,
    add_compose_from_route_parser,
    add_compose_slms_parser,
    add_composer_app_parser,
    add_doctor_parser,
    add_infer_parser,
    add_provision_backend_parser,
    add_route_parser,
    add_serve_parser,
    add_validate_runtime_parser,
)
from .descriptions import ROOT_EXAMPLES
from .factory import (
    add_factory_parser,
    add_generate_dataset_parser,
    add_import_lora_parser,
    add_package_slm_parser,
    add_train_plasticity_lora_parser,
    add_train_slm_parser,
    add_validate_dataset_parser,
    add_validate_slm_package_parser,
)


def build_parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        prog="slmcortex",
        **parser_kwargs("Compose, validate, run, and optionally author Slm Cortex packages.", ROOT_EXAMPLES),
    )
    root.add_argument("--product-mode", choices=PRODUCT_MODES, default="composer")
    commands = root.add_subparsers(
        dest="command",
        required=True,
        title="product commands",
        metavar=(
            "{doctor,provision-backend,composer-app,compose-folder,validate-runtime,route,"
            "compose-from-route,infer,serve,agent,factory,compose-slms}"
        ),
    )
    add_doctor_parser(commands)
    add_provision_backend_parser(commands)
    add_composer_app_parser(commands)
    add_compose_from_folder_parser(commands)
    add_validate_runtime_parser(commands)
    add_route_parser(commands)
    add_compose_from_route_parser(commands)
    add_infer_parser(commands)
    add_serve_parser(commands)
    add_agent_parser(commands)
    add_factory_parser(commands)
    add_generate_dataset_parser(commands, hidden=True)
    add_validate_dataset_parser(commands, hidden=True)
    add_train_slm_parser(commands, hidden=True)
    add_train_plasticity_lora_parser(commands, hidden=True)
    add_import_lora_parser(commands, hidden=True)
    add_package_slm_parser(commands, hidden=True)
    add_validate_slm_package_parser(commands, hidden=True)
    add_compose_slms_parser(commands)
    commands._choices_actions = [
        action for action in commands._choices_actions if action.help != argparse.SUPPRESS
    ]
    return root
