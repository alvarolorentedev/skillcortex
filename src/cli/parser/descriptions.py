from __future__ import annotations

from textwrap import dedent


FACTORY_COMMAND_DESCRIPTION = dedent(
    """
    Advanced Factory mode exposes dataset generation, validation, training,
    packaging, and import workflows without changing the default Composer path.

    Prerequisites:
    - dataset generation and validation work in the base install
    - training workflows require optional local training dependencies
    - imported or trained packages remain compatible with Composer discovery
    """
).strip()


ROOT_EXAMPLES = dedent(
    """
    slmcortex doctor
    slmcortex compose-folder --folder . --task "Create a FastAPI endpoint with request validation"
    slmcortex factory doctor
    slmcortex factory package-slm --slm-id python_slm --name "Python Slm" --adapter-dir artifacts/adapters/python_slm --train-dataset tests/fixtures/slmcortex_demo/train.jsonl --eval-dataset tests/fixtures/slmcortex_demo/eval.jsonl --eval-summary tests/fixtures/slmcortex_demo/eval-summary.json --output /tmp/slmcortex-demo/python_slm
    slmcortex compose-slms --slms /tmp/slmcortex-demo/python_slm,/tmp/slmcortex-demo/debugging_slm --strategy routed --output /tmp/slmcortex-demo/runtime
    slmcortex compose-from-route --slms-dir slms --repo . --task "Create a FastAPI endpoint" --runtime-out /tmp/slmcortex-demo/runtime
    slmcortex validate-runtime --runtime /tmp/slmcortex-demo/runtime
    slmcortex infer --runtime /tmp/slmcortex-demo/runtime --prompt "Fix this Python traceback" --dry-run
    slmcortex agent run --runtime /tmp/slmcortex-demo/runtime --repo /tmp/slmcortex-demo/toy-repo
    """
).strip()
