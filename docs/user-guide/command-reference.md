# Command Reference

This page covers the public CLI surface in one place. Use it as the detailed
follow-up to the [quickstart](quickstart.md).

## Common Conventions

- Use `slmcortex` or `python -m slmcortex`.
- Prefer `--dry-run` when you only want to inspect routing, composition, or
  agent behavior.
- `doctor`, `compose-folder`, `route`, `compose-from-route`, `infer`, `serve`,
  and `agent run` are the Composer-first product path.
- `generate-dataset`, `train-slm`, `package-slm`, and related authoring
  commands are advanced Factory commands.
- `train-slm` accepts either a preset positional slm name or an explicit
  `--slm-id`.
- `infer` requires exactly one of `--runtime` or `--slms-dir`.
- `agent run` requires exactly one of `--runtime` or `--slms-dir`.
- `train-plasticity-lora` requires exactly one of `--output` or
  `--publish-dir`.
- Backend selection comes from base config: `backend: auto | mlx | gguf`.
  `auto` resolves to MLX only on macOS Apple Silicon and GGUF elsewhere.
- GGUF requires a `.gguf` runtime model path. MLX is rejected outside macOS
  arm64/aarch64.

## At A Glance

| Command | Reads | Writes | Best For |
| --- | --- | --- | --- |
| `doctor` | platform, optional deps | diagnostics result | checking packaged-app readiness |
| `compose-folder` | folder, workspace, task | runtime bundle, export descriptor, logs | one-step folder-to-runtime composition |
| `generate-dataset` | slm id, domain | train/eval JSONL, report | bootstrapping training data |
| `validate-dataset` | dataset paths | validation report | checking schema and leakage |
| `train-slm` | datasets, adapter config | slm package | turning datasets into a packaged slm |
| `train-plasticity-lora` | prompt/target JSONL | slm package | on-demand local adapter training |
| `import-lora` | Hugging Face source, datasets | slm package | wrapping an external LoRA |
| `package-slm` | existing adapter, datasets, eval summary | slm package | converting a trained adapter into a package |
| `validate-slm-package` | packaged slm path | validation result only | verifying package integrity |
| `compose-slms` | slm package paths | runtime bundle | building a deterministic runtime |
| `route` | slms dir, repo, task | routing result only | understanding which slms match a task |
| `compose-from-route` | slms dir, repo, task | runtime bundle | one-shot routing plus composition |
| `validate-runtime` | runtime bundle | validation result only | checking a bundle before use |
| `infer` | runtime or slms dir, prompt or request file | inference result or dry-run route | running model-backed or dry-run inference |
| `serve` | runtime bundle | server process | exposing the OpenAI-compatible API |
| `agent run` | runtime or slms dir, repo, task | trace, diffs, optional writes | bounded repo work on a local checkout |

## `doctor`

Report the packaged product contract for the current machine.

Optional flags:

- `--workspace` overrides the default external app workspace root

Writes:

- diagnostics output only

Example:

```bash
slmcortex doctor
slmcortex doctor --workspace /tmp/slmcortex-app
```

Use this first when validating a packaged install or when you need to confirm backend and optional Factory dependency availability.

## `compose-folder`

Compose a runtime from one local folder and the external app workspace contract.

Required flags:

- `--folder`
- `--task`

Optional flags:

- `--workspace`
- `--slms-dir`
- `--runtime-name`
- `--export-descriptor`
- `--allow-base`
- `--overwrite`

Writes:

- a runtime under `runtimes/`
- an optional export descriptor under `exports/`
- a compose log under `logs/`

Example:

```bash
slmcortex compose-folder \
  --workspace /tmp/slmcortex-app \
  --folder /path/to/repo \
  --task "Create a FastAPI endpoint with request validation" \
  --export-descriptor /tmp/slmcortex-app/exports/repo.json
```

Use this as the default packaged-product entry point when you want folder scan, routing, composition, validation, and export metadata in one call.

## `generate-dataset`

Generate deterministic train and eval JSONL datasets.

Required flags:

- `--slm-id`
- `--domain`

Important flags:

- `--task-type` defaults to `python_generation`
- `--num-examples` defaults to `100`
- `--seed` defaults to the built-in dataset seed
- `--output` and `--eval-output` override the default
  `datasets/<slm_id>/...` paths
- `--eval-size` controls the eval split size
- `--report-output` writes a machine-readable report

Writes:

- a train JSONL file
- an eval JSONL file
- an optional dataset report JSON

Example:

```bash
slmcortex generate-dataset \
  --slm-id fastapi_contract \
  --domain fastapi \
  --report-output /tmp/fastapi_contract-report.json
```

Use this when you want a reproducible dataset bundle for `train-slm`.

## `validate-dataset`

Validate one dataset, and optionally check leakage against an eval dataset.

Required input:

- `dataset` positional path to the train JSONL

Important flags:

- `--eval-dataset` checks cross-split leakage
- `--min-target-length` defaults to `24`
- `--report-output` writes the validation report JSON

Writes:

- validation output only
- an optional validation report JSON

Example:

```bash
slmcortex validate-dataset datasets/fastapi_contract/train.jsonl \
  --eval-dataset datasets/fastapi_contract/eval.jsonl
```

Use this before training when you want an explicit quality gate.

## `train-slm`

Train a LoRA slm from datasets and package it as a Slm Cortex artifact.

You can call it in two ways:

- preset mode: `slmcortex train-slm python_slm --output ...`
- generic mode: `slmcortex train-slm --slm-id fastapi_contract ...`

Behavior:

- preset mode uses built-in preset metadata
- generic mode builds default routing metadata when you do not provide it
- dataset validation runs before training
- the command fails early on malformed or leaky datasets

Required flags:

- `--output`

Common flags:

- `--train-dataset` defaults to `data/train.jsonl`
- `--eval-dataset` defaults to `data/eval.jsonl`
- `--name`
- `--version` defaults to `0.1.1`
- `--description`
- `--examples`
- `--allowed-task-types`
- `--activation-scope`
- `--semantic-families`
- `--compatible-slms`
- `--incompatible-slms`
- `--seed`
- `--force`
- `--dry-run`

Writes:

- a packaged slm directory
- package metadata, fingerprints, and training artifacts

Example:

```bash
slmcortex train-slm \
  --slm-id fastapi_contract \
  --name "FastAPI Contract Slm" \
  --train-dataset datasets/fastapi_contract/train.jsonl \
  --eval-dataset datasets/fastapi_contract/eval.jsonl \
  --output slms/fastapi_contract
```

Notes:

- Generic mode applies default routing metadata if you do not provide it.
- Use `--examples` when you want example snippets attached to the package.
- Validation happens before training and fails early on malformed or leaky
  data.
- MLX training writes `adapter/adapters.safetensors`; GGUF training writes
  `adapter/adapter.gguf` after PEFT training and llama.cpp conversion.

## `train-plasticity-lora`

Train an on-demand LoRA from a prompt/target dataset.

Required flags:

- `--slm-id`
- `--name`
- `--prompt-file`

One of:

- `--output`
- `--publish-dir`

Optional flags:

- `--eval-dataset`
- `--version`
- `--description`
- `--seed`
- `--force`
- `--dry-run`

Writes:

- a packaged slm directory at `--output`
- or a package under `--publish-dir/<slm-id>`

Example:

```bash
slmcortex train-plasticity-lora \
  --slm-id local_fix \
  --name "Local Fix" \
  --prompt-file data/train.jsonl \
  --publish-dir slms
```

Use this when you want the shortest path from prompt/target data to a reusable
local package.

## `import-lora`

Import a public Hugging Face LoRA into a local Slm Cortex package.

Required flags:

- `--source`
- `--slm-id`
- `--name`
- `--output`
- `--train-dataset`
- `--eval-dataset`

Optional flags:

- `--cache-dir`
- `--max-download-bytes`
- `--version`
- `--description`
- `--force`

Writes:

- a packaged slm directory
- cached remote downloads when `--cache-dir` is set

Example:

```bash
slmcortex import-lora \
  --source hf://owner/repo \
  --slm-id fastapi_slm \
  --name "FastAPI Slm" \
  --output slms/fastapi_slm \
  --train-dataset data/train.jsonl \
  --eval-dataset data/eval.jsonl
```

Use this when you already have a remote LoRA and only need it wrapped in the
local package contract.

When the resolved backend is GGUF, import converts the downloaded PEFT LoRA to
`adapter/adapter.gguf`; set `gguf_converter` in the selected base config.

## `package-slm`

Package an existing adapter into a self-describing slm artifact.

Required flags:

- `--slm-id`
- `--name`
- `--adapter-dir`
- `--output`
- `--train-dataset`
- `--eval-dataset`
- `--eval-summary`

`--adapter-dir` may contain either MLX weights (`adapters.safetensors`) or GGUF
weights (`adapter.gguf`). Mixed backend packages cannot be composed together.

Optional flags:

- `--version`
- `--description`
- `--examples`
- `--allowed-task-types`
- `--activation-scope`
- `--semantic-families`
- `--compatible-slms`
- `--incompatible-slms`
- `--force`
- `--dry-run`

Writes:

- a self-describing slm package directory
- package fingerprints and provenance metadata

Example:

```bash
slmcortex package-slm \
  --slm-id python_slm \
  --name "Python Slm" \
  --adapter-dir artifacts/adapters/python_slm \
  --train-dataset tests/fixtures/slmcortex_demo/train.jsonl \
  --eval-dataset tests/fixtures/slmcortex_demo/eval.jsonl \
  --eval-summary tests/fixtures/slmcortex_demo/eval-summary.json \
  --output /tmp/slmcortex-demo/python_slm
```

Use this when the adapter already exists and you only need a package wrapper.

## `validate-slm-package`

Validate a packaged slm artifact and its recorded fingerprints.

Required flags:

- `--path`

Writes:

- validation output only
- no package files are modified

Example:

```bash
slmcortex validate-slm-package --path /tmp/slmcortex-demo/python_slm
```

Use this after packaging and before composition.

## `compose-slms`

Compose validated slm packages into a deterministic runtime bundle.

Required flags:

- `--slms`
- `--output`

Important flags:

- `--strategy` currently supports `routed`
- `--registry` is optional
- `--force`
- `--dry-run`

Writes:

- a runtime bundle directory
- bundle manifest and routed slm metadata

Example:

```bash
slmcortex compose-slms \
  --slms /tmp/slmcortex-demo/python_slm,/tmp/slmcortex-demo/debugging_slm \
  --output /tmp/slmcortex-demo/runtime
```

Use this when you want a runtime that is stable and repeatable from a fixed set
of packaged slms.

## `route`

Route a task against discovered slm packages without loading adapters.

Required flags:

- `--slms-dir`
- `--repo`
- `--task`

Optional flags:

- `--base-model`
- `--explain`

Writes:

- routing output only
- no bundle or package artifacts

Example:

```bash
slmcortex route \
  --slms-dir slms \
  --repo . \
  --task "Create a FastAPI endpoint" \
  --explain
```

Use this when you want to inspect the routing decision before composing a
runtime.

## `compose-from-route`

Route a task and compose the selected slm packages into a runtime bundle.

Required flags:

- `--slms-dir`
- `--repo`
- `--task`
- `--runtime-out`

Optional flags:

- `--explain`
- `--allow-base`
- `--overwrite`

Writes:

- a runtime bundle at `--runtime-out`
- routing metadata for the selected slms

Example:

```bash
slmcortex compose-from-route \
  --slms-dir slms \
  --repo . \
  --task "Create a FastAPI endpoint" \
  --runtime-out runtime/generated
```

Use this when you want one command to select slms and materialize a runtime.

## `validate-runtime`

Validate a composed runtime bundle before inference or serving.

Required flags:

- `--runtime`

Writes:

- validation output only
- no runtime files are modified

Example:

```bash
slmcortex validate-runtime --runtime /tmp/slmcortex-demo/runtime
```

Use this after composition and before serving or inference.

## `infer`

Run inference or a dry-run route decision against a runtime bundle.

Required input:

- exactly one of `--runtime` or `--slms-dir`
- exactly one of `--prompt` or `--request-file`

Behavior:

- runtime mode loads the composed bundle
- slms-dir mode resolves slms directly and can fetch remote LoRAs if
  allowed
- `--dry-run` returns routing metadata without generating text

Optional flags:

- `--allow-remote-loras`
- `--lora-cache-dir`
- `--system`
- `--task-type`
- `--semantic-family`
- `--slm-override`
- `--max-tokens`
- `--temperature`
- `--dry-run`

Returns:

- a dry-run route decision, or
- an inference payload with generated text and token counts

Example:

```bash
slmcortex infer \
  --runtime /tmp/slmcortex-demo/runtime \
  --prompt "Fix this Python traceback" \
  --dry-run
```

Use `--request-file` when you already have a chat payload on disk.
Use `--slms-dir` when you want routing against discovered slm packages
without building a runtime first.

## `serve`

Start the minimal OpenAI-compatible server for a runtime bundle or SLM directory.

Required input:

- exactly one of `--runtime` or `--slms-dir`

Optional flags:

- `--allow-remote-loras` when serving from `--slms-dir`
- `--lora-cache-dir` when serving from `--slms-dir`
- `--host` defaults to `127.0.0.1`
- `--port` defaults to `8000`
- `--dry-run`

Behavior:

- runtime mode serves an already composed runtime bundle
- slms-dir mode serves directly from discovered slm packages
- `--dry-run` validates the serving configuration without starting the server
- non-dry-run mode starts a blocking local HTTP server

Example:

```bash
slmcortex serve --runtime /tmp/slmcortex-demo/runtime --host 127.0.0.1 --port 8000
slmcortex serve --slms-dir slms --allow-remote-loras --dry-run
```

Use `--dry-run` to check the serving configuration without starting the server.
Use `--slms-dir` when you want the folder-based runtime path without composing a bundle first.
Use the real server mode when you want a drop-in compatibility endpoint.

## `agent`

Agent is the command group for bounded local repository workflows.

Public subcommands:

- `run`

Use `slmcortex agent run ...` for actual execution.

## `agent run`

Run the bounded local agent on top of a runtime bundle.

Required input:

- exactly one of `--runtime` or `--slms-dir`
- `--repo`

Behavior:

- runtime mode runs the agent against an already composed bundle
- slms-dir mode routes and composes first, then runs the agent
- `--task` can be repeated to preload multiple tasks
- `--dry-run` plans the work without applying changes

Optional flags:

- `--task` can be repeated to preload multiple tasks
- `--writes` or `--write-mode` accepts `off`, `confirm`, or `on`
- `--test-command`
- `--trace-out`
- `--compose-runtime-out`
- `--overwrite`
- `--dry-run`

Writes:

- a trace JSON when `--trace-out` is set
- optional review artifacts in confirm mode
- optional file writes in `--write-mode on`

Example:

```bash
slmcortex agent run \
  --runtime /tmp/slmcortex-demo/runtime \
  --repo /tmp/slmcortex-demo/toy-repo \
  --task "Fix the failing answer implementation." \
  --dry-run
```

Notes:

- `--slms-dir` mode only supports `--dry-run` or `--write-mode confirm`.
- If you omit `--task`, the command reads tasks from stdin or prompts
  interactively.
- `--trace-out` writes the run trace JSON to disk.
- `--write-mode off` keeps the sandbox read-only.
