# Command Reference

This page covers the public CLI surface in one place. Use it as the detailed
follow-up to the [quickstart](quickstart.md).

## Common Conventions

- Use `slmcortex` or `python -m slmcortex`.
- Prefer `--dry-run` when you only want to inspect routing, composition, or
  agent behavior.
- `generate-dataset`, `train-skill`, `package-skill`, and `compose-skills`
  are the packaging pipeline.
- `route`, `compose-from-route`, `infer`, `serve`, and `agent run` are the
  runtime pipeline.
- `train-skill` accepts either a preset positional skill name or an explicit
  `--skill-id`.
- `infer` requires exactly one of `--runtime` or `--skills-dir`.
- `agent run` requires exactly one of `--runtime` or `--skills-dir`.
- `train-plasticity-lora` requires exactly one of `--output` or
  `--publish-dir`.
- Backend selection comes from base config: `backend: auto | mlx | gguf`.
  `auto` resolves to MLX only on macOS Apple Silicon and GGUF elsewhere.
- GGUF requires a `.gguf` runtime model path. MLX is rejected outside macOS
  arm64/aarch64.

## At A Glance

| Command | Reads | Writes | Best For |
| --- | --- | --- | --- |
| `generate-dataset` | skill id, domain | train/eval JSONL, report | bootstrapping training data |
| `validate-dataset` | dataset paths | validation report | checking schema and leakage |
| `train-skill` | datasets, adapter config | skill package | turning datasets into a packaged skill |
| `train-plasticity-lora` | prompt/target JSONL | skill package | on-demand local adapter training |
| `import-lora` | Hugging Face source, datasets | skill package | wrapping an external LoRA |
| `package-skill` | existing adapter, datasets, eval summary | skill package | converting a trained adapter into a package |
| `validate-skill-package` | packaged skill path | validation result only | verifying package integrity |
| `compose-skills` | skill package paths | runtime bundle | building a deterministic runtime |
| `route` | skills dir, repo, task | routing result only | understanding which skills match a task |
| `compose-from-route` | skills dir, repo, task | runtime bundle | one-shot routing plus composition |
| `validate-runtime` | runtime bundle | validation result only | checking a bundle before use |
| `infer` | runtime or skills dir, prompt or request file | inference result or dry-run route | running model-backed or dry-run inference |
| `serve` | runtime bundle | server process | exposing the OpenAI-compatible API |
| `agent run` | runtime or skills dir, repo, task | trace, diffs, optional writes | bounded repo work on a local checkout |

## `generate-dataset`

Generate deterministic train and eval JSONL datasets.

Required flags:

- `--skill-id`
- `--domain`

Important flags:

- `--task-type` defaults to `python_generation`
- `--num-examples` defaults to `100`
- `--seed` defaults to the built-in dataset seed
- `--output` and `--eval-output` override the default
  `datasets/<skill_id>/...` paths
- `--eval-size` controls the eval split size
- `--report-output` writes a machine-readable report

Writes:

- a train JSONL file
- an eval JSONL file
- an optional dataset report JSON

Example:

```bash
slmcortex generate-dataset \
  --skill-id fastapi_contract \
  --domain fastapi \
  --report-output /tmp/fastapi_contract-report.json
```

Use this when you want a reproducible dataset bundle for `train-skill`.

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

## `train-skill`

Train a LoRA skill from datasets and package it as a Slm Cortex artifact.

You can call it in two ways:

- preset mode: `slmcortex train-skill python_skill --output ...`
- generic mode: `slmcortex train-skill --skill-id fastapi_contract ...`

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
- `--version` defaults to `0.1.0`
- `--description`
- `--examples`
- `--allowed-task-types`
- `--activation-scope`
- `--semantic-families`
- `--compatible-skills`
- `--incompatible-skills`
- `--seed`
- `--force`
- `--dry-run`

Writes:

- a packaged skill directory
- package metadata, fingerprints, and training artifacts

Example:

```bash
slmcortex train-skill \
  --skill-id fastapi_contract \
  --name "FastAPI Contract Skill" \
  --train-dataset datasets/fastapi_contract/train.jsonl \
  --eval-dataset datasets/fastapi_contract/eval.jsonl \
  --output skills/fastapi_contract
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

- `--skill-id`
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

- a packaged skill directory at `--output`
- or a package under `--publish-dir/<skill-id>`

Example:

```bash
slmcortex train-plasticity-lora \
  --skill-id local_fix \
  --name "Local Fix" \
  --prompt-file data/train.jsonl \
  --publish-dir skills
```

Use this when you want the shortest path from prompt/target data to a reusable
local package.

## `import-lora`

Import a public Hugging Face LoRA into a local Slm Cortex package.

Required flags:

- `--source`
- `--skill-id`
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

- a packaged skill directory
- cached remote downloads when `--cache-dir` is set

Example:

```bash
slmcortex import-lora \
  --source hf://owner/repo \
  --skill-id fastapi_skill \
  --name "FastAPI Skill" \
  --output skills/fastapi_skill \
  --train-dataset data/train.jsonl \
  --eval-dataset data/eval.jsonl
```

Use this when you already have a remote LoRA and only need it wrapped in the
local package contract.

When the resolved backend is GGUF, import converts the downloaded PEFT LoRA to
`adapter/adapter.gguf`; set `gguf_converter` in the selected base config.

## `package-skill`

Package an existing adapter into a self-describing skill artifact.

Required flags:

- `--skill-id`
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
- `--compatible-skills`
- `--incompatible-skills`
- `--force`
- `--dry-run`

Writes:

- a self-describing skill package directory
- package fingerprints and provenance metadata

Example:

```bash
slmcortex package-skill \
  --skill-id python_skill \
  --name "Python Skill" \
  --adapter-dir artifacts/adapters/python_skill \
  --train-dataset tests/fixtures/slmcortex_demo/train.jsonl \
  --eval-dataset tests/fixtures/slmcortex_demo/eval.jsonl \
  --eval-summary tests/fixtures/slmcortex_demo/eval-summary.json \
  --output /tmp/slmcortex-demo/python_skill
```

Use this when the adapter already exists and you only need a package wrapper.

## `validate-skill-package`

Validate a packaged skill artifact and its recorded fingerprints.

Required flags:

- `--path`

Writes:

- validation output only
- no package files are modified

Example:

```bash
slmcortex validate-skill-package --path /tmp/slmcortex-demo/python_skill
```

Use this after packaging and before composition.

## `compose-skills`

Compose validated skill packages into a deterministic runtime bundle.

Required flags:

- `--skills`
- `--output`

Important flags:

- `--strategy` currently supports `routed`
- `--registry` is optional
- `--force`
- `--dry-run`

Writes:

- a runtime bundle directory
- bundle manifest and routed skill metadata

Example:

```bash
slmcortex compose-skills \
  --skills /tmp/slmcortex-demo/python_skill,/tmp/slmcortex-demo/debugging_skill \
  --output /tmp/slmcortex-demo/runtime
```

Use this when you want a runtime that is stable and repeatable from a fixed set
of packaged skills.

## `route`

Route a task against discovered skill packages without loading adapters.

Required flags:

- `--skills-dir`
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
  --skills-dir skills \
  --repo . \
  --task "Create a FastAPI endpoint" \
  --explain
```

Use this when you want to inspect the routing decision before composing a
runtime.

## `compose-from-route`

Route a task and compose the selected skill packages into a runtime bundle.

Required flags:

- `--skills-dir`
- `--repo`
- `--task`
- `--runtime-out`

Optional flags:

- `--explain`
- `--allow-base`
- `--overwrite`

Writes:

- a runtime bundle at `--runtime-out`
- routing metadata for the selected skills

Example:

```bash
slmcortex compose-from-route \
  --skills-dir skills \
  --repo . \
  --task "Create a FastAPI endpoint" \
  --runtime-out runtime/generated
```

Use this when you want one command to select skills and materialize a runtime.

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

- exactly one of `--runtime` or `--skills-dir`
- exactly one of `--prompt` or `--request-file`

Behavior:

- runtime mode loads the composed bundle
- skills-dir mode resolves skills directly and can fetch remote LoRAs if
  allowed
- `--dry-run` returns routing metadata without generating text

Optional flags:

- `--allow-remote-loras`
- `--lora-cache-dir`
- `--system`
- `--task-type`
- `--semantic-family`
- `--skill-override`
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
Use `--skills-dir` when you want routing against discovered skill packages
without building a runtime first.

## `serve`

Start the minimal OpenAI-compatible server for a runtime bundle.

Required flags:

- `--runtime`

Optional flags:

- `--host` defaults to `127.0.0.1`
- `--port` defaults to `8000`
- `--dry-run`

Behavior:

- `--dry-run` validates the serving configuration without starting the server
- non-dry-run mode starts a blocking local HTTP server

Example:

```bash
slmcortex serve --runtime /tmp/slmcortex-demo/runtime --host 127.0.0.1 --port 8000
```

Use `--dry-run` to check the serving configuration without starting the server.
Use the real server mode when you want a drop-in compatibility endpoint.

## `agent`

Agent is the command group for bounded local repository workflows.

Public subcommands:

- `run`

Use `slmcortex agent run ...` for actual execution.

## `agent run`

Run the bounded local agent on top of a runtime bundle.

Required input:

- exactly one of `--runtime` or `--skills-dir`
- `--repo`

Behavior:

- runtime mode runs the agent against an already composed bundle
- skills-dir mode routes and composes first, then runs the agent
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

- `--skills-dir` mode only supports `--dry-run` or `--write-mode confirm`.
- If you omit `--task`, the command reads tasks from stdin or prompts
  interactively.
- `--trace-out` writes the run trace JSON to disk.
- `--write-mode off` keeps the sandbox read-only.
