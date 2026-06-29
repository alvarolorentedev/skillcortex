# Quickstart

Use this path if you want the first successful end-to-end run with the fewest moving parts.

For the packaged Composer-first contract and installer artifacts, see [Packaged Install](packaged-install.md).

## 1. Install

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install --upgrade pip
pip install -e '.[test]'
```

Install one real-model backend only when you need actual training or inference:

```bash
pip install -e '.[mlx]'   # macOS Apple Silicon
pip install -e '.[gguf]'  # Linux, Windows, or GGUF on any supported OS
```

`backend: auto` uses MLX on macOS arm64/aarch64 and GGUF everywhere else.
GGUF configs must use a `.gguf` runtime model path.

## 2. Inspect the Composer-first workspace contract

```bash
slmcortex doctor
slmcortex compose-folder --help
```

This confirms the packaged workspace layout, backend availability, and the folder-first composition entry point before you touch any advanced Factory commands.

## 3. Run the no-model demo

```bash
python scripts/run_slmcortex_demo.py
```

This exercises the full public flow without loading a real model:

- package two checked-in adapters
- compose them into one runtime bundle
- validate the runtime
- run inference in `--dry-run` mode
- run the bounded agent in `--dry-run` mode

The script prints or writes outputs under a temporary directory and is the fastest way to verify the repo is healthy.

## 4. Run the same flow by hand

If you want the command-by-command version, use the same sequence the demo script wraps:

```bash
DEMO_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/slmcortex-demo.XXXXXX")"

slmcortex package-slm \
  --slm-id python_slm \
  --name "Python Slm" \
  --adapter-dir artifacts/adapters/python_slm \
  --train-dataset tests/fixtures/slmcortex_demo/train.jsonl \
  --eval-dataset tests/fixtures/slmcortex_demo/eval.jsonl \
  --eval-summary tests/fixtures/slmcortex_demo/eval-summary.json \
  --output "$DEMO_ROOT/python_slm"

slmcortex package-slm \
  --slm-id debugging_slm \
  --name "Debugging Slm" \
  --adapter-dir artifacts/adapters/debugging_slm \
  --train-dataset tests/fixtures/slmcortex_demo/train.jsonl \
  --eval-dataset tests/fixtures/slmcortex_demo/eval.jsonl \
  --eval-summary tests/fixtures/slmcortex_demo/eval-summary.json \
  --output "$DEMO_ROOT/debugging_slm"

slmcortex compose-slms \
  --slms "$DEMO_ROOT/python_slm,$DEMO_ROOT/debugging_slm" \
  --output "$DEMO_ROOT/runtime"

slmcortex validate-runtime --runtime "$DEMO_ROOT/runtime"

slmcortex infer \
  --runtime "$DEMO_ROOT/runtime" \
  --prompt "Fix this Python traceback" \
  --dry-run

slmcortex agent run \
  --runtime "$DEMO_ROOT/runtime" \
  --repo /path/to/local/repo \
  --task "Fix the failing answer implementation." \
  --dry-run
```

## 5. Try the built-in smoke checks

The default arbitrary-slm smoke stays no-model:

```bash
python scripts/run_slmcortex_arbitrary_slm_smoke.py
python scripts/run_package_product_smoke.py
```

If you explicitly want the slower local training path:

```bash
python scripts/run_slmcortex_arbitrary_slm_smoke.py --real-training
```

For GGUF training/import conversion, set `gguf_converter` in the selected base
config to llama.cpp's `convert_lora_to_gguf.py`.

## 6. Set up the adaptive local coding agent

If you want Slm Cortex to act as your local coding agent instead of only
running the static runtime demo, continue with the
[local coding agent setup](local-coding-agent-setup.md).

## 7. Read the command reference

Once the quickstart works, move to the [command reference](command-reference.md) for the full flag-by-flag guide.
