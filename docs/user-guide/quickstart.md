# Quickstart

Use this path if you want the first successful end-to-end run with the fewest moving parts.

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

## 2. Run the no-model demo

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

## 3. Run the same flow by hand

If you want the command-by-command version, use the same sequence the demo script wraps:

```bash
DEMO_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/slmcortex-demo.XXXXXX")"

slmcortex package-skill \
  --skill-id python_skill \
  --name "Python Skill" \
  --adapter-dir artifacts/adapters/python_skill \
  --train-dataset tests/fixtures/slmcortex_demo/train.jsonl \
  --eval-dataset tests/fixtures/slmcortex_demo/eval.jsonl \
  --eval-summary tests/fixtures/slmcortex_demo/eval-summary.json \
  --output "$DEMO_ROOT/python_skill"

slmcortex package-skill \
  --skill-id debugging_skill \
  --name "Debugging Skill" \
  --adapter-dir artifacts/adapters/debugging_skill \
  --train-dataset tests/fixtures/slmcortex_demo/train.jsonl \
  --eval-dataset tests/fixtures/slmcortex_demo/eval.jsonl \
  --eval-summary tests/fixtures/slmcortex_demo/eval-summary.json \
  --output "$DEMO_ROOT/debugging_skill"

slmcortex compose-skills \
  --skills "$DEMO_ROOT/python_skill,$DEMO_ROOT/debugging_skill" \
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

## 4. Try the built-in smoke checks

The default arbitrary-skill smoke stays no-model:

```bash
python scripts/run_slmcortex_arbitrary_skill_smoke.py
```

If you explicitly want the slower local training path:

```bash
python scripts/run_slmcortex_arbitrary_skill_smoke.py --real-training
```

For GGUF training/import conversion, set `gguf_converter` in the selected base
config to llama.cpp's `convert_lora_to_gguf.py`.

## 5. Read the command reference

Once the quickstart works, move to the [command reference](command-reference.md) for the full flag-by-flag guide.
