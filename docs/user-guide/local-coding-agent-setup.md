# Local Coding Agent Setup

Use this page when you want Slm Cortex to act as your local coding agent instead of only running the static package-and-compose demo.

This guide uses the current adaptive prototype path:

- local slm selection from a `--slms-dir`
- optional remote LoRA import from the configured catalog
- optional plasticity LoRA training when routing decides a new adapter is needed
- bounded local agent execution against a repository checkout

Start with the safe no-model flow, then enable the adaptive runtime only after the basic checks pass.

## 1. Pick your backend

The runtime selects a backend from base config:

- `backend: auto` uses MLX on macOS arm64/aarch64
- `backend: auto` uses GGUF everywhere else

Install the matching extras only when you need real inference or training:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install --upgrade pip
pip install -e '.[test]'
pip install -e '.[mlx]'   # macOS Apple Silicon
pip install -e '.[gguf]'  # Linux, Windows, macOS Intel, or explicit GGUF use
```

Check that the CLI is available:

```bash
python -m slmcortex --help
```

## 2. Verify the repo with the no-model demo

Run the fastest healthy-path check first:

```bash
python scripts/run_slmcortex_demo.py
```

This validates the public flow without loading a real model:

- packages checked-in adapters
- composes one runtime bundle
- validates the runtime
- runs inference in `--dry-run`
- runs the bounded agent in `--dry-run`

Do not move on until this works.

## 3. Verify the adaptive prototype in mock mode

Run the adaptive smoke before turning on real downloads or training:

```bash
python scripts/run_dynamic_adaptive_smoke.py
```

This validates the adaptive branches with mocked model and adaptation work:

- local slm branch
- remote LoRA branch
- plasticity training branch
- branch tracing and selected-skill output

If this fails, fix the runtime or packaging path before trying real adaptive mode.

## 4. Enable the real adaptive prototype config

Use the shipped prototype config:

```bash
export SLMCORTEX_BASE_CONFIG=src/slmcortex_resources/configs/prototype.yaml
```

This profile already enables:

- remote LoRA downloads
- the smaller router model
- plasticity training
- a plasticity publish directory
- a small curated remote LoRA catalog

Read it before using it:

- [src/slmcortex_resources/configs/prototype.yaml](../../src/slmcortex_resources/configs/prototype.yaml)

If you are using GGUF, make sure the runtime model path is a `.gguf` file and set `gguf_converter` in your selected config before real GGUF training or conversion.

## 5. Run the real adaptive smoke

Once the prototype config is exported, run:

```bash
python scripts/run_dynamic_adaptive_smoke.py --real
```

This is the first meaningful proof that the system can behave like an adaptive coding runtime on your machine.

Expect this path to be slower than the no-model and mock flows because it may:

- download model assets
- download a remote LoRA
- run local plasticity training
- load a real runtime backend

## 6. Use dynamic inference directly

The adaptive runtime path uses `--slms-dir` rather than a prebuilt runtime bundle.

Start in `--dry-run` so you can inspect routing before you allow real adaptation or model execution:

```bash
python -m slmcortex infer \
  --slms-dir .slmcortex/prototype-slms \
  --prompt "Fix a FastAPI validation bug" \
  --allow-remote-loras \
  --dry-run
```

Then run the same prompt without `--dry-run`:

```bash
python -m slmcortex infer \
  --slms-dir .slmcortex/prototype-slms \
  --prompt "Fix a FastAPI validation bug" \
  --allow-remote-loras
```

Look for fields such as:

- `route_branch`
- `selected_skills`
- `remote_loras`
- `train_new_lora`
- `route_trace`

These tell you whether the runtime used a local slm, fetched a remote LoRA, trained a plasticity adapter, or fell back to the base model.

If you want the OpenAI-compatible API without prebuilding a runtime bundle, use the same `--slms-dir` path:

```bash
python -m slmcortex serve \
  --slms-dir .slmcortex/prototype-slms \
  --allow-remote-loras \
  --dry-run
```

Then start the real server without `--dry-run` when the route looks sane.

## 7. Point it at your repository

Once direct inference looks sane, use the bounded agent path.

Start with `--dry-run`:

```bash
python -m slmcortex agent run \
  --slms-dir .slmcortex/prototype-slms \
  --repo /path/to/your/repo \
  --task "Fix the failing answer implementation." \
  --dry-run
```

When routing and task output look correct, move to confirm mode:

```bash
python -m slmcortex agent run \
  --slms-dir .slmcortex/prototype-slms \
  --repo /path/to/your/repo \
  --task "Fix the failing answer implementation." \
  --write-mode confirm
```

Use confirm mode before you allow broader write automation.

## 8. Recommended first-user flow

Use this exact sequence the first time:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install --upgrade pip
pip install -e '.[test]'
python scripts/run_slmcortex_demo.py
python scripts/run_dynamic_adaptive_smoke.py
export SLMCORTEX_BASE_CONFIG=src/slmcortex_resources/configs/prototype.yaml
python scripts/run_dynamic_adaptive_smoke.py --real
python -m slmcortex infer --slms-dir .slmcortex/prototype-slms --prompt "Fix a FastAPI validation bug" --allow-remote-loras --dry-run
python -m slmcortex serve --slms-dir .slmcortex/prototype-slms --allow-remote-loras --dry-run
python -m slmcortex agent run --slms-dir .slmcortex/prototype-slms --repo /path/to/your/repo --task "Fix the failing answer implementation." --dry-run
```

If each step works, the next step is to remove `--dry-run` for the inference path and then for the bounded agent path.

## 9. Current limitations

This is still a prototype local coding agent.

Important constraints today:

- remote discovery is curated through the configured `remote_lora_catalog`
- plasticity training still depends on configured datasets
- GGUF currently supports one active adapter until adapter merge is configured
- local training and real inference depend on your machine, backend, and model/tooling availability

Treat the adaptive path as a local power-user workflow, not as a zero-config product.

## 10. Next references

- [quickstart](quickstart.md)
- [command-reference](command-reference.md)