# SkillLatticeCoder

SkillLatticeCoder is a local research prototype for testing whether a frozen
small coding model becomes more capable by composing sparse task-specific LoRA
skills instead of using one monolithic fine-tune.

## Hypothesis

The experiment compares:

1. frozen base model;
2. rank-24 generic LoRA trained on all examples;
3. one routed rank-8 skill LoRA;
4. up to two routed rank-8 skill LoRAs composed at inference.

The lattice is supported only when mode 4 beats the generic adapter overall
and per active adapter parameter. It is falsified for this experiment when it
does not beat the generic adapter overall.

This remains generative AI: the model generates code, tests, explanations, and
fixes. Skills alter model weights during generation. They are not retrieval,
repo indexing, MCP tools, or deterministic patch rules.

## Model and skills

The frozen source model is
`Qwen/Qwen2.5-Coder-1.5B-Instruct`; Apple Silicon execution uses
`mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit`.

Initial skills:

- `python_skill`
- `debugging_skill`
- `test_generation_skill`

The rule router selects at most two skills. Explicit dataset tags are supported
for controlled tests, but normal lattice evaluation routes from prompt text.

## Install

Requires native ARM Python 3.11+ on Apple Silicon.

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e '.[test]'
pytest
```

## Dry-run the complete control flow

Dry runs do not download or load a model.

```bash
skill-lattice train-skill python_skill --dry-run
skill-lattice train-generic --dry-run
skill-lattice infer --mode base --prompt "Write a Python function" --dry-run
skill-lattice infer --mode lattice --prompt "Fix this Python traceback" --dry-run
skill-lattice eval --dataset data/eval.jsonl --dry-run
```

## Train

```bash
skill-lattice train-skill python_skill
skill-lattice train-skill debugging_skill
skill-lattice train-skill test_generation_skill
skill-lattice train-generic
```

Adapters are saved under `artifacts/adapters/`. The base model remains frozen;
MLX-LM trains only LoRA parameters.

## Infer

```bash
skill-lattice infer --mode base --prompt "Write a Python function"
skill-lattice infer --mode generic --prompt "Write a Python function"
skill-lattice infer --mode single-skill --skill debugging_skill --prompt "Fix this traceback"
skill-lattice infer --mode lattice --prompt "Fix this Python traceback"
```

Lattice composition concatenates compatible MLX LoRA ranks so that the
resulting delta equals the weighted sum of the selected adapters. It never
fuses adapters into the quantized base.

## Evaluate

```bash
skill-lattice eval --dataset data/eval.jsonl
```

Evaluation writes raw JSONL, `summary.json`, and `report.md` under
`artifacts/evaluations/`. Metrics include fuzzy and exact match, Python syntax
validity, trusted-fixture subprocess results, latency, token counts, peak MLX
memory, and active adapter parameters.

The checked-in benchmark contains 150 execution-backed cases: 50 Python
generation, 50 debugging, and 50 test-generation tasks. Test-generation cases
must pass against the correct implementation and fail against a paired mutant.
Rebuild it deterministically with:

```bash
python scripts/build_eval.py data/eval.jsonl
```

Confidence intervals bootstrap 50 semantic families rather than treating the
three task formulations of each behavior as independent samples.

## Five-seed experiment

The conclusive experiment trains three rank-8 skills for 100 iterations each
and the rank-24 generic baseline for 300 iterations, matching aggregate
optimizer steps. It evaluates routed and oracle lattice modes:

```bash
python scripts/run_seeds.py --seeds 11 22 33 44 55
```

Adapters and per-seed results are stored under
`artifacts/experiments/five-seed/`. The combined `summary.json` bootstraps
seed × semantic-family pairs. Use `--dry-run` to validate the full control
flow without training or model loading.

Execution fixtures are for trusted local toy data only. They are isolated in a
temporary directory with a timeout, but they are not a security sandbox.

## Roadmap

- Stage 2: hypernetwork-generated micro-skills.
- Stage 3: adaptive-depth coding model.

Learned routing, extra skills, PEFT compatibility, distributed training, and
untrusted-code sandboxing are intentionally deferred.
