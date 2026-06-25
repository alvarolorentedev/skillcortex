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

The primary statistical claim is supported when the lattice's paired execution
improvement over generic has a 95% confidence interval above zero. A separate,
stricter practical criterion requires at least five percentage points of
absolute improvement and better performance per active parameter.

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
- `alternating_skill` (promoted behind a strict semantic gate)

The default `skillcortex_router_v1` uses frozen-base fallback for Python
generation, validated skill pairs for debugging and test generation, and
activates `alternating_skill` only for the `alternating` semantic family's
debugging and test-generation tasks. The pre-promotion baseline remains
available as `protected_skill_router_without_failure_born`; the previous
prompt-rule router remains available as `legacy_rule_router`.

## Install

Requires native ARM Python 3.11+ on Apple Silicon.

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e '.[test]'
pytest
```

Repository boundaries and the current phase-0 split are documented in
[docs/repo-boundary-map.md](docs/repo-boundary-map.md). Runnable smoke
examples live in [examples/README.md](examples/README.md), and curated report
copies live in [reports/research-results.md](reports/research-results.md) and
[reports/five-seed-artifact-resume.md](reports/five-seed-artifact-resume.md).

## Dry-run the complete control flow

Dry runs do not download or load a model.

```bash
skill-lattice train-skill python_skill --dry-run
skill-lattice train-generic --dry-run
skill-lattice infer --mode base --prompt "Write a Python function" --dry-run
skill-lattice infer --mode lattice --task-type debugging --prompt "Fix this Python traceback" --dry-run
skill-lattice eval --dataset data/eval.jsonl --dry-run
```

Product packaging lives under `skillcortex` and keeps the research workflow
unchanged:

```bash
skillcortex package-skill \
	--skill-id python_skill \
	--name "Python Skill" \
	--adapter-dir artifacts/adapters/python_skill \
	--train-dataset data/train.jsonl \
	--eval-dataset data/eval.jsonl \
	--eval-summary artifacts/evaluations/20260620T152056Z/summary.json \
	--output skills/python_skill

skillcortex validate-skill-package --path skills/python_skill
skillcortex train-skill python_skill --output skills/python_skill_run --force
	skillcortex compose-skills \
		--skills skills/python_skill,skills/debugging_skill \
		--strategy routed \
		--output runtime/debugging_bundle
```

The product package contract, expected files, and reproducibility guarantees
are documented in [docs/skill-package-contract.md](docs/skill-package-contract.md).

Phase 2 Composer is package-first: valid self-describing skill packages can be
composed without the internal registry. Official/internal registry files remain
available as optional enrichment inputs only and are recorded as provenance,
not source of truth.

Phase 3A Runtime Core consumes the composed runtime bundle directly. The bundle
is the deployment artifact and source of truth at runtime; registry input is
not required.

## Serve A Runtime Bundle

Validate a runtime bundle before startup:

```bash
skillcortex validate-runtime --runtime runtime/debugging_bundle
```

Run local CLI inference against the same runtime service used by the server:

```bash
skillcortex infer \
	--runtime runtime/debugging_bundle \
	--prompt "Fix this Python traceback" \
	--dry-run
```

You can also pass an OpenAI-style chat request file:

```json
{
  "messages": [
    {"role": "system", "content": "You are a debugging assistant."},
    {"role": "user", "content": "Fix this Python traceback and failing test."}
  ],
  "task_type": "debugging"
}
```

```bash
skillcortex infer \
	--runtime runtime/debugging_bundle \
	--request-file request.json \
	--dry-run
```

Start the minimal OpenAI-compatible compatibility server:

```bash
skillcortex serve --runtime runtime/debugging_bundle --host 127.0.0.1 --port 8000
```

Minimal API checks:

```bash
curl http://127.0.0.1:8000/v1/models

curl http://127.0.0.1:8000/v1/chat/completions \
	-H 'Content-Type: application/json' \
	-d '{
	  "model": "debugging_bundle",
	  "messages": [
	    {"role": "user", "content": "Fix this Python traceback"}
	  ]
	}'
```

Current Phase 3A limits:

- non-streaming responses only
- dry-run support for `infer` and `serve`
- runtime bundle is the only required deployment input
- registry enrichment remains optional provenance only
- the OpenAI-compatible server is a thin compatibility layer over the shared runtime service

## Run A Local Agent Task

Phase 3B adds a native local agent loop on top of Runtime Core. The OpenAI API
remains a compatibility surface; the agent uses the shared runtime service
directly.

Run a bounded local task against a repository:

```bash
skillcortex agent run \
	--runtime runtime/debugging_bundle \
	--repo ./toy-repo \
	--task "Fix the failing answer implementation." \
	--test-command "PYTHONPATH=. uv run pytest -q" \
	--trace-out artifacts/agent-trace.json
```

Safety modes:

- `--writes confirm` is the default and keeps file mutation human-gated
- `--writes off` never mutates files and only emits proposed diffs
- `--writes on` applies supported file-replacement actions inside the repo root

Current Phase 3B limits:

- local, single-run execution only
- bounded safe tool loop: list files, read files, propose/apply file replacement, run one opt-in validation command
- dynamic skill routing happens per inference step
- execution traces record selected skills per step
- no IDE orchestration, background tasks, or distributed coordination

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
skill-lattice infer --mode lattice --task-type debugging --prompt "Fix this Python traceback"
```

Lattice composition concatenates compatible MLX LoRA ranks so that the
resulting delta equals the weighted sum of the selected adapters. It never
fuses adapters into the quantized base.

Use `--router-policy legacy_rule_router` for the original prompt-rule behavior.
Use `--router-policy protected_skill_router_without_failure_born` for the
validated pre-promotion baseline.

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

### Phase 1 result

Across five seeds and 750 executions per mode:

- Generic LoRA: **40.67%**
- Single skill: **44.27%**
- Routed lattice: **45.60%**
- Oracle lattice: **47.60%**

Routed lattice improved over generic by **+4.93 percentage points**, with a
cluster-bootstrap 95% confidence interval of **+1.47 to +8.40 points**, while
using 54.2% fewer active adapter parameters. It won in all five seeds.

This statistically supports the claim that the lattice beats generic, but
narrowly misses the preregistered five-point practical threshold. Oracle
routing improved by **+6.93 points**, showing that routing is the main next
target. Python generation regressed after fine-tuning, so the result does not
yet support a claim of general coding improvement.

See [the complete Phase 1 results](reports/research-results.md) for methodology,
per-task and per-seed tables, limitations, and reproduction details.

### Recent artifact outcomes

- FastAPI contract baseline: see `artifacts/experiments/fastapi-contract-baseline/summary.md` for per-behavior fuzzy scores, mean latency, and active-parameter counts. The candidate `skillcortex_router_v1` uses 466,944 active adapter parameters; per-behavior summaries and latency comparisons are recorded in `summary.json`.
- Evaluation report (2026-06-20): see `artifacts/evaluations/20260620T152056Z/report.md`. The evaluation labels its hypothesis inconclusive and reports per-mode fuzzy/exact/syntax/execution diagnostics; parameter-efficiency details are available in the corresponding `summary.json` files under `artifacts/`.

Execution fixtures are for trusted local toy data only. They are isolated in a
temporary directory with a timeout, but they are not a security sandbox.

## Roadmap

- Stage 2: hypernetwork-generated micro-skills.
- Stage 3: adaptive-depth coding model.

Learned routing, extra skills, PEFT compatibility, distributed training, and
untrusted-code sandboxing are intentionally deferred.
