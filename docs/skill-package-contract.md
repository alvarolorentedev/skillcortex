# Skill Package Contract

`skillcortex` packages a trained LoRA adapter as a reusable skill artifact
without changing router semantics, registry semantics, accepted datasets, or
checked-in benchmark artifacts.

## Commands

Package an existing adapter:

```bash
skillcortex package-skill \
  --skill-id python_skill \
  --name "Python Skill" \
  --adapter-dir artifacts/adapters/python_skill \
  --train-dataset data/train.jsonl \
  --eval-dataset data/eval.jsonl \
  --eval-summary tests/fixtures/skillcortex_demo/eval-summary.json \
  --output skills/python_skill
```

Train and package one of the built-in skills:

```bash
skillcortex train-skill python_skill --output skills/python_skill_run --force
```

Validate a package:

```bash
skillcortex validate-skill-package --path skills/python_skill
```

Compose validated packages into a deterministic runtime bundle:

```bash
skillcortex compose-skills \
  --skills skills/python_skill,skills/debugging_skill \
  --strategy routed \
  --output runtime/debugging_bundle
```

## Expected Output

```text
skills/python_skill/
├── adapter/
│   ├── adapters.safetensors
│   └── adapter_config.json
├── skill.yaml
├── README.md
├── eval.json
├── training_config.json
├── metadata.json
└── examples.jsonl
```

`examples.jsonl` is optional and is only written when supplied.

`skill.yaml` and `metadata.json` may also include a `composition` section.
When present, it makes the package self-describing for package-first
composition.

Product `train-skill` also creates an isolated sibling run directory named
`.PACKAGE_NAME.run` containing the temporary training data, adapter output, and
evaluation summary used to build the final package.

## Package-First Composition Metadata

Phase 2 Composer treats package metadata as the source of truth. The internal
registry is optional enrichment only.

Minimal required fields for a self-describing package:

```yaml
composition:
  capabilities:
    allowed_task_types: [debugging]
  activation:
    default_route_type: adapter
    scope: task
  compatibility:
    compatible_skills: []
    incompatible_skills: []
  routing:
    tasks: {}
```

Required fields:

- `composition.capabilities.allowed_task_types`
- `composition.activation.default_route_type`
- `composition.activation.scope`

Optional fields:

- `composition.activation.semantic_families`
- `composition.compatibility.compatible_skills`
- `composition.compatibility.incompatible_skills`
- `composition.routing.tasks`

`composition.routing.tasks` is optional, but official/internal skills use it to
encode routing order and companion requirements so Composer can mirror the
validated router behavior without consulting the registry.

Task routing entries currently support:

- `order`: lower values are selected earlier in a route
- `requires_all_of`: all listed skills must be present in the composition
- `requires_any_of`: at least one listed skill must be present in the composition

Self-describing external packages work without any registry input. Older
packages that do not carry `composition` metadata remain valid Phase 1 skill
packages, but they are not composable by Phase 2 Composer unless future
non-authoritative enrichment support is used to fill missing declarations.

## Validation Rules

- `skill.yaml`, `metadata.json`, `training_config.json`, `eval.json`, and the
  adapter weights must exist.
- `metadata.json` must record deterministic per-file checksums for the package.
- `metadata.json` must record protected input snapshots and confirm they stayed
  unchanged.
- If `composition` metadata is present, `skill.yaml` and `metadata.json` must
  record the same value.
- If `composition` metadata is present, `validate-skill-package` validates its
  schema and task declarations.
- Validation rechecks package file checksums.
- Validation rechecks the current hashes of protected inputs when those source
  files still exist in the workspace.

## Protected Inputs

Packaging and product training snapshot these inputs before and after work:

- the requested train dataset
- the requested eval dataset
- `configs/base.yaml`
- `configs/training.yaml`
- `configs/skill_registry.json`
- `configs/skills.yaml`
- files under `artifacts/adapters/`
- files under `data/benchmarks/`

If any protected input changes during packaging, the command fails.

## Reproducibility Guarantees

- package manifests are written deterministically
- training config values are copied into `training_config.json`
- package metadata records the resolved base model, runtime model, rank,
  target modules, dataset hashes, and training command when available
- package metadata records the run directory and source artifact locations
- package composition metadata, when present, is written to both `skill.yaml`
  and `metadata.json`

## Compose-Skills Runtime Bundle

`compose-skills` writes a deterministic runtime bundle:

```text
runtime/debugging_bundle/
├── composition.yaml
├── router_config.json
├── active_skills.json
├── compatibility_report.json
├── budget_report.json
├── checksums.json
└── README.md
```

Bundle files:

- `composition.yaml`: source-of-truth composition manifest with skills, routes,
  runtime base model, and provenance
- `router_config.json`: projected route table for runtime consumption
- `active_skills.json`: flat view of active packaged skills and route membership
- `compatibility_report.json`: compatibility checks plus optional enrichment
  provenance
- `budget_report.json`: stored and active adapter parameter and file-size budget
- `checksums.json`: deterministic hashes for emitted bundle files plus source
  package fingerprints
- `README.md`: human-readable summary

`compose-skills` never mutates source packages, adapters, datasets, registries,
or benchmark artifacts.

Optional registry enrichment:

- is never required for a complete self-describing package
- is never treated as the source of truth when package metadata is present
- is reported only as enrichment and provenance
- does not override explicit package metadata unless a future explicit override
  mode is added

## Runtime Core Usage

Phase 3A Runtime Core consumes the emitted runtime bundle directly. It does not
require registry state at startup or inference time.

Validate a bundle before loading any model state:

```bash
skillcortex validate-runtime --runtime runtime/debugging_bundle
```

Run local CLI inference with a single prompt:

```bash
skillcortex infer \
  --runtime runtime/debugging_bundle \
  --prompt "Fix this Python traceback" \
  --dry-run
```

Run local CLI inference with an OpenAI-style request file:

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
  --request-file request.json
```

Start the minimal OpenAI-compatible compatibility server:

```bash
skillcortex serve --runtime runtime/debugging_bundle --host 127.0.0.1 --port 8000
```

Minimal HTTP examples:

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

- non-streaming only
- runtime bundles remain the source of truth
- registry enrichment is optional and non-authoritative
- `infer` and `serve` support dry-run startup/control-flow checks
- the compatibility server delegates to the shared runtime service layer

## Current Scope

Product `train-skill` reuses the existing research training internals and is
currently limited to the existing research skills exposed by the repository.
It does not promote skills, update the registry, change router behavior, or
rewrite accepted datasets or benchmark artifacts.