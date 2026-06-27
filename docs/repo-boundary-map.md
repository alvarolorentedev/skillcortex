# Repo Boundary Map

Skill Cortex is the canonical public product surface for this repository.
Public documentation and runtime behavior should stay product-first.

## Canonical boundaries

- `src/`: public product CLI, package/runtime tooling, and bounded
	agent surface.
- `configs/`: runtime defaults and governed skill registry inputs.
- `data/`: current canonical datasets and benchmarks.
- `skills/`: skill catalog mirror and package artifacts.
- `examples/`: runnable examples and smoke snippets.
- `docs/`: specs, architecture notes, and user-facing documentation.
- `scripts/`: validation, demo, and governance helpers.
- `tests/`: unit, integration, and regression coverage.
- `artifacts/`: immutable adapters, validation fixtures, and generated governance outputs.

## Stability policy

- Do not change model behavior, adapters, registry semantics, or benchmark data.
- Keep `skillcortex` as the canonical public identity.
- Keep generated artifacts immutable.
- Keep public documentation product-first.

## Current source of truth

- Public CLI entry point: `skillcortex`
- Product runtime/package implementation: `src/`
- Skill registry: `configs/skill_registry.json`
- Skill metadata: `configs/skills.yaml`
- Skill mirror: `skills/skill_registry.json`, `skills/skills.yaml`
- Datasets and benchmarks: `data/`
- Checked-in adapters and validation fixtures: `artifacts/`
