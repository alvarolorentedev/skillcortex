# Repo Boundary Map

Slm Cortex is the canonical public product surface for this repository.
Public documentation and runtime behavior should stay product-first.

## Canonical boundaries

- `src/`: public product CLI, package/runtime tooling, and bounded
	agent surface.
- `configs/`: runtime defaults and governed slm registry inputs.
- `data/`: current canonical datasets and benchmarks.
- `slms/`: slm catalog mirror and package artifacts.
- `examples/`: runnable examples and smoke snippets.
- `docs/`: specs, architecture notes, and user-facing documentation.
- `scripts/`: validation, demo, and governance helpers.
- `tests/`: unit, integration, and regression coverage.
- `artifacts/`: immutable adapters, validation fixtures, and generated governance outputs.

## Stability policy

- Do not change model behavior, adapters, registry semantics, or benchmark data.
- Keep `slmcortex` as the canonical public identity.
- Keep generated artifacts immutable.
- Keep public documentation product-first.

## Current source of truth

- Public CLI entry point: `slmcortex`
- Product runtime/package implementation: `src/`
- Slm registry: `configs/slm_registry.json`
- Slm metadata: `configs/slms.yaml`
- Slm mirror: `slms/slm_registry.json`, `slms/slms.yaml`
- Datasets and benchmarks: `data/`
- Checked-in adapters and validation fixtures: `artifacts/`
