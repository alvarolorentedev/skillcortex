# Slm Composer

Slm Composer is the deterministic assembly layer in SLMCortex v0.1. It
loads validated slm packages, checks compatibility, and writes a runtime
bundle that Runtime Core can consume directly.

## Responsibilities

- validate that selected packages can coexist
- derive task and semantic-family routes from package composition metadata
- emit a runtime bundle with stable manifests, reports, and checksums
- keep registry enrichment optional and non-authoritative

## Inputs

- one or more validated slm packages
- optional registry enrichment file
- composition strategy, currently `routed`

## Outputs

- one runtime bundle containing `composition.yaml`, `router_config.json`,
  `active_slms.json`, `compatibility_report.json`, `budget_report.json`,
  `checksums.json`, and `README.md`

## Role In The Product Flow

Slm Composer owns the `compose-slms` stage in the quickstart and scripted
demo. The runtime bundle it emits is the deployment artifact and source of
truth for runtime loading.

## v0.1 Boundaries

- only the routed composition strategy is supported
- package metadata stays authoritative when registry enrichment is present
- source packages and checked-in artifacts are never mutated during composition