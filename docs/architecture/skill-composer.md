# Skill Composer

Skill Composer is the deterministic assembly layer in Skill Cortex v0.1. It
loads validated skill packages, checks compatibility, and writes a runtime
bundle that Runtime Core can consume directly.

## Responsibilities

- validate that selected packages can coexist
- derive task and semantic-family routes from package composition metadata
- emit a runtime bundle with stable manifests, reports, and checksums
- keep registry enrichment optional and non-authoritative

## Inputs

- one or more validated skill packages
- optional registry enrichment file
- composition strategy, currently `routed`

## Outputs

- one runtime bundle containing `composition.yaml`, `router_config.json`,
  `active_skills.json`, `compatibility_report.json`, `budget_report.json`,
  `checksums.json`, and `README.md`

## Role In The Product Flow

Skill Composer owns the `compose-skills` stage in the quickstart and scripted
demo. The runtime bundle it emits is the deployment artifact and source of
truth for runtime loading.

## v0.1 Boundaries

- only the routed composition strategy is supported
- package metadata stays authoritative when registry enrichment is present
- source packages and checked-in artifacts are never mutated during composition