# Runtime Core

Runtime Core is the serving and local inference layer for Skill Cortex v0.1. It
loads a runtime bundle, validates its manifests and checksums, selects the
route for a request, and either performs inference or returns a dry-run routing
decision.

## Responsibilities

- validate runtime bundles before use
- normalize chat-style inference requests
- select task and semantic-family routes from the bundle
- load the base model plus selected adapters when a real inference is requested
- expose the same core logic through local CLI inference and the compatibility
  server

## Inputs

- one runtime bundle
- prompt or OpenAI-style request payload
- optional runtime overrides such as `task_type`, `semantic_family`, and
  `skill_override`

## Outputs

- validation status for `validate-runtime`
- dry-run route selection details for `infer --dry-run`
- generated completion payloads for real inference or server requests

## Role In The Product Flow

Runtime Core owns `validate-runtime`, `infer`, and `serve`. The demo path uses
it in dry-run mode so a developer can verify the full routing flow without
downloading or loading a model.

## v0.1 Boundaries

- runtime bundles are required; raw registries are not a runtime dependency
- the compatibility server is non-streaming and intentionally thin
- dry-run validates control flow, not model quality