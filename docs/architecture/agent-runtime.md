# Agent Runtime

Agent Runtime is the bounded local task runner layered on top of Runtime Core in
SLMCortex v0.1. It inspects a repository, asks Runtime Core for a plan and a
patch proposal, optionally materializes a file replacement, and can run one
developer-supplied validation command.

## Responsibilities

- inspect a local repository with a small safe tool surface
- request step-wise routing decisions from Runtime Core
- capture traces of selected slms, proposed changes, and validation results
- keep writes scoped to the repository root and controlled by `--writes`

## Inputs

- runtime bundle path
- local repository path
- task description
- optional validation command and trace output path

## Outputs

- one JSON result describing the agent run
- optional trace file with per-step routing and tool activity
- optional repository file replacement when `--writes on` is selected

## Role In The Product Flow

Agent Runtime owns `agent run`, the final step in the documented product flow.
The no-model demo uses `--dry-run` so developers can inspect the end-to-end
agent control flow without model downloads.

## v0.1 Boundaries

- local, single-run execution only
- bounded tool loop rather than a full IDE agent
- no background orchestration, long-lived memory, or distributed execution