# Skill Factory

Skill Factory is the packaging boundary for Skill Cortex v0.1. Its job is to
turn an existing adapter directory plus its provenance into a self-describing
skill package without retraining a model or mutating the checked-in research
artifacts.

## Responsibilities

- package an existing adapter into a deterministic skill artifact
- optionally run the product `train-skill` wrapper for built-in research skills
- record provenance, checksums, protected input snapshots, and composition
  metadata
- validate that emitted package files are internally consistent

## Inputs

- adapter directory with `adapters.safetensors` and adapter config
- train and eval dataset paths for provenance
- eval summary JSON
- optional examples and composition metadata

## Outputs

- one skill package directory containing `skill.yaml`, `metadata.json`,
  `training_config.json`, `eval.json`, `README.md`, and `adapter/`

## Role In The Product Flow

Skill Factory owns the `package-skill` and product `train-skill` stages of the
demo flow. It is the only layer that emits package-first skill artifacts for
Composer.

## v0.1 Boundaries

- does not retrain models unless the user explicitly runs product `train-skill`
- does not require registry state to produce a valid package
- does not promote skills, update router defaults, or publish to a marketplace