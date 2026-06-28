# Slm Factory

Slm Factory is the packaging boundary for SLMCortex v0.1. Its job is to
turn an existing adapter directory plus its provenance into a self-describing
slm package without retraining a model or mutating the checked-in research
artifacts.

## Responsibilities

- package an existing adapter into a deterministic slm artifact
- optionally run the product `train-slm` wrapper for built-in research slms
- record provenance, checksums, protected input snapshots, and composition
  metadata
- validate that emitted package files are internally consistent

## Inputs

- adapter directory with `adapters.safetensors` and adapter config
- train and eval dataset paths for provenance
- eval summary JSON
- optional examples and composition metadata

## Outputs

- one slm package directory containing `slm.yaml`, `metadata.json`,
  `training_config.json`, `eval.json`, `README.md`, and `adapter/`

## Role In The Product Flow

Slm Factory owns the `package-slm` and product `train-slm` stages of the
demo flow. It is the only layer that emits package-first slm artifacts for
Composer.

## v0.1 Boundaries

- does not retrain models unless the user explicitly runs product `train-slm`
- does not require registry state to produce a valid package
- does not promote slms, update router defaults, or publish to a marketplace