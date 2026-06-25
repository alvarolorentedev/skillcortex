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
  --eval-summary artifacts/evaluations/20260620T152056Z/summary.json \
  --output skills/python_skill
```

Train and package one of the current research skills:

```bash
skillcortex train-skill python_skill --output skills/python_skill_run --force
```

Validate a package:

```bash
skillcortex validate-skill-package --path skills/python_skill
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

Product `train-skill` also creates an isolated sibling run directory named
`.PACKAGE_NAME.run` containing the temporary training data, adapter output, and
evaluation summary used to build the final package.

## Validation Rules

- `skill.yaml`, `metadata.json`, `training_config.json`, `eval.json`, and the
  adapter weights must exist.
- `metadata.json` must record deterministic per-file checksums for the package.
- `metadata.json` must record protected input snapshots and confirm they stayed
  unchanged.
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

## Current Scope

Product `train-skill` reuses the existing research training internals and is
currently limited to the existing research skills exposed by the repository.
It does not promote skills, update the registry, change router behavior, or
rewrite accepted datasets or benchmark artifacts.