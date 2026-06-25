# Examples

## Skill Cortex v0.1

Run the no-model end-to-end demo:

```bash
python scripts/run_skillcortex_demo.py
```

For the manual command-by-command quickstart, see [README.md](../README.md).

## Research Dry-Run Smoke Checks

```bash
skill-lattice train-skill python_skill --dry-run
skillcortex train-skill python_skill --dry-run
skill-lattice train-generic --dry-run
skill-lattice infer --mode lattice --task-type debugging --prompt "Fix this Python traceback" --dry-run
skill-lattice eval --dataset data/eval.jsonl --dry-run
```

## Repository Map

- [docs/repo-boundary-map.md](../docs/repo-boundary-map.md)
