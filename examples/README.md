# Examples

## Slm Cortex v0.1

Run the no-model end-to-end demo:

```bash
python scripts/run_slmcortex_demo.py
```

For the manual command-by-command quickstart and product overview, see
[README.md](../README.md).

## Arbitrary Slm Smoke

Run the default no-model arbitrary-slm smoke flow for the tiny `fastapi_contract` fixture:

```bash
python scripts/run_slmcortex_arbitrary_slm_smoke.py
```

Run the opt-in real local training path:

```bash
python scripts/run_slmcortex_arbitrary_slm_smoke.py --real-training
```

The default mode stages a demo adapter and validates package, compose, runtime, infer dry-run, and agent dry-run without real model training. The `--real-training` mode is slow, local-only, and intentionally excluded from normal CI.

## Repository Map

- [docs/repo-boundary-map.md](../docs/repo-boundary-map.md)
