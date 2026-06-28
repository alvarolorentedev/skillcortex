# FastAPI Contract Tiny Fixture

This fixture is the smallest documented arbitrary-slm dataset for Slm Cortex product training smoke checks.

- `train.jsonl`: tiny supervised training set for `slmcortex train-slm --slm-id fastapi_contract`
- `eval.jsonl`: tiny evaluation set for packaging and runtime validation
- `request.json`: dry-run runtime request for the composed bundle

Dataset contract:

- required fields: `id`, `task_type`, `prompt`, `target`
- optional fields used here: `semantic_family`, `slms`

Supported task types remain the current product vocabulary:

- `python_generation`
- `debugging`
- `test_generation`

This fixture is intended for local validation and examples. It is not a research benchmark artifact and is not used by default CI.