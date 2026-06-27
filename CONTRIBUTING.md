# Contributing

Thanks for contributing to Skill Cortex.

## Scope

Skill Cortex v0.1 is intentionally narrow. Contributions are welcome, but
changes should preserve the current product boundaries:

- do not change adapters, benchmarks, or checked-in validation fixtures unless
  the change explicitly targets those assets
- do not change the skill package contract or runtime bundle contract without a
  coordinated design update
- keep the public product surface `skillcortex` clear and documented

## Local setup

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install --upgrade pip
pip install -e '.[test]'
```

## Recommended checks

Run the full test suite:

```bash
pytest -q
```

Run the focused product CLI and demo checks:

```bash
pytest tests/test_skillcortex_cli.py tests/test_skillcortex_demo.py -q
```

Run the no-model demo manually:

```bash
python scripts/run_skillcortex_demo.py
```

## Documentation expectations

When you change the public product surface, update the relevant docs:

- `README.md` for first-time user experience
- `docs/architecture/` for product-layer responsibilities
- `docs/skill-package-contract.md` for package/runtime contract changes
- `CHANGELOG.md` for user-visible release changes

## Pull requests

Please keep pull requests focused and explain:

- what changed
- why it changed
- whether it affects product behavior, docs only, or release engineering only
- how you validated it

If your change only improves docs or release readiness, say that explicitly.

## Reporting issues

Use the issue templates when possible.

Bug reports are most useful when they include:

- operating system and Python version
- exact command run
- expected behavior
- actual behavior
- minimal reproduction steps

## Release posture

For the current public v0.1 line, maintainers should favor:

- clarity over breadth
- deterministic demos over aspirational claims
- explicit limitations over ambiguous promises
