# Packaged Install

This guide defines the Phase 1 packaged-product contract for Slm Cortex.

The default product path is Composer-first:

1. Install the launcher
2. Run `slmcortex doctor`
3. Point the product at a local folder
4. Compose a runtime
5. Run or export the result

Advanced Factory commands remain available, but they are optional and are not part of the normal install path.

## Supported Platform Matrix

| Target | Baseline artifact | Notes |
| --- | --- | --- |
| macOS | `artifacts/installers/install-slmcortex-macos.sh` | MLX is available on Apple Silicon; GGUF remains optional |
| Linux | `artifacts/installers/install-slmcortex-linux.sh` | Composer-first path works without training extras |
| Windows | `artifacts/installers/install-slmcortex-windows.ps1` | PowerShell installer creates a local launcher |

Each artifact expects a wheel, source distribution, or package source path and creates an isolated virtual environment plus a launcher that runs `python -m slmcortex`.

## App Workspace Contract

The packaged app workspace is external to the repository checkout.

| Path | Purpose |
| --- | --- |
| `state/` | local state, copied demo repos, and future support metadata |
| `packages/` | imported or authored slm packages |
| `runtimes/` | emitted runtime bundles |
| `exports/` | export descriptors for launcher or UI handoff |
| `logs/` | compose and diagnostics logs |
| `diagnostics/` | future support bundles and environment reports |

Default roots:

- macOS: `~/Library/Application Support/SlmCortex`
- Linux: `${XDG_STATE_HOME:-~/.local/state}/slmcortex`
- Windows: `%APPDATA%\SlmCortex`

Inspect the resolved contract at any time:

```bash
slmcortex doctor
slmcortex doctor --workspace /tmp/slmcortex-app
```

## Composer-First Flow

Compose a runtime directly from a folder and the external app workspace:

```bash
slmcortex compose-folder \
  --workspace /tmp/slmcortex-app \
  --folder /path/to/repo \
  --task "Create a FastAPI endpoint with request validation" \
  --export-descriptor /tmp/slmcortex-app/exports/repo.json
```

The command returns a structured result with:

- task hints inferred from the folder scan
- the routing decision and selected packages
- runtime composition and validation status
- an optional export descriptor path
- machine-readable diagnostics and warnings

## Smoke Validation Paths

External workspace smoke:

```bash
python scripts/run_package_product_smoke.py
```

Clean-machine style install-and-launch smoke:

```bash
python scripts/run_packaged_install_smoke.py --package-source .
```

The first script validates the external workspace layout, package import, compose, validation, export descriptor, and log output. The second script validates that an isolated install can launch the Composer-first entry point without relying on repository-relative runtime state.