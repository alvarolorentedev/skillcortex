# Packaged Install

This guide defines the packaged-product contract for Slm Cortex through Phase 3 distribution hardening.

The default product path is Composer-first:

1. Install the launcher
2. Run `slmcortex doctor` or open the Composer launcher
3. Point the product at a local folder
4. Compose a runtime
5. Run or export the result

Advanced Factory commands remain available, but they are optional and are not part of the normal install path.

## Supported Platform Matrix

| Target | Baseline artifact | Notes |
| --- | --- | --- |
| macOS | `scripts/installers/install-slmcortex-macos.sh` | MLX is available on Apple Silicon; GGUF remains optional |
| Linux | `scripts/installers/install-slmcortex-linux.sh` | Composer-first path works without training extras |
| Windows | `scripts/installers/install-slmcortex-windows.ps1` | PowerShell installer creates a local launcher |

Each artifact expects a wheel, source distribution, or package source path and creates an isolated virtual environment plus two launchers:

- `slmcortex`: the full product CLI
- `slmcortex-composer`: the Composer App launcher that opens directly into the guided folder-to-runtime flow

## App Workspace Contract

The packaged app workspace is external to the repository checkout.

| Path | Purpose |
| --- | --- |
| `state/` | local state, copied demo repos, and future support metadata |
| `packages/` | imported or authored slm packages |
| `runtimes/` | emitted runtime bundles |
| `exports/` | export descriptors for launcher or UI handoff |
| `logs/` | compose and diagnostics logs |
| `diagnostics/` | support bundles, doctor exports, and environment reports |

Default roots:

- macOS: `~/Library/Application Support/SlmCortex`
- Linux: `${XDG_STATE_HOME:-~/.local/state}/slmcortex`
- Windows: `%APPDATA%\SlmCortex`

Inspect the resolved contract at any time:

```bash
slmcortex doctor
slmcortex doctor --workspace /tmp/slmcortex-app
slmcortex doctor --export-support-bundle
slmcortex doctor --workspace /tmp/slmcortex-app --export-support-bundle
slmcortex-composer --help
```

`slmcortex doctor` now reports:

- workspace schema version and resolved workspace paths
- installed and optional runtime backend capabilities
- optional backend provisioning status for `mlx` and `gguf`
- support bundle export availability

Install optional runtime dependencies as a separate step when you need real local inference:

```bash
slmcortex provision-backend --backend mlx --dry-run
slmcortex provision-backend --backend gguf
```

The provisioning command installs backend-specific dependencies into the current product environment without changing the base Composer install contract. Failed provisioning should leave the base dry-run Composer path usable.

Support bundle exports intentionally exclude environment variable dumps and source file contents. The bundle captures platform details, app version, workspace layout, diagnostics, and recent warnings or errors into one JSON artifact.

## Upgrade And Recovery Contract

The Composer app state file is versioned explicitly in `state/composer-app-state.json`.

- schema upgrades migrate forward when the repo knows how to rewrite the older state
- unsupported future schemas fail explicitly instead of guessing
- migrating from older state writes a backup file into the same state directory before rewriting
- project state records selected package fingerprints and runtime bundle checksum provenance
- failed migrations should be handled by moving the state file aside, exporting a doctor support bundle, and retrying with a clean workspace root

This keeps imported packages and emitted runtime bundles attributable after upgrades.

## Composer-First Flow

Launch the guided Composer App workflow directly from the packaged launcher:

```bash
slmcortex-composer \
  --workspace /tmp/slmcortex-app \
  --folder /path/to/repo \
  --task "Create a FastAPI endpoint with request validation" \
  --outcome export_bundle \
  --export-logs
```

The full CLI can still compose a runtime directly from a folder and the external app workspace:

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

The first script validates the external workspace layout, package import, guided Composer App export, runtime validation, local-run dry-run agent flow, and log output. The second script validates that an isolated install can launch both the main CLI and the dedicated Composer launcher, export a doctor support bundle, and install, compose, and export from the packaged launchers without relying on repository-relative runtime state.

## Install And Uninstall Notes

- install artifacts create an isolated virtual environment plus launcher scripts under the platform-specific install root
- uninstall is currently a directory removal operation for that install root and any optional app workspace cleanup the user chooses to do separately
- backend provisioning is optional and separate from the base install; a backend provisioning failure should not invalidate the base CLI or dry-run Composer flow