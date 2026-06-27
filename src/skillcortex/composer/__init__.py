import shutil
import tempfile
from pathlib import Path

from .bundle import build_budget_report, build_bundle, bundle_files, write_checksums
from .compatibility import build_compatibility_report, load_registry_enrichment
from .loading import load_skill_package, validate_unique_skill_ids
from .routing import build_routes


def compose_skill_packages(
    *,
    skills: list[Path],
    strategy: str,
    output: Path,
    registry: Path | None = None,
    force: bool = False,
    dry_run: bool = False,
) -> dict:
    if strategy != "routed":
        raise ValueError("only the routed composition strategy is currently supported")
    if not skills:
        raise ValueError("at least one skill package is required")

    loaded = [load_skill_package(path) for path in skills]
    validate_unique_skill_ids(loaded)
    enrichment = load_registry_enrichment(registry, loaded)
    compatibility = build_compatibility_report(loaded, enrichment)
    if compatibility["errors"]:
        raise ValueError(compatibility["errors"][0])

    routes = build_routes(loaded)
    bundle = build_bundle(loaded, routes, enrichment)
    budget = build_budget_report(loaded, routes)

    output = output.resolve()
    output_exists = output.exists()
    if dry_run:
        return {
            "status": "dry-run",
            "strategy": strategy,
            "output": str(output),
            "skills": [item["skill_id"] for item in loaded],
            "files": sorted(bundle_files(bundle, compatibility, budget)),
        }
    if output_exists and any(output.iterdir()) and not force:
        raise FileExistsError(f"{output} exists; pass --force to replace it")
    if output_exists:
        shutil.rmtree(output)

    with tempfile.TemporaryDirectory(prefix="skillcortex-compose-") as directory:
        staging = Path(directory) / output.name
        staging.mkdir(parents=True, exist_ok=True)
        files = bundle_files(bundle, compatibility, budget)
        for relative, content in sorted(files.items()):
            destination = staging / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(content)
        write_checksums(staging, files, loaded, enrichment)
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(staging), str(output))
    return {
        "status": "complete",
        "strategy": strategy,
        "output": str(output),
        "skills": [item["skill_id"] for item in loaded],
    }
