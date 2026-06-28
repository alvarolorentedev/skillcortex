import json
import random
from pathlib import Path

from ..datasets import default_report_path, validate_training_datasets, write_validation_report
from .constants import BLUEPRINTS, DEFAULT_DATASET_SEED, DEFAULT_EVAL_RATIO, ENTITY_VARIANTS, SUPPORTED_DOMAINS
from .rendering import render_example
from .reports import coverage_report, template_distribution


def generate_dataset_bundle(
    *,
    slm_id: str,
    domain: str,
    task_type: str,
    num_examples: int,
    output: str | Path,
    eval_output: str | Path,
    seed: int = DEFAULT_DATASET_SEED,
    eval_size: int | None = None,
    report_output: str | Path | None = None,
) -> dict:
    if num_examples <= 0:
        raise ValueError("--num-examples must be greater than zero")
    resolved_domain = resolve_domain(domain)
    if task_type != "python_generation":
        raise ValueError("fastapi_contract generation currently supports only python_generation")
    resolved_eval_size = eval_size if eval_size is not None else max(1, round(num_examples * DEFAULT_EVAL_RATIO))
    total_examples = num_examples + resolved_eval_size
    rows = build_examples(slm_id=slm_id, task_type=task_type, total_examples=total_examples, seed=seed)
    train_rows = assign_split(rows[:num_examples], split="train", seed=seed, slm_id=slm_id)
    eval_rows = assign_split(rows[num_examples:], split="eval", seed=seed, slm_id=slm_id)
    output_path = Path(output)
    eval_output_path = Path(eval_output)
    write_jsonl(output_path, train_rows)
    write_jsonl(eval_output_path, eval_rows)
    report = validate_training_datasets(output_path, eval_dataset=eval_output_path)
    report["generation"] = {
        "slm_id": slm_id,
        "domain": resolved_domain,
        "task_type": task_type,
        "seed": seed,
        "train_examples": len(train_rows),
        "eval_examples": len(eval_rows),
        "template_distribution": template_distribution(train_rows + eval_rows),
    }
    report["coverage"] = coverage_report(train_rows + eval_rows)
    resolved_report = Path(report_output or default_report_path(output_path, eval_output=eval_output_path))
    write_validation_report(report, resolved_report)
    if report["errors"]:
        raise RuntimeError(f"generated dataset failed validation; report written to {resolved_report}")
    return {
        "status": report["status"],
        "slm_id": slm_id,
        "domain": resolved_domain,
        "task_type": task_type,
        "seed": seed,
        "train_dataset": str(output_path),
        "eval_dataset": str(eval_output_path),
        "report_output": str(resolved_report),
        "counts": {"train": len(train_rows), "eval": len(eval_rows)},
        "coverage": report["coverage"],
        "warnings": report["warnings"],
    }


def resolve_domain(domain: str) -> str:
    normalized = domain.strip().lower().replace("-", "_")
    resolved = SUPPORTED_DOMAINS.get(normalized)
    if resolved is None:
        supported = ", ".join(sorted(SUPPORTED_DOMAINS))
        raise ValueError(f"unsupported domain: {domain}; supported values: {supported}")
    return resolved


def build_examples(*, slm_id: str, task_type: str, total_examples: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    entity_indices = list(range(len(ENTITY_VARIANTS)))
    blueprint_indices = list(range(len(BLUEPRINTS)))
    rng.shuffle(entity_indices)
    rng.shuffle(blueprint_indices)
    rows: list[dict] = []
    for offset in range(total_examples):
        entity_round = offset // len(entity_indices)
        blueprint_round = offset // len(blueprint_indices)
        entity = ENTITY_VARIANTS[entity_indices[(offset + entity_round) % len(entity_indices)]]
        blueprint = BLUEPRINTS[blueprint_indices[(offset + blueprint_round) % len(blueprint_indices)]]
        rows.append(render_example(entity=entity, blueprint=blueprint, task_type=task_type, slm_id=slm_id, variant_number=offset + 1))
    return rows


def assign_split(rows: list[dict], *, split: str, seed: int, slm_id: str) -> list[dict]:
    assigned: list[dict] = []
    for index, row in enumerate(rows, 1):
        payload = dict(row)
        payload["id"] = f"{slm_id.replace('_', '-')}-{split}-{index:04d}"
        payload["semantic_family"] = "fastapi_contract"
        payload["slms"] = [slm_id]
        metadata = dict(payload.get("metadata") or {})
        metadata.update({"split": split, "seed": seed, "domain": "fastapi_contract"})
        payload["metadata"] = metadata
        assigned.append(payload)
    return assigned


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
