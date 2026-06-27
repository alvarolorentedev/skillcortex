import json
from collections import Counter
from pathlib import Path

from .analysis import (
    PREVIEW_LIMIT,
    dataset_hash,
    detect_leakage,
    detect_repeated_output,
    example_hash,
    format_validation_error,
    hash_text,
    length_stats,
    normalize_text,
    strip_internal_rows,
    truncate,
    unique_ratio,
)


REQUIRED_FIELDS = ("id", "task_type", "prompt", "target")
DEFAULT_MIN_TARGET_LENGTH = 24


def validate_training_datasets(
    train_dataset: str | Path,
    *,
    eval_dataset: str | Path | None = None,
    min_target_length: int = DEFAULT_MIN_TARGET_LENGTH,
) -> dict:
    train_report = validate_dataset(train_dataset, min_target_length=min_target_length)
    eval_report = validate_dataset(eval_dataset, min_target_length=min_target_length) if eval_dataset is not None else None
    leakage = detect_leakage(train_report, eval_report)
    cross_split = {
        "leakage_count": len(leakage),
        "leakage_examples": leakage[:PREVIEW_LIMIT],
        "warnings": [],
        "errors": [],
    }
    if leakage:
        cross_split["errors"].append(f"train/eval leakage detected for {len(leakage)} example(s)")
    errors = list(train_report["errors"])
    warnings = list(train_report["warnings"])
    if eval_report is not None:
        errors.extend(eval_report["errors"])
        warnings.extend(eval_report["warnings"])
    errors.extend(cross_split["errors"])
    warnings.extend(cross_split["warnings"])
    return {
        "status": "invalid" if errors else "ok",
        "schema": {"required_fields": list(REQUIRED_FIELDS), "min_target_length": min_target_length},
        "train": train_report,
        "eval": eval_report,
        "cross_split": cross_split,
        "warnings": warnings,
        "errors": errors,
    }


def validate_dataset(
    dataset_path: str | Path,
    *,
    min_target_length: int = DEFAULT_MIN_TARGET_LENGTH,
) -> dict:
    path = Path(dataset_path)
    rows: list[dict] = []
    normalized_hashes: Counter[str] = Counter()
    prompt_hashes: Counter[str] = Counter()
    target_hashes: Counter[str] = Counter()
    prompt_lengths: list[int] = []
    target_lengths: list[int] = []
    invalid_count = 0
    errors: list[str] = []
    warnings: list[str] = []
    repeated_output_examples: list[dict] = []
    preview: list[dict] = []
    with path.open() as handle:
        for line_number, raw_line in enumerate(handle, 1):
            if not raw_line.strip():
                continue
            try:
                payload = json.loads(raw_line)
            except json.JSONDecodeError as error:
                invalid_count += 1
                errors.append(f"{path}:{line_number}: invalid JSON: {error.msg}")
                continue
            if not isinstance(payload, dict):
                invalid_count += 1
                errors.append(f"{path}:{line_number}: row must be a JSON object")
                continue
            missing_fields = [field for field in REQUIRED_FIELDS if field not in payload]
            if missing_fields:
                invalid_count += 1
                errors.append(f"{path}:{line_number}: missing required field(s): {', '.join(missing_fields)}")
                continue
            prompt = payload.get("prompt")
            target = payload.get("target")
            example_id = payload.get("id")
            task_type = payload.get("task_type")
            row_errors: list[str] = []
            if not isinstance(example_id, str) or not example_id.strip():
                row_errors.append("id must be a non-empty string")
            if not isinstance(task_type, str) or not task_type.strip():
                row_errors.append("task_type must be a non-empty string")
            if not isinstance(prompt, str) or not prompt.strip():
                row_errors.append("prompt must be a non-empty string")
            if not isinstance(target, str) or not target.strip():
                row_errors.append("target must be a non-empty string")
            elif len(target.strip()) < min_target_length:
                row_errors.append(f"target must be at least {min_target_length} characters")
            if row_errors:
                invalid_count += 1
                errors.append(f"{path}:{line_number}: {'; '.join(row_errors)}")
                continue
            prompt_text = str(prompt).strip()
            target_text = str(target).strip()
            normalized = example_hash(prompt_text, target_text)
            normalized_hashes[normalized] += 1
            prompt_hashes[hash_text(normalize_text(prompt_text))] += 1
            target_hashes[hash_text(normalize_text(target_text))] += 1
            prompt_lengths.append(len(prompt_text))
            target_lengths.append(len(target_text))
            repeated_findings = detect_repeated_output(target_text)
            if repeated_findings:
                repeated_output_examples.append({"id": example_id, "line": line_number, "findings": repeated_findings})
            row = {
                "id": example_id,
                "task_type": task_type,
                "prompt": prompt_text,
                "target": target_text,
                "metadata": payload.get("metadata") or {},
                "line": line_number,
                "hash": normalized,
            }
            rows.append(row)
            if len(preview) < PREVIEW_LIMIT:
                preview.append({
                    "id": example_id,
                    "task_type": task_type,
                    "prompt": truncate(prompt_text, 140),
                    "target_preview": truncate(target_text, 160),
                })
    if not rows and invalid_count == 0:
        errors.append(f"{path} is empty")
    duplicates = sorted(hash_value for hash_value, count in normalized_hashes.items() if count > 1)
    if duplicates:
        errors.append(f"{path}: duplicate examples detected: {len(duplicates)}")
    if repeated_output_examples:
        errors.append(f"{path}: repeated-token or repeated-punctuation outputs detected: {len(repeated_output_examples)}")
    prompt_ratio = unique_ratio(prompt_hashes, len(rows))
    target_ratio = unique_ratio(target_hashes, len(rows))
    example_ratio = unique_ratio(normalized_hashes, len(rows))
    if len(rows) >= 10 and (prompt_ratio < 0.7 or target_ratio < 0.7 or example_ratio < 0.75):
        warnings.append(
            f"{path}: suspiciously low diversity (prompt={prompt_ratio:.2f}, target={target_ratio:.2f}, example={example_ratio:.2f})"
        )
    return {
        "dataset_path": str(path),
        "dataset_hash": dataset_hash(path),
        "counts": {"total": len(rows) + invalid_count, "valid": len(rows), "invalid": invalid_count, "duplicates": len(duplicates)},
        "stats": {
            "prompt_length": length_stats(prompt_lengths),
            "target_length": length_stats(target_lengths),
            "unique_prompt_ratio": prompt_ratio,
            "unique_target_ratio": target_ratio,
            "unique_example_ratio": example_ratio,
        },
        "warnings": warnings,
        "errors": errors,
        "preview": preview,
        "repeated_output_examples": repeated_output_examples[:PREVIEW_LIMIT],
        "rows": rows,
    }


def write_validation_report(report: dict, output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = strip_internal_rows(report)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return output


def default_report_path(
    dataset_path: str | Path,
    *,
    eval_output: str | Path | None = None,
    filename: str = "dataset-report.json",
) -> Path:
    dataset = Path(dataset_path)
    if eval_output is not None:
        eval_path = Path(eval_output)
        if dataset.parent == eval_path.parent:
            return dataset.parent / filename
    return dataset.with_suffix(".report.json")


def validate_dataset_command(
    dataset: str | Path,
    *,
    eval_dataset: str | Path | None = None,
    min_target_length: int = DEFAULT_MIN_TARGET_LENGTH,
    report_output: str | Path | None = None,
) -> dict:
    report = validate_training_datasets(dataset, eval_dataset=eval_dataset, min_target_length=min_target_length)
    resolved_report = Path(report_output or default_report_path(dataset, eval_output=eval_dataset, filename="validation-report.json"))
    write_validation_report(report, resolved_report)
    result = {
        "status": report["status"],
        "dataset": str(Path(dataset)),
        "eval_dataset": str(Path(eval_dataset)) if eval_dataset is not None else None,
        "report_output": str(resolved_report),
        "warnings": report["warnings"],
        "errors": report["errors"],
        "counts": {"train": report["train"]["counts"], "eval": report["eval"]["counts"] if report["eval"] else None},
    }
    if report["errors"]:
        raise ValueError(f"dataset validation failed; report written to {resolved_report}")
    return result


def ensure_datasets_are_trainable(
    train_dataset: str | Path,
    *,
    eval_dataset: str | Path | None = None,
    min_target_length: int = DEFAULT_MIN_TARGET_LENGTH,
) -> dict:
    report = validate_training_datasets(train_dataset, eval_dataset=eval_dataset, min_target_length=min_target_length)
    if report["errors"]:
        raise ValueError(format_validation_error(report))
    return report
