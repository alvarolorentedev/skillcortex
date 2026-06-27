import hashlib
import json
import re
from collections import Counter
from pathlib import Path


REQUIRED_FIELDS = ("id", "task_type", "prompt", "target")
DEFAULT_MIN_TARGET_LENGTH = 24
PREVIEW_LIMIT = 3
REPEATED_PUNCTUATION_PATTERN = re.compile(r"([!?.,;:=_\-#*/])\\1{7,}")
TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def validate_training_datasets(
    train_dataset: str | Path,
    *,
    eval_dataset: str | Path | None = None,
    min_target_length: int = DEFAULT_MIN_TARGET_LENGTH,
) -> dict:
    train_report = validate_dataset(
        train_dataset,
        min_target_length=min_target_length,
    )
    eval_report = (
        validate_dataset(eval_dataset, min_target_length=min_target_length)
        if eval_dataset is not None
        else None
    )
    leakage = _detect_leakage(train_report, eval_report)
    cross_split = {
        "leakage_count": len(leakage),
        "leakage_examples": leakage[:PREVIEW_LIMIT],
        "warnings": [],
        "errors": [],
    }
    if leakage:
        cross_split["errors"].append(
            f"train/eval leakage detected for {len(leakage)} example(s)"
        )

    errors = list(train_report["errors"])
    warnings = list(train_report["warnings"])
    if eval_report is not None:
        errors.extend(eval_report["errors"])
        warnings.extend(eval_report["warnings"])
    errors.extend(cross_split["errors"])
    warnings.extend(cross_split["warnings"])

    return {
        "status": "invalid" if errors else "ok",
        "schema": {
            "required_fields": list(REQUIRED_FIELDS),
            "min_target_length": min_target_length,
        },
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
                errors.append(
                    f"{path}:{line_number}: missing required field(s): {', '.join(missing_fields)}"
                )
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
                row_errors.append(
                    f"target must be at least {min_target_length} characters"
                )

            if row_errors:
                invalid_count += 1
                errors.append(f"{path}:{line_number}: {'; '.join(row_errors)}")
                continue

            prompt_text = str(prompt).strip()
            target_text = str(target).strip()
            normalized = _example_hash(prompt_text, target_text)
            normalized_hashes[normalized] += 1
            prompt_hashes[_hash_text(_normalize_text(prompt_text))] += 1
            target_hashes[_hash_text(_normalize_text(target_text))] += 1
            prompt_lengths.append(len(prompt_text))
            target_lengths.append(len(target_text))

            repeated_findings = _detect_repeated_output(target_text)
            if repeated_findings:
                repeated_output_examples.append(
                    {
                        "id": example_id,
                        "line": line_number,
                        "findings": repeated_findings,
                    }
                )

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
                preview.append(
                    {
                        "id": example_id,
                        "task_type": task_type,
                        "prompt": _truncate(prompt_text, 140),
                        "target_preview": _truncate(target_text, 160),
                    }
                )

    if not rows and invalid_count == 0:
        errors.append(f"{path} is empty")

    duplicates = sorted(
        hash_value for hash_value, count in normalized_hashes.items() if count > 1
    )
    if duplicates:
        errors.append(f"{path}: duplicate examples detected: {len(duplicates)}")

    if repeated_output_examples:
        errors.append(
            f"{path}: repeated-token or repeated-punctuation outputs detected: {len(repeated_output_examples)}"
        )

    prompt_ratio = _unique_ratio(prompt_hashes, len(rows))
    target_ratio = _unique_ratio(target_hashes, len(rows))
    example_ratio = _unique_ratio(normalized_hashes, len(rows))
    if len(rows) >= 10 and (prompt_ratio < 0.7 or target_ratio < 0.7 or example_ratio < 0.75):
        warnings.append(
            (
                f"{path}: suspiciously low diversity "
                f"(prompt={prompt_ratio:.2f}, target={target_ratio:.2f}, example={example_ratio:.2f})"
            )
        )

    return {
        "dataset_path": str(path),
        "dataset_hash": _dataset_hash(path),
        "counts": {
            "total": len(rows) + invalid_count,
            "valid": len(rows),
            "invalid": invalid_count,
            "duplicates": len(duplicates),
        },
        "stats": {
            "prompt_length": _length_stats(prompt_lengths),
            "target_length": _length_stats(target_lengths),
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
    payload = _strip_internal_rows(report)
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
    report = validate_training_datasets(
        dataset,
        eval_dataset=eval_dataset,
        min_target_length=min_target_length,
    )
    resolved_report = Path(
        report_output
        or default_report_path(
            dataset,
            eval_output=eval_dataset,
            filename="validation-report.json",
        )
    )
    write_validation_report(report, resolved_report)
    result = {
        "status": report["status"],
        "dataset": str(Path(dataset)),
        "eval_dataset": str(Path(eval_dataset)) if eval_dataset is not None else None,
        "report_output": str(resolved_report),
        "warnings": report["warnings"],
        "errors": report["errors"],
        "counts": {
            "train": report["train"]["counts"],
            "eval": report["eval"]["counts"] if report["eval"] else None,
        },
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
    report = validate_training_datasets(
        train_dataset,
        eval_dataset=eval_dataset,
        min_target_length=min_target_length,
    )
    if report["errors"]:
        raise ValueError(_format_validation_error(report))
    return report


def _detect_leakage(train_report: dict, eval_report: dict | None) -> list[dict]:
    if eval_report is None:
        return []
    train_index = {row["hash"]: row for row in train_report["rows"]}
    leakage: list[dict] = []
    for row in eval_report["rows"]:
        match = train_index.get(row["hash"])
        if match is None:
            continue
        leakage.append(
            {
                "hash": row["hash"],
                "train_id": match["id"],
                "eval_id": row["id"],
            }
        )
    return leakage


def _detect_repeated_output(target: str) -> list[str]:
    findings: list[str] = []
    punct_match = REPEATED_PUNCTUATION_PATTERN.search(target)
    if punct_match:
        findings.append(f"repeated punctuation: {punct_match.group(0)[:16]}")

    tokens = TOKEN_PATTERN.findall(target)
    if not tokens:
        return findings
    max_run = 1
    current_run = 1
    repeated_token = tokens[0]
    for previous, current in zip(tokens, tokens[1:]):
        if current == previous:
            current_run += 1
            if current_run > max_run:
                max_run = current_run
                repeated_token = current
        else:
            current_run = 1
    if max_run >= 8:
        findings.append(f"repeated token '{repeated_token}' run length {max_run}")

    for window_size in (2, 3):
        if len(tokens) < window_size * 4:
            continue
        for index in range(0, len(tokens) - window_size * 4 + 1):
            window = tokens[index : index + window_size]
            repeats = 1
            cursor = index + window_size
            while cursor + window_size <= len(tokens) and tokens[cursor : cursor + window_size] == window:
                repeats += 1
                cursor += window_size
            if repeats >= 4:
                findings.append(
                    f"repeated token sequence {' '.join(window[:window_size])} x{repeats}"
                )
                return findings
    return findings


def _dataset_hash(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _example_hash(prompt: str, target: str) -> str:
    return _hash_text(f"{_normalize_text(prompt)}\n---\n{_normalize_text(target)}")


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _normalize_text(value: str) -> str:
    return " ".join(value.split())


def _truncate(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[: limit - 3] + "..."


def _unique_ratio(counter: Counter[str], total: int) -> float:
    if total <= 0:
        return 0.0
    return round(len(counter) / total, 4)


def _length_stats(lengths: list[int]) -> dict:
    if not lengths:
        return {"min": 0, "max": 0, "mean": 0.0}
    return {
        "min": min(lengths),
        "max": max(lengths),
        "mean": round(sum(lengths) / len(lengths), 2),
    }


def _strip_internal_rows(report: dict) -> dict:
    payload = dict(report)
    if "rows" in payload:
        payload.pop("rows")
    if payload.get("train") and "rows" in payload["train"]:
        payload["train"] = dict(payload["train"])
        payload["train"].pop("rows", None)
    if payload.get("eval") and "rows" in payload["eval"]:
        payload["eval"] = dict(payload["eval"])
        payload["eval"].pop("rows", None)
    return payload


def _format_validation_error(report: dict) -> str:
    details = report["errors"][:5]
    return "dataset validation failed: " + " | ".join(details)