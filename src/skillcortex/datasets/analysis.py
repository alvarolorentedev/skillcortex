import hashlib
import re
from collections import Counter
from pathlib import Path


PREVIEW_LIMIT = 3
REPEATED_PUNCTUATION_PATTERN = re.compile(r"([!?.,;:=_\-#*/])\\1{7,}")
TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def detect_leakage(train_report: dict, eval_report: dict | None) -> list[dict]:
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


def detect_repeated_output(target: str) -> list[str]:
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
                findings.append(f"repeated token sequence {' '.join(window[:window_size])} x{repeats}")
                return findings
    return findings


def dataset_hash(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def example_hash(prompt: str, target: str) -> str:
    return hash_text(f"{normalize_text(prompt)}\n---\n{normalize_text(target)}")


def hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def normalize_text(value: str) -> str:
    return " ".join(value.split())


def truncate(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[: limit - 3] + "..."


def unique_ratio(counter: Counter[str], total: int) -> float:
    if total <= 0:
        return 0.0
    return round(len(counter) / total, 4)


def length_stats(lengths: list[int]) -> dict:
    if not lengths:
        return {"min": 0, "max": 0, "mean": 0.0}
    return {
        "min": min(lengths),
        "max": max(lengths),
        "mean": round(sum(lengths) / len(lengths), 2),
    }


def strip_internal_rows(report: dict) -> dict:
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


def format_validation_error(report: dict) -> str:
    details = report["errors"][:5]
    return "dataset validation failed: " + " | ".join(details)
