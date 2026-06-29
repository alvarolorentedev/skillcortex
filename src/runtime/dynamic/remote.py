from __future__ import annotations

import json
import urllib.request


def fetch_remote_lora_catalog(catalog_url: str) -> list[dict]:
    with urllib.request.urlopen(catalog_url) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("entries") or payload.get("remote_lora_catalog") or []
    if not isinstance(payload, list):
        raise ValueError("remote catalog payload must be a list of entries")
    return [entry for entry in payload if isinstance(entry, dict)]


def remote_catalog_match(words: set[str], config: dict, *, fetch_catalog=fetch_remote_lora_catalog) -> tuple[list[str], list[str]]:
    scored = []
    remote_entries = list(config.get("remote_lora_catalog") or [])
    catalog_url = config.get("remote_lora_catalog_url")
    if isinstance(catalog_url, str) and catalog_url.strip():
        remote_entries.extend(fetch_catalog(catalog_url))
    for entry in remote_entries:
        if not isinstance(entry, dict):
            continue
        slm_id = entry.get("slm_id")
        source = entry.get("source")
        cues = entry.get("cues") or []
        if not isinstance(slm_id, str) or not isinstance(source, str):
            continue
        fields = [entry.get("name"), entry.get("description"), *cues]
        fields.extend(entry.get("task_types") or [])
        fields.extend(entry.get("semantic_families") or [])
        score = sum(
            1
            for field in fields
            if isinstance(field, str)
            and any(word in field.lower() or field.lower() in word for word in words)
        )
        if score:
            scored.append((score, slm_id, source))
    if not scored:
        return [], []
    _score, slm_id, source = sorted(scored, reverse=True)[0]
    return [slm_id], [source]
