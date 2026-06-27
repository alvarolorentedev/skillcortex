from __future__ import annotations

import hashlib
import json
from pathlib import Path


def package_fingerprint(skill_manifest: dict, metadata: dict) -> str:
    payload = json.dumps(
        {
            "skill_id": skill_manifest["skill_id"],
            "version": skill_manifest["version"],
            "base": metadata["base"],
            "adapter": metadata["adapter"],
            "checksums": metadata["checksums"],
        },
        sort_keys=True,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
