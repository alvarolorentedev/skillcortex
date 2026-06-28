from __future__ import annotations

import hashlib
import json
from pathlib import Path


def package_fingerprint(slm_manifest: dict, metadata: dict) -> str:
    payload = json.dumps(
        {
            "slm_id": slm_manifest["slm_id"],
            "version": slm_manifest["version"],
            "base": metadata["base"],
            "adapter": metadata["adapter"],
            "checksums": metadata["checksums"],
        },
        sort_keys=True,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
