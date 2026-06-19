from __future__ import annotations

import re

SECRET = re.compile(
    r"(?i)\b(token|api[_-]?key|password|secret)\b\s*[:=]\s*[^\s,;]+"
)
HOME = re.compile(r"/Users/[^/\s]+|/home/[^/\s]+")


def redact(value: str) -> str:
    value = SECRET.sub(lambda match: f"{match.group(1)}=[REDACTED]", value)
    return HOME.sub("~", value)

