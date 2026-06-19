from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from repo_brain.models import TraceEvent, TraceRun
from repo_brain.tracing.redaction import redact


def _now() -> str:
    return datetime.now(UTC).isoformat()


class TraceStore:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        with sqlite3.connect(path) as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS trace_runs (
                    id TEXT PRIMARY KEY,
                    started_at TEXT NOT NULL,
                    repository_revision TEXT,
                    task TEXT,
                    status TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS trace_events (
                    run_id TEXT NOT NULL,
                    sequence INTEGER NOT NULL,
                    timestamp TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    PRIMARY KEY (run_id, sequence),
                    FOREIGN KEY (run_id) REFERENCES trace_runs(id)
                );
                """
            )

    def start_run(
        self, run_id: str, repository_revision: str | None, task: str | None = None
    ) -> None:
        safe_task = redact(task) if task is not None else None
        with sqlite3.connect(self.path) as connection:
            connection.execute(
                "INSERT INTO trace_runs VALUES (?, ?, ?, ?, ?)",
                (run_id, _now(), repository_revision, safe_task, "running"),
            )

    def add_event(self, run_id: str, event_type: str, payload: dict[str, object]) -> None:
        encoded = redact(json.dumps(payload, sort_keys=True))
        with sqlite3.connect(self.path) as connection:
            row = connection.execute(
                "SELECT COALESCE(MAX(sequence), 0) + 1 FROM trace_events WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            sequence = int(row[0]) if row else 1
            connection.execute(
                "INSERT INTO trace_events VALUES (?, ?, ?, ?, ?)",
                (run_id, sequence, _now(), event_type, encoded),
            )

    def finish_run(self, run_id: str, status: str) -> None:
        with sqlite3.connect(self.path) as connection:
            connection.execute(
                "UPDATE trace_runs SET status = ? WHERE id = ?", (status, run_id)
            )

    def list_runs(self) -> list[TraceRun]:
        with sqlite3.connect(self.path) as connection:
            rows = connection.execute(
                "SELECT id, started_at, repository_revision, task, status "
                "FROM trace_runs ORDER BY started_at DESC"
            )
            return [TraceRun(*row) for row in rows]

    def events(self, run_id: str) -> list[TraceEvent]:
        with sqlite3.connect(self.path) as connection:
            rows = connection.execute(
                "SELECT run_id, sequence, timestamp, event_type, payload "
                "FROM trace_events WHERE run_id = ? ORDER BY sequence",
                (run_id,),
            )
            return [
                TraceEvent(row[0], row[1], row[2], row[3], json.loads(row[4])) for row in rows
            ]
