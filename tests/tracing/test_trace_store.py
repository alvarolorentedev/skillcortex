from pathlib import Path

from repo_brain.tracing.trace_store import TraceStore


def test_trace_store_records_ordered_events(tmp_path: Path) -> None:
    store = TraceStore(tmp_path / "traces.sqlite3")
    store.start_run("run-1", "abc", "fix login")
    store.add_event("run-1", "localize", {"paths": ["auth.py"]})
    store.add_event("run-1", "validate", {"passed": True})
    store.finish_run("run-1", "passed")
    runs = store.list_runs()
    events = store.events("run-1")
    assert runs[0].status == "passed"
    assert [event.sequence for event in events] == [1, 2]
    assert events[0].payload["paths"] == ["auth.py"]
