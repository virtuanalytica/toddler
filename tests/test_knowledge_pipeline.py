"""The public research queue may schedule work, never grant mastery."""

import json
from datetime import date, timedelta

import pytest

from scripts.knowledge_pipeline import ROOT, inventory, load_sources, read_ledger, tick


def test_six_source_bundles_are_registered_and_counted():
    sources = load_sources(ROOT)
    assert len(sources) == 6
    assert all(len(row["sources"]) >= 2 for row in sources.values())
    state = inventory(ROOT)
    assert state["total"] == 10_000
    assert state["stages"] == {"needs_sources": 9_390, "needs_lesson": 594,
                                "needs_review": 16, "reviewed": 0}
    assert state["fully_reviewed"] is False


def test_daily_queue_is_idempotent_and_auditable(tmp_path):
    today = date(2026, 10, 11)
    first = tick(ROOT, tmp_path, daily_target=64, day=today)
    assert first["status"] == "queued" and first["count"] == 64
    assert first["position_before"] == 0
    assert len((tmp_path / "queues" / "2026-10-11.jsonl").read_text().splitlines()) == 64
    assert tick(ROOT, tmp_path, daily_target=64, day=today)["status"] == "already_queued"
    second = tick(ROOT, tmp_path, daily_target=64, day=today + timedelta(days=1))
    assert second["position_before"] == 64
    assert len(read_ledger(tmp_path / "ledger.jsonl")) == 2
    row = json.loads((tmp_path / "queues" / "2026-10-11.jsonl").read_text().splitlines()[0])
    assert row["mastery_claim"] is False
    assert "answer_key" not in row
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text(ledger.read_text().replace('"count": 64', '"count": 65', 1))
    with pytest.raises(ValueError, match="ledger event"):
        read_ledger(ledger)
