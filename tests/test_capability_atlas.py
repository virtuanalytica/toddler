"""The crowdsourcing atlas must retain its stable addresses and honest labels."""

from pathlib import Path

from scripts.capability_atlas import ROOT, validate


def test_materialized_capability_atlas_is_complete():
    assert validate(ROOT)["validated_slots"] == 10_000
    assert len(list(Path(ROOT).rglob("lesson.md"))) >= 10
