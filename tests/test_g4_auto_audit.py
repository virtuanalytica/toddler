"""The nightly handoff must preserve the private trial's single-attempt gate."""

import json
import pytest

from scripts import g4_auto_audit as audit


def _manifest(tmp_path, selected=5):
    path = tmp_path / "cohort" / "manifest.json"
    path.parent.mkdir()
    path.write_text(json.dumps({"created_utc": "20261010T001936Z-886c2e",
                                "navigation": {"children": [
                                    {"selected_specialist": i < selected} for i in range(5)]}}))
    return path


def test_public_failure_never_prepares_private_seeds(tmp_path, monkeypatch):
    path = _manifest(tmp_path, selected=4)
    monkeypatch.setattr(audit.G4, "validate_cohort", lambda manifest, root: (json.loads(path.read_text()), {}))
    monkeypatch.setattr(audit.G4, "prepare", lambda *args: pytest.fail("private seeds used"))
    result = audit.audit(path, tmp_path / "private", tmp_path / "registry")
    assert result["status"] == "public_gate_failed"
    assert result["private_seeds_created"] is False
    assert not (tmp_path / "private").exists()


def test_passed_gate_runs_once_then_resumes_same_protocol(tmp_path, monkeypatch):
    path = _manifest(tmp_path)
    family = tmp_path / "private"
    calls = []
    monkeypatch.setattr(audit.G4, "validate_cohort", lambda manifest, root: (json.loads(path.read_text()), {}))
    monkeypatch.setattr(audit.G4, "digest", lambda file: "fixed-sha")

    def prepare(manifest, out, root):
        calls.append("prepare")
        out.mkdir()
        protocol = out / "protocol.json"
        protocol.write_text("{}")
        (family / "family-attempt.json").write_text(json.dumps({"manifest_sha256": "fixed-sha", "out": str(out)}))
        return {"protocol": str(protocol)}

    def run(protocol, index, root):
        calls.append(f"run-{index}")
        (protocol.parent / f"trial-{index + 1}.json").write_text("{}")

    monkeypatch.setattr(audit.G4, "prepare", prepare)
    monkeypatch.setattr(audit.G4, "run", run)
    monkeypatch.setattr(audit.G4, "assess", lambda *args: {"promote": True})
    monkeypatch.setattr(audit.G4, "promote", lambda *args: {
        "generation": "G4-search", "children": 5, "ledger_verified": True})
    assert audit.audit(path, family, tmp_path / "registry")["status"] == "promoted"
    assert audit.audit(path, family, tmp_path / "registry")["status"] == "promoted"
    assert calls == ["prepare", "run-0", "run-1"]


def test_locked_family_rejects_another_cohort(tmp_path, monkeypatch):
    path = _manifest(tmp_path)
    family = tmp_path / "private"
    family.mkdir()
    (family / "family-attempt.json").write_text(json.dumps({"manifest_sha256": "older"}))
    monkeypatch.setattr(audit.G4, "validate_cohort", lambda manifest, root: (json.loads(path.read_text()), {}))
    monkeypatch.setattr(audit.G4, "digest", lambda file: "current")
    assert audit.audit(path, family, tmp_path / "registry")["status"] == "family_attempt_already_spent"


def test_newest_incomplete_run_does_not_fall_back(tmp_path):
    older = tmp_path / "20261010T010000Z-old"
    newer = tmp_path / "20261010T020000Z-new"
    older.mkdir()
    newer.mkdir()
    (older / "manifest.json").write_text(json.dumps({"status": "completed_candidate",
                                                     "created_utc": "20261010T010000Z-old"}))
    (newer / "manifest.json").write_text(json.dumps({"status": "incomplete"}))
    assert audit.latest_completed_manifest(tmp_path, max_age_hours=100000) is None


def test_stale_run_is_not_accepted(tmp_path):
    older = tmp_path / "20200101T000000Z-old"
    older.mkdir()
    (older / "manifest.json").write_text(json.dumps({"status": "completed_candidate",
                                                     "created_utc": "20200101T000000Z-old"}))
    assert audit.latest_completed_manifest(tmp_path) is None
