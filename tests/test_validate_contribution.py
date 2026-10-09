"""Contributed trial PRs disclose provenance without publishing private seeds."""

import hashlib
import json
from pathlib import Path

import pytest

from scripts.validate_contribution import validate


def _write(tmp_path, *, private_model=True):
    root = tmp_path / "repo"
    folder = root / "submissions" / "alice" / "g4-test"
    folder.mkdir(parents=True)
    protocol = {"seed_set_id": "public-a", "tasks": ["cartpole"], "control": "G3-recombined"}
    protocol_path = folder / "protocol.json"
    protocol_path.write_text(json.dumps(protocol))
    protocol_hash = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    report_path = folder / "report.json"
    report_path.write_text(json.dumps({"seed_set_id": "public-a", "protocol_sha256": protocol_hash,
                                       "candidate_iqm": 0.9, "control_iqm": 0.8}))
    report_hash = hashlib.sha256(report_path.read_bytes()).hexdigest()
    manifest = {"schema": "toddler-contribution/v1",
                "contributor": {"github": "alice", "name": "Alice"},
                "candidate": {"generation": "G4-alice", "id": "t4001",
                              "weights_sha256": "a" * 64, "parent_refs": ["G3-recombined/t5001"]},
                "recipe": {"summary": "Train from G3 on new tasks", "commands": ["python train.py"],
                           "environment": {"torch": "2.6"},
                           "data_sources": [{"id": "MiniGrid", "provenance": "gymnasium package",
                                             "visibility": "public"}],
                           "models_used": [{"provider": "local", "model_id": "model-x", "revision": "sha123",
                                            "role": "teacher", "usage": "feedback on actions",
                                            "visibility": "private" if private_model else "public"}],
                           "interactions": [{"kind": "human feedback", "count": 5,
                                             "log_sha256": "b" * 64}]},
                "trials": [{"kind": "contributor_public", "seed_set_id": "public-a",
                            "matched_control": "G3-recombined/t5001", "eval_mode": "sample",
                            "protocol": "protocol.json", "protocol_sha256": protocol_hash,
                            "report": "report.json", "report_sha256": report_hash}]}
    path = folder / "submission.json"
    path.write_text(json.dumps(manifest))
    return root, path, manifest, report_path


def test_valid_disclosure_accepts_private_model_metadata_without_private_weights(tmp_path):
    root, path, _, _ = _write(tmp_path)
    result = validate(path, root)
    assert result["models_disclosed"] == 1
    assert result["promotion"] == "requires independent official hidden trials"


def test_missing_model_disclosure_or_changed_report_is_rejected(tmp_path):
    root, path, manifest, report_path = _write(tmp_path)
    del manifest["recipe"]["models_used"]
    path.write_text(json.dumps(manifest))
    with pytest.raises(KeyError):
        validate(path, root)
    manifest["recipe"]["models_used"] = []
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="disclose every model"):
        validate(path, root)
    manifest["recipe"]["models_used"] = [{"provider": "local", "model_id": "x", "revision": "r",
                                            "role": "teacher", "usage": "feedback", "visibility": "private"}]
    path.write_text(json.dumps(manifest))
    report_path.write_text('{"changed":true}')
    with pytest.raises(ValueError, match="hash mismatch"):
        validate(path, root)


def test_raw_private_seed_values_and_path_escape_are_rejected(tmp_path):
    root, path, manifest, _ = _write(tmp_path)
    manifest["trials"][0]["seeds"] = [123456]
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="raw secrets"):
        validate(path, root)
    del manifest["trials"][0]["seeds"]
    manifest["trials"][0]["protocol"] = "../protocol.json"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="inside the submission"):
        validate(path, root)
