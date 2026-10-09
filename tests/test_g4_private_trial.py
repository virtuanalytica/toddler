"""Guard the irreversible boundary around private G4 promotion."""

import json

import pytest

from scripts import g4_private_trial as G4


def test_evaluator_code_change_blocks_private_run(tmp_path):
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps({"schema": "toddler-g4-private-trial/v1",
                                    "tasks": G4.TASKS, "children": G4.CHILDREN,
                                    "parents": G4.PARENTS, "eval_mode": "sample",
                                    "evaluator_sha256": "0" * 64}))
    with pytest.raises(ValueError, match="evaluator code changed"):
        G4.run(protocol, 0, tmp_path)


def test_failed_private_gate_cannot_import_generation(tmp_path, monkeypatch):
    monkeypatch.setattr(G4, "assess", lambda protocol, root: {"promote": False})
    with pytest.raises(ValueError, match="did not pass"):
        G4.promote(tmp_path / "protocol.json", tmp_path)
    assert not (tmp_path / "G4-oracle").exists()
