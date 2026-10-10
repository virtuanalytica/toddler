"""The searched cohort has a distinct, stricter frozen evaluation contract."""

import json
from types import SimpleNamespace

import pytest

from scripts import g4_search_private_trial as G4S


def _decision(p):
    arm = SimpleNamespace(p_value=p)
    trial = SimpleNamespace(passed=True, promotion_vs_ancestor=arm, promotion_vs_control=arm)
    return SimpleNamespace(promote=True, trials=(trial, trial))


def test_second_family_alpha_rejects_old_five_percent_result():
    assert not G4S.confirmatory_pass(_decision(0.037))
    assert G4S.confirmatory_pass(_decision(0.00397))


def test_new_evaluator_code_change_blocks_private_run(tmp_path):
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps({"schema": "toddler-g4-search-private-trial/v1",
                                    "tasks": G4S.TASKS, "children": G4S.CHILDREN,
                                    "parents": G4S.PARENTS, "eval_mode": "sample",
                                    "evaluator_sha256": "0" * 64}))
    with pytest.raises(ValueError, match="evaluator code changed"):
        G4S.run(protocol, 0, tmp_path)


def test_failed_confirmation_cannot_import_generation(tmp_path, monkeypatch):
    monkeypatch.setattr(G4S, "assess", lambda protocol, root: {"promote": False})
    with pytest.raises(ValueError, match="did not pass"):
        G4S.promote(tmp_path / "protocol.json", tmp_path)
    assert not (tmp_path / "G4-search").exists()
