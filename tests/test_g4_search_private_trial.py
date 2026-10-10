"""The searched cohort has a distinct, stricter frozen evaluation contract."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import pytest

from scripts import g4_search_private_trial as G4S


def test_paired_hidden_seed_gate_uses_thirty_matched_maps():
    reference = np.full(30, 1.0)
    gain = np.full(30, 1.1)
    evidence = G4S.paired_seed_evidence(gain, reference)
    assert evidence["passed"] is True
    assert evidence["wins"] == 30 and evidence["p_value"] < G4S.CONFIRMATORY_ALPHA
    stalled = G4S.paired_seed_evidence(np.r_[np.full(10, 1.1), np.full(20, 0.9)], reference)
    assert stalled["passed"] is False


def test_private_seed_aggregate_tampering_is_refused():
    tasks = {task: [1.0] * 5 for task in G4S.TASKS}
    matrices = {task: [[1.0] * 30 for _ in range(5)] for task in G4S.TASKS}
    raw = {"seed_count": 30, "candidate": tasks, "control": tasks,
           "ancestors": {"G3-recombined": tasks},
           "per_seed": {arm: matrices for arm in ("candidate", "control", "G3-recombined")}}
    assert G4S._validated_seed_means(raw)["candidate"].shape == (30,)
    raw["candidate"]["unlockpickup"] = [0.0] * 5
    with pytest.raises(ValueError, match="aggregate differs"):
        G4S._validated_seed_means(raw)


def test_two_matched_private_sets_promote_only_after_both_pass(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.json"
    manifest.write_text("frozen cohort")
    weights = {child: "a" * 64 for child in G4S.CHILDREN}
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps({"schema": "toddler-g4-search-private-trial/v2",
                                    "evaluator_sha256": G4S.digest(Path(G4S.__file__)),
                                    "manifest": str(manifest), "manifest_sha256": G4S.digest(manifest),
                                    "candidate_weights": weights,
                                    "seed_commitments": [{"set_id": "set-a"}, {"set_id": "set-b"}]}))

    def raw(set_id, gain):
        control_rows = {task: [[1.0] * 30 for _ in range(5)] for task in G4S.TASKS}
        candidate_rows = json.loads(json.dumps(control_rows))
        for child in range(4):
            candidate_rows["unlockpickup"][child] = [1.0 + gain] * 30
        def aggregate(matrices):
            return {task: [sum(values) / 30 for values in rows]
                    for task, rows in matrices.items()}
        return {"seed_set_id": set_id, "frozen_plan_sha256": G4S.digest(protocol),
                "child_ids": G4S.CHILDREN, "candidate_weights": weights, "seed_count": 30,
                "candidate": aggregate(candidate_rows), "control": aggregate(control_rows),
                "ancestors": {"G3-recombined": aggregate(control_rows)},
                "per_seed": {"candidate": candidate_rows, "control": control_rows,
                             "G3-recombined": control_rows}}

    for number, set_id in enumerate(("set-a", "set-b"), 1):
        (tmp_path / f"trial-{number}.json").write_text(json.dumps(raw(set_id, 0.5)))
    monkeypatch.setattr(G4S, "Ledger", lambda root: SimpleNamespace(verify=lambda: (True, None)))
    retention = SimpleNamespace(strongest_ancestor="G3-recombined", regressions={}, retention_floors={})
    monkeypatch.setattr(G4S, "assess_lineage_replication", lambda *args: SimpleNamespace(
        required_ancestors=("G3-recombined",), trials=(retention, retention)))
    assert G4S.assess(protocol, tmp_path)["promote"] is True
    (tmp_path / "trial-2.json").write_text(json.dumps(raw("set-b", 0.0)))
    assert G4S.assess(protocol, tmp_path)["promote"] is False


def test_new_evaluator_code_change_blocks_private_run(tmp_path):
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps({"schema": "toddler-g4-search-private-trial/v2",
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


def test_self_reward_winner_requires_real_training_provenance():
    row = {"search": {"winner": "g3-self-ppo-4096",
                      "selected_profile": "g3-self-ppo-4096", "profiles": [{
        "profile": "g3-self-ppo-4096",
        "training": {"method": "self_reward_ppo", "steps": 4096,
                     "uses_teacher_grid": False},
    }]}}
    assert G4S.winner_training(row)["steps"] == 4096
    row["search"]["profiles"][0]["training"]["uses_teacher_grid"] = True
    with pytest.raises(ValueError, match="fixed CPU training contract"):
        G4S.winner_training(row)


def test_combined_profile_and_retained_prior_require_distinct_provenance():
    combined = {"method": "oracle_then_self_reward", "disjoint_training_seeds": True,
                "imitation": {"method": "oracle_imitation", "oracle_steps": 100,
                              "teacher_privileged_grid": True},
                "own_interaction": {"method": "self_reward_ppo", "steps": 4096,
                                    "uses_teacher_grid": False}}
    row = {"search": {"selected_profile": "g3-continue-h64-128x12-then-self-ppo-4096",
                      "profiles": [{"profile": "g3-continue-h64-128x12-then-self-ppo-4096", "training": combined}]}}
    assert G4S.winner_training(row)["method"] == "oracle_then_self_reward"
    combined["disjoint_training_seeds"] = False
    with pytest.raises(ValueError, match="combined profile"):
        G4S.winner_training(row)
    row["search"]["selected_profile"] = "previous-unchanged"
    row["search"]["profiles"] = [{"profile": "previous-unchanged",
                                   "training": {"method": "unchanged_hash_verified_prior", "steps": 0}}]
    assert G4S.winner_training(row)["steps"] == 0


def test_public_selection_recomputes_prior_rollback_from_raw_scores():
    profiles = [{"profile": name, "hidden": 64,
                 "development_mean_raw": 0.1, "development_mean": 0.1,
                 "training": {"method": method}}
                for name, method in (G4S.BASE_PROFILES | G4S.PRIOR_PROFILES).items()]
    for profile in profiles:
        if profile["profile"] == "previous-unchanged":
            profile["development_mean_raw"] = profile["development_mean"] = 1.00111
        if profile["profile"] == "previous-self-ppo-4096":
            profile["development_mean_raw"] = profile["development_mean"] = 1.01243
    search = {"selection_rule": G4S.SELECTION_RULE, "profiles": profiles,
              "winner": "previous-self-ppo-4096", "selected_profile": "previous-unchanged",
              "selected_specialist": True, "winner_public_check_mean": 0.96736,
              "selection_evidence": {"winner_development_mean": 1.01243,
                                     "winner_public_check_mean": 0.96736,
                                     "parent_development_mean": 0.0,
                                     "parent_public_check_mean": 0.0,
                                     "previous_development_mean": 1.00111,
                                     "previous_public_check_mean": 1.01146},
              "previous_baseline": {"development_mean": 1.00111,
                                    "public_check_mean": 1.01146},
              "development": {"candidate_mean": 1.00111, "parent_mean": 0.0},
              "public_check": {"candidate_mean": 1.01146, "parent_mean": 0.0}}
    assert G4S.verify_search_selection(search, True) == "previous-unchanged"
    search["selected_profile"] = "previous-self-ppo-4096"
    with pytest.raises(ValueError, match="route selection"):
        G4S.verify_search_selection(search, True)
