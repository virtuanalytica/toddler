"""A later skill cannot buy promotion by hiding regression on ancestor skills."""

from dataclasses import replace

import pytest

from toddler.learn.ancestor_gate import Trial, assess_lineage_replication, surviving_ancestors
from toddler.learn.lineage import Inheritance, Ledger


def _ledger(tmp_path):
    ledger = Ledger(tmp_path)
    parents = ()
    for generation in ("G1", "G2", "G3"):
        ref = f"{generation}/t1"
        inherit = (Inheritance(parents[0], ledger.weights_of(parents[0]), ("policy",)),) if parents else ()
        ledger.born(ref, generation, role="population", inherits=inherit,
                    code={}, data={}, budget={}, hardware={}, software={})
        ledger.selected(generation, "survived", {"trial": generation})
        parents = (ref,)
    return ledger


def _trial(seed_set_id, old_score=1.04):
    children = tuple(f"t{i}" for i in range(5))
    constant = lambda value: tuple([value] * 5)
    return Trial(seed_set_id, "b" * 64, children, {child: "a" * 64 for child in children},
                 {"old_skill": constant(old_score), "new_skill": constant(1.30)},
                 {"old_skill": constant(1.04), "new_skill": constant(0.60)},
                 {"G1": {"old_skill": constant(1.00)},
                  "G2": {"old_skill": constant(1.02), "new_skill": constant(0.20)},
                  "G3": {"old_skill": constant(1.04), "new_skill": constant(0.60)}})


def test_all_surviving_ancestors_get_a_matched_vote(tmp_path):
    ledger = _ledger(tmp_path)
    assert surviving_ancestors(ledger, ("G3/t1",)) == ("G1", "G2", "G3")
    outcome = assess_lineage_replication(ledger, ("G3/t1",), (_trial("fresh-a"), _trial("fresh-b")))
    assert outcome.promote is True
    assert all(trial.strongest_ancestor == "G3" for trial in outcome.trials)
    assert all(trial.promotion_vs_control.promote for trial in outcome.trials)


def test_new_skill_gain_does_not_hide_forgetting(tmp_path):
    ledger = _ledger(tmp_path)
    a, b = _trial("fresh-a", old_score=0.80), _trial("fresh-b", old_score=0.80)
    outcome = assess_lineage_replication(ledger, ("G3/t1",), (a, b))
    assert outcome.promote is False
    assert all(trial.promotion_vs_ancestor.promote for trial in outcome.trials)
    assert all(trial.regressions["old_skill"] > 0 for trial in outcome.trials)


def test_missing_ancestor_or_reused_secret_set_blocks_promotion(tmp_path):
    ledger = _ledger(tmp_path)
    first = _trial("fresh-a")
    with pytest.raises(ValueError, match="distinct hidden seed sets"):
        assess_lineage_replication(ledger, ("G3/t1",), (first, first))
    without_g1 = replace(first, ancestors={k: v for k, v in first.ancestors.items() if k != "G1"})
    with pytest.raises(ValueError, match="all and only surviving ancestors"):
        assess_lineage_replication(ledger, ("G3/t1",), (without_g1, replace(without_g1, seed_set_id="fresh-b")))
    changed_weights = replace(_trial("fresh-b"), candidate_weights={"t0": "c" * 64, **{
        f"t{i}": "a" * 64 for i in range(1, 5)}})
    with pytest.raises(ValueError, match="changed children, weights"):
        assess_lineage_replication(ledger, ("G3/t1",), (first, changed_weights))
