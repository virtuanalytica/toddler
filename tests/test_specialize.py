import pytest

from toddler import fastpath, objective, specialize as sp, stop

ROLE = sp.Role("Data Steward", "data-governance", "senior", ("datakwaliteitscontrole",))


def _expert(mid, q=0.8, eur=0.001, j=50.0, domains=("dq",)):
    return sp.Expert(mid, frozenset(domains), q, eur, j, evidence="mlflow run 123")


def test_cannot_shadow_base_stop_rule():
    weak = stop.StopRule("no-secrets", "weaker", lambda a: False)
    with pytest.raises(sp.UnsafeSpecialisation):
        sp.build(ROLE, extra_stop_rules=[weak])


def test_extra_rule_is_added_and_enforced():
    rule = stop.StopRule("no-prod-delete", "never delete production data", lambda a: "delete-prod" in a.tags)
    spec = sp.build(ROLE, extra_stop_rules=[rule])
    cand = objective.Candidate(stop.Action("wipe", tags=frozenset({"delete-prod"})), 9, 0, 0, 0, 0, 0)
    assert objective.choose([cand], rules=spec.stop_rules()).chosen is None
    assert len(spec.stop_rules()) == len(stop.DEFAULT_RULES) + 1


def test_jev_question_added_but_base_kept():
    q = fastpath.PhysicalQuestion("spill_risk", "Will the liquid spill?", 0.4, 0.2)
    spec = sp.build(ROLE, jev_questions=[q])
    assert {x.qid for x in spec.questions()} >= {"person_in_zone", "spill_risk"}
    with pytest.raises(sp.UnsafeSpecialisation):
        sp.build(ROLE, jev_questions=[fastpath.PhysicalQuestion("person_in_zone", "x", 0.9, 0.8)])


def test_expert_needs_evidence():
    with pytest.raises(ValueError):
        sp.Expert("m", frozenset({"dq"}), 0.9, 0, 0, evidence="")


def test_router_prefers_cheaper_equal_expert_and_falls_back_to_base():
    spec = sp.build(ROLE, experts=[_expert("big", 0.85, eur=0.02, j=5000), _expert("small", 0.85, eur=0.001, j=50)])
    r = sp.route(spec, "dq", tokens_k=10, eur_per_joule=0.30 / 3.6e6)
    assert r.model_id == "small"
    assert sp.route(spec, "legal", tokens_k=10, eur_per_joule=0).model_id == "mixture-of-models"


def _fast(mid, q, tps, domains=("dq",)):
    return sp.Expert(mid, frozenset(domains), q, 0.0, 50.0, evidence="report row", tokens_per_second=tps)


def test_speed_floor_excludes_slow_expert_even_if_better():
    spec = sp.build(ROLE, experts=[_fast("slow-best", 0.95, 20), _fast("fast-ok", 0.85, 100)])
    assert sp.route(spec, "dq", tokens_k=1, eur_per_joule=0).model_id == "slow-best"
    r = sp.route(spec, "dq", tokens_k=1, eur_per_joule=0, min_tokens_per_second=88)
    assert r.model_id == "fast-ok"


def test_speed_floor_excludes_expert_without_measured_speed():
    spec = sp.build(ROLE, experts=[_expert("unmeasured", 0.99)])
    r = sp.route(spec, "dq", tokens_k=1, eur_per_joule=0, min_tokens_per_second=10)
    assert r.model_id == "mixture-of-models"


def test_speed_floor_must_be_positive_and_speed_must_be_valid():
    spec = sp.build(ROLE, experts=[_fast("a", 0.9, 100)])
    with pytest.raises(ValueError):
        sp.route(spec, "dq", tokens_k=1, eur_per_joule=0, min_tokens_per_second=0)
    with pytest.raises(ValueError):
        sp.Expert("m", frozenset({"dq"}), 0.9, 0, 0, evidence="x", tokens_per_second=0)
