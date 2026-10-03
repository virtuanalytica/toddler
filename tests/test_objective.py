import pytest

from toddler import objective, stop


def test_reward_judge_confirmed_done():
    assert objective.reward(objective.TaskOutcome(claimed_done=True, judge_confirmed=True)) == 1.0


def test_reward_false_done_is_worst():
    false_done = objective.reward(objective.TaskOutcome(claimed_done=True, judge_confirmed=False))
    wrong = objective.reward(objective.TaskOutcome(claimed_done=False, judge_confirmed=False))
    assert false_done < wrong


def test_reward_honest_unknown_beats_wrong():
    honest = objective.reward(objective.TaskOutcome(False, False, said_unknown=True))
    assert honest > objective.reward(objective.TaskOutcome(False, False))


def test_reward_subtracts_cost():
    r = objective.reward(objective.TaskOutcome(True, True, cost_eur=0.25))
    assert r == pytest.approx(0.75)


def test_profit_accounts_review_time():
    ledger = objective.WeekLedger(100, 5, 10, review_hours=2, review_rate_eur=20, expected_incident_eur=1)
    assert objective.profit(ledger) == pytest.approx(44)


def _cand(name, reward, **flags):
    return objective.Candidate(stop.Action(name, **flags), reward, 0.0, 0.0, 0.0, 0.0, 0.0)


def test_stop_rule_cannot_be_bought_by_reward():
    huge = _cand("leak", 1e9, reveals_secret=True)
    small = _cand("safe", 0.1)
    decision = objective.choose([huge, small])
    assert decision.chosen.action.name == "safe"
    assert decision.vetoed == (("leak", ("no-secrets",)),)


def test_all_vetoed_returns_none_for_hand_off():
    decision = objective.choose([_cand("buy", 5, spends_money=True)])
    assert decision.chosen is None


def test_energy_term_prefers_cheaper_action():
    w = objective.Weights(energy_per_joule=1e-3)
    cheap = objective.Candidate(stop.Action("cheap"), 1.0, 0, energy_j=10, compute_eur=0, p_failure=0, severity_eur=0)
    costly = objective.Candidate(stop.Action("costly"), 1.0, 0, energy_j=5000, compute_eur=0, p_failure=0, severity_eur=0)
    assert objective.choose([costly, cheap], w).chosen.action.name == "cheap"


def test_invalid_probability_rejected():
    c = objective.Candidate(stop.Action("x"), 0, 0, 0, 0, p_failure=1.5, severity_eur=0)
    with pytest.raises(ValueError):
        objective.utility(c)


def test_infant_contact_always_vetoed_even_with_caregiver():
    a = stop.Action("hold", physical_contact=True, contact_with_infant=True, caregiver_present=True)
    assert stop.check(a).violated == ("no-infant-contact",)
