from datetime import date

import pytest

from toddler import fastpath as fp
from toddler import provenance as pv

CALM = fp.Sensors(force_n=10, speed_m_s=0.1, person_distance_m=2.0)


def _answers(**p):
    base = {"grasp_slip": 0.05, "person_in_zone": 0.02, "collision_soon": 0.01, "unstable_object": 0.05}
    base.update(p)
    return [fp.Answer(k, v, 0.02) for k, v in base.items()]


def test_hard_limit_overrides_confident_jev():
    r = fp.reflex(fp.Sensors(200, 0.1, 2.0), _answers(), elapsed_ms=5)
    assert r.command == "stop" and "force" in r.because[0]


def test_late_answer_means_stop():
    assert fp.reflex(CALM, _answers(), elapsed_ms=50).command == "stop"


def test_missing_answer_means_stop():
    assert fp.reflex(CALM, _answers()[:2], elapsed_ms=5).command == "stop"


def test_slow_and_continue():
    assert fp.reflex(CALM, _answers(), elapsed_ms=5).command == "continue"
    assert fp.reflex(CALM, _answers(person_in_zone=0.15), elapsed_ms=5).command == "slow"


def test_high_uncertainty_counts_at_upper_bound():
    answers = [a if a.qid != "collision_soon" else fp.Answer("collision_soon", 0.05, 0.4) for a in _answers()]
    assert fp.reflex(CALM, answers, elapsed_ms=5).command == "stop"


def _src(**kw):
    base = dict(source_id="s1", url="https://doi.org/10.1002/hbm.23459", licence="Taverne",
                retrieved=date(2026, 10, 3), sha256="a" * 64)
    base.update(kw)
    return pv.Source(**base)


def test_register_rejects_synthetic_and_unconsented():
    reg = pv.Register()
    with pytest.raises(pv.RejectedSource):
        reg.admit(_src(synthetic=True))
    with pytest.raises(pv.RejectedSource):
        reg.admit(_src(involves_people=True))
    reg.admit(_src(involves_people=True, consent_reference="NHG DSRB / CIRB"))
    assert "doi.org" in reg.cite("s1")


def test_negative_uncertainty_or_bad_probability_stops():
    bad = [a if a.qid != "grasp_slip" else fp.Answer("grasp_slip", 0.05, -0.1) for a in _answers()]
    assert fp.reflex(CALM, bad, elapsed_ms=5).command == "stop"
    bad = [a if a.qid != "grasp_slip" else fp.Answer("grasp_slip", 1.5, 0.0) for a in _answers()]
    assert fp.reflex(CALM, bad, elapsed_ms=5).command == "stop"


def test_register_rejects_non_hex_hash_and_names_unknown_ids():
    reg = pv.Register()
    with pytest.raises(pv.RejectedSource):
        reg.admit(_src(sha256="z" * 64))
    with pytest.raises(KeyError, match="not registered"):
        reg.cite("nope")
