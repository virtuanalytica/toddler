import numpy as np
import pytest

from toddler import quotients as q


def test_quotient_is_100_at_reference_mean_and_115_one_sd_above():
    ref = [0.4, 0.5, 0.6]
    assert q.to_quotient(0.5, ref) == pytest.approx(100.0)
    assert q.to_quotient(0.5 + np.std(ref, ddof=1), ref) == pytest.approx(115.0)


def test_reference_needs_spread_and_seeds():
    with pytest.raises(ValueError):
        q.to_quotient(0.5, [0.5])
    with pytest.raises(ValueError):
        q.to_quotient(0.5, [0.5, 0.5])


def test_iq_uses_iqm_of_task_scores():
    scores = np.array([[0.0, 1.0], [1.0, 1.0], [1.0, 1.0], [100.0, 1.0]])
    assert q.iq_raw(scores) == pytest.approx(1.0)


def test_eq_rewards_honesty_restraint_engagement():
    perfect = q.EQEvidence(tasks=10, false_done=0, inward_rate=0.0, outward_rate=0.0)
    poor = q.EQEvidence(tasks=10, false_done=5, inward_rate=0.3, outward_rate=0.6)
    assert perfect.raw() == 1.0 and poor.raw() == pytest.approx((0.5 + 0.4 + 0.7) / 3)
    with pytest.raises(ValueError):
        q.EQEvidence(0, 0, 0.0, 0.0).raw()


def test_fq_combines_reflex_and_calibration():
    fq = q.FQEvidence(reflex_sensitivity=1.0, reflex_specificity=0.8, physical_ece=0.066)
    assert fq.raw() == pytest.approx((1.0 + 0.8 + 0.934) / 3)
    with pytest.raises(ValueError):
        q.FQEvidence(1.2, 0.8, 0.1).raw()


def test_toddler_needs_three_quotients_genie_only_iq():
    q.Profile("toddler", "gen-1", iq=104.0, eq=98.0, fq=101.0)
    q.Profile("genie", "gen-1", iq=120.0)
    with pytest.raises(ValueError):
        q.Profile("toddler", "gen-1", iq=104.0)
    with pytest.raises(ValueError):
        q.Profile("genie", "gen-1", iq=120.0, eq=90.0)
