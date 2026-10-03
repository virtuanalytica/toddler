"""Evaluation tests. Classification/CCA steps use scikit-learn's breast-cancer dataset
(real clinical measurements, Wisconsin diagnostic study) so no data is fabricated."""

import numpy as np
import pytest
from sklearn.datasets import load_breast_cancer

from toddler import evaluation as ev


def test_toddler_cannot_rate_itself():
    with pytest.raises(ValueError):
        ev.JudgedScore("scrape", 1.0, rater="Toddler")


def test_t_scores_standardised_per_task_type():
    scores = [ev.JudgedScore("a", v, "judge") for v in (1, 2, 3)] + [ev.JudgedScore("b", 10, "judge")]
    t = ev.t_scores(scores)
    assert t[1] == pytest.approx(50) and t[0] == pytest.approx(40) and t[3] == pytest.approx(50)


def test_behaviour_axes_are_separate_rates():
    ax = ev.BehaviourAxes(refused_doable=1, stalled=1, needless_handoff=0,
                          unrequested_action=3, vetoed_attempt=1, tasks=10)
    assert ax.inward == pytest.approx(0.2) and ax.outward == pytest.approx(0.4)


def test_horizon_report_exposes_missingness():
    rep = ev.horizon_report({"24m": 120, "48m": 120}, {"24m": 80, "48m": 72})
    assert [round(r.missing_rate, 3) for r in rep] == [0.333, 0.4]


def test_sens_spec_reported_apart():
    sens, spec = ev.sens_spec([1, 1, 1, 0, 0], [1, 1, 0, 0, 1])
    assert sens == pytest.approx(2 / 3) and spec == pytest.approx(0.5)


def test_nested_selection_on_real_data():
    data = load_breast_cancer()
    X, y = data.data[:, :10], data.target
    res = ev.nested_feature_selection(X, y, outer=3, inner=3, seed=0)
    assert res.accuracy > 0.85
    assert 0 <= res.selection_rate.min() and res.selection_rate.max() <= 1
    assert all(res.selection_rate[j] > 0.65 for j in res.selected)


def test_cca_bonferroni_never_below_raw_p():
    data = load_breast_cancer()
    res = ev.cca_test(data.data[:, :8], data.data[:, 20], n_tests=4)
    assert res.df == 8 and res.p_bonferroni >= res.p_value and 0 < res.canonical_r <= 1


def test_robust_without_extremes_returns_two_r():
    data = load_breast_cancer()
    r_all, r_trim = ev.robust_without_extremes(data.data[:, 0], data.data[:, 2])
    assert r_all > 0.9 and r_trim > 0.9


def test_promotion_needs_confirmatory_replication():
    explor = ev.Finding("x", 0.001, confirmatory=False)
    conf = ev.Finding("x", 0.01, confirmatory=True)
    assert not ev.promotion_allowed([explor, explor, conf])
    assert ev.promotion_allowed([conf, conf])


def test_compare_extremes_welch():
    f = ev.compare_extremes([1, 2, 3, 4], [10, 11, 12, 13], "worst differs", confirmatory=True)
    assert f.p_value < 0.01
