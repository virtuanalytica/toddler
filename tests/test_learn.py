"""Learning and benchmark-comparability tests on real Gymnasium environments (no fabricated data)."""

import numpy as np
import pytest
import torch

pytest.importorskip("gymnasium")

from toddler.learn import ppo, scoring  # noqa: E402
from toddler.learn import tasks as T  # noqa: E402

SMALL = dict(total_steps=4096, rollout=1024, epochs=2, minibatch=256)


def test_random_anchor_is_measured_and_deterministic():
    a, b = T.random_anchor("cartpole", T.EVAL_SEEDS[:5]), T.random_anchor("cartpole", T.EVAL_SEEDS[:5])
    assert a == b and 5 < a < 60


def test_normalise_anchors():
    assert T.normalise("cartpole", 22.0, 22.0) == 0.0
    assert T.normalise("cartpole", T.TASKS["cartpole"].solved, 22.0) == 1.0


def test_same_seed_same_toddler_on_cpu():
    """Comparability across generations rests on this: a run is reproducible from its seed and
    step budget, independent of wall-clock time."""
    torch.set_num_threads(2)
    n1, l1 = ppo.train("cartpole", ppo.PPOConfig(seed=7, **SMALL))
    n2, l2 = ppo.train("cartpole", ppo.PPOConfig(seed=7, **SMALL))
    for p1, p2 in zip(n1.parameters(), n2.parameters()):
        assert torch.equal(p1, p2)
    assert l1.steps == l2.steps == SMALL["total_steps"]


def test_evaluation_is_deterministic_and_uses_held_out_seeds():
    net, _ = ppo.train("cartpole", ppo.PPOConfig(seed=3, **SMALL))
    r1, r2 = scoring.evaluate(net, "cartpole", T.EVAL_SEEDS[:5]), scoring.evaluate(net, "cartpole", T.EVAL_SEEDS[:5])
    assert np.array_equal(r1, r2) and min(T.EVAL_SEEDS) >= 10_000


def test_teacher_distillation_runs():
    teacher, _ = ppo.train("cartpole", ppo.PPOConfig(seed=1, **SMALL))
    student, log = ppo.train("cartpole", ppo.PPOConfig(seed=2, **SMALL), teacher=teacher)
    assert log.updates == 4 and student is not teacher


def test_iqm_ignores_tails():
    assert scoring.iqm(np.array([0, 1, 1, 1, 1, 1, 1, 100.0])) == 1.0


def test_bootstrap_ci_brackets_iqm():
    rng = np.random.default_rng(0)
    scores = rng.normal(0.5, 0.1, size=(20, 3))
    lo, hi = scoring.bootstrap_ci(scores, reps=500)
    assert lo <= scoring.iqm(scores) <= hi


def test_probability_of_improvement():
    better = np.full((5, 2), 0.9)
    worse = np.full((5, 2), 0.1)
    assert scoring.prob_improvement(better, worse) == 1.0
    assert scoring.prob_improvement(worse, better) == 0.0
    assert scoring.prob_improvement(better, better) == 0.5
