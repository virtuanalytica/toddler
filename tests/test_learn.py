"""Learning and benchmark-comparability tests on real Gymnasium environments (no fabricated data)."""

import numpy as np
import pytest

pytest.importorskip("gymnasium")
torch = pytest.importorskip("torch")

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


def test_peer_experiment_has_both_controls_and_decision_rule():
    from toddler.learn import peer

    r = peer.run(teacher_steps=2048, student_steps=2048, student_seeds=(1, 2), eval_seeds=T.EVAL_SEEDS[:3])
    assert set(r.groups) == {"real_teacher", "no_teacher", "random_teacher"}
    assert all(len(v) == 2 for v in r.groups.values())
    # teaching may only be declared helpful when the real teacher beats BOTH controls
    assert r.teaching_helps == (r.p_vs_no_teacher < 0.05 and r.p_vs_random_teacher < 0.05)


def test_training_seeds_never_hit_held_out_seeds():
    rng = np.random.default_rng(0)
    draws = {T.train_seed(rng) for _ in range(20_000)}
    assert not draws & set(T.EVAL_SEEDS)


def test_aggregate_iqm_is_mean_of_per_task_iqms():
    scores = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    assert scoring.aggregate_iqm(scores) == 0.5


def test_gae_truncation_bootstraps_from_truncated_state_not_next_episode():
    # step 0 ends by truncation (boot V=10), step 1 is the first step of a new episode (V=99)
    adv = ppo.gae_advantages([1.0, 1.0], [0.0, 99.0], [True, False], [10.0, 0.0], last_value=0.0, gamma=1.0, lam=1.0)
    assert adv[0] == 11.0          # r + V(truncated state); no leak of the next episode's 99


def test_gae_termination_bootstraps_zero():
    adv = ppo.gae_advantages([1.0], [0.0], [True], [0.0], last_value=50.0, gamma=0.9, lam=0.95)
    assert adv[0] == 1.0


def test_gae_mid_episode_uses_last_value():
    adv = ppo.gae_advantages([1.0], [0.0], [False], [0.0], last_value=5.0, gamma=0.5, lam=1.0)
    assert adv[0] == 3.5


def test_return_scaler_brings_long_negative_returns_to_unit_scale():
    s = ppo.ReturnScaler(gamma=0.99)
    scaled = [s(-1.0, end=(i % 500 == 499)) for i in range(5000)]
    # unscaled discounted returns reach about -99; scaled rewards keep their sign and shrink
    assert all(x < 0 for x in scaled) and abs(scaled[-1]) < 0.2


def test_unscaled_method_version_stays_reproducible():
    """Generations recorded before return scaling became the default must stay reproducible."""
    torch.set_num_threads(2)
    n1, _ = ppo.train("cartpole", ppo.PPOConfig(seed=7, scale_rewards=False, **SMALL))
    n2, _ = ppo.train("cartpole", ppo.PPOConfig(seed=7, scale_rewards=False, **SMALL))
    n3, _ = ppo.train("cartpole", ppo.PPOConfig(seed=7, **SMALL))
    assert all(torch.equal(a, b) for a, b in zip(n1.parameters(), n2.parameters()))
    assert not all(torch.equal(a, b) for a, b in zip(n1.parameters(), n3.parameters()))


def test_behaviour_clone_and_run_bc_smoke():
    from toddler.learn import peer

    teacher, _ = ppo.train("cartpole", ppo.PPOConfig(seed=1, **SMALL))
    student = peer.behaviour_clone(teacher, "cartpole", steps=512, seed=3, epochs=2)
    assert student is not teacher and sum(p.numel() for p in student.parameters()) > 0
    r = peer.run_bc(teacher_steps=2048, budget=3072, clone_steps=1024, student_seeds=(1, 2), eval_seeds=T.EVAL_SEEDS[:3])
    assert set(r.groups) == {"real_teacher", "no_teacher", "random_teacher"} and r.student_steps == 3072


def test_statistics_refuse_degenerate_input():
    with pytest.raises(ValueError):
        scoring.iqm(np.array([]))
    with pytest.raises(ValueError):
        scoring.bootstrap_ci(np.array([0.1, 0.2]))
    with pytest.raises(ValueError):
        scoring.prob_improvement(np.array([0.1]), np.array([0.2]))
    with pytest.raises(ValueError):
        T.normalise("cartpole", 100.0, 500.0)          # anchor above the solve threshold
