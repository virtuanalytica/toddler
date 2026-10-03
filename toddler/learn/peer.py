"""Toddlers training each other.

Teacher -> student: a trained toddler teaches a younger one by policy distillation (KL on the
student's own states, see ppo.train(teacher=...)).

The experiment that decides whether teaching helps compares, at an EQUAL step budget and over
several seeds, three groups of students:
  * real teacher     - distilled from a trained toddler
  * no teacher       - plain self-reinforcement learning
  * random teacher   - distilled from an untrained network (control: is it the teacher's
                       knowledge that helps, or just the extra loss term?)
Scores are normalised on held-out seeds on the CPU. Teaching "works" only when the real
teacher beats BOTH controls (one-sided Mann-Whitney U, and probability of improvement).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy import stats

from toddler.learn import ppo, scoring
from toddler.learn import tasks as T
from toddler.learn.policy import ActorCritic


@dataclass(frozen=True)
class PeerResult:
    task: str
    student_steps: int
    teacher_score: float
    groups: dict[str, list[float]]          # normalised scores per student seed
    iqm: dict[str, float]
    p_vs_no_teacher: float
    p_vs_random_teacher: float
    prob_improvement_vs_no_teacher: float
    prob_improvement_vs_random_teacher: float
    teaching_helps: bool

    def to_dict(self) -> dict:
        return asdict(self)


def _score(net: ActorCritic, task: str, anchor: float, seeds: tuple[int, ...]) -> float:
    return float(T.normalise(task, scoring.evaluate(net, task, seeds).mean(), anchor))


def run(task: str = "cartpole", teacher_steps: int = 150_000, student_steps: int = 30_000,
        student_seeds: tuple[int, ...] = (11, 12, 13, 14, 15), teacher_seed: int = 2,
        eval_seeds: tuple[int, ...] = T.EVAL_SEEDS, alpha: float = 0.05) -> PeerResult:
    anchor = T.random_anchor(task, eval_seeds)
    teacher, _ = ppo.train(task, ppo.PPOConfig(total_steps=teacher_steps, seed=teacher_seed))
    env = T.make(task)
    random_teacher = ActorCritic(env.observation_space.shape[0], env.action_space.n)
    env.close()
    groups: dict[str, list[float]] = {"real_teacher": [], "no_teacher": [], "random_teacher": []}
    for s in student_seeds:
        cfg = ppo.PPOConfig(total_steps=student_steps, seed=s)
        for name, t in (("real_teacher", teacher), ("no_teacher", None), ("random_teacher", random_teacher)):
            net, _ = ppo.train(task, cfg, teacher=t)
            groups[name].append(_score(net, task, anchor, eval_seeds))
    real = np.asarray(groups["real_teacher"])
    p_none = float(stats.mannwhitneyu(real, groups["no_teacher"], alternative="greater").pvalue)
    p_rand = float(stats.mannwhitneyu(real, groups["random_teacher"], alternative="greater").pvalue)
    pi_none = scoring.prob_improvement(real[:, None], np.asarray(groups["no_teacher"])[:, None])
    pi_rand = scoring.prob_improvement(real[:, None], np.asarray(groups["random_teacher"])[:, None])
    return PeerResult(task, student_steps, _score(teacher, task, anchor, eval_seeds), groups,
                      {k: scoring.iqm(np.asarray(v)) for k, v in groups.items()},
                      p_none, p_rand, pi_none, pi_rand, bool(p_none < alpha and p_rand < alpha))
