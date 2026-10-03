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

Why an unpaired test: the groups share student seeds, but their training diverges from the
first update (different losses), so a seed does not pair two outcomes; Mann-Whitney U is the
conservative, distribution-free choice and is also the test registered for the follow-up.

The behaviour-cloning variant (behaviour_clone, run_bc) is that pre-registered follow-up
(docs/learn/PREREG_peer_bc.md): the teacher's knowledge is passed by cloning its actions on
teacher rollouts that count against the student's step budget, then self-reinforcement learning.

Method version: both registered experiments ran before return scaling became the PPO default
(docs/learn/return_scaling.json), so run() and run_bc() pin scale_rewards=False to stay
reproducible. A run with scaling is a new experiment and needs its own registration.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch
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
        eval_seeds: tuple[int, ...] = T.EVAL_SEEDS, alpha: float = 0.05,
        scale_rewards: bool = False) -> PeerResult:
    anchor = T.random_anchor(task, eval_seeds)
    teacher, _ = ppo.train(task, ppo.PPOConfig(total_steps=teacher_steps, seed=teacher_seed, scale_rewards=scale_rewards))
    env = T.make(task)
    random_teacher = ActorCritic(env.observation_space.shape[0], env.action_space.n)
    env.close()
    groups: dict[str, list[float]] = {"real_teacher": [], "no_teacher": [], "random_teacher": []}
    for s in student_seeds:
        cfg = ppo.PPOConfig(total_steps=student_steps, seed=s, scale_rewards=scale_rewards)
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


def behaviour_clone(teacher: ActorCritic, task: str, steps: int, seed: int,
                    epochs: int = 20, minibatch: int = 256, lr: float = 1e-3) -> ActorCritic:
    """Collect `steps` environment steps of the teacher's greedy policy and train a fresh student
    to predict the teacher's actions (cross-entropy). Those steps count against the student's
    budget, so groups stay comparable."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    env = T.make(task)
    obs_list, act_list = [], []
    obs, _ = env.reset(seed=T.train_seed(rng))
    teacher = teacher.to("cpu").eval()
    for _ in range(steps):
        with torch.no_grad():
            a = int(torch.argmax(teacher(torch.as_tensor(obs, dtype=torch.float32))[0]))
        obs_list.append(obs)
        act_list.append(a)
        obs, _, term, trunc, _ = env.step(a)
        if term or trunc:
            obs, _ = env.reset(seed=T.train_seed(rng))
    student = ActorCritic(env.observation_space.shape[0], env.action_space.n)
    env.close()
    X = torch.as_tensor(np.asarray(obs_list), dtype=torch.float32)
    Y = torch.as_tensor(act_list)
    opt = torch.optim.Adam(student.parameters(), lr=lr)
    idx = np.arange(len(X))
    for _ in range(epochs):
        rng.shuffle(idx)
        for s in range(0, len(idx), minibatch):
            b = torch.as_tensor(idx[s:s + minibatch])
            loss = torch.nn.functional.cross_entropy(student(X[b])[0], Y[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
    return student


def run_bc(task: str = "cartpole", teacher_steps: int = 150_000, budget: int = 30_000, clone_steps: int = 10_000,
           student_seeds: tuple[int, ...] = (11, 12, 13, 14, 15), teacher_seed: int = 2,
           eval_seeds: tuple[int, ...] = T.EVAL_SEEDS, alpha: float = 0.05,
           scale_rewards: bool = False) -> PeerResult:
    """Pre-registered follow-up (docs/learn/PREREG_peer_bc.md): behaviour-cloning warm start."""
    anchor = T.random_anchor(task, eval_seeds)
    teacher, _ = ppo.train(task, ppo.PPOConfig(total_steps=teacher_steps, seed=teacher_seed, scale_rewards=scale_rewards))
    env = T.make(task)
    random_teacher = ActorCritic(env.observation_space.shape[0], env.action_space.n)
    env.close()
    groups: dict[str, list[float]] = {"real_teacher": [], "no_teacher": [], "random_teacher": []}
    for s in student_seeds:
        for name, t in (("real_teacher", teacher), ("no_teacher", None), ("random_teacher", random_teacher)):
            if t is None:
                net, _ = ppo.train(task, ppo.PPOConfig(total_steps=budget, seed=s, scale_rewards=scale_rewards))
            else:
                warm = behaviour_clone(t, task, clone_steps, seed=s)
                net, _ = ppo.train(task, ppo.PPOConfig(total_steps=budget - clone_steps, seed=s, scale_rewards=scale_rewards), net=warm)
            groups[name].append(_score(net, task, anchor, eval_seeds))
    real = np.asarray(groups["real_teacher"])
    p_none = float(stats.mannwhitneyu(real, groups["no_teacher"], alternative="greater").pvalue)
    p_rand = float(stats.mannwhitneyu(real, groups["random_teacher"], alternative="greater").pvalue)
    return PeerResult(task, budget, _score(teacher, task, anchor, eval_seeds), groups,
                      {k: scoring.iqm(np.asarray(v)) for k, v in groups.items()}, p_none, p_rand,
                      scoring.prob_improvement(real[:, None], np.asarray(groups["no_teacher"])[:, None]),
                      scoring.prob_improvement(real[:, None], np.asarray(groups["random_teacher"])[:, None]),
                      bool(p_none < alpha and p_rand < alpha))
