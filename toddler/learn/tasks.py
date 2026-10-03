"""Benchmark tasks and their normalisation anchors.

A score is normalised as (return - random) / (solved - random): 0 = a random policy, 1 = the
task's documented solve threshold (Gymnasium `reward_threshold`). The random anchor is
MEASURED with a uniform random policy on the same fixed evaluation seeds, never assumed, so
scores stay comparable across generations and hardware.
"""

from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import numpy as np


@dataclass(frozen=True)
class Task:
    env_id: str
    solved: float          # documented solve threshold (Gymnasium's own TimeLimit applies)


TASKS: dict[str, Task] = {
    "cartpole": Task("CartPole-v1", 475.0),
    "acrobot": Task("Acrobot-v1", -100.0),
    "mountaincar": Task("MountainCar-v0", -110.0),
}

EVAL_SEEDS: tuple[int, ...] = tuple(range(10_000, 10_030))   # held out: never used for training
TRAIN_SEED_LOW = 1_000_000     # training resets draw from [TRAIN_SEED_LOW, 2**31), disjoint from EVAL_SEEDS
assert max(EVAL_SEEDS) < TRAIN_SEED_LOW


def train_seed(rng: np.random.Generator) -> int:
    """Seed for a training reset; can never coincide with a held-out evaluation seed."""
    return int(rng.integers(TRAIN_SEED_LOW, 2**31))


def make(task: str, seed: int | None = None):
    env = gym.make(TASKS[task].env_id)
    if seed is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)
    return env


def random_anchor(task: str, seeds: tuple[int, ...] = EVAL_SEEDS) -> float:
    """Mean return of a uniform random policy on the evaluation seeds (measured)."""
    returns = []
    for s in seeds:
        env = make(task)
        rng = np.random.default_rng(s)
        obs, _ = env.reset(seed=s)
        total, done = 0.0, False
        while not done:
            obs, r, term, trunc, _ = env.step(int(rng.integers(env.action_space.n)))
            total += r
            done = term or trunc
        returns.append(total)
        env.close()
    return float(np.mean(returns))


def normalise(task: str, ret: float, random_ret: float) -> float:
    return (ret - random_ret) / (TASKS[task].solved - random_ret)
