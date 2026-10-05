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
    grid: bool = False     # MiniGrid: image observation flattened to a vector (needs `minigrid`)


TASKS: dict[str, Task] = {
    "cartpole": Task("CartPole-v1", 475.0),
    "acrobot": Task("Acrobot-v1", -100.0),
    "mountaincar": Task("MountainCar-v0", -110.0),
    # MiniGrid has no Gymnasium reward_threshold; its return is 1 - 0.9 * steps / max_steps on
    # success and 0 otherwise. Our threshold 0.9 = goal reached within 11 % of the step limit
    # (an operator choice, documented here, part of the reference fingerprint).
    "empty5": Task("MiniGrid-Empty-5x5-v0", 0.9, grid=True),
    "doorkey5": Task("MiniGrid-DoorKey-5x5-v0", 0.9, grid=True),
    # Harder procedural MiniGrid tasks (added 2026-10-05, beyond the G1 ceiling). Same 147-d
    # observation and 7 actions, so a multi-task net only needs a new adapter. Measured random
    # anchors on EVAL_SEEDS: 0.000-0.021. The 0.9 threshold leaves 26-64 steps here (10 % of the
    # step limit), within reach of a good policy. MultiRoom-N2-S4 is left out on purpose: its
    # 40-step limit makes 0.9 mean "goal in 4 steps", a ceiling no policy can reach.
    "doorkey8": Task("MiniGrid-DoorKey-8x8-v0", 0.9, grid=True),
    "unlock": Task("MiniGrid-Unlock-v0", 0.9, grid=True),
    "unlockpickup": Task("MiniGrid-UnlockPickup-v0", 0.9, grid=True),
    "keycorridor3": Task("MiniGrid-KeyCorridorS3R1-v0", 0.9, grid=True),
    "lavacross9": Task("MiniGrid-LavaCrossingS9N1-v0", 0.9, grid=True),
}

EVAL_SEEDS: tuple[int, ...] = tuple(range(10_000, 10_030))   # held out: never used for training
TRAIN_SEED_LOW = 1_000_000     # training resets draw from [TRAIN_SEED_LOW, 2**31), disjoint from EVAL_SEEDS
# Secret evaluation seeds (toddler.learn.secret_seeds) come from their own band, disjoint from both
# the public EVAL_SEEDS and every training seed, so no training run can ever have seen them.
SECRET_SEED_LOW, SECRET_SEED_HIGH = 100_000, TRAIN_SEED_LOW
if max(EVAL_SEEDS) >= SECRET_SEED_LOW or SECRET_SEED_HIGH > TRAIN_SEED_LOW:  # survives python -O
    raise RuntimeError("EVAL_SEEDS, secret seeds and training seeds must be disjoint")


def train_seed(rng: np.random.Generator) -> int:
    """Seed for a training reset; can never coincide with a held-out evaluation seed."""
    return int(rng.integers(TRAIN_SEED_LOW, 2**31))


def make(task: str, seed: int | None = None):
    t = TASKS[task]
    if t.grid:
        import minigrid  # noqa: F401  (registers the MiniGrid environments)
        from gymnasium.wrappers import FlattenObservation
        from minigrid.wrappers import ImgObsWrapper

        env = FlattenObservation(ImgObsWrapper(gym.make(t.env_id)))
    else:
        env = gym.make(t.env_id)
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
    if TASKS[task].solved <= random_ret:
        raise ValueError(f"{task}: solve threshold {TASKS[task].solved} is not above the random anchor {random_ret}")
    return (ret - random_ret) / (TASKS[task].solved - random_ret)
