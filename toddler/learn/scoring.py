"""Hardware-independent evaluation and robust comparison of toddler generations.

Evaluation always runs on the CPU, greedy actions, on the held-out EVAL_SEEDS, so the score of
a toddler does not depend on which GPU trained it or how busy it was. Aggregation follows
Agarwal et al. (2021, "Deep RL at the edge of the statistical precipice"): normalised scores,
interquartile mean (IQM), stratified bootstrap confidence intervals, and the probability of
improvement P(X > Y) between two generations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from toddler.learn import tasks as T
from toddler.learn.policy import ActorCritic


def evaluate(net: ActorCritic, task: str, seeds: tuple[int, ...] = T.EVAL_SEEDS) -> np.ndarray:
    net = net.to("cpu").eval()
    out = []
    for s in seeds:
        env = T.make(task)
        obs, _ = env.reset(seed=s)
        total, done = 0.0, False
        while not done:
            with torch.no_grad():
                a = int(torch.argmax(net(torch.as_tensor(obs, dtype=torch.float32))[0]))
            obs, r, term, trunc, _ = env.step(a)
            total += r
            done = term or trunc
        out.append(total)
        env.close()
    return np.asarray(out)


def iqm(x: np.ndarray) -> float:
    x = np.sort(np.asarray(x, float).ravel())
    n = len(x)
    lo, hi = int(np.floor(0.25 * n)), int(np.ceil(0.75 * n))
    return float(x[lo:hi].mean()) if hi > lo else float(x.mean())


def bootstrap_ci(scores: np.ndarray, stat=iqm, reps: int = 2000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    """Stratified bootstrap over runs: scores has shape (runs, tasks); resample runs per task."""
    rng = np.random.default_rng(seed)
    s = np.asarray(scores, float)
    runs, tasks = s.shape
    vals = []
    for _ in range(reps):
        cols = [s[rng.integers(0, runs, runs), j] for j in range(tasks)]
        vals.append(stat(np.stack(cols, axis=1)))
    return float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))


def prob_improvement(x: np.ndarray, y: np.ndarray) -> float:
    """Average over tasks of P(X > Y) + 0.5 P(X = Y) between runs of two generations."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    per_task = []
    for j in range(x.shape[1]):
        a, b = x[:, j][:, None], y[:, j][None, :]
        per_task.append(float(np.mean((a > b) + 0.5 * (a == b))))
    return float(np.mean(per_task))


@dataclass(frozen=True)
class GenerationScore:
    generation: str
    iqm: float
    ci_low: float
    ci_high: float
    runs: int
    tasks: tuple[str, ...]


def score_generation(name: str, normalised: np.ndarray, tasks: tuple[str, ...]) -> GenerationScore:
    lo, hi = bootstrap_ci(normalised)
    return GenerationScore(name, iqm(normalised), lo, hi, normalised.shape[0], tasks)
