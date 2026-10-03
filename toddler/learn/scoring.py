"""Hardware-independent evaluation and robust comparison of toddler generations.

Evaluation always runs on the CPU on the held-out EVAL_SEEDS, greedy by default (every recorded
score is greedy; a sampled mode exists and is recorded in the reference fingerprint), so the score of
a toddler does not depend on which GPU trained it or how busy it was. Aggregation follows
Agarwal et al. (2021, "Deep RL at the edge of the statistical precipice"): normalised scores,
aggregate interquartile mean (mean of per-task IQMs), stratified bootstrap confidence intervals, and the probability of
improvement P(X > Y) between two generations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from toddler.learn.policy import ActorCritic


def evaluate(net: "ActorCritic", task: str, seeds: tuple[int, ...] | None = None,
             mode: str = "greedy") -> np.ndarray:
    """Returns on held-out seeds (default T.EVAL_SEEDS), on the CPU. torch and gymnasium are
    imported here so the statistics below work without the `learn` extra.

    mode "greedy" (default, used by every recorded score): argmax action.
    mode "sample": actions drawn from the policy with a generator seeded by the evaluation seed,
    so the score is still deterministic per (network, seed). On tasks where a greedy policy can
    loop (MiniGrid DoorKey) the two can differ widely; a report must say which mode it used and
    never compare scores across modes."""
    import torch

    from toddler.learn import tasks as T

    if mode not in ("greedy", "sample"):
        raise ValueError(f"unknown evaluation mode {mode!r}")
    seeds = T.EVAL_SEEDS if seeds is None else seeds
    net = net.to("cpu").eval()
    out = []
    for s in seeds:
        env = T.make(task)
        obs, _ = env.reset(seed=s)
        gen = torch.Generator().manual_seed(int(s)) if mode == "sample" else None
        total, done = 0.0, False
        while not done:
            with torch.no_grad():
                logits = net(torch.as_tensor(obs, dtype=torch.float32))[0]
                if mode == "greedy":
                    a = int(torch.argmax(logits))
                else:
                    a = int(torch.multinomial(torch.softmax(logits, -1), 1, generator=gen))
            obs, r, term, trunc, _ = env.step(a)
            total += r
            done = term or trunc
        out.append(total)
        env.close()
    return np.asarray(out)


def iqm(x: np.ndarray) -> float:
    x = np.sort(np.asarray(x, float).ravel())
    n = len(x)
    if n == 0:
        raise ValueError("IQM of an empty sample")
    lo, hi = int(np.floor(0.25 * n)), int(np.ceil(0.75 * n))
    return float(x[lo:hi].mean()) if hi > lo else float(x.mean())


def aggregate_iqm(scores: np.ndarray) -> float:
    """Agarwal et al. (2021) aggregate: the mean over tasks of each task's IQM over runs, so a
    weak task cannot be trimmed away by pooling. scores: (runs, tasks)."""
    s = np.asarray(scores, float)
    if s.ndim == 1:
        return iqm(s)
    return float(np.mean([iqm(s[:, j]) for j in range(s.shape[1])]))


def bootstrap_ci(scores: np.ndarray, stat=aggregate_iqm, reps: int = 2000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    """Stratified bootstrap over runs: scores has shape (runs, tasks); resample runs per task."""
    rng = np.random.default_rng(seed)
    s = np.asarray(scores, float)
    if s.ndim != 2:
        raise ValueError("bootstrap_ci expects scores of shape (runs, tasks)")
    runs, tasks = s.shape
    vals = []
    for _ in range(reps):
        cols = [s[rng.integers(0, runs, runs), j] for j in range(tasks)]
        vals.append(stat(np.stack(cols, axis=1)))
    return float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))


def prob_improvement(x: np.ndarray, y: np.ndarray) -> float:
    """Average over tasks of P(X > Y) + 0.5 P(X = Y) between runs of two generations."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1]:
        raise ValueError("prob_improvement expects (runs, tasks) arrays with the same tasks")
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
    return GenerationScore(name, aggregate_iqm(normalised), lo, hi, normalised.shape[0], tasks)
