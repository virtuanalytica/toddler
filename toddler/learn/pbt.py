"""Population-based training (Jaderberg et al. 2017, arXiv:1711.09846) for toddlers.

A population trains in equal intervals of environment steps. After every interval the members
are ranked on their own TRAINING episodes (never on the held-out evaluation seeds). In the PBT
arm the worst member copies the weights and hyperparameters of the best one (exploit) and then
perturbs its learning rate and entropy coefficient by a factor 0.8 or 1.2 (explore). The control
arm runs the identical schedule without exploit/explore, so both arms spend exactly the same
number of environment steps.

Each population returns ONE toddler: the member with the best training score after the last
interval. That toddler is scored on the held-out seeds; the selection never sees them.

Every interval is a fresh `ppo.train` call that continues from the member's weights, so the
optimiser state, curiosity weight and return scaler restart per interval, in both arms alike.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field, replace

import numpy as np

from toddler.learn import ppo


@dataclass
class Member:
    idx: int
    cfg: ppo.PPOConfig
    net: object = None
    train_score: float = float("-inf")
    history: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class PopulationResult:
    arm: str
    seed: int
    selected_member: int
    selected_config: dict
    train_scores: list[float]
    steps_per_member: int
    events: list[str]


def _train_score(log: ppo.TrainLog, last: int = 20) -> float:
    eps = log.episode_returns[-last:]
    return float(np.mean(eps)) if eps else float("-inf")


def run_population(task: str, seed: int, pbt: bool, members: int = 4, intervals: int = 10,
                   interval_steps: int = 10_000, base: ppo.PPOConfig | None = None):
    """Train one population; returns (selected net, PopulationResult)."""
    rng = np.random.default_rng(seed)
    base = base or ppo.PPOConfig()
    pop = [Member(i, replace(base, total_steps=interval_steps)) for i in range(members)]
    events: list[str] = []
    for k in range(intervals):
        for m in pop:
            cfg = replace(m.cfg, seed=int(rng.integers(0, 2**31)))       # fresh resets each interval
            m.net, log = ppo.train(task, cfg, net=m.net)
            m.train_score = _train_score(log)
        if pbt and k < intervals - 1:
            ranked = sorted(pop, key=lambda m: m.train_score)
            worst, best = ranked[0], ranked[-1]
            if best.train_score > worst.train_score:
                worst.net = copy.deepcopy(best.net)
                f_lr, f_ent = rng.choice([0.8, 1.2]), rng.choice([0.8, 1.2])
                worst.cfg = replace(best.cfg, lr=best.cfg.lr * f_lr, ent_coef=best.cfg.ent_coef * f_ent)
                events.append(f"interval {k}: member {worst.idx} <- member {best.idx} "
                              f"(lr x{f_lr}, ent x{f_ent})")
    chosen = max(pop, key=lambda m: m.train_score)
    return chosen.net, PopulationResult("pbt" if pbt else "control", seed, chosen.idx,
                                        {k: v for k, v in asdict(chosen.cfg).items() if k in ("lr", "ent_coef")},
                                        [round(m.train_score, 4) for m in pop], intervals * interval_steps, events)
