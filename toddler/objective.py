"""Toddler's objective: reward, profit, energy/compute cost and the decision rule.

U(a) = E[reward] + w_profit * E[profit] - w_energy * energy_J - w_compute * cost_eur
       - w_risk * P(failure) * severity,   subject to stop.check(a).allowed

All weights are documented starting points, changed only after a measured and
replicated improvement (wee2017-mapping rows 11, 12, 24).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from toddler import stop

# Reward points per task outcome, as published on the explainer page.
REWARD_DONE = 1.0
REWARD_HONEST_UNKNOWN = 0.2
REWARD_NEW_KNOWLEDGE = 0.1
PENALTY_WRONG = -1.0
PENALTY_FALSE_DONE = -5.0


@dataclass(frozen=True)
class TaskOutcome:
    """One finished task. `judge_confirmed` comes from the independent judge, never from Toddler."""

    claimed_done: bool
    judge_confirmed: bool
    said_unknown: bool = False
    new_knowledge: bool = False
    cost_eur: float = 0.0


def reward(outcome: TaskOutcome) -> float:
    """Per-task learning signal. Claiming done without judge confirmation is the worst outcome."""
    if outcome.claimed_done and outcome.judge_confirmed:
        r = REWARD_DONE
    elif outcome.claimed_done:
        r = PENALTY_FALSE_DONE
    elif outcome.said_unknown:
        r = REWARD_HONEST_UNKNOWN
    else:
        r = PENALTY_WRONG
    if outcome.new_knowledge:
        r += REWARD_NEW_KNOWLEDGE
    return r - outcome.cost_eur


@dataclass(frozen=True)
class WeekLedger:
    value_eur: float
    energy_eur: float
    compute_eur: float
    review_hours: float
    review_rate_eur: float
    expected_incident_eur: float


def profit(ledger: WeekLedger) -> float:
    """Weekly system-level account; decides whether a phase continues, does not train Toddler."""
    return (
        ledger.value_eur
        - ledger.energy_eur
        - ledger.compute_eur
        - ledger.review_hours * ledger.review_rate_eur
        - ledger.expected_incident_eur
    )


@dataclass(frozen=True)
class Weights:
    profit: float = 1.0
    energy_per_joule: float = 1e-6
    compute_per_eur: float = 1.0
    risk: float = 1.0


@dataclass(frozen=True)
class Candidate:
    action: stop.Action
    expected_reward: float
    expected_profit_eur: float
    energy_j: float
    compute_eur: float
    p_failure: float
    severity_eur: float


def utility(c: Candidate, w: Weights = Weights()) -> float:
    if not 0.0 <= c.p_failure <= 1.0:
        raise ValueError("p_failure must be a probability")
    return (
        c.expected_reward
        + w.profit * c.expected_profit_eur
        - w.energy_per_joule * c.energy_j
        - w.compute_per_eur * c.compute_eur
        - w.risk * c.p_failure * c.severity_eur
    )


@dataclass(frozen=True)
class Decision:
    chosen: Candidate | None
    vetoed: tuple[tuple[str, tuple[str, ...]], ...]


def choose(candidates: Sequence[Candidate], w: Weights = Weights(),
           rules: Iterable[stop.StopRule] = stop.DEFAULT_RULES) -> Decision:
    """Pick the highest-utility allowed candidate. Returns None when every option is vetoed:
    the caller must then hand off to a human rather than improvise."""
    rules = tuple(rules)
    allowed: list[Candidate] = []
    vetoed: list[tuple[str, tuple[str, ...]]] = []
    for c in candidates:
        verdict = stop.check(c.action, rules)
        if verdict.allowed:
            allowed.append(c)
        else:
            vetoed.append((c.action.name, verdict.violated))
    best = max(allowed, key=lambda c: utility(c, w), default=None)
    return Decision(chosen=best, vetoed=tuple(vetoed))
