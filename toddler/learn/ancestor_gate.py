"""Promotion gate for descendants of surviving Toddler generations.

Ancestor policies are rerun on the same hidden seeds as each candidate. Their
best mastered-task score supplies a retention floor, and the strongest whole-
battery ancestor is a matched promotion reference. This module consumes trial
results; it never reads or publishes hidden seeds.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from toddler.learn.generations import Promotion, decide_promotion
from toddler.learn.lineage import Ledger
from toddler.learn.scoring import iqm

MASTERY = 0.90
MAX_REGRESSION = 0.10


@dataclass(frozen=True)
class Trial:
    seed_set_id: str
    frozen_plan_sha256: str
    child_ids: tuple[str, ...]
    candidate_weights: dict[str, str]
    candidate: dict[str, tuple[float, ...]]
    control: dict[str, tuple[float, ...]]
    ancestors: dict[str, dict[str, tuple[float, ...]]]


@dataclass(frozen=True)
class TrialDecision:
    passed: bool
    strongest_ancestor: str
    promotion_vs_ancestor: Promotion
    promotion_vs_control: Promotion
    retention_floors: dict[str, float]
    regressions: dict[str, float]


@dataclass(frozen=True)
class ReplicationDecision:
    promote: bool
    required_ancestors: tuple[str, ...]
    trials: tuple[TrialDecision, ...]


def surviving_ancestors(ledger: Ledger, parents: tuple[str, ...]) -> tuple[str, ...]:
    """Every surviving parent or earlier ancestor is entitled to a comparison."""
    if not parents:
        raise ValueError("a successor needs registered parents")
    generations = set()
    for parent in parents:
        if ledger.birth(parent) is None:
            raise ValueError(f"unregistered parent: {parent}")
        for ref in (parent, *(item["parent"] for item in ledger.ancestry(parent))):
            generation = ref.split("/", 1)[0]
            verdict = ledger.verdict(generation)
            if verdict and verdict["verdict"] == "survived":
                generations.add(generation)
    if not generations:
        raise ValueError("no surviving ancestor exists")
    return tuple(sorted(generations))


def _series(values: tuple[float, ...], children: int, label: str) -> np.ndarray:
    out = np.asarray(values, dtype=float)
    if out.shape != (children,) or not np.isfinite(out).all():
        raise ValueError(f"{label} needs one finite score per matched child")
    return out


def assess_trial(trial: Trial, required_ancestors: tuple[str, ...], *,
                 mastery: float = MASTERY, max_regression: float = MAX_REGRESSION) -> TrialDecision:
    children = len(trial.child_ids)
    if children < 5 or len(set(trial.child_ids)) != children:
        raise ValueError("promotion needs at least five distinct matched children")
    if not trial.seed_set_id or not re.fullmatch(r"[0-9a-f]{64}", trial.frozen_plan_sha256):
        raise ValueError("trial needs a seed-set ID and frozen plan hash")
    if (set(trial.candidate_weights) != set(trial.child_ids)
            or any(not re.fullmatch(r"[0-9a-f]{64}", value) for value in trial.candidate_weights.values())):
        raise ValueError("candidate weight hashes must cover every child")
    tasks = tuple(trial.candidate)
    if not tasks or set(trial.control) != set(tasks):
        raise ValueError("candidate and control need the same nonempty task battery")
    if (not required_ancestors or len(set(required_ancestors)) != len(required_ancestors)
            or set(trial.ancestors) != set(required_ancestors)):
        raise ValueError("all and only surviving ancestors must be rerun")
    if not 0 <= mastery <= 1 or not 0 <= max_regression < 1:
        raise ValueError("invalid preregistered retention thresholds")
    candidate = {task: _series(trial.candidate[task], children, f"candidate/{task}") for task in tasks}
    control = {task: _series(trial.control[task], children, f"control/{task}") for task in tasks}
    ancestral = {}
    for name, scores in trial.ancestors.items():
        if not scores or set(scores) - set(tasks):
            raise ValueError(f"{name} has no applicable tasks or an unknown task")
        ancestral[name] = {task: _series(values, children, f"{name}/{task}")
                           for task, values in scores.items()}
    cand_means = np.mean(np.stack([candidate[t] for t in tasks], axis=1), axis=1)
    control_means = np.mean(np.stack([control[t] for t in tasks], axis=1), axis=1)
    ancestor_means = {name: np.mean(np.stack([scores.get(t, np.zeros(children)) for t in tasks], axis=1), axis=1)
                      for name, scores in ancestral.items()}
    strongest = max(sorted(ancestral), key=lambda name: iqm(ancestor_means[name]))
    floors, regressions = {}, {}
    for task in tasks:
        mastered = [iqm(scores[task]) for scores in ancestral.values()
                    if task in scores and iqm(scores[task]) >= mastery]
        if mastered:
            floor = max(mastery, max(mastered) - max_regression)
            floors[task] = float(floor)
            actual = iqm(candidate[task])
            if actual + 1e-12 < floor:
                regressions[task] = float(floor - actual)
    vs_ancestor = decide_promotion(cand_means.tolist(), ancestor_means[strongest].tolist())
    vs_control = decide_promotion(cand_means.tolist(), control_means.tolist())
    return TrialDecision(not regressions and vs_ancestor.promote and vs_control.promote,
                         strongest, vs_ancestor, vs_control, floors, regressions)


def assess_replication(trials: tuple[Trial, ...], required_ancestors: tuple[str, ...], *,
                       mastery: float = MASTERY, max_regression: float = MAX_REGRESSION) -> ReplicationDecision:
    """Same frozen candidate and plan must pass at least two independent seed sets."""
    if len(trials) < 2 or len({trial.seed_set_id for trial in trials}) != len(trials):
        raise ValueError("at least two distinct hidden seed sets are required")
    first = trials[0]
    for trial in trials[1:]:
        if (trial.child_ids != first.child_ids or trial.candidate_weights != first.candidate_weights
                or trial.frozen_plan_sha256 != first.frozen_plan_sha256
                or set(trial.candidate) != set(first.candidate)):
            raise ValueError("replication changed children, weights, plan or task battery")
    decisions = tuple(assess_trial(trial, required_ancestors,
                                   mastery=mastery, max_regression=max_regression)
                      for trial in trials)
    return ReplicationDecision(all(decision.passed for decision in decisions), required_ancestors, decisions)


def assess_lineage_replication(ledger: Ledger, parents: tuple[str, ...], trials: tuple[Trial, ...], *,
                               mastery: float = MASTERY,
                               max_regression: float = MAX_REGRESSION) -> ReplicationDecision:
    """Use the actual ledger ancestry, so the caller cannot omit a strong ancestor."""
    return assess_replication(trials, surviving_ancestors(ledger, parents),
                              mastery=mastery, max_regression=max_regression)
