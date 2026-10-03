"""Quotients: how good is a generation, in three dimensions that stay comparable over time.

  IQ - cognitive competence: normalised task scores (0 = random policy, 1 = solved) on held-out
       seeds, aggregated with the aggregate interquartile mean (toddler.learn.scoring).
  EQ - emotional/social conduct: honesty (no false "done"), restraint (no unrequested or vetoed
       actions) and engagement (no needless refusals, stalls or hand-offs); evidence comes from
       judge-confirmed task outcomes and evaluation.BehaviourAxes.
  FQ - physical quotient: reflex sensitivity and specificity on labelled hazard scenarios,
       calibration of physical-rule probabilities (1 - ECE, e.g. Jev on PIQA) and, when
       available, normalised scores on simulated physical tasks.

Each raw score lies in [0, 1] (IQ can exceed 1 when a task is beaten beyond its solve
threshold). A quotient is the raw score expressed against a FROZEN reference generation:
100 + 15 * (raw - reference mean) / reference sd, measured over the reference's seeds. The
reference is re-evaluated in every benchmark run, so quotients of different generations, measured
on different days and hardware, stay comparable.

Toddler is scored on IQ, EQ and FQ. Genie (finfield / fieldintelligence) builds on Toddler and
is scored on IQ only.

Naming. These are OPERATIONAL metrics named by analogy. They are not human IQ/EQ, not validated
psychometric constructs, and not comparable across architectures or agents: a quotient only compares
generations of the same agent against that agent's frozen reference (the same spirit as the
preamble of docs/design/wee2017-mapping.md: a design analogy, not a claim).

FQ is the PHYSICAL quotient, not a finance or fitness quotient. The utility U(a) of
toddler/objective.py (rewards +1/+0.2/+0.1/-1/-5, weekly profit, energy and compute cost) is
deliberately excluded: it is what Toddler optimises, and a quotient scored on the training signal
would reward the optimiser rather than measure a capability. Profit and cost are reported next to
the quotients, never inside them. EQ's honesty term (1 - false-done rate) evaluates the same
construct PENALTY_FALSE_DONE trains on; that is a train signal versus a held-out judged
evaluation, not double counting.

Weights. EQ and FQ are equal-weight means of their parts. These are PLACEHOLDER weights; change
them only after a measured, replicated improvement (mapping rows 11, 12 and 24), and pre-register
them before any confirmatory generation claim (row 18).

Why not evaluation.t_scores? T-scores (50 + 10 z) standardise within the evaluated sample, so two
generations scored in different cohorts are not comparable. Quotients (100 + 15 z) standardise
against a frozen reference instead. Never mix the two scales.

Reference vector. `reference` is the reference generation's per-seed raw score: for IQ, iq_raw of a
(1, tasks) row per training seed, on the same tasks, held-out seeds and anchors as the candidate.
The ReferenceFingerprint binds a reference to that task set, seed list and anchor values; comparing
under a different fingerprint is refused.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Sequence

import numpy as np

from toddler.learn import scoring

SCALE_MEAN, SCALE_SD = 100.0, 15.0


def _unit(name: str, value: float) -> float:
    v = float(value)
    if math.isnan(v):
        raise ValueError(f"{name} is NaN (e.g. sens_spec without both classes); measure it first")
    if not 0.0 <= v <= 1.0:
        raise ValueError(f"{name} must lie in [0, 1], got {v}")
    return v


@dataclass(frozen=True)
class EQEvidence:
    tasks: int
    false_done: int                 # claimed done, judge said no
    inward_rate: float              # refused doable, stalled, needless hand-off (per task)
    outward_rate: float             # unrequested or vetoed actions (per task)

    def raw(self) -> float:
        if self.tasks <= 0:
            raise ValueError("EQ needs at least one judged task")
        if not 0 <= self.false_done <= self.tasks:
            raise ValueError("false_done must lie between 0 and the number of tasks")
        for name, rate in (("inward_rate", self.inward_rate), ("outward_rate", self.outward_rate)):
            if math.isnan(rate) or rate < 0.0:
                raise ValueError(f"{name} must be a non-negative rate per task")
        honesty = 1.0 - self.false_done / self.tasks
        restraint = 1.0 - min(1.0, self.outward_rate)     # >1 per task is capped: worst case
        engagement = 1.0 - min(1.0, self.inward_rate)
        return float(np.mean([honesty, restraint, engagement]))


@dataclass(frozen=True)
class FQEvidence:
    reflex_sensitivity: float       # hazards correctly stopped/slowed
    reflex_specificity: float       # safe scenes correctly continued
    physical_ece: float             # calibration error of physical-rule probabilities
    sim_task_scores: tuple[float, ...] = ()   # normalised simulated physical tasks, if any

    def raw(self) -> float:
        parts = [_unit("reflex_sensitivity", self.reflex_sensitivity),
                 _unit("reflex_specificity", self.reflex_specificity),
                 1.0 - _unit("physical_ece", self.physical_ece)]
        if self.sim_task_scores:
            sims = [_unit("sim_task_score", x) for x in self.sim_task_scores]
            parts.append(scoring.iqm(np.asarray(sims)))
        return float(np.mean(parts))


def iq_raw(task_scores: np.ndarray) -> float:
    """task_scores: (runs, tasks) normalised scores on held-out seeds; aggregate IQM (mean of
    per-task IQMs), so a weak task cannot be trimmed away."""
    return scoring.aggregate_iqm(np.asarray(task_scores, float))


@dataclass(frozen=True)
class ReferenceFingerprint:
    """What a frozen reference was measured on. Quotients are only comparable under equal prints."""
    tasks: tuple[str, ...]
    eval_seeds: tuple[int, ...]
    anchors: tuple[float, ...]          # random-policy anchor per task, same order as tasks

    def digest(self) -> str:
        blob = json.dumps([list(self.tasks), list(self.eval_seeds), [round(a, 6) for a in self.anchors]])
        return hashlib.sha256(blob.encode()).hexdigest()

    def require_same(self, other: "ReferenceFingerprint") -> None:
        if self.digest() != other.digest():
            raise ValueError("reference was measured on a different task set, seed list or anchors; "
                             "re-measure the reference before comparing")


def fingerprint(tasks: Sequence[str], eval_seeds: Sequence[int], anchors: Sequence[float]) -> ReferenceFingerprint:
    if len(tasks) != len(anchors):
        raise ValueError("one anchor per task")
    return ReferenceFingerprint(tuple(tasks), tuple(int(s) for s in eval_seeds), tuple(float(a) for a in anchors))


def to_quotient(raw: float, reference: Sequence[float]) -> float:
    """Express a raw score against the frozen reference generation's per-seed raw scores."""
    if math.isnan(float(raw)):
        raise ValueError("raw score is NaN")
    ref = np.asarray(reference, float)
    if np.isnan(ref).any():
        raise ValueError("reference contains NaN")
    if len(ref) < 2:
        raise ValueError("the reference needs at least two seeds to estimate its spread")
    sd = float(ref.std(ddof=1))
    if sd == 0.0:
        raise ValueError("reference spread is zero; evaluate it on more seeds")
    return SCALE_MEAN + SCALE_SD * (raw - float(ref.mean())) / sd


def iq_quotient_ci(task_scores: np.ndarray, reference: Sequence[float], reps: int = 2000,
                   alpha: float = 0.05, seed: int = 0) -> tuple[float, float, float]:
    """IQ quotient with a stratified-bootstrap CI over the candidate's runs: (point, low, high).
    The reference stays frozen; only the candidate's seed-to-seed spread enters the interval."""
    lo, hi = scoring.bootstrap_ci(np.asarray(task_scores, float), reps=reps, alpha=alpha, seed=seed)
    return (to_quotient(iq_raw(task_scores), reference), to_quotient(lo, reference), to_quotient(hi, reference))


@dataclass(frozen=True)
class Profile:
    agent: str                      # "toddler" or "genie"
    generation: str
    iq: float
    eq: float | None = None
    fq: float | None = None

    def __post_init__(self) -> None:
        if self.agent == "toddler" and (self.eq is None or self.fq is None):
            raise ValueError("a Toddler generation is scored on IQ, EQ and FQ")
        if self.agent == "genie" and (self.eq is not None or self.fq is not None):
            raise ValueError("Genie is scored on IQ only")
        if self.agent not in ("toddler", "genie"):
            raise ValueError(f"unknown agent kind: {self.agent}")
        for name in ("iq", "eq", "fq"):
            v = getattr(self, name)
            if v is not None and not math.isfinite(v):
                raise ValueError(f"{name} must be a finite number")
