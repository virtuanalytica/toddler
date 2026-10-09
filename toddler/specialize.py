"""Specialisation: Toddler is the general base a developer or consultant builds on, towards one
role in a company's function house (functiehuis) or one very specific task.

Two extension points, both declarative so a deployment is reviewable:
  * Jev rules  - extra typed questions with stop/slow thresholds for the fast path, and extra
                 STOP rules for the logic layer;
  * experts    - additional helper models on top of the mixture-of-models base, routed by
                 domain, quality and energy/compute cost.

Safety is monotonic: a specialisation can add constraints but can never remove or weaken a
base STOP rule, the judge, or the audit trail.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from toddler import fastpath, stop


@dataclass(frozen=True)
class Role:
    """A position in the function house, or a single task when `family` is 'task'."""

    name: str
    family: str                       # e.g. "finance", "care", "data-governance", "task"
    level: str = ""                   # e.g. "medior", "senior", or a salary scale
    tasks: tuple[str, ...] = ()


@dataclass(frozen=True)
class Expert:
    """A helper model added to the mixture-of-models base."""

    model_id: str
    domains: frozenset[str]
    quality: float                    # measured on the role's evaluation set, 0..1
    eur_per_1k_tokens: float
    joules_per_1k_tokens: float
    evidence: str                     # where `quality` was measured (MLflow run, report)
    tokens_per_second: float | None = None  # measured decode speed; needed for a speed floor

    def __post_init__(self) -> None:
        if not 0.0 <= self.quality <= 1.0:
            raise ValueError("quality must be in [0, 1]")
        if not self.evidence:
            raise ValueError("an expert needs measured evidence for its quality")
        if self.tokens_per_second is not None and self.tokens_per_second <= 0:
            raise ValueError("tokens_per_second must be positive when given")


def company_guardrail(rule_id: str, reason: str, owner: str, forbidden_tags: frozenset[str]) -> stop.StopRule:
    """A company guardrail is a STOP rule owned by the company (e.g. its compliance officer):
    any action carrying one of `forbidden_tags` is vetoed. The owner is part of the reason so
    the audit trail shows whose rule fired."""
    if not owner:
        raise ValueError("a company guardrail needs an owner")
    tags = frozenset(forbidden_tags)
    return stop.StopRule(f"company:{rule_id}", f"{reason} (owner: {owner})", lambda a: bool(tags & a.tags))


@dataclass(frozen=True)
class Specialisation:
    role: Role
    jev_questions: tuple[fastpath.PhysicalQuestion, ...] = ()
    extra_stop_rules: tuple[stop.StopRule, ...] = ()
    experts: tuple[Expert, ...] = ()
    base_model: str = "mixture-of-models"

    def stop_rules(self) -> tuple[stop.StopRule, ...]:
        return stop.DEFAULT_RULES + self.extra_stop_rules

    def questions(self) -> tuple[fastpath.PhysicalQuestion, ...]:
        return fastpath.DEFAULT_QUESTIONS + self.jev_questions


class UnsafeSpecialisation(ValueError):
    pass


def build(role: Role, jev_questions: Sequence[fastpath.PhysicalQuestion] = (),
          extra_stop_rules: Sequence[stop.StopRule] = (), experts: Sequence[Expert] = ()) -> Specialisation:
    """Validate and freeze a deployment. Rejects anything that would weaken the base."""
    base_ids = {r.rule_id for r in stop.DEFAULT_RULES}
    seen: set[str] = set()
    for r in extra_stop_rules:
        if r.rule_id in base_ids:
            raise UnsafeSpecialisation(f"rule id {r.rule_id} would shadow a base STOP rule")
        if r.rule_id in seen:
            raise UnsafeSpecialisation(f"duplicate rule id {r.rule_id}")
        seen.add(r.rule_id)
    base_q = {q.qid for q in fastpath.DEFAULT_QUESTIONS}
    qids = [q.qid for q in jev_questions]
    if len(qids) != len(set(qids)):
        raise UnsafeSpecialisation("duplicate Jev question ids")
    for q in jev_questions:
        if q.qid in base_q:
            raise UnsafeSpecialisation(f"question {q.qid} would replace a base reflex question")
        if not 0.0 <= q.slow_if_above <= q.stop_if_above <= 1.0:
            raise UnsafeSpecialisation(f"question {q.qid}: need 0 <= slow <= stop <= 1")
    ids = [e.model_id for e in experts]
    if len(ids) != len(set(ids)):
        raise UnsafeSpecialisation("duplicate expert model ids")
    return Specialisation(role, tuple(jev_questions), tuple(extra_stop_rules), tuple(experts))


@dataclass(frozen=True)
class Route:
    model_id: str
    score: float
    reasons: tuple[str, ...] = field(default_factory=tuple)


def route(spec: Specialisation, domain: str, tokens_k: float, eur_per_joule: float,
          w_cost: float = 1.0, base_quality: float = 0.5,
          min_tokens_per_second: float | None = None) -> Route:
    """Mixture-of-models gate: pick the expert with the best quality minus cost for this
    domain; fall back to the base model when no expert covers the domain or none beats it.

    `min_tokens_per_second` is a hard speed floor: an expert without a measured speed, or
    slower than the floor, is never routed to. None keeps the quality-minus-cost behaviour."""
    if min_tokens_per_second is not None and min_tokens_per_second <= 0:
        raise ValueError("min_tokens_per_second must be positive when given")
    best = Route(spec.base_model, base_quality, ("base model",))
    for e in spec.experts:
        if domain not in e.domains:
            continue
        if min_tokens_per_second is not None and (
                e.tokens_per_second is None or e.tokens_per_second < min_tokens_per_second):
            continue
        cost = tokens_k * (e.eur_per_1k_tokens + e.joules_per_1k_tokens * eur_per_joule)
        score = e.quality - w_cost * cost
        if score > best.score:
            best = Route(e.model_id, score, (f"domain {domain}", f"quality {e.quality:.2f}", f"cost EUR {cost:.4f}"))
    return best
