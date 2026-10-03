"""Fast path (reflex layer) for the future physical Toddler.

Jev answers many typed physical-rule questions in parallel and returns probabilities with
uncertainty (as in virtualpc-jev-finance/docs/JEV-FINANCE-EXPERIMENT.md, where Jev is a
feature generator, never an oracle). Hard sensor limits always override Jev: a measured force
above the limit stops the robot whatever the probabilities say.

Mapping rows 19-21: the safety/threat detector and the inhibition layer are the first and best
connected modules; deliberate planning only runs when the reflex says it is safe to think.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

REFLEX_BUDGET_MS = 20.0   # decide stop/slow/continue within this budget, or stop


@dataclass(frozen=True)
class PhysicalQuestion:
    qid: str
    text: str
    stop_if_above: float        # probability that forces STOP
    slow_if_above: float        # probability that forces SLOW


@dataclass(frozen=True)
class Answer:
    qid: str
    probability: float
    uncertainty: float          # e.g. width of a calibrated interval


class JevClient(Protocol):
    def ask(self, state: dict, questions: Sequence[PhysicalQuestion]) -> Sequence[Answer]: ...


@dataclass(frozen=True)
class HardLimits:
    max_force_n: float = 140.0          # ISO/TS 15066-style transient contact limit, set per body region
    max_speed_m_s: float = 0.25
    min_person_distance_m: float = 0.5


@dataclass(frozen=True)
class Sensors:
    force_n: float
    speed_m_s: float
    person_distance_m: float


@dataclass(frozen=True)
class Reflex:
    command: str                # "stop", "slow", "continue"
    because: tuple[str, ...]


DEFAULT_QUESTIONS: tuple[PhysicalQuestion, ...] = (
    PhysicalQuestion("grasp_slip", "Will the current grasp slip within one second?", 0.5, 0.2),
    PhysicalQuestion("person_in_zone", "Is a person inside the robot's safety zone?", 0.3, 0.1),
    PhysicalQuestion("collision_soon", "Will the planned path collide within 0.5 s?", 0.3, 0.1),
    PhysicalQuestion("unstable_object", "Is the carried object unstable?", 0.6, 0.3),
)


def hard_limit_check(s: Sensors, lim: HardLimits) -> tuple[str, ...]:
    reasons = []
    if s.force_n > lim.max_force_n:
        reasons.append(f"force {s.force_n:.0f} N > {lim.max_force_n:.0f} N")
    if s.speed_m_s > lim.max_speed_m_s:
        reasons.append(f"speed {s.speed_m_s:.2f} m/s > {lim.max_speed_m_s:.2f} m/s")
    if s.person_distance_m < lim.min_person_distance_m:
        reasons.append(f"person at {s.person_distance_m:.2f} m < {lim.min_person_distance_m:.2f} m")
    return tuple(reasons)


def reflex(sensors: Sensors, answers: Sequence[Answer] | None, elapsed_ms: float,
           questions: Sequence[PhysicalQuestion] = DEFAULT_QUESTIONS,
           limits: HardLimits = HardLimits()) -> Reflex:
    """Decide stop/slow/continue. Hard limits first; then a missing, late or invalid answer
    means stop; then each answer is judged at its upper bound p = min(1, probability +
    uncertainty), so more uncertainty can only make Toddler more careful."""
    hard = hard_limit_check(sensors, limits)
    if hard:
        return Reflex("stop", hard)
    if answers is None or elapsed_ms > REFLEX_BUDGET_MS:
        return Reflex("stop", ("no timely physical-rule answer",))
    by_id = {a.qid: a for a in answers}
    slow: list[str] = []
    for q in questions:
        a = by_id.get(q.qid)
        if a is None:
            return Reflex("stop", (f"missing answer: {q.qid}",))
        if a.uncertainty < 0 or not 0.0 <= a.probability <= 1.0:
            return Reflex("stop", (f"invalid answer: {q.qid}",))
        p = min(1.0, a.probability + a.uncertainty)
        if p > q.stop_if_above:
            return Reflex("stop", (f"{q.qid} p={p:.2f}",))
        if p > q.slow_if_above:
            slow.append(f"{q.qid} p={p:.2f}")
    return Reflex("slow", tuple(slow)) if slow else Reflex("continue", ())
