"""Compare CLM with JEV without permitting CLM to control the reflex."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Sequence

from toddler.fastpath import (DEFAULT_QUESTIONS, Answer, HardLimits, JevClient,
                              PhysicalQuestion, Reflex, Sensors, hard_limit_check, reflex)


@dataclass(frozen=True)
class ShadowResult:
    jev: Reflex
    clm: Reflex
    jev_ms: float
    clm_ms: float
    jev_error: str | None
    clm_error: str | None


def _ask(client: JevClient, state: dict, questions: Sequence[PhysicalQuestion]) -> tuple[Sequence[Answer] | None, float, str | None]:
    start = time.perf_counter()
    try:
        answers = client.ask(state, questions)
        return answers, (time.perf_counter() - start) * 1000, None
    except Exception as exc:  # one failed comparison must never override the reflex
        return None, (time.perf_counter() - start) * 1000, type(exc).__name__


def compare(state: dict, sensors: Sensors, jev_client: JevClient, clm_client: JevClient,
            questions: Sequence[PhysicalQuestion] = DEFAULT_QUESTIONS,
            limits: HardLimits = HardLimits()) -> ShadowResult:
    """Return both decisions. Only ``jev`` is the existing decision; ``clm`` is hypothetical.

    The hard-limit path never sends the state to either model. Sequential calls avoid
    competition for the same scarce GPU; latency is measured separately per client.
    """
    hard = hard_limit_check(sensors, limits)
    if hard:
        stop = Reflex("stop", hard)
        return ShadowResult(stop, stop, 0.0, 0.0, None, None)
    jev_answers, jev_ms, jev_error = _ask(jev_client, state, questions)
    clm_answers, clm_ms, clm_error = _ask(clm_client, state, questions)
    return ShadowResult(reflex(sensors, jev_answers, jev_ms, questions, limits),
                        reflex(sensors, clm_answers, clm_ms, questions, limits),
                        jev_ms, clm_ms, jev_error, clm_error)
