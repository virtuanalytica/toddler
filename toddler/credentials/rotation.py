"""Key rotation as an explicit, auditable sequence: create -> store -> verify -> swap -> revoke.

The planner is pure so it can be tested and reviewed; the executor runs one step at a time
and stops at the first failure, leaving the old key active (never revoke before the new key
is verified).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Callable

STEPS = ("create", "store", "verify", "swap", "revoke_old")


@dataclass(frozen=True)
class KeyState:
    provider: str
    identity_email: str
    key_name: str
    created_at: datetime
    max_age: timedelta


def due(state: KeyState, now: datetime | None = None) -> bool:
    now = now or datetime.now(timezone.utc)
    return now - state.created_at >= state.max_age


def plan(state: KeyState, now: datetime | None = None) -> tuple[str, ...]:
    return STEPS if due(state, now) else ()


@dataclass(frozen=True)
class StepResult:
    step: str
    ok: bool
    detail: str = ""


def execute(steps: tuple[str, ...], handlers: dict[str, Callable[[], str]]) -> list[StepResult]:
    """Run steps in order; stop at the first failure so the old key is never revoked early.
    Steps and handlers are validated up front: unknown steps, missing handlers, or a revoke
    that is not preceded by verify are refused before anything runs."""
    unknown = [st for st in steps if st not in STEPS]
    if unknown:
        raise ValueError(f"unknown rotation steps: {unknown}")
    missing = [st for st in steps if st not in handlers]
    if missing:
        raise ValueError(f"no handler for steps: {missing}")
    if "revoke_old" in steps and ("verify" not in steps or steps.index("verify") > steps.index("revoke_old")):
        raise ValueError("revoke_old requires a preceding verify step")
    results: list[StepResult] = []
    for step in steps:
        try:
            results.append(StepResult(step, True, handlers[step]()))
        except Exception as exc:  # each failure is reported, never swallowed silently
            results.append(StepResult(step, False, f"{type(exc).__name__}: {exc}"))
            break
    return results
