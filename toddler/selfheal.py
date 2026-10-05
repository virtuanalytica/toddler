"""Self-healing watchdog: classify crashes, recover with reduced budget, quarantine repeat offenders.

Design (patterns observed in the fillslava ClaudeClaw codebase, re-implemented
natively for Toddler — that repository is read-only reference for us):

- A pure classifier (`classify_failure`) maps an exception (plus optional
  stderr) to a failure kind and a green/amber/red severity band.  No I/O, no
  state, so it is trivially testable.
- The runner (`run_guarded`) separates classification from recovery: retryable
  kinds get another attempt with a *reduced* budget (fewer threads, forced
  CPU — recovery must never claim more resources than the crash allowed);
  deterministic kinds (assertion failures, missing modules) are not retried
  because a retry cannot change their outcome.
- Three consecutive failures (per task configurable: `max_failures`) quarantine a task (`TaskGuard`, the heartbeat
  "3-strikes" pattern): later calls fail fast with an honest
  `Outcome(disabled=True)` — the work is *deferred*, not crashed, until a
  human or an operator script calls `reset_task`.
- Every decision lands in the hash-chained audit trail (`toddler.audit`), so
  what crashed, what we tried and what was disabled is provable afterwards.
  The watchdog never throws, but it never hides a lost audit entry either:
  every `Outcome` carries `audit_ok`, and `SelfHealer.audit_failures` counts
  entries that could not be written. The shared instance persists its trail
  to `$TODDLER_SELFHEAL_AUDIT` (default ~/.local/share/toddler/selfheal_audit.jsonl).
- Lifting a quarantine (`reset_task`) requires a named `requested_by` and a
  reason; the audit records exactly who asked, never an assumed "operator".
- Forward note: when this wraps learn/PPO runs, a cuda->cpu fallback retry is a
  different method configuration and must be recorded like any other
  (TrainState `_check_resume`, reference fingerprint, pre-registration).
  `ppo.train_guarded` keeps the step budget on that fallback and train() logs the switch in
  TrainLog.device_switches; consumers comparing runs must read that field.

Nothing here re-raises application errors: `run_guarded` returns an `Outcome`.
Only `KeyboardInterrupt` / `SystemExit` pass through untouched — an operator
must always be able to stop the process.
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Mapping, Sequence

from . import audit as _audit

MAX_CONSECUTIVE_FAILURES = 3          # default quarantine threshold (overridable per task)
DEFAULT_MAX_CONSECUTIVE_FAILURES = MAX_CONSECUTIVE_FAILURES
GREEN, AMBER, RED = "green", "amber", "red"


class FailureKind(str, Enum):
    NONE = "none"
    CUDA_OOM = "cuda_oom"            # GPU memory: retry on CPU / smaller model
    HOST_OOM = "host_oom"            # host memory: retry with smaller budget
    SEGFAULT = "segfault"            # native crash: one retry at most
    TIMEOUT = "timeout"              # wall clock: retry with smaller budget
    NETWORK = "network"              # transient: retry after backoff
    ASSERTION = "assertion"          # deterministic: retry cannot help
    IMPORT_ERROR = "import_error"    # environment: deterministic for this box
    UNKNOWN = "unknown"


# Signature fragments searched (case-insensitive) in str(exc) and stderr.
_SIGNATURES: tuple[tuple[FailureKind, tuple[str, ...]], ...] = (
    (FailureKind.CUDA_OOM, ("cuda out of memory", "outofmemoryerror", "cudnn status alloc failed",
                            "hip out of memory")),
    (FailureKind.HOST_OOM, ("memoryerror", "cannot allocate memory", "std::bad_alloc",
                            "out of memory")),
    (FailureKind.SEGFAULT, ("segmentation fault", "core dumped", "illegal instruction")),
    (FailureKind.TIMEOUT, ("timed out", "timeout expired", "took longer than")),
    (FailureKind.NETWORK, ("connection reset", "connection refused", "connection aborted",
                           "ssl:", "temporary failure in name resolution", "remote disconnected")),
)

# Which kinds get another attempt (with a reduced budget), and which are
# deterministic enough that a retry would only hide the real problem.
RETRYABLE: dict[FailureKind, bool] = {
    FailureKind.CUDA_OOM: True,
    FailureKind.HOST_OOM: True,
    FailureKind.SEGFAULT: True,
    FailureKind.TIMEOUT: True,
    FailureKind.NETWORK: True,
    FailureKind.ASSERTION: False,
    FailureKind.IMPORT_ERROR: False,
    FailureKind.UNKNOWN: True,
}


class TaskDisabled(RuntimeError):
    """Raised lazily by TaskGuard.check() — callers decide to skip or surface."""


def classify_failure(exc: BaseException | None, stderr: str = "") -> tuple[FailureKind, str]:
    """Map an exception (plus optional stderr) to (FailureKind, severity band).

    Pure function: no I/O, no global state, never raises.
    """
    if exc is None:
        return FailureKind.NONE, GREEN
    if isinstance(exc, (KeyboardInterrupt, SystemExit)):
        # Operator action is not a failure of the task; severity is band-neutral.
        return FailureKind.UNKNOWN, AMBER
    if isinstance(exc, AssertionError):
        return FailureKind.ASSERTION, RED
    if isinstance(exc, ImportError):
        return FailureKind.IMPORT_ERROR, RED
    if isinstance(exc, TimeoutError):
        return FailureKind.TIMEOUT, AMBER
    if isinstance(exc, MemoryError):
        return FailureKind.HOST_OOM, RED
    if isinstance(exc, ConnectionError) or isinstance(exc, OSError) and "connection" in str(exc).lower():
        return FailureKind.NETWORK, AMBER

    text = f"{exc}\n{stderr}".lower()
    for kind, fragments in _SIGNATURES:
        if any(fragment in text for fragment in fragments):
            severity = RED if kind in (FailureKind.CUDA_OOM, FailureKind.HOST_OOM,
                                       FailureKind.SEGFAULT) else AMBER
            return kind, severity
    return FailureKind.UNKNOWN, AMBER


@dataclass(frozen=True)
class Report:
    """What happened to one guarded run — the honest status surface."""

    task: str
    kind: FailureKind
    severity: str
    attempt: int                 # 1-based attempt that produced this report
    recovered: bool = False      # True when a later attempt succeeded
    disabled: bool = False       # True when the task is quarantined
    message: str = ""
    backoff_sec: float = 0.0     # sleep applied before the next attempt


@dataclass(frozen=True)
class Outcome:
    ok: bool
    value: object = None
    report: Report | None = None
    attempts: int = 0
    audit_ok: bool = True        # False when an audit entry for this run could not be written

    def __bool__(self) -> bool:  # convenience: `if outcome:`
        return self.ok


@dataclass
class TaskGuard:
    """3-strikes quarantine state for one task (heartbeat pattern).

    `max_failures` is the quarantine threshold, configurable per task (default:
    DEFAULT_MAX_CONSECUTIVE_FAILURES).  `consecutive_failures` counts *task
    runs* (not attempts).  Reaching `max_failures` disables the task: later
    runs are refused without executing the callable until `reset()` is called.
    """

    task: str
    consecutive_failures: int = 0
    disabled: bool = False
    last_kind: FailureKind = FailureKind.NONE
    total_failures: int = 0
    total_recoveries: int = 0
    max_failures: int = DEFAULT_MAX_CONSECUTIVE_FAILURES
    history: list = field(default_factory=list)   # last N outcome strings

    def record(self, kind: FailureKind, ok: bool) -> None:
        self.history.append(f"{'ok' if ok else kind.value}")
        del self.history[:-20]
        if ok:
            # success clears the strike counter but never the quarantine itself:
            # only reset() (an audited, attributed request) lifts `disabled`
            self.consecutive_failures = 0
            self.last_kind = FailureKind.NONE
            return
        self.consecutive_failures += 1
        self.total_failures += 1
        self.last_kind = kind
        if self.consecutive_failures >= self.max_failures:
            self.disabled = True

    def check(self) -> None:
        if self.disabled:
            raise TaskDisabled(
                f"task {self.task!r} is quarantined after {self.consecutive_failures} consecutive "
                f"failures (threshold {self.max_failures}, last: {self.last_kind.value}); "
                f"reset_task() to re-enable")

    def reset(self) -> None:
        self.consecutive_failures = 0
        self.disabled = False


class SelfHealer:
    """Classify + retry + quarantine runner, audited, and never throwing.

    `budgets` are per-attempt keyword overrides for `fn` (attempt 1 uses none);
    give e.g. `[{"threads": 4}, {"device": "cpu", "threads": 2}]` so a CUDA OOM
    second attempt runs on CPU — recovery only ever *shrinks* the budget.
    """

    def __init__(self, audit_log: _audit.AuditLog | None = None, *, actor: str = "toddler",
                 now_ms: Callable[[], int] | None = None,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        self.audit = audit_log or _audit.AuditLog()
        self.audit_failures = 0
        self.actor = actor
        self._now_ms = now_ms or (lambda: int(time.time() * 1000))
        self._sleep = sleep
        self.guards: dict[str, TaskGuard] = {}
        self._lock = threading.Lock()

    # -- guard registry ---------------------------------------------------
    def guard(self, task: str, max_failures: int | None = None) -> TaskGuard:
        with self._lock:
            return self.guard_unlocked(task, max_failures)

    def guard_unlocked(self, task: str, max_failures: int | None = None) -> TaskGuard:
        if task not in self.guards:
            self.guards[task] = TaskGuard(task, max_failures=max_failures or DEFAULT_MAX_CONSECUTIVE_FAILURES)
        elif max_failures is not None:
            self.guards[task].max_failures = max_failures
        return self.guards[task]

    def reset_task(self, task: str, *, requested_by: str, reason: str) -> bool:
        """Lift a quarantine. `requested_by` names who asked (a person, a script, a service);
        the audit records that name, never an assumed operator. Returns audit_ok."""
        if not requested_by.strip() or not reason.strip():
            raise ValueError("reset_task needs a non-empty requested_by and reason")
        with self._lock:
            self.guard_unlocked(task).reset()
        return self._audit("selfheal.reset", {"task": task, "requested_by": requested_by, "reason": reason})

    # -- main entry -------------------------------------------------------
    def run_guarded(self, fn: Callable, *args, task: str, budgets: Sequence[Mapping] = (),
                    backoff_sec: float = 1.0, stderr_of: Callable[[BaseException], str] = lambda e: "",
                    max_failures: int | None = None, **kwargs) -> Outcome:
        try:
            guard = self.guard(task, max_failures)
            with self._lock:
                guard.check()
            return self._run(fn, *args, task=task, guard=guard, budgets=budgets,
                             backoff_sec=backoff_sec, stderr_of=stderr_of, **kwargs)
        except TaskDisabled as e:
            return Outcome(ok=False, report=Report(task, FailureKind.UNKNOWN, RED, attempt=0,
                                                   disabled=True, message=str(e)))
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:  # the watchdog itself must never propagate
            ok = self._audit("selfheal.watchdog_error", {"task": task, "error": repr(e)})
            return Outcome(ok=False, report=Report(task, FailureKind.UNKNOWN, RED, attempt=0,
                                                   message=f"watchdog failure: {e!r}"), audit_ok=ok)

    # -- internals --------------------------------------------------------
    def _run(self, fn, *args, task, guard, budgets, backoff_sec, stderr_of, **kwargs) -> Outcome:
        max_attempts = 1 + max(len(budgets), 0)
        attempt = 0
        first_report: Report | None = None
        last_kind = FailureKind.UNKNOWN
        audit_ok = True
        while attempt < max_attempts:
            attempt += 1
            override = budgets[attempt - 2] if attempt >= 2 else {}
            try:
                value = fn(*args, **{**kwargs, **override})
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception as e:
                kind, severity = classify_failure(e, stderr_of(e))
                last_kind = kind
                report = Report(task, kind, severity, attempt=attempt, message=repr(e))
                first_report = first_report or report
                audit_ok &= self._audit("selfheal.crash", {"task": task, "kind": kind.value, "attempt": attempt,
                                               "budget": dict(override), "error": repr(e)})
                # NB: guard state is recorded once per RUN (after the attempt
                # loop), never per attempt — otherwise one run that exhausts a
                # 3-attempt budget would instantly trip the 3-strikes quarantine.
                if not RETRYABLE[kind]:
                    break                       # deterministic: retry cannot help
                if attempt < max_attempts and not guard.disabled:
                    report = Report(task, kind, severity, attempt=attempt,
                                    message=repr(e), backoff_sec=backoff_sec)
                    audit_ok &= self._audit("selfheal.retry", {"task": task, "attempt": attempt,
                                                   "next_budget": dict(budgets[attempt - 1])})
                    self._sleep(backoff_sec)
                    continue
                break
            else:
                with self._lock:
                    guard.record(kind=FailureKind.NONE, ok=True)
                    if attempt > 1:
                        guard.total_recoveries += 1
                if attempt > 1:
                    audit_ok &= self._audit("selfheal.recovered", {"task": task, "attempt": attempt,
                                                                   "budget": dict(override)})
                return Outcome(ok=True, value=value,
                               report=first_report and Report(task, first_report.kind,
                                                              first_report.severity,
                                                              attempt=attempt, recovered=True,
                                                              message=first_report.message),
                               attempts=attempt, audit_ok=audit_ok)
        # One guard record per run: a failed run counts one strike, a
        # successful run clears the consecutive-failure counter.
        with self._lock:
            guard.record(last_kind, ok=False)
        if guard.disabled and not (first_report and first_report.disabled):
            audit_ok &= self._audit("selfheal.disabled", {"task": task,
                                              "consecutive_failures": guard.consecutive_failures})
        final = first_report or Report(task, FailureKind.UNKNOWN, AMBER, attempt=attempt)
        return Outcome(ok=False, report=Report(task, final.kind, final.severity, attempt=final.attempt,
                                               disabled=guard.disabled, message=final.message,
                                               backoff_sec=final.backoff_sec),
                       attempts=attempt, audit_ok=audit_ok)

    def _audit(self, event_type: str, details: Mapping) -> bool:
        """Append one entry; never raises, but a failure is counted and reported (audit_ok)."""
        try:
            self.audit.append(self._now_ms(), self.actor, event_type, dict(details))
            return True
        except Exception:
            with self._lock:
                self.audit_failures += 1
            return False


def _default_audit_path() -> Path:
    p = Path(os.environ.get("TODDLER_SELFHEAL_AUDIT")
             or Path.home() / ".local" / "share" / "toddler" / "selfheal_audit.jsonl")
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


_SHARED = SelfHealer(_audit.AuditLog(_default_audit_path()))


def shared_healer() -> SelfHealer:
    """Process-wide healer: quarantine state is shared across all guarded callers."""
    return _SHARED


def run_guarded(fn, *args, task: str, **kwargs) -> Outcome:
    """Module-level convenience on a shared SelfHealer (shared quarantine state)."""
    return _SHARED.run_guarded(fn, *args, task=task, **kwargs)


def reset_task(task: str, *, requested_by: str, reason: str) -> bool:
    return _SHARED.reset_task(task, requested_by=requested_by, reason=reason)


class GuardFailure(RuntimeError):
    """Raised by guarded wrappers (train_guarded, benchmark_toddler) when the
    guarded run did not succeed. Carries the honest Report, including whether
    the task is now quarantined (deferred, not crashed)."""

    def __init__(self, report: Report) -> None:
        super().__init__(
            f"guarded run failed ({report.kind.value}, attempt {report.attempt}, "
            f"quarantined={report.disabled}): {report.message}")
        self.report = report
