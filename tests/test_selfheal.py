import pytest
from toddler import audit, selfheal as sh


def _healer(**kw):
    return sh.SelfHealer(audit_log=audit.AuditLog(), sleep=lambda s: None, **kw)


# ---------------------------------------------------------------- classifier
def test_classify_cuda_oom_from_message_or_stderr():
    kind, sev = sh.classify_failure(RuntimeError("CUDA out of memory. Tried to allocate 2 GiB"))
    assert kind is sh.FailureKind.CUDA_OOM and sev == sh.RED

    class torch_oom(RuntimeError):
        pass

    kind, sev = sh.classify_failure(torch_oom("what?"), stderr="torch.OutOfMemoryError: on device 0")
    assert kind is sh.FailureKind.CUDA_OOM and sev == sh.RED


def test_classify_host_oom_timeout_network():
    assert sh.classify_failure(MemoryError())[0] is sh.FailureKind.HOST_OOM
    assert sh.classify_failure(TimeoutError("job timed out"))[0] is sh.FailureKind.TIMEOUT
    assert sh.classify_failure(ConnectionError("Connection reset by peer"))[0] is sh.FailureKind.NETWORK


def test_classify_deterministic_kinds_are_red_and_not_retryable():
    for exc, kind in [(AssertionError("3 != 4"), sh.FailureKind.ASSERTION),
                      (ImportError("No module named 'zlib2'"), sh.FailureKind.IMPORT_ERROR)]:
        kind_got, sev = sh.classify_failure(exc)
        assert kind_got is kind and sev == sh.RED
        assert not sh.RETRYABLE[kind_got]


def test_classify_none_is_green_and_unknown_is_amber_retryable():
    assert sh.classify_failure(None) == (sh.FailureKind.NONE, sh.GREEN)
    kind, sev = sh.classify_failure(ValueError("??"))
    assert kind is sh.FailureKind.UNKNOWN and sev == sh.AMBER and sh.RETRYABLE[kind]


def test_keyboard_interrupt_is_not_treated_as_task_failure():
    kind, sev = sh.classify_failure(KeyboardInterrupt())
    assert sev == sh.AMBER  # band-neutral; run_guarded re-raises it untouched


# ------------------------------------------------------------------ guarded runs
def test_success_first_try_no_crash_events():
    h = _healer()
    out = h.run_guarded(lambda: 42, task="t1")
    assert out.ok and out.value == 42 and out.attempts == 1
    kinds = [e.event_type for e in h.audit.entries]
    assert "selfheal.crash" not in kinds


def test_cuda_oom_is_retried_with_reduced_budget_and_recovers():
    h = _healer()
    seen = []

    def train(threads=16, device="cuda:1"):
        seen.append((threads, device))
        if device != "cpu":
            raise RuntimeError("CUDA out of memory on device cuda:1")
        return "done"

    out = h.run_guarded(train, task="train", budgets=[{"device": "cpu", "threads": 2}],
                        backoff_sec=0)
    assert out.ok and out.value == "done" and out.attempts == 2
    assert seen == [(16, "cuda:1"), (2, "cpu")]
    kinds = [e.event_type for e in h.audit.entries]
    assert kinds == ["selfheal.crash", "selfheal.retry", "selfheal.recovered"]


def test_assertion_failure_is_not_retried_and_quarantines_on_third_strike():
    h = _healer()
    calls = []

    def flaky():
        calls.append(1)
        raise AssertionError("deterministic bug")

    for _ in range(sh.MAX_CONSECUTIVE_FAILURES):
        out = h.run_guarded(flaky, task="t2")
        assert not out.ok and not out.report.recovered
    assert len(calls) == sh.MAX_CONSECUTIVE_FAILURES          # no wasted retries
    assert h.guard("t2").disabled

    # 4th call fails fast without executing the callable (deferred, not crashed)
    out = h.run_guarded(flaky, task="t2")
    assert not out.ok and out.report.disabled and len(calls) == sh.MAX_CONSECUTIVE_FAILURES
    assert "selfheal.disabled" in [e.event_type for e in h.audit.entries]


def test_unknown_failure_uses_all_budgets_then_quarantines_after_three_strikes():
    h = _healer()
    calls = []

    def always(value_at=None):
        calls.append(value_at)
        raise ValueError("mysterious")

    budgets = [{"value_at": "smaller"}, {"value_at": "cpu"}]
    for _ in range(sh.MAX_CONSECUTIVE_FAILURES):
        out = h.run_guarded(always, task="t3", budgets=budgets, backoff_sec=0)
        assert not out.ok
    # every run: attempt 1 without budget, then the two reduced budgets
    assert calls == [None, "smaller", "cpu"] * sh.MAX_CONSECUTIVE_FAILURES
    assert h.guard("t3").disabled


def test_reset_task_re_enables_quarantined_task():
    h = _healer()

    def bad():
        raise AssertionError("nope")

    for _ in range(sh.MAX_CONSECUTIVE_FAILURES):
        h.run_guarded(bad, task="t4")
    assert h.guard("t4").disabled
    h.reset_task("t4", requested_by="test-suite", reason="hotfix deployed")
    out = h.run_guarded(bad, task="t4")
    assert not out.ok and not out.report.disabled          # runs again, fails normally


def test_recovery_resets_consecutive_failures():
    h = _healer()
    state = {"n": 0}

    def sometimes():
        state["n"] += 1
        if state["n"] % 2:
            raise TimeoutError("timed out")
        return state["n"]

    for _ in range(4):
        out = h.run_guarded(sometimes, task="t5", budgets=[{}], backoff_sec=0)
    assert h.guard("t5").consecutive_failures == 0
    assert h.guard("t5").total_recoveries == 4 and not h.guard("t5").disabled   # every run recovered on attempt 2
    assert state["n"] == 8                                  # 4 runs x 2 attempts


def test_success_after_manual_reset_clears_disabled_flag():
    h = _healer()

    def flip():
        raise AssertionError("x")

    for _ in range(sh.MAX_CONSECUTIVE_FAILURES):
        h.run_guarded(flip, task="t6")
    h.reset_task("t6", requested_by="test-suite", reason="fixed")

    def good():
        return "ok"

    out = h.run_guarded(good, task="t6")
    assert out.ok and not h.guard("t6").disabled


# ---------------------------------------------------------------- never-throws guarantee
def test_watchdog_never_propagates_and_reports_watchdog_failure(monkeypatch):
    h = _healer()

    def boom(**kwargs):
        raise RuntimeError("budget explosion")

    monkeypatch.setattr(h, "guard", lambda task: (_ for _ in ()).throw(RuntimeError("registry broken")))
    out = h.run_guarded(lambda: 1, task="t7")
    assert not out.ok and "watchdog failure" in out.report.message


def test_audit_failures_are_swallowed():
    class broken_log:
        def append(self, *a, **kw):
            raise OSError("disk full")

    h = sh.SelfHealer(audit_log=broken_log(), sleep=lambda s: None)
    out = h.run_guarded(lambda: 7, task="t8")
    assert out.ok and out.value == 7


def test_keyboard_interrupt_passes_through():
    h = _healer()

    def stop():
        raise KeyboardInterrupt()

    try:
        h.run_guarded(stop, task="t9")
    except KeyboardInterrupt:
        pass
    else:
        raise AssertionError("KeyboardInterrupt must pass through")


def test_audit_events_are_hash_chained():
    h = _healer()

    def train(device="cuda"):
        raise RuntimeError("CUDA out of memory")

    out = h.run_guarded(train, task="t10", budgets=[{"device": "cpu"}], backoff_sec=0)
    assert not out.ok
    entries = h.audit.entries
    assert all(e.row_hash for e in entries)
    for prev, cur in zip(entries, entries[1:]):
        assert cur.prev_hash == prev.row_hash
    kinds = [e.event_type for e in entries]
    assert kinds[0] == "selfheal.crash" and "selfheal.retry" in kinds


# ---------------------------------------------------------------- review #27: audit honesty, attributed resets
def test_lost_audit_entries_are_reported_not_hidden():
    h = _healer()

    class Broken:
        def append(self, *a, **k):
            raise OSError("disk full")

    h.audit = Broken()

    def bad():
        raise AssertionError("x")

    out = h.run_guarded(bad, task="a1")
    assert not out.ok and out.audit_ok is False and h.audit_failures >= 1


def test_reset_requires_a_named_requester_and_records_it():
    h = _healer()
    with pytest.raises(ValueError):
        h.reset_task("r1", requested_by="", reason="x")
    with pytest.raises(ValueError):
        h.reset_task("r1", requested_by="bob", reason=" ")
    assert h.reset_task("r1", requested_by="nightly-script", reason="dependency fixed") is True
    last = h.audit.entries[-1]
    assert last.event_type == "selfheal.reset" and last.details["requested_by"] == "nightly-script"


def test_success_does_not_lift_a_quarantine_without_reset():
    g = sh.TaskGuard("q1")
    for _ in range(sh.MAX_CONSECUTIVE_FAILURES):
        g.record(sh.FailureKind.ASSERTION, ok=False)
    assert g.disabled
    g.record(sh.FailureKind.NONE, ok=True)        # e.g. a concurrent run that started before the trip
    assert g.disabled and g.consecutive_failures == 0
