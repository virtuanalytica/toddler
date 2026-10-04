import pytest

pytest.importorskip("gymnasium")
pytest.importorskip("torch")

from toddler import audit, selfheal  # noqa: E402
from toddler.learn import benchmark as B  # noqa: E402
from toddler.learn import ppo  # noqa: E402


def _healer():
    return selfheal.SelfHealer(audit_log=audit.AuditLog(), sleep=lambda s: None)


CUDA_OOM = "CUDA out of memory. Tried to allocate 2.00 GiB"


# ---------------------------------------------------------------- ppo.train_guarded
def test_train_guarded_falls_back_to_cpu_after_cuda_oom_with_same_budget(monkeypatch):
    calls = []

    def fake_train(task, cfg, device="cpu", **kw):
        calls.append((task, device, cfg.total_steps))
        if device != "cpu":
            raise RuntimeError(CUDA_OOM)
        return "net", "log"

    monkeypatch.setattr(ppo, "train", fake_train)
    cfg = ppo.PPOConfig(total_steps=1234, seed=3)
    h = _healer()
    net, log = ppo.train_guarded("cartpole", cfg, device="cuda:0", healer=h)
    assert (net, log) == ("net", "log")
    # fallback ran on the CPU with the SAME step budget: steps, not wall-clock, are the discipline
    assert calls == [("cartpole", "cuda:0", 1234), ("cartpole", "cpu", 1234)]
    kinds = [e.event_type for e in h.audit.entries]
    assert kinds == ["selfheal.crash", "selfheal.retry", "selfheal.recovered"]


def test_train_guarded_cpu_run_is_not_retried():
    calls = []

    def fake_train(task, cfg, device="cpu", **kw):
        calls.append(device)
        raise RuntimeError("boom")

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(ppo, "train", fake_train)
    h = _healer()
    with pytest.raises(selfheal.GuardFailure) as ei:
        ppo.train_guarded("cartpole", ppo.PPOConfig(), device="cpu", healer=h, max_failures=1)
    assert ei.value.report.kind is selfheal.FailureKind.UNKNOWN
    assert calls == ["cpu"]                            # no CPU->CPU retry budgets
    assert h.guard("ppo.cartpole").disabled            # threshold 1 reached
    monkeypatch.undo()


def test_train_guarded_quarantine_fails_fast_without_calling_train(monkeypatch):
    calls = []

    def fake_train(task, cfg, device="cpu", **kw):
        calls.append(device)
        raise AssertionError("deterministic bug")

    monkeypatch.setattr(ppo, "train", fake_train)
    h = _healer()
    for _ in range(selfheal.DEFAULT_MAX_CONSECUTIVE_FAILURES):
        with pytest.raises(selfheal.GuardFailure):
            ppo.train_guarded("cartpole", ppo.PPOConfig(), healer=h)
    n_after_strikes = len(calls)
    with pytest.raises(selfheal.GuardFailure) as ei:
        ppo.train_guarded("cartpole", ppo.PPOConfig(), healer=h)
    assert ei.value.report.disabled and len(calls) == n_after_strikes
    # assertion failures are deterministic: exactly one call per run, no retries
    assert n_after_strikes == selfheal.DEFAULT_MAX_CONSECUTIVE_FAILURES


# ---------------------------------------------------------------- benchmark guards
def _cfg(name="gpu0", device="cuda:0", threads=None):
    return B.HardwareConfig(name, device, gpu_index=0 if device.startswith("cuda") else None,
                            gpu_name="Test GPU" if device.startswith("cuda") else None,
                            threads=threads)


def test_measure_cell_degrades_broken_cell_and_names_the_kind(monkeypatch):
    def broken(net, states, cfg, **kw):
        raise RuntimeError(CUDA_OOM)

    monkeypatch.setattr(B, "measure_efficiency", broken)
    eff = B.measure_cell(object(), object(), _cfg(), "cartpole", healer=_healer())
    assert not eff.measured and "watchdog: cuda_oom" in eff.reason


def test_measure_cell_recovers_on_reduced_retry(monkeypatch):
    seen = []

    def flaky(net, states, cfg, single_reps=300, batch_reps=50, **kw):
        seen.append((single_reps, batch_reps, cfg.threads))
        if len(seen) == 1:
            raise RuntimeError(CUDA_OOM)
        return B.Efficiency(cfg.name, True, latency_ms_median=1.0)

    monkeypatch.setattr(B, "measure_efficiency", flaky)
    eff = B.measure_cell(object(), object(), _cfg(threads=8), "cartpole", healer=_healer())
    assert eff.measured and eff.latency_ms_median == 1.0
    # second attempt: strictly smaller budget (fewer reps, fewer threads)
    assert seen[1][0] < 300 and seen[1][1] < 50 and seen[1][2] == 4


def test_benchmark_toddler_cpu_fallback_keeps_quality_score(monkeypatch):
    from toddler.learn import scoring
    from toddler.learn import tasks as T

    quality_called = []

    def fake_benchmark(net, generation, toddler_id, task, anchor, recorded, hw, tol=1e-6):
        quality_called.append([c.device for c in hw])
        if any(c.device.startswith("cuda") for c in hw):
            raise RuntimeError(CUDA_OOM)               # a GPU cell killed the whole run
        return B.ToddlerBenchmark(generation, toddler_id, 0.5, 0.5, True)

    monkeypatch.setattr(B, "_benchmark_toddler", fake_benchmark)
    monkeypatch.setattr(scoring, "evaluate", lambda *a, **kw: [])  # unused in the fake, keep import honest
    monkeypatch.setattr(T, "normalise", lambda *a, **kw: 0.0)
    hw = [B.HardwareConfig("cpu-4t", "cpu", threads=4), _cfg()]
    h = _healer()
    out = B.benchmark_toddler(object(), "gen-x", "t1", "cartpole", 0.0, [], hw, healer=h)
    assert out.quality_reproduced
    # retry ran with CPU-only cells: quality never depends on a GPU
    assert quality_called[0][1] == "cuda:0" and quality_called[1] == ["cpu"]
