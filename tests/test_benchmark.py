import pytest

pytest.importorskip("gymnasium")
pytest.importorskip("torch")

from toddler import resources  # noqa: E402
from toddler.learn import benchmark as B  # noqa: E402
from toddler.learn import ppo  # noqa: E402
from toddler.learn import tasks as T  # noqa: E402

SMALL = dict(total_steps=2048, rollout=1024, epochs=1, minibatch=256)


def _host(free_mib=10_000, foreign=0, lease=False, guard=True):
    g = resources.GpuState(0, "Test GPU", "0000:01:00.0", 16_000, free_mib, foreign)
    return resources.HostState((g,), 8, 0.0, lease, guard)


def test_governor_decides_per_gpu():
    assert B.gpu_skip_reason(_host(), 0) is None
    assert "foreign" in B.gpu_skip_reason(_host(foreign=1), 0)
    assert "usable" in B.gpu_skip_reason(_host(free_mib=4000), 0)       # 4000 - 25% of 16000 < 512
    assert "lease" in B.gpu_skip_reason(_host(lease=True), 0)
    assert B.gpu_skip_reason(_host(), 3) == "gpu3 not present"


def test_configs_list_cpu_first_then_every_gpu():
    cfgs = B.configs(_host(), 4)
    assert [c.name for c in cfgs] == ["cpu-4t", "gpu0"] and cfgs[1].device == "cuda:0"


def test_cpu_benchmark_reproduces_quality_and_measures_efficiency():
    net, _ = ppo.train("cartpole", ppo.PPOConfig(seed=5, **SMALL))
    anchor = T.random_anchor("cartpole", T.EVAL_SEEDS[:3])
    from toddler.learn import scoring

    recorded = [float(T.normalise("cartpole", r, anchor)) for r in scoring.evaluate(net, "cartpole")]
    r = B.benchmark_toddler(net, "gen-x", "t1", "cartpole", anchor, recorded, [B.HardwareConfig("cpu-2t", "cpu", threads=2)])
    assert r.quality_reproduced
    e = r.efficiency[0]
    assert e.measured and e.latency_ms_median > 0 and e.throughput_states_per_s > 0
    assert e.energy_note == "not measured (CPU: no energy counter read)"


def test_states_come_from_real_rollouts():
    net, _ = ppo.train("cartpole", ppo.PPOConfig(seed=5, **SMALL))
    x = B.collect_states(net, "cartpole", n=64)
    assert x.shape == (64, 4)
