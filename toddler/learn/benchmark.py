"""Benchmark generations per hardware configuration without breaking comparability.

Two kinds of measurement, deliberately kept apart:

  quality     - normalised task score on the held-out seeds, always measured on the CPU with
                greedy actions. It does not depend on the hardware, so it is measured once per
                toddler and checked against the score recorded at training time
                (`quality_reproduced`). Generations are compared on quality only.
  efficiency  - per hardware configuration: single-state latency (median and p95) and batched
                throughput of the policy, plus GPU energy per 1000 forward passes when the
                driver exposes an energy counter. Efficiency says what a generation costs on a
                given machine; it never enters a quotient.

Hardware is dynamic: a GPU configuration is only measured when the resource governor allows it
at that moment (free memory minus margin, no foreign compute process, guard and lease). A
configuration that is not measured is reported as skipped with its reason, never filled in.
The frozen reference generation is re-measured in every run so drift is visible.

Inputs for efficiency are real observations collected by running the policy itself on the first
held-out seed, not fabricated states.
"""

from __future__ import annotations

import os
import time
from dataclasses import asdict, dataclass, field

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")   # CUDA index == nvidia-smi index

import numpy as np
import torch

from toddler import resources
from toddler.learn import scoring
from toddler.learn import tasks as T
from toddler.learn.policy import ActorCritic

NEED_MIB = 512            # CUDA context + a tiny MLP; the governor still applies its margin


@dataclass(frozen=True)
class HardwareConfig:
    name: str              # "cpu-8t", "gpu0", ...
    device: str            # "cpu" or "cuda:<pci index>"
    threads: int = 0
    gpu_index: int | None = None
    gpu_name: str = ""


@dataclass
class Efficiency:
    config: str
    measured: bool
    reason: str = ""
    latency_ms_median: float | None = None
    latency_ms_p95: float | None = None
    throughput_states_per_s: float | None = None
    energy_mj_per_1000: float | None = None
    energy_note: str = ""


@dataclass
class ToddlerBenchmark:
    generation: str
    toddler_id: str
    quality: float
    recorded_quality: float
    quality_reproduced: bool
    efficiency: list[Efficiency] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def configs(host: resources.HostState, cpu_threads: int) -> list[HardwareConfig]:
    out = [HardwareConfig(f"cpu-{cpu_threads}t", "cpu", threads=cpu_threads)]
    out += [HardwareConfig(f"gpu{g.index}", f"cuda:{g.index}", gpu_index=g.index, gpu_name=g.name) for g in host.gpus]
    return out


def gpu_skip_reason(host: resources.HostState, index: int, need_mib: int = NEED_MIB) -> str | None:
    """Ask the governor about exactly this GPU; None means it may be used now."""
    only = [g for g in host.gpus if g.index == index]
    if not only:
        return f"gpu{index} not present"
    placement = resources.plan_device(
        resources.HostState(tuple(only), host.cores, host.load_1m, host.lease_held_by_other, host.guard_allows_gpu),
        need_mib)
    return None if placement.gpu_index == index else "; ".join(placement.reasons)


def collect_states(net: ActorCritic, task: str, n: int = 512) -> torch.Tensor:
    """Real observations visited by the policy on the first held-out seed (repeated episodes)."""
    net = net.to("cpu").eval()
    env, states, k = T.make(task), [], 0
    obs, _ = env.reset(seed=T.EVAL_SEEDS[0])
    while len(states) < n:
        states.append(obs)
        with torch.no_grad():
            a = int(torch.argmax(net(torch.as_tensor(obs, dtype=torch.float32))[0]))
        obs, _, term, trunc, _ = env.step(a)
        if term or trunc:
            k += 1
            obs, _ = env.reset(seed=T.EVAL_SEEDS[k % len(T.EVAL_SEEDS)])
    env.close()
    return torch.as_tensor(np.asarray(states), dtype=torch.float32)


def _energy_mj(index: int) -> tuple[float | None, str]:
    """(millijoules since driver load, "") or (None, why it could not be read)."""
    try:
        import pynvml
    except ImportError:
        return None, "pynvml not installed"
    try:
        pynvml.nvmlInit()
        try:
            h = pynvml.nvmlDeviceGetHandleByIndex(index)
            return float(pynvml.nvmlDeviceGetTotalEnergyConsumption(h)), ""
        finally:
            pynvml.nvmlShutdown()
    except pynvml.NVMLError_NotSupported:
        return None, "no energy counter on this device"
    except pynvml.NVMLError as exc:
        return None, f"energy read failed: {type(exc).__name__}"


def measure_efficiency(net: ActorCritic, states: torch.Tensor, cfg: HardwareConfig,
                       single_reps: int = 300, batch_reps: int = 50) -> Efficiency:
    if cfg.device.startswith("cuda"):
        if not torch.cuda.is_available():
            return Efficiency(cfg.name, False, "torch has no CUDA")
        torch_name = torch.cuda.get_device_name(cfg.gpu_index)
        if torch_name != cfg.gpu_name:
            return Efficiency(cfg.name, False, f"CUDA index maps to {torch_name}, not {cfg.gpu_name}")
    prev_threads = torch.get_num_threads()
    if cfg.threads:
        torch.set_num_threads(cfg.threads)
    dev = torch.device(cfg.device)
    model = ActorCritic(**net.spec()).to(dev).eval()
    model.load_state_dict(net.state_dict())
    x = states.to(dev)
    sync = (lambda: torch.cuda.synchronize(dev)) if dev.type == "cuda" else (lambda: None)
    try:
        with torch.no_grad():
            for i in range(20):                      # warm-up
                model(x[i:i + 1])
            sync()
            lat = []
            for i in range(single_reps):
                s = x[i % len(x):i % len(x) + 1]
                t0 = time.perf_counter()
                model(s)
                sync()
                lat.append((time.perf_counter() - t0) * 1e3)
            e0, why0 = _energy_mj(cfg.gpu_index) if cfg.gpu_index is not None else (None, "CPU: no energy counter read")
            t0 = time.perf_counter()
            for _ in range(batch_reps):
                model(x)
            sync()
            dt = time.perf_counter() - t0
            e1, why1 = _energy_mj(cfg.gpu_index) if cfg.gpu_index is not None else (None, why0)
    finally:
        torch.set_num_threads(prev_threads)
        del model, x
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    passes = batch_reps * len(states)
    eff = Efficiency(cfg.name, True, latency_ms_median=round(float(np.median(lat)), 4),
                     latency_ms_p95=round(float(np.quantile(lat, 0.95)), 4),
                     throughput_states_per_s=round(passes / dt, 1))
    if e0 is not None and e1 is not None and e1 >= e0:
        # board energy over the window, shared with anything else on that GPU: an upper bound
        eff.energy_mj_per_1000 = round((e1 - e0) / passes * 1000, 3)
        eff.energy_note = "GPU board counter over the window (upper bound: includes idle and other users)"
    elif e0 is not None and e1 is not None:
        eff.energy_note = "not measured (energy counter went backwards during the window)"
    else:
        eff.energy_note = f"not measured ({why0 or why1})"
    return eff


def benchmark_toddler(net: ActorCritic, generation: str, toddler_id: str, task: str, anchor: float,
                      recorded: list[float], hw: list[HardwareConfig],
                      tol: float = 1e-6) -> ToddlerBenchmark:
    q = [float(T.normalise(task, r, anchor)) for r in scoring.evaluate(net, task)]
    out = ToddlerBenchmark(generation, toddler_id, float(np.mean(q)), float(np.mean(recorded)),
                           bool(np.allclose(q, recorded, rtol=0.0, atol=tol)))
    states = collect_states(net, task)
    for cfg in hw:
        if cfg.gpu_index is not None:
            reason = gpu_skip_reason(resources.probe(), cfg.gpu_index)   # re-probe: capacity is dynamic
            if reason:
                out.efficiency.append(Efficiency(cfg.name, False, f"skipped by governor: {reason}"))
                continue
        out.efficiency.append(measure_efficiency(net, states, cfg))
    return out
