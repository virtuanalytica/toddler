"""Resource governor: Toddler only uses capacity that is free, and gives it back.

Customer and other workloads come first. Toddler measures free GPU memory per device (PCI
order, so it matches nvidia-smi), honours the shared GPU safety guard and the GPU
coordination lease, keeps a safety margin, caps CPU threads at half the cores minus the current
load, and backs off during training when a foreign process appears or free memory falls below
the margin. It never stops or touches another process.

Decision logic is pure (plan_device, cpu_threads, should_back_off) and tested with explicit
inputs; probe() is the thin I/O layer.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

GUARD_FILE = Path("/media/knight2/EDS2/projects/numerai-signals/reports/gpu_safety_guard.json")
COORD = Path.home() / ".claude/coordination/coord.py"
GPU_LEASE = "numerai/gpu-rmse-experiments"
DEFAULT_MARGIN = 0.25


@dataclass(frozen=True)
class GpuState:
    index: int                 # nvidia-smi / PCI order
    name: str
    bus_id: str
    total_mib: int
    free_mib: int
    foreign_processes: int     # compute processes not owned by Toddler


@dataclass(frozen=True)
class HostState:
    gpus: tuple[GpuState, ...]
    cores: int
    load_1m: float
    lease_held_by_other: bool
    guard_allows_gpu: bool


@dataclass(frozen=True)
class Placement:
    device: str                # "cpu" or "cuda:<pci index>"
    gpu_index: int | None
    budget_mib: int
    cuda_visible: str          # value for CUDA_VISIBLE_DEVICES under CUDA_DEVICE_ORDER=PCI_BUS_ID
    reasons: tuple[str, ...] = field(default_factory=tuple)


def usable_mib(g: GpuState, margin: float = DEFAULT_MARGIN) -> int:
    return max(0, g.free_mib - int(g.total_mib * margin))


def plan_device(host: HostState, need_mib: int, margin: float = DEFAULT_MARGIN, allow_gpu: bool = True) -> Placement:
    """Pick the GPU with the most usable memory that fits `need_mib` and has no foreign compute
    process; otherwise CPU. Never plans more than free minus margin."""
    reasons: list[str] = []
    if not allow_gpu:
        reasons.append("GPU not requested")
    elif not host.guard_allows_gpu:
        reasons.append("GPU safety guard / preflight does not allow a GPU job")
    elif host.lease_held_by_other:
        reasons.append(f"GPU lease {GPU_LEASE} held by another session")
    else:
        best = None
        for g in host.gpus:
            if g.foreign_processes:
                reasons.append(f"gpu{g.index} busy ({g.foreign_processes} foreign process(es))")
                continue
            free = usable_mib(g, margin)
            if free < need_mib:
                reasons.append(f"gpu{g.index} usable {free} MiB < need {need_mib} MiB")
                continue
            if best is None or free > usable_mib(best, margin):
                best = g
        if best is not None:
            return Placement(f"cuda:{best.index}", best.index, min(need_mib, usable_mib(best, margin)),
                             str(best.index), tuple(reasons) + (f"gpu{best.index} {best.name}",))
    return Placement("cpu", None, 0, "", tuple(reasons) or ("no GPU fits",))


def cpu_threads(cores: int, load_1m: float, cap_share: float = 0.5) -> int:
    """At most half the cores, minus what is already busy; at least one thread."""
    return max(1, min(int(cores * cap_share), int(cores - load_1m)))


def should_back_off(placement: Placement, now: GpuState | None, margin: float = DEFAULT_MARGIN) -> str | None:
    """Return a reason to leave the GPU, or None. Called by the training watchdog."""
    if placement.gpu_index is None or now is None:
        return None
    if now.foreign_processes:
        return f"foreign process appeared on gpu{now.index}"
    if now.free_mib < int(now.total_mib * margin):
        return f"free memory on gpu{now.index} fell below the {int(margin * 100)} % margin"
    return None


def _own_pids() -> set[int]:
    return {os.getpid(), os.getppid()}


def probe(own_pids: set[int] | None = None) -> HostState:
    """Read the live host state (pynvml, guard file, preflight, coordination lease, load)."""
    import pynvml

    own = own_pids or _own_pids()
    pynvml.nvmlInit()
    gpus = []
    try:
        for i in range(pynvml.nvmlDeviceGetCount()):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            procs = [p for p in pynvml.nvmlDeviceGetComputeRunningProcesses(h) if p.pid not in own]
            name = pynvml.nvmlDeviceGetName(h)
            bus = pynvml.nvmlDeviceGetPciInfo(h).busId
            gpus.append(GpuState(i, name.decode() if isinstance(name, bytes) else name,
                                 bus.decode() if isinstance(bus, bytes) else bus,
                                 mem.total // 2**20, mem.free // 2**20, len(procs)))
    finally:
        pynvml.nvmlShutdown()
    return HostState(tuple(gpus), os.cpu_count() or 1, os.getloadavg()[0], _lease_held_by_other(), _guard_allows())


def _guard_allows() -> bool:
    if not GUARD_FILE.exists():
        return True
    try:
        rules = json.loads(GUARD_FILE.read_text())["rules"]
    except (OSError, ValueError, KeyError):
        return False
    script = rules.get("preflight_script", "")
    if rules.get("preflight_required") and script:
        alt = Path(script)
        if not alt.exists():
            alt = Path("/media/knight2/EDS2/projects/numerai-signals/signals-repo/scripts/gpu_preflight.sh")
        if not alt.exists():
            return False
        return subprocess.run(["bash", str(alt)], capture_output=True, timeout=60).returncode == 0
    return True


def _lease_held_by_other(agent_id: str | None = None) -> bool:
    if not COORD.exists():
        return False
    me = agent_id or os.environ.get("CLAUDE_AGENT_ID", "toddler")
    out = subprocess.run(["python3", str(COORD), "status"], capture_output=True, text=True, timeout=30).stdout
    for line in out.splitlines():
        parts = line.split()
        if parts and parts[0] == GPU_LEASE and "held" in parts:
            return parts[1] != me
    return False


def cuda_env(p: Placement) -> dict[str, str]:
    """Environment for a child process so CUDA numbering matches nvidia-smi."""
    return {"CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": p.cuda_visible}
