"""Compare saved G3-recombined policies with their G2 parents on public states.

This is an operational cost probe. It never opens the secret seed sets and never
uses latency, weight size or a public task score to promote a generation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from toddler.learn import tasks as T
from toddler.learn.generations import Registry
from toddler.learn.multitask import MultiTaskNet, TaskView
from toddler.learn.routing import TaskExpertRouter

TASKS = ("doorkey8", "unlockpickup")
CHILDREN = tuple(range(1, 6))


def observations(task: str, count: int = 128) -> torch.Tensor:
    """Record real environment observations from a seeded uniform-action walk."""
    env = T.make(task, seed=T.EVAL_SEEDS[0])
    rng = np.random.default_rng(T.EVAL_SEEDS[0])
    values = []
    episode = 0
    try:
        state, _ = env.reset(seed=T.EVAL_SEEDS[episode])
        while len(values) < count:
            values.append(np.asarray(state, dtype=np.float32))
            state, _, terminated, truncated, _ = env.step(int(rng.integers(env.action_space.n)))
            if terminated or truncated:
                episode += 1
                state, _ = env.reset(seed=T.EVAL_SEEDS[episode % len(T.EVAL_SEEDS)])
    finally:
        env.close()
    return torch.from_numpy(np.stack(values))


def measure(net: MultiTaskNet | TaskExpertRouter, task: str, states: torch.Tensor,
            singles: int = 300, batches: int = 40) -> dict:
    if singles < 30 or batches < 5:
        raise ValueError("timing needs at least 30 single and 5 batch passes")
    policy = TaskView(net.to("cpu").eval(), task)
    with torch.inference_mode():
        for _ in range(30):
            policy(states[:1])
        latencies = []
        for index in range(singles):
            state = states[index % len(states):index % len(states) + 1]
            begin = time.perf_counter_ns()
            policy(state)
            latencies.append((time.perf_counter_ns() - begin) / 1_000_000)
        begin = time.perf_counter_ns()
        for _ in range(batches):
            policy(states)
        elapsed = (time.perf_counter_ns() - begin) / 1_000_000_000
    return {"single_ms_median": round(statistics.median(latencies), 4),
            "single_ms_p95": round(float(np.quantile(latencies, .95)), 4),
            "batch_states_per_second": round(batches * len(states) / elapsed, 1),
            "single_passes": singles, "batch_passes": batches, "batch_size": len(states)}


def verify_sources(child: TaskExpertRouter, record, parent_id: str, registry: Registry,
                   protocol: dict, protocol_sha256: str) -> None:
    """The saved router must still contain the exact pre-registered source weights."""
    if record.config.get("protocol_sha256") != protocol_sha256:
        raise ValueError("candidate record cites another source protocol")
    if record.config.get("route") != child.route:
        raise ValueError("saved route differs from the recorded route")
    for generation, embedded in child.experts.items():
        source_id = parent_id if generation.startswith("G2") else f"t{int(parent_id[1:]) + 1000}"
        source, source_record = registry.load(generation, source_id)
        if source_record.weights_sha256 != protocol["source_weights"][generation][parent_id]:
            raise ValueError(f"{generation}/{source_id} differs from pre-registration")
        embedded_state, source_state = embedded.state_dict(), source.state_dict()
        if embedded_state.keys() != source_state.keys() or any(
                not torch.equal(value, source_state[key]) for key, value in embedded_state.items()):
            raise ValueError(f"{generation}/{source_id} embedded weights differ from source")


def run(baseline_root: Path, candidate_root: Path, source_protocol: Path,
        tasks: tuple[str, ...] = TASKS,
        repeats: int = 3) -> dict:
    if repeats < 2:
        raise ValueError("use at least two repeated timing rounds")
    if not tasks or len(tasks) != len(set(tasks)) or any(task not in T.TASKS for task in tasks):
        raise ValueError("tasks must be distinct registered tasks")
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    rows = []
    try:
        baseline, candidate = Registry(baseline_root), Registry(candidate_root)
        protocol_raw = source_protocol.read_bytes()
        protocol = json.loads(protocol_raw)
        protocol_sha256 = hashlib.sha256(protocol_raw).hexdigest()
        samples = {task: observations(task) for task in tasks}
        for index in CHILDREN:
            parent_id, child_id = f"t{2000 + index}", f"t{5000 + index}"
            parent, parent_record = baseline.load("G2", parent_id)
            child, child_record = candidate.load("G3-recombined", child_id)
            if not isinstance(parent, MultiTaskNet) or not isinstance(child, TaskExpertRouter):
                raise TypeError("comparison requires a G2 multitask parent and a routed child")
            if f"G2/{parent_id}" not in child_record.parents:
                raise ValueError(f"{child_id} does not cite its matched G2 parent")
            verify_sources(child, child_record, parent_id, baseline, protocol, protocol_sha256)
            if any(child.route[task] != "G2" for task in ("cartpole", "acrobot", "empty5", "doorkey5")):
                raise ValueError(f"{child_id} reroutes an original G1 task")
            for task in tasks:
                arms = (("G2", parent, parent_record, baseline_root / "G2" / parent_id / "weights.pt"),
                        ("G3-recombined", child, child_record,
                         candidate_root / "G3-recombined" / child_id / "weights.pt"))
                timings: dict[str, list[dict]] = {name: [] for name, *_ in arms}
                for repetition in range(repeats):
                    for name, net, _, _ in (arms if repetition % 2 == 0 else reversed(arms)):
                        timings[name].append(measure(net, task, samples[task]))
                for name, _, record, weights in arms:
                    measured = timings[name]
                    rows.append({"task": task, "child_index": index, "arm": name,
                                 "model_id": record.toddler_id, "weights_sha256": record.weights_sha256,
                                 "weights_bytes": weights.stat().st_size,
                                 "single_ms_median": round(statistics.median(x["single_ms_median"] for x in measured), 4),
                                 "single_ms_p95": round(statistics.median(x["single_ms_p95"] for x in measured), 4),
                                 "batch_states_per_second": round(statistics.median(x["batch_states_per_second"] for x in measured), 1),
                                 "timing_rounds": repeats})
    finally:
        torch.set_num_threads(previous_threads)
    summary = {}
    for task in tasks:
        summary[task] = {}
        for arm in ("G2", "G3-recombined"):
            selected = [row for row in rows if row["task"] == task and row["arm"] == arm]
            summary[task][arm] = {
                "single_ms_median_across_children": round(statistics.median(row["single_ms_median"] for row in selected), 4),
                "batch_states_per_second_median_across_children": round(statistics.median(row["batch_states_per_second"] for row in selected), 1),
                "weights_bytes_mean": round(statistics.mean(row["weights_bytes"] for row in selected)),
            }
    return {"protocol": "public-state-routed-successor-efficiency-v1",
            "measured_at": datetime.now(timezone.utc).isoformat(),
            "cpu": platform.processor() or platform.machine(), "torch": torch.__version__,
            "torch_threads": 1, "tasks": list(tasks), "children_per_arm": len(CHILDREN),
            "source_protocol_sha256": protocol_sha256, "source_weights_verified": True,
            "input": "real observations from seeded uniform-action walks on public evaluation seeds",
            "measurement": "300 single-state and 40 batch-128 forwards per round; three rounds by default",
            "energy": {"status": "not_measured", "reason": "shared host CPU energy cannot be attributed to these small forwards"},
            "quality": {"status": "not_measured", "reason": "public states are used only for timing, not a promotion test"},
            "summary": summary, "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", required=True, type=Path)
    parser.add_argument("--candidate-root", required=True, type=Path)
    parser.add_argument("--source-protocol", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("output already exists; use a new result path")
    report = run(args.baseline_root, args.candidate_root, args.source_protocol,
                 repeats=args.repeats)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
