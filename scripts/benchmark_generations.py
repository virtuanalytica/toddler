"""Benchmark every registered generation per hardware configuration (fase E).

Quality is measured on the CPU (hardware-independent) and checked against the training record;
efficiency is measured per configuration the resource governor allows right now; the rest is
reported as skipped. Quotients use the frozen gen-0 reference, whose fingerprint must match.

Run: TODDLER_GPU_GUARD=... TODDLER_COORD=... PYTHONPATH=. python3 scripts/benchmark_generations.py [--root DIR]
The registry must exist first (scripts/build_generations.py; same --root / TODDLER_GENERATIONS_ROOT).
GPU rows are a snapshot: which GPUs the governor allows depends on the moment and on whether the
guard and lease are configured (TODDLER_GPU_GUARD, TODDLER_COORD); without them, any idle GPU is used.
Report: docs/learn/generation_benchmarks.json
"""

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

from toddler import business, quotients, resources
from toddler.learn import benchmark as B
from toddler.learn import generations as G
from toddler.learn import scoring
from toddler.learn import tasks as T

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_generations import default_root  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=default_root())
    ap.add_argument("--reference", default="gen-0")
    a = ap.parse_args()
    t0 = time.time()
    host = resources.probe()
    threads = min(8, resources.cpu_threads(host.cores, host.load_1m))
    torch.set_num_threads(threads)
    reg = G.Registry(Path(a.root))
    gens = sorted(p.name for p in Path(a.root).iterdir() if p.is_dir() and p.name.startswith("gen-"))
    hw = B.configs(host, threads)

    results, by_gen, task, anchors = [], {}, None, {}
    for gen in gens:
        for rec in reg.generation(gen):
            task = task or rec.task
            if rec.task != task:
                raise SystemExit(f"{gen}/{rec.toddler_id} is on {rec.task}, not {task}; benchmark one task per run")
            net, _ = reg.load(gen, rec.toddler_id)
            anchors.setdefault(rec.task, T.random_anchor(rec.task))      # deterministic: measure once
            r = B.benchmark_toddler(net, gen, rec.toddler_id, rec.task, anchors[rec.task], rec.eval_scores, hw)
            results.append(r)
            by_gen.setdefault(gen, []).append(r.quality)
            print(f"{gen}/{rec.toddler_id} quality {r.quality:.3f} reproduced={r.quality_reproduced}", flush=True)

    fp = quotients.fingerprint([task], T.EVAL_SEEDS, [anchors[task]], [T.TASKS[task].solved])
    frozen = reg.reference(a.reference, fp)
    remeasured = by_gen[a.reference]
    gens_out = {}
    for gen, q in by_gen.items():
        point, lo, hi = quotients.iq_quotient_ci(np.asarray(q)[:, None], frozen)
        gens_out[gen] = {"toddlers": len(q), "iq_raw": round(quotients.iq_raw(np.asarray(q)[:, None]), 3),
                         "iq_quotient": round(point, 1), "iq_quotient_95ci": [round(lo, 1), round(hi, 1)]}

    per_config = {}
    for cfg in hw:
        rows = [e for r in results for e in r.efficiency if e.config == cfg.name]
        ok = [e for e in rows if e.measured]
        per_config[cfg.name] = {
            "device": cfg.device, "gpu_name": cfg.gpu_name or None, "measured": len(ok), "skipped": len(rows) - len(ok),
            "skip_reasons": sorted({e.reason for e in rows if not e.measured}),
            "latency_ms_median": round(float(np.median([e.latency_ms_median for e in ok])), 4) if ok else None,
            "throughput_states_per_s_median": round(float(np.median([e.throughput_states_per_s for e in ok])), 1) if ok else None,
        }

    report = {
        "task": task, "measured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "seconds": round(time.time() - t0, 1),
        "reference": {"generation": a.reference, "fingerprint": fp.digest(),
                      "frozen_raw_iqm": round(scoring.iqm(frozen), 4),
                      "remeasured_raw_iqm": round(scoring.iqm(remeasured), 4),
                      "drift": round(scoring.iqm(remeasured) - scoring.iqm(frozen), 6)},
        "quality_reproduced_all": all(r.quality_reproduced for r in results),
        "generations": gens_out,
        "hardware_configs": per_config,
        "host": {"cpu": platform.processor() or platform.machine(), "cores": host.cores,
                 "load_1m_at_start": round(host.load_1m, 1), "torch": torch.__version__},
        "notes": ["Quality is CPU-measured and hardware-independent; efficiency never enters a quotient.",
                  "GPU configs are measured only when the governor allows them at that moment; skipped is reported, not filled in.",
                  "Efficiency rows are a point-in-time snapshot; quality rows must be identical on every re-run."],
        "toddlers": [r.to_dict() for r in results],
        **business.FIELDS,
    }
    out = Path(__file__).resolve().parents[1] / "docs" / "learn" / "generation_benchmarks.json"
    out.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({k: report[k] for k in ("reference", "quality_reproduced_all", "generations", "hardware_configs")}, indent=1))


if __name__ == "__main__":
    main()
