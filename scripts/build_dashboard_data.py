"""Build the data file behind the Toddler learning dashboard (site/toddler/learning.json).

Reads every generation in the registry (meta.json per toddler, reference.json where frozen) and
writes one point per generation: when it was trained, how (method, network, evaluation mode) and
the interquartile mean of the normalised held-out score per task over its toddlers. Scores from
different evaluation modes are never merged; the dashboard draws them as separate series.

Run: PYTHONPATH=. python3 scripts/build_dashboard_data.py [--root DIR]
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from toddler.learn.scoring import iqm

REPO = Path(__file__).resolve().parents[1]


def default_root() -> str:
    return os.environ.get("TODDLER_GENERATIONS_ROOT") or str(Path.home() / ".local/share/toddler/generations")


def _per_task(meta: dict) -> dict[str, float]:
    """Normalised held-out score per task for one toddler record."""
    cfg = meta.get("config", {})
    if cfg.get("per_task"):
        return {k: float(v) for k, v in cfg["per_task"].items()}
    return {meta["task"]: float(np.mean(meta["eval_scores"]))}


def collect(root: Path) -> list[dict]:
    points = []
    for gdir in sorted(p for p in root.iterdir() if p.is_dir()):
        metas = [json.loads(m.read_text()) for m in sorted(gdir.glob("*/meta.json"))]
        if not metas:
            continue
        cfg = metas[0].get("config", {})
        scores: dict[str, list[float]] = defaultdict(list)
        for m in metas:
            for task, s in _per_task(m).items():
                scores[task].append(s)
        tasks = {t: {"iqm": round(iqm(np.asarray(v)), 4), "n": len(v),
                     "values": [round(x, 4) for x in v]} for t, v in sorted(scores.items())}
        ref = gdir / "reference.json"
        points.append({
            "generation": gdir.name,
            "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(max(m["created_at"] for m in metas))),
            "toddlers": len(metas),
            "network": "multitask" if str(metas[0].get("task", "")).startswith("multitask:") else "single-task",
            "method": cfg.get("method", "ppo"),
            "scale_rewards": cfg.get("scale_rewards", False),
            "eval_mode": cfg.get("eval_mode", "greedy"),
            "parents": sorted({p for m in metas for p in m.get("parents", [])}),
            "frozen_reference": ref.exists(),
            "tasks": tasks,
            "iq_raw": round(float(np.mean([t["iqm"] for t in tasks.values()])), 4),
        })
    return sorted(points, key=lambda p: p["trained_at"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=default_root())
    ap.add_argument("--out", default=str(REPO / "site" / "toddler" / "learning.json"))
    a = ap.parse_args()
    data = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "score": "normalised held-out score: 0 = random policy, 1 = task solved; interquartile mean over toddlers",
        "rule": "scores with different eval_mode or network are separate series and are never joined",
        "generations": collect(Path(a.root)),
    }
    Path(a.out).write_text(json.dumps(data, indent=1) + "\n")
    print(f"{len(data['generations'])} generations -> {a.out}")


if __name__ == "__main__":
    main()
