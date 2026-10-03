"""Confirmatory test of return scaling (docs/learn/PREREG_return_scaling.md).

Run: PYTHONPATH=. python3 scripts/return_scaling_experiment.py
Report: docs/learn/return_scaling.json
"""

import json
from pathlib import Path

import numpy as np
import torch
from scipy import stats

from toddler import resources
from toddler.learn import ppo, scoring
from toddler.learn import tasks as T

SEEDS, BUDGET, TASKS = (21, 22, 23, 24, 25), 150_000, ("acrobot", "cartpole")


def main() -> None:
    host = resources.probe()
    torch.set_num_threads(min(4, resources.cpu_threads(host.cores, host.load_1m)))
    report = {"prereg": "docs/learn/PREREG_return_scaling.md", "budget": BUDGET, "seeds": SEEDS, "tasks": {}}
    for task in TASKS:
        anchor = T.random_anchor(task)
        groups = {"scaled": [], "unscaled": []}
        for s in SEEDS:
            for name, flag in (("scaled", True), ("unscaled", False)):
                net, _ = ppo.train(task, ppo.PPOConfig(total_steps=BUDGET, seed=s, scale_rewards=flag))
                groups[name].append(round(float(T.normalise(task, scoring.evaluate(net, task).mean(), anchor)), 4))
                print(task, s, name, groups[name][-1], flush=True)
        x, y = np.asarray(groups["scaled"]), np.asarray(groups["unscaled"])
        p = float(stats.mannwhitneyu(x, y, alternative="greater").pvalue)
        report["tasks"][task] = {"anchor": anchor, "groups": groups, "iqm": {k: scoring.iqm(np.asarray(v)) for k, v in groups.items()},
                                 "p": round(p, 4), "prob_improvement": scoring.prob_improvement(x[:, None], y[:, None]),
                                 "passes": p < 0.05}
    report["default_changes"] = all(t["passes"] for t in report["tasks"].values())
    out = Path(__file__).resolve().parents[1] / "docs" / "learn" / "return_scaling.json"
    out.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({k: (v["p"], v["iqm"]) for k, v in report["tasks"].items()}), report["default_changes"])


if __name__ == "__main__":
    main()
