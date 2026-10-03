"""Confirmatory PBT test (docs/learn/PREREG_pbt.md).

Run: PYTHONPATH=. python3 scripts/pbt_experiment.py
Report: docs/learn/pbt.json
"""

import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from scipy import stats

from toddler import business, resources
from toddler.learn import pbt, scoring
from toddler.learn import tasks as T

TASK, SEEDS, INTERVALS, INTERVAL_STEPS, MEMBERS = "doorkey5", (21, 22, 23, 24, 25), 8, 10_000, 4


def main() -> None:
    host = resources.probe()
    torch.set_num_threads(min(4, resources.cpu_threads(host.cores, host.load_1m)))
    t0, anchor = time.time(), T.random_anchor(TASK)
    groups, runs = {"pbt": [], "control": []}, []
    for s in SEEDS:
        for arm in ("pbt", "control"):
            net, r = pbt.run_population(TASK, seed=s, pbt=arm == "pbt", members=MEMBERS,
                                        intervals=INTERVALS, interval_steps=INTERVAL_STEPS)
            score = round(float(T.normalise(TASK, scoring.evaluate(net, TASK).mean(), anchor)), 4)
            groups[arm].append(score)
            runs.append({**asdict(r), "held_out_score": score})
            print(arm, s, score, r.train_scores, flush=True)
    x, y = np.asarray(groups["pbt"]), np.asarray(groups["control"])
    p = float(stats.mannwhitneyu(x, y, alternative="greater").pvalue)
    report = {"prereg": "docs/learn/PREREG_pbt.md", "task": TASK, "anchor": anchor, "seeds": SEEDS,
              "members": MEMBERS, "steps_per_member": INTERVALS * INTERVAL_STEPS, "groups": groups,
              "iqm": {k: scoring.iqm(np.asarray(v)) for k, v in groups.items()}, "p": round(p, 4),
              "prob_improvement": scoring.prob_improvement(x[:, None], y[:, None]), "pbt_helps": p < 0.05,
              "seconds": round(time.time() - t0, 1), "runs": runs, **business.FIELDS}
    out = Path(__file__).resolve().parents[1] / "docs" / "learn" / "pbt.json"
    out.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({k: report[k] for k in ("iqm", "p", "prob_improvement", "pbt_helps")}))


if __name__ == "__main__":
    main()
