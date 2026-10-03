"""Measure and calibrate the Jev server's 'noul' probabilities on PIQA (physical commonsense).

PIQA (Bisk et al., 2020): each item has a goal, two solutions and the correct one. Every
solution becomes a yes/no question ("does this achieve the goal?"), so the labels are real
human annotations. Half of the items fit an isotonic calibration, the other half measures
Brier score, log loss, expected calibration error (ECE) and accuracy, raw and calibrated.
Nothing is assumed: the report says what the model actually does.

Run: PYTHONPATH=. python3 -m jevserver.calibrate --items 150 --out ~/.config/toddler-jev/calibration.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.isotonic import IsotonicRegression

from jevserver.backend import Backend, LlamaCppBackend

QUESTION = "Does the proposed solution achieve the goal in the physical world?"


def piqa_items(n: int, seed: int = 0) -> list[dict]:
    from datasets import load_dataset

    d = load_dataset("ybisk/piqa", split="validation", revision="refs/convert/parquet")
    idx = np.random.default_rng(seed).choice(len(d), size=min(n, len(d)), replace=False)
    return [d[int(i)] for i in idx]


def raw_probabilities(items: list[dict], backend: Backend) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (p_yes, label, item index) for both solutions of every item."""
    ps, ys, ix = [], [], []
    for i, it in enumerate(items):
        for k, sol in enumerate((it["sol1"], it["sol2"])):
            probs, _, _ = backend.option_probs({"goal": it["goal"], "solution": sol}, QUESTION, ["yes", "no"])
            ps.append(probs["yes"])
            ys.append(1 if it["label"] == k else 0)
            ix.append(i)
    return np.asarray(ps), np.asarray(ys), np.asarray(ix)


def ece(p: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi) if hi < 1 else (p >= lo) & (p <= hi)
        if m.any():
            total += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(total)


def metrics(p: np.ndarray, y: np.ndarray, ix: np.ndarray) -> dict:
    eps = 1e-6
    pc = np.clip(p, eps, 1 - eps)
    pair_correct = []
    for i in np.unique(ix):
        m = ix == i
        pair_correct.append(int(np.argmax(p[m]) == np.argmax(y[m])))
    return {"brier": round(float(np.mean((p - y) ** 2)), 4),
            "log_loss": round(float(-np.mean(y * np.log(pc) + (1 - y) * np.log(1 - pc))), 4),
            "ece": round(ece(p, y), 4),
            "accuracy_at_0_5": round(float(np.mean((p >= 0.5) == (y == 1))), 4),
            "piqa_pair_accuracy": round(float(np.mean(pair_correct)), 4)}


def calibrate(items: list[dict], backend: Backend) -> dict:
    t0 = time.time()
    p, y, ix = raw_probabilities(items, backend)
    fit = ix < len(items) // 2
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(p[fit], y[fit])
    pe, ye, ie = p[~fit], y[~fit], ix[~fit]
    return {
        "dataset": "PIQA validation (ybisk/piqa, refs/convert/parquet)",
        "items": len(items), "questions": int(len(p)), "fit_questions": int(fit.sum()),
        "seconds": round(time.time() - t0, 1),
        "raw_eval": metrics(pe, ye, ie),
        "calibrated_eval": metrics(iso.predict(pe), ye, ie),
        "isotonic": {"x": [float(v) for v in iso.X_thresholds_], "y": [float(v) for v in iso.y_thresholds_]},
    }


def load_calibrator(path: str):
    cal = json.loads(Path(path).expanduser().read_text())["isotonic"]
    x, yv = np.asarray(cal["x"]), np.asarray(cal["y"])
    return lambda p: float(np.interp(p, x, yv))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="~/.config/toddler-jev/calibration.json")
    a = ap.parse_args()
    report = calibrate(piqa_items(a.items, a.seed), LlamaCppBackend())
    out = Path(a.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != "isotonic"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
