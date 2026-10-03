"""Build gen-0 (self-reinforcement from scratch) and gen-1 (taught by the best gen-0 toddler via
behaviour cloning, same step budget), register both, decide promotion and compute IQ quotients
against the frozen gen-0 reference.

Run: PYTHONPATH=. python3 scripts/build_generations.py [--root DIR]
Weights go to the registry root (outside git); the report goes to docs/learn/generations_report.json.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from toddler import quotients, resources
from toddler.learn import generations as G
from toddler.learn import peer, ppo, scoring
from toddler.learn import tasks as T

TASK, BUDGET, CLONE = "cartpole", 150_000, 10_000


def evaluate(net, anchor) -> list[float]:
    return [float(T.normalise(TASK, r, anchor)) for r in scoring.evaluate(net, TASK)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/media/knight2/EDS2/toddler-generations")
    a = ap.parse_args()
    host = resources.probe()
    torch.set_num_threads(min(8, resources.cpu_threads(host.cores, host.load_1m)))
    reg, anchor, hw, t0 = G.Registry(Path(a.root)), T.random_anchor(TASK), G.hardware_fingerprint(), time.time()

    gen0 = []
    for s in (101, 102, 103, 104, 105):
        tid = f"t{s}"
        net, log = ppo.train(TASK, ppo.PPOConfig(total_steps=BUDGET, seed=s),
                             checkpoint=reg.checkpoint_fn("gen-0", tid), checkpoint_every=25)
        rec = G.ToddlerRecord("gen-0", tid, TASK, {"seed": s, "method": "ppo"}, log.steps, [], evaluate(net, anchor), hw)
        reg.save(net, rec)
        gen0.append(rec)
    best = max(gen0, key=lambda r: r.score)
    teacher, _ = reg.load("gen-0", best.toddler_id)

    gen1 = []
    for s in (201, 202, 203, 204, 205):
        tid = f"t{s}"
        warm = peer.behaviour_clone(teacher, TASK, CLONE, seed=s)
        net, log = ppo.train(TASK, ppo.PPOConfig(total_steps=BUDGET - CLONE, seed=s), net=warm,
                             checkpoint=reg.checkpoint_fn("gen-1", tid), checkpoint_every=25)
        rec = G.ToddlerRecord("gen-1", tid, TASK, {"seed": s, "method": "behaviour_clone+ppo", "clone_steps": CLONE},
                              CLONE + log.steps, [f"gen-0/{best.toddler_id}"], evaluate(net, anchor), hw)
        reg.save(net, rec)
        gen1.append(rec)

    ref = [r.score for r in gen0]
    cand = [r.score for r in gen1]
    promo = G.decide_promotion(cand, ref)
    report = {
        "task": TASK, "step_budget_per_toddler": BUDGET, "registry_root": a.root, "seconds": round(time.time() - t0, 1),
        "reference_generation": "gen-0 (frozen)",
        "gen-0": {r.toddler_id: round(r.score, 3) for r in gen0},
        "gen-1": {r.toddler_id: round(r.score, 3) for r in gen1},
        "gen-1_parent": f"gen-0/{best.toddler_id}",
        "promotion_gen1_over_gen0": {"promote": promo.promote, "p": round(promo.p_value, 4),
                                     "prob_improvement": round(promo.prob_improvement, 3),
                                     "iqm_gen1": round(promo.candidate_iqm, 3), "iqm_gen0": round(promo.reference_iqm, 3)},
        "IQ_quotient": {"gen-0": round(quotients.to_quotient(float(np.mean(ref)), ref), 1),
                        "gen-1": round(quotients.to_quotient(float(np.mean(cand)), ref), 1)},
        "hardware": hw, **G.BUSINESS,
    }
    out = Path(__file__).resolve().parents[1] / "docs" / "learn" / "generations_report.json"
    out.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
