"""Build gen-0 (self-reinforcement from scratch) and gen-1 (taught by the best gen-0 toddler via
behaviour cloning, same step budget), register both, decide promotion and compute IQ quotients
against the frozen gen-0 reference.

Run: PYTHONPATH=. python3 scripts/build_generations.py [--root DIR] [--report-only] [--refreeze]
--root defaults to $TODDLER_GENERATIONS_ROOT, else ~/.local/share/toddler/generations.
--report-only rebuilds the report from the registry without training (weights are sha256-checked).
--refreeze replaces an existing gen-0 reference frozen under another fingerprint (this invalidates
every quotient published against it); without it such a run is refused.
Weights live outside git: on a fresh clone, run this script first (CPU runs are reproducible from
seed and step budget), then scripts/benchmark_generations.py.
Weights go to the registry root (outside git); the report goes to docs/learn/generations_report.json.
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from toddler import quotients, resources
from toddler.learn import generations as G
from toddler.learn import peer, ppo, scoring
from toddler.learn import tasks as T

TASK, BUDGET, CLONE = "cartpole", 150_000, 10_000
# Method version of gen-0/gen-1: trained before return scaling became the PPO default (#18).
# Pinned so a rebuild reproduces the registry; a scaled generation is a new method version.
SCALE_REWARDS = False


def evaluate(net, anchor) -> list[float]:
    return [float(T.normalise(TASK, r, anchor)) for r in scoring.evaluate(net, TASK)]


def default_root() -> str:
    return os.environ.get("TODDLER_GENERATIONS_ROOT") or str(Path.home() / ".local/share/toddler/generations")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=default_root())
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--refreeze", action="store_true")
    a = ap.parse_args()
    host = resources.probe()
    torch.set_num_threads(min(8, resources.cpu_threads(host.cores, host.load_1m)))
    reg, anchor, hw, t0 = G.Registry(Path(a.root)), T.random_anchor(TASK), G.hardware_fingerprint(), time.time()
    fp = quotients.fingerprint([TASK], T.EVAL_SEEDS, [anchor], [T.TASKS[TASK].solved])
    sw = G.software_versions()
    if a.report_only:
        gen0, gen1 = reg.generation("gen-0"), reg.generation("gen-1")
        if not gen0 or not gen1 or not gen1[0].parents:
            raise SystemExit("--report-only needs a registry with gen-0 and gen-1 (with parents); train first")
        for r in gen0 + gen1:
            reg.load(r.generation, r.toddler_id)          # integrity check only
        reg.freeze_reference("gen-0", fp, refreeze=a.refreeze)   # no-op when already frozen under fp
        write_report(reg, fp, gen0, gen1, gen1[0].parents[0], None, hw)
        return

    gen0 = []
    for s in (101, 102, 103, 104, 105):
        tid = f"t{s}"
        net, log = ppo.train(TASK, ppo.PPOConfig(total_steps=BUDGET, seed=s, scale_rewards=SCALE_REWARDS),
                             checkpoint=reg.checkpoint_fn("gen-0", tid), checkpoint_every=25)
        rec = G.ToddlerRecord("gen-0", tid, TASK, {"seed": s, "method": "ppo", "scale_rewards": SCALE_REWARDS}, log.steps, [], evaluate(net, anchor), hw,
                              device_switches=log.device_switches, software=sw)
        reg.save(net, rec)
        gen0.append(rec)
    best = max(gen0, key=lambda r: r.score)          # per-run score, the quantity the IQM aggregates
    teacher, _ = reg.load("gen-0", best.toddler_id)

    reg.freeze_reference("gen-0", fp, refreeze=a.refreeze)

    gen1 = []
    for s in (201, 202, 203, 204, 205):
        tid = f"t{s}"
        warm = peer.behaviour_clone(teacher, TASK, CLONE, seed=s)
        net, log = ppo.train(TASK, ppo.PPOConfig(total_steps=BUDGET - CLONE, seed=s, scale_rewards=SCALE_REWARDS), net=warm,
                             checkpoint=reg.checkpoint_fn("gen-1", tid), checkpoint_every=25)
        rec = G.ToddlerRecord("gen-1", tid, TASK, {"seed": s, "method": "behaviour_clone+ppo", "clone_steps": CLONE, "scale_rewards": SCALE_REWARDS},
                              CLONE + log.steps, [f"gen-0/{best.toddler_id}"], evaluate(net, anchor), hw,
                              device_switches=log.device_switches, software=sw)
        reg.save(net, rec)
        gen1.append(rec)

    write_report(reg, fp, gen0, gen1, f"gen-0/{best.toddler_id}", round(time.time() - t0, 1), hw)


def write_report(reg, fp, gen0, gen1, parent, seconds, hw) -> None:
    ref = reg.reference("gen-0", fp)                 # refuses a changed task set, seed list or anchor
    cand = [r.score for r in gen1]
    promo = G.decide_promotion(cand, ref)
    q1 = quotients.iq_quotient_ci(np.asarray(cand)[:, None], ref)
    report = {
        "task": TASK, "step_budget_per_toddler": BUDGET,
        "training_seconds": seconds if seconds is not None else "not re-measured (report rebuilt from registry)",
        "method_version": {"scale_rewards": SCALE_REWARDS},
        "reference_generation": "gen-0 (frozen)", "reference_fingerprint": fp.digest(),
        "gen-0": {r.toddler_id: round(r.score, 3) for r in gen0},
        "gen-1": {r.toddler_id: round(r.score, 3) for r in gen1},
        "gen-1_parent": parent,
        "promotion_gen1_over_gen0": {"promote": promo.promote, "p": round(promo.p_value, 4),
                                     "prob_improvement": round(promo.prob_improvement, 3),
                                     "iqm_gen1": round(promo.candidate_iqm, 3), "iqm_gen0": round(promo.reference_iqm, 3)},
        "IQ_raw": {"gen-0": round(quotients.iq_raw(np.asarray(ref)[:, None]), 3),
                   "gen-1": round(quotients.iq_raw(np.asarray(cand)[:, None]), 3)},
        "IQ_quotient": {"gen-0": round(quotients.to_quotient(quotients.iq_raw(np.asarray(ref)[:, None]), ref), 1),
                        "gen-1": round(q1[0], 1)},
        "IQ_note": "the reference is centred on its own IQM, so gen-0 scores 100 against itself",
        "IQ_quotient_gen-1_95ci": [round(q1[1], 1), round(q1[2], 1)],
        "EQ_FQ": "not measured for these RL generations (no judged tasks or reflex scenarios yet)",
        "hardware": hw, **G.BUSINESS,
    }
    out = Path(__file__).resolve().parents[1] / "docs" / "learn" / "generations_report.json"
    out.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
