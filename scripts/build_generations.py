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

--g1 builds Generation 1 (G1): 5 toddlers, each ONE multi-task network (shared trunk, per-task
adapters and heads; toddler/learn/multitask.py) trained on cartpole, acrobot, empty5 and doorkey5,
150k environment steps per task in interleaved blocks of 15k, return scaling on, evaluated with
seeded sampling (mode "sample"). G1 is frozen as the quotient reference; IQ = aggregate IQM over
the four tasks. Report: docs/learn/generation_g1_report.json.
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
    ap.add_argument("--g1", action="store_true", help="build Generation 1 (multi-task)")
    a = ap.parse_args()
    if a.g1:
        build_g1(a.root, a.refreeze)
        return
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


G1_TASKS = ["cartpole", "acrobot", "empty5", "doorkey5"]
G1_SEEDS = (1001, 1002, 1003, 1004, 1005)
G1_STEPS_PER_TASK, G1_BLOCK, G1_EVAL_MODE = 150_000, 15_000, "sample"


def _train_g1_toddler(seed: int, threads: int) -> dict:
    """Worker: train one G1 toddler and return weights + per-task held-out scores."""
    torch.set_num_threads(threads)
    from toddler.learn import multitask as M

    t0 = time.time()
    net, log = M.train_multitask(G1_TASKS, G1_STEPS_PER_TASK, G1_BLOCK, seed)
    anchors = {t: T.random_anchor(t) for t in G1_TASKS}
    scores = M.evaluate_multitask(net, G1_TASKS, anchors, mode=G1_EVAL_MODE)
    return {"seed": seed, "state": {k: v.cpu() for k, v in net.state_dict().items()}, "spec": net.spec(),
            "steps": log.steps_per_task, "scores": scores, "seconds": round(time.time() - t0, 1)}


def build_g1(root: str, refreeze: bool) -> None:
    from concurrent.futures import ProcessPoolExecutor

    from toddler import business
    from toddler.learn import multitask as M

    host = resources.probe()
    threads = 4
    workers = max(1, min(len(G1_SEEDS), resources.cpu_threads(host.cores, host.load_1m) // threads))
    reg, hw, sw, t0 = G.Registry(Path(root)), G.hardware_fingerprint(), G.software_versions(), time.time()
    anchors = {t: T.random_anchor(t) for t in G1_TASKS}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        results = sorted(ex.map(_train_g1_toddler, G1_SEEDS, [threads] * len(G1_SEEDS)), key=lambda r: r["seed"])
    rows = []
    for r in results:
        net = M.MultiTaskNet({k: tuple(v) for k, v in r["spec"]["task_dims"].items()}, r["spec"]["hidden"])
        net.load_state_dict(r["state"])
        per_task = {t: float(np.mean(r["scores"][t])) for t in G1_TASKS}
        rec = G.ToddlerRecord("G1", f"t{r['seed']}", "multitask:" + "+".join(G1_TASKS),
                              {"seed": r["seed"], "method": "ppo-multitask", "scale_rewards": True,
                               "eval_mode": G1_EVAL_MODE, "block_steps": G1_BLOCK, "per_task": per_task},
                              sum(r["steps"].values()), [], [per_task[t] for t in G1_TASKS], hw, software=sw)
        reg.save(net, rec)
        rows.append([per_task[t] for t in G1_TASKS])
    fp = quotients.fingerprint(G1_TASKS, T.EVAL_SEEDS, [anchors[t] for t in G1_TASKS],
                               [T.TASKS[t].solved for t in G1_TASKS], eval_mode=G1_EVAL_MODE)
    reg.freeze_reference("G1", fp, refreeze=refreeze)
    scores = np.asarray(rows)                                   # (toddlers, tasks)
    reference = reg.reference("G1", fp)
    point, lo, hi = quotients.iq_quotient_ci(scores, reference)
    report = {
        "generation": "G1", "tasks": G1_TASKS, "toddlers": len(results), "steps_per_task": G1_STEPS_PER_TASK,
        "block_steps": G1_BLOCK, "eval_mode": G1_EVAL_MODE, "method_version": {"scale_rewards": True, "network": "multitask"},
        "measured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "training_seconds": round(time.time() - t0, 1),
        "reference_fingerprint": fp.digest(),
        "per_task_iqm": {t: round(scoring.iqm(scores[:, j]), 4) for j, t in enumerate(G1_TASKS)},
        "per_toddler": {f"t{r['seed']}": {t: round(float(np.mean(r["scores"][t])), 4) for t in G1_TASKS} for r in results},
        "IQ_raw": round(quotients.iq_raw(scores), 4),
        "IQ_quotient": round(point, 1), "IQ_quotient_95ci": [round(lo, 1), round(hi, 1)],
        "IQ_note": "G1 is the frozen reference; it scores 100 against itself by construction",
        "EQ_FQ": "not measured (no judged tasks or reflex scenarios yet)",
        "hardware": hw, "software": sw, **business.FIELDS,
    }
    out = Path(__file__).resolve().parents[1] / "docs" / "learn" / "generation_g1_report.json"
    out.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({k: report[k] for k in ("per_task_iqm", "IQ_raw", "IQ_quotient", "IQ_quotient_95ci", "training_seconds")}, indent=1))


if __name__ == "__main__":
    main()
