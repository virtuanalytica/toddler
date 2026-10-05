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
    ap.add_argument("--report", metavar="GENERATION", help="write the development report PDF of a surviving generation")
    ap.add_argument("--allow-dirty", action="store_true", help="throw-away run from uncommitted code")
    ap.add_argument("--g3", action="store_true", help="build Generation 3 (inheritance variants, selection across arms)")
    ap.add_argument("--g2", action="store_true", help="build Generation 2 (+5 harder tasks, inherit vs scratch, secret seeds)")
    a = ap.parse_args()
    if a.g3:
        build_g3(a.root, a.refreeze, a.allow_dirty)
        return
    if a.report:
        from toddler.learn import devreport
        from toddler.learn import lineage as L

        backfill_g1_lineage(G.Registry(Path(a.root)), L.Ledger(Path(a.root)))
        docs = Path(__file__).resolve().parents[1] / "docs" / "learn"
        pdf = devreport.write_pdf(Path(a.root), a.report, docs / "reports" / f"{a.report}_ontwikkelverslag.pdf",
                                  docs / f"generation_{a.report.lower()}_report.json")
        print(pdf)
        return
    if a.g2:
        build_g2(a.root, a.refreeze, a.allow_dirty)
        return
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


G2_NEW = ["doorkey8", "unlock", "unlockpickup", "keycorridor3", "lavacross9"]
G2_TASKS = G1_TASKS + G2_NEW
G2_SEEDS = (2001, 2002, 2003, 2004, 2005)               # parent of t200x is G1/t100x
G2_STEPS = {**{t: 150_000 for t in G1_TASKS}, **{t: 1_000_000 for t in G2_NEW}}
G2_BLOCK, G2_SECRET_N, G2_SECRET_DAYS = 25_000, 30, 7


G2_FORGETTING_MIN = 0.25    # P(G2 > G1) on the G1 tasks below this = G2 forgot what G1 could do


def _seed_digest(seeds) -> str:
    import hashlib

    return hashlib.sha256(",".join(str(int(x)) for x in seeds).encode()).hexdigest()


def _task_data(tasks: list[str]) -> dict:
    import gymnasium
    import minigrid

    return {"tasks": {t: T.TASKS[t].env_id for t in tasks}, "solve_thresholds": {t: T.TASKS[t].solved for t in tasks},
            "env_versions": {"gymnasium": gymnasium.__version__, "minigrid": minigrid.__version__},
            "public_eval_seeds_sha256": _seed_digest(T.EVAL_SEEDS), "training_seed_band": [T.TRAIN_SEED_LOW, 2**31]}


def backfill_g1_lineage(reg: "G.Registry", led) -> None:
    """G1 was trained before the evolution ledger existed: record its births and its verdict once,
    marked as backfilled, from the registry metadata and the G1 report."""
    if led.members("G1"):
        return
    report = json.loads((Path(__file__).resolve().parents[1] / "docs/learn/generation_g1_report.json").read_text())
    for r in reg.generation("G1"):
        led.born(f"G1/{r.toddler_id}", r.weights_sha256, role="population", backfilled=True,
                 code={"commit": "", "note": "not recorded at training time; built by scripts/build_generations.py --g1 (PR #28)"},
                 data=_task_data(G1_TASKS),
                 budget={"steps_per_task": G1_STEPS_PER_TASK, "block_steps": G1_BLOCK, "seed": r.config.get("seed"),
                         "eval_mode": G1_EVAL_MODE},
                 hardware=r.hardware, software=r.software)
    led.selected("G1", "survived", {"basis": "first multi-task generation, frozen as the IQ reference",
                                    "IQ_quotient": report["IQ_quotient"], "per_task_iqm": report["per_task_iqm"],
                                    "reference_fingerprint": report["reference_fingerprint"], "backfilled": True})


def _train_g2_toddler(job: dict) -> dict:
    """Worker: one G2 toddler (inheriting from its G1 parent, or from scratch for the control arm),
    scored on the public EVAL_SEEDS and on the secret seed set (scores only; seeds never returned)."""
    torch.set_num_threads(job["threads"])
    from toddler.learn import multitask as M

    t0, parent = time.time(), None
    if job["parent"]:
        spec = job["parent"]["spec"]
        parent = M.MultiTaskNet({k: tuple(v) for k, v in spec["task_dims"].items()}, spec["hidden"])
        parent.load_state_dict(job["parent"]["state"])
    tasks, steps = job.get("tasks", G2_TASKS), job.get("steps", G2_STEPS)
    net, log = M.train_multitask(tasks, 0, job.get("block", G2_BLOCK), job["seed"], parent=parent, steps=steps,
                                 inherit_mode=job.get("inherit_mode", "full"))
    pub = M.evaluate_multitask(net, tasks, job["anchors_pub"], mode=G1_EVAL_MODE)
    sec = M.evaluate_multitask(net, tasks, job["anchors_sec"], mode=G1_EVAL_MODE, seeds=job["secret_seeds"])
    return {"arm": job["arm"], "seed": job["seed"], "state": {k: v.cpu() for k, v in net.state_dict().items()},
            "spec": net.spec(), "steps": log.steps_per_task, "pub": pub, "sec": sec,
            "seconds": round(time.time() - t0, 1)}


def build_g2(root: str, refreeze: bool, allow_dirty: bool = False) -> None:
    """Generation 2: G1 tasks + five harder procedural tasks. Arm "G2" inherits its G1 parent's
    trunk and task modules; arm "G2-scratch" is the same recipe from random init (control for the
    inheritance effect). Evaluated on the public seeds and on a fresh secret seed set (commit-reveal,
    toddler.learn.secret_seeds); only the commitment is reported until the window closes."""
    from concurrent.futures import ProcessPoolExecutor
    from dataclasses import asdict
    from datetime import date, timedelta

    from toddler import business
    from toddler.learn import lineage as L
    from toddler.learn import multitask as M
    from toddler.learn import secret_seeds as SS

    code = L.code_version()
    if code["dirty"] and not allow_dirty:
        raise SystemExit("uncommitted changes in the toddler repo: a generation must be born from a commit "
                         "(commit first, or pass --allow-dirty for a throw-away run)")
    led = L.Ledger(Path(root))
    backfill_g1_lineage(G.Registry(Path(root)), led)
    host, threads = resources.probe(), 4
    jobs_n = 2 * len(G2_SEEDS)
    workers = max(1, min(jobs_n, resources.cpu_threads(host.cores, host.load_1m) // threads))
    reg, hw, sw, t0 = G.Registry(Path(root)), G.hardware_fingerprint(), G.software_versions(), time.time()
    commitment = SS.new_set(G2_SECRET_N, date.today() + timedelta(days=G2_SECRET_DAYS))
    _, secret, _ = SS.load_private(commitment.set_id)
    anchors_pub = {t: T.random_anchor(t) for t in G2_TASKS}
    anchors_sec = {t: T.random_anchor(t, secret) for t in G2_TASKS}
    jobs = []
    for arm in ("G2", "G2-scratch"):
        for s in G2_SEEDS:
            parent = None
            if arm == "G2":
                pnet, prec = reg.load("G1", f"t{s - 1000}")
                parent = {"spec": pnet.spec(), "state": pnet.state_dict(), "ref": f"G1/t{s - 1000}",
                          "sha": prec.weights_sha256, "tasks": list(pnet.task_dims)}
            jobs.append({"arm": arm, "seed": s, "parent": parent, "threads": threads, "anchors_pub": anchors_pub,
                         "anchors_sec": anchors_sec, "secret_seeds": secret})
    print(f"G2: {len(jobs)} toddlers, {workers} workers x {threads} threads, secret set {commitment.set_id}", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(_train_g2_toddler, jobs))
    parents_of = {(j["arm"], j["seed"]): j["parent"] for j in jobs}
    data = {**_task_data(G2_TASKS), "hidden_seed_set": asdict(commitment)}
    arms: dict[str, dict] = {"G2": {"pub": [], "sec": [], "ids": []}, "G2-scratch": {"pub": [], "sec": [], "ids": []}}
    for r in sorted(results, key=lambda r: (r["arm"], r["seed"])):
        net = M.MultiTaskNet({k: tuple(v) for k, v in r["spec"]["task_dims"].items()}, r["spec"]["hidden"])
        net.load_state_dict(r["state"])
        pub = {t: float(np.mean(r["pub"][t])) for t in G2_TASKS}
        sec = {t: float(np.mean(r["sec"][t])) for t in G2_TASKS}
        parents = [f"G1/t{r['seed'] - 1000}"] if r["arm"] == "G2" else []
        rec = G.ToddlerRecord(r["arm"], f"t{r['seed']}", "multitask:" + "+".join(G2_TASKS),
                              {"seed": r["seed"], "method": "ppo-multitask", "scale_rewards": True,
                               "eval_mode": G1_EVAL_MODE, "block_steps": G2_BLOCK, "steps": G2_STEPS,
                               "inherits": bool(parents), "per_task": pub, "per_task_secret": sec,
                               "secret_set": commitment.set_id},
                              sum(r["steps"].values()), parents, [pub[t] for t in G2_TASKS], hw, software=sw)
        reg.save(net, rec)
        par = parents_of[(r["arm"], r["seed"])]
        inherits = [L.Inheritance(par["ref"], par["sha"], ("trunk", *(f"task:{t}" for t in par["tasks"])))] if par else []
        led.born(f"{r['arm']}/t{r['seed']}", rec.weights_sha256, role="population" if r["arm"] == "G2" else "control",
                 inherits=inherits, code=code, data=data,
                 budget={"steps": G2_STEPS, "block_steps": G2_BLOCK, "seed": r["seed"], "eval_mode": G1_EVAL_MODE,
                         "threads": threads, "seconds": r["seconds"]},
                 hardware=hw, software=sw)
        a = arms[r["arm"]]
        a["pub"].append([pub[t] for t in G2_TASKS]); a["sec"].append([sec[t] for t in G2_TASKS]); a["ids"].append(f"t{r['seed']}")
    for a in arms.values():
        a["pub"], a["sec"] = np.asarray(a["pub"]), np.asarray(a["sec"])
    g1_fp = quotients.fingerprint(G1_TASKS, T.EVAL_SEEDS, [anchors_pub[t] for t in G1_TASKS],
                                  [T.TASKS[t].solved for t in G1_TASKS], eval_mode=G1_EVAL_MODE)
    g1_ref = reg.reference("G1", g1_fp)          # raises if the G1 reference was frozen under other anchors
    g2_fp = quotients.fingerprint(G2_TASKS, T.EVAL_SEEDS, [anchors_pub[t] for t in G2_TASKS],
                                  [T.TASKS[t].solved for t in G2_TASKS], eval_mode=G1_EVAL_MODE)
    reg.freeze_reference("G2", g2_fp, refreeze=refreeze)
    n1, new = len(G1_TASKS), slice(len(G1_TASKS), None)

    def arm_summary(a: dict) -> dict:
        lo, hi = scoring.bootstrap_ci(a["pub"][:, new])
        slo, shi = scoring.bootstrap_ci(a["sec"][:, new])
        iq, iq_lo, iq_hi = quotients.iq_quotient_ci(a["pub"][:, :n1], g1_ref)
        return {"per_task_iqm_public": {t: round(scoring.iqm(a["pub"][:, j]), 4) for j, t in enumerate(G2_TASKS)},
                "per_task_iqm_secret": {t: round(scoring.iqm(a["sec"][:, j]), 4) for j, t in enumerate(G2_TASKS)},
                "new_tasks_iqm_public": [round(scoring.aggregate_iqm(a["pub"][:, new]), 4), [round(lo, 4), round(hi, 4)]],
                "new_tasks_iqm_secret": [round(scoring.aggregate_iqm(a["sec"][:, new]), 4), [round(slo, 4), round(shi, 4)]],
                "IQ_quotient_vs_G1_on_G1_tasks": [round(iq, 1), [round(iq_lo, 1), round(iq_hi, 1)]],
                "toddlers": a["ids"]}

    report = {
        "generation": "G2", "tasks": G2_TASKS, "new_tasks": G2_NEW, "steps": G2_STEPS, "block_steps": G2_BLOCK,
        "eval_mode": G1_EVAL_MODE, "method_version": {"scale_rewards": True, "network": "multitask", "inherit": "G1"},
        "measured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "training_seconds": round(time.time() - t0, 1),
        "reference_fingerprint": g2_fp.digest(), "g1_reference_fingerprint": g1_fp.digest(),
        "secret_seed_commitment": asdict(commitment),
        "arms": {k: arm_summary(v) for k, v in arms.items()},
        "inheritance_effect": {
            "P(G2 > G2-scratch) public, new tasks": round(scoring.prob_improvement(arms["G2"]["pub"][:, new], arms["G2-scratch"]["pub"][:, new]), 3),
            "P(G2 > G2-scratch) secret, new tasks": round(scoring.prob_improvement(arms["G2"]["sec"][:, new], arms["G2-scratch"]["sec"][:, new]), 3),
            "P(G2 > G1) public, G1 tasks (forgetting check)": None,
        },
        "public_vs_secret_gap_new_tasks": {k: round(float(scoring.aggregate_iqm(v["pub"][:, new]) - scoring.aggregate_iqm(v["sec"][:, new])), 4)
                                           for k, v in arms.items()},
        "IQ_note": "IQ quotient on the four G1 tasks against the frozen G1 reference; G2 is the frozen reference for the nine-task battery",
        "EQ_FQ": "not measured (no judged tasks or reflex scenarios yet)",
        "hardware": hw, "software": sw, **business.FIELDS,
    }
    g1 = reg.generation("G1")
    if g1:
        g1_rows = np.asarray([[r.config["per_task"][t] for t in G1_TASKS] for r in g1])
        report["inheritance_effect"]["P(G2 > G1) public, G1 tasks (forgetting check)"] = round(
            scoring.prob_improvement(arms["G2"]["pub"][:, :n1], g1_rows), 3)
    # Selection (Darwin): G2 survives when it beats G1 on the nine-task battery (G1 has no capability
    # on the new tasks, scored as 0 = random, never invented) AND did not forget the G1 tasks.
    g1_battery = np.hstack([g1_rows, np.zeros((len(g1_rows), len(G2_NEW)))]) if g1 else None
    promo = G.decide_promotion(arms["G2"]["pub"].mean(axis=1).tolist(), g1_battery.mean(axis=1).tolist())
    forget = report["inheritance_effect"]["P(G2 > G1) public, G1 tasks (forgetting check)"]
    verdict = "survived" if promo.promote and forget >= G2_FORGETTING_MIN else "extinct"
    report["selection"] = {"verdict": verdict, "promotion": asdict(promo), "forgetting_P": forget,
                           "forgetting_min": G2_FORGETTING_MIN,
                           "rule": "decide_promotion(G2, G1 on the 9-task battery; G1 = 0 on new tasks) and "
                                   "P(G2 > G1 on G1 tasks) >= forgetting_min"}
    led.selected("G2", verdict, report["selection"])
    led.selected("G2-scratch", "control", {"basis": "same recipe from random init; control for the inheritance effect"})
    report["lineage_ledger"] = {"path": str(led.path), "verified": led.verify()[0], "code": code}
    out = Path(__file__).resolve().parents[1] / "docs" / "learn" / "generation_g2_report.json"
    out.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({k: report[k] for k in ("arms", "inheritance_effect", "public_vs_secret_gap_new_tasks", "training_seconds")}, indent=1))

G3_SEEDS = (3001, 3002, 3003, 3004, 3005)               # parent of t300x is G2/t200x
G3_ARMS = {"G3-sp": "shrink_perturb", "G3-trunk": "trunk_only", "G3-scratch": None}
G3_THREADS = 3


def build_g3(root: str, refreeze: bool, allow_dirty: bool = False) -> None:
    """Generation 3: same nine tasks and budget as G2, parents = surviving G2. G2 showed that copying
    the whole parent (mode "full") hampers the harder tasks, so G3 tests two plasticity-keeping ways
    to inherit against a scratch control, and selection runs ACROSS the arms (pre-registered here):

      eligible  arm beats G2 on the nine-task battery (decide_promotion on per-run means) AND did not
                forget the G1 tasks (P(arm > G1 on G1 tasks) >= G2_FORGETTING_MIN)
      survivor  the eligible arm with the highest aggregate IQM over the nine tasks; every other
                arm is "extinct" (the scratch arm too: if it wins, it is a new root without G2 lineage)
      none      no arm eligible -> all G3 arms extinct, G2 stays the line
    """
    from concurrent.futures import ProcessPoolExecutor
    from dataclasses import asdict
    from datetime import date, timedelta

    from toddler import business
    from toddler.learn import lineage as L
    from toddler.learn import multitask as M
    from toddler.learn import secret_seeds as SS

    code = L.code_version()
    if code["dirty"] and not allow_dirty:
        raise SystemExit("uncommitted changes in the toddler repo: a generation must be born from a commit")
    reg, led = G.Registry(Path(root)), L.Ledger(Path(root))
    v = led.verdict("G2")
    if not v or v["verdict"] != "survived":
        raise SystemExit("G3 needs a surviving G2 in the ledger")
    hw, sw, t0 = G.hardware_fingerprint(), G.software_versions(), time.time()
    commitment = SS.new_set(G2_SECRET_N, date.today() + timedelta(days=G2_SECRET_DAYS))
    _, secret, _ = SS.load_private(commitment.set_id)
    anchors_pub = {t: T.random_anchor(t) for t in G2_TASKS}
    anchors_sec = {t: T.random_anchor(t, secret) for t in G2_TASKS}
    jobs = []
    for arm, mode in G3_ARMS.items():
        for s in G3_SEEDS:
            parent = None
            if mode:
                pnet, prec = reg.load("G2", f"t{s - 1000}")
                parent = {"spec": pnet.spec(), "state": pnet.state_dict(), "ref": f"G2/t{s - 1000}",
                          "sha": prec.weights_sha256, "tasks": list(pnet.task_dims)}
            jobs.append({"arm": arm, "seed": s, "parent": parent, "inherit_mode": mode or "full", "threads": G3_THREADS,
                         "anchors_pub": anchors_pub, "anchors_sec": anchors_sec, "secret_seeds": secret})
    host = resources.probe()
    workers = max(1, min(len(jobs), resources.cpu_threads(host.cores, host.load_1m) // G3_THREADS))
    print(f"G3: {len(jobs)} toddlers, {workers} workers x {G3_THREADS} threads, secret set {commitment.set_id}", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(_train_g2_toddler, jobs))
    data = {**_task_data(G2_TASKS), "hidden_seed_set": asdict(commitment)}
    modules = {"shrink_perturb": lambda ts: ("trunk~0.4p+0.1fresh", *(f"task:{t}~0.4p+0.1fresh" for t in ts)),
               "trunk_only": lambda ts: ("trunk",)}
    arms = {a: {"pub": [], "sec": []} for a in G3_ARMS}
    by_key = {(j["arm"], j["seed"]): j for j in jobs}
    for r in sorted(results, key=lambda r: (r["arm"], r["seed"])):
        net = M.MultiTaskNet({k: tuple(v) for k, v in r["spec"]["task_dims"].items()}, r["spec"]["hidden"])
        net.load_state_dict(r["state"])
        job = by_key[(r["arm"], r["seed"])]
        pub = {t: float(np.mean(r["pub"][t])) for t in G2_TASKS}
        sec = {t: float(np.mean(r["sec"][t])) for t in G2_TASKS}
        par, mode = job["parent"], G3_ARMS[r["arm"]]
        rec = G.ToddlerRecord(r["arm"], f"t{r['seed']}", "multitask:" + "+".join(G2_TASKS),
                              {"seed": r["seed"], "method": "ppo-multitask", "inherit_mode": mode, "steps": G2_STEPS,
                               "block_steps": G2_BLOCK, "eval_mode": G1_EVAL_MODE, "per_task": pub,
                               "per_task_secret": sec, "secret_set": commitment.set_id},
                              sum(r["steps"].values()), [par["ref"]] if par else [], [pub[t] for t in G2_TASKS], hw,
                              software=sw)
        reg.save(net, rec)
        inherits = [L.Inheritance(par["ref"], par["sha"], modules[mode](par["tasks"]))] if par else []
        led.born(f"{r['arm']}/t{r['seed']}", rec.weights_sha256, role="population", inherits=inherits, code=code,
                 data=data, budget={"steps": G2_STEPS, "block_steps": G2_BLOCK, "seed": r["seed"], "inherit_mode": mode,
                                    "eval_mode": G1_EVAL_MODE, "threads": G3_THREADS, "seconds": r["seconds"]},
                 hardware=hw, software=sw)
        arms[r["arm"]]["pub"].append([pub[t] for t in G2_TASKS])
        arms[r["arm"]]["sec"].append([sec[t] for t in G2_TASKS])
    g2_rows = np.asarray([[r.config["per_task"][t] for t in G2_TASKS] for r in reg.generation("G2")])
    g1_rows = np.asarray([[r.config["per_task"][t] for t in G1_TASKS] for r in reg.generation("G1")])
    n1, new = len(G1_TASKS), slice(len(G1_TASKS), None)
    summary = {}
    for a, d in arms.items():
        pub_, sec_ = np.asarray(d["pub"]), np.asarray(d["sec"])
        promo = G.decide_promotion(pub_.mean(axis=1).tolist(), g2_rows.mean(axis=1).tolist())
        forget = scoring.prob_improvement(pub_[:, :n1], g1_rows)
        lo, hi = scoring.bootstrap_ci(pub_)
        summary[a] = {"inherit_mode": G3_ARMS[a], "battery_iqm_public": round(scoring.aggregate_iqm(pub_), 4),
                      "battery_iqm_public_95ci": [round(lo, 4), round(hi, 4)],
                      "battery_iqm_secret": round(scoring.aggregate_iqm(sec_), 4),
                      "new_tasks_iqm_public": round(scoring.aggregate_iqm(pub_[:, new]), 4),
                      "per_task_iqm_public": {t: round(scoring.iqm(pub_[:, j]), 4) for j, t in enumerate(G2_TASKS)},
                      "per_task_iqm_secret": {t: round(scoring.iqm(sec_[:, j]), 4) for j, t in enumerate(G2_TASKS)},
                      "promotion_vs_G2": asdict(promo), "P_vs_G1_on_G1_tasks": round(forget, 3),
                      "eligible": bool(promo.promote and forget >= G2_FORGETTING_MIN),
                      "P_vs_G2_per_task_mean": round(scoring.prob_improvement(pub_, g2_rows), 3)}
    eligible = [a for a in summary if summary[a]["eligible"]]
    winner = max(eligible, key=lambda a: summary[a]["battery_iqm_public"]) if eligible else None
    rule = ("eligible = decide_promotion(arm, G2 on the 9-task battery) and P(arm > G1 on G1 tasks) >= "
            f"{G2_FORGETTING_MIN}; survivor = eligible arm with the highest battery IQM; others extinct")
    for a in G3_ARMS:
        led.selected(a, "survived" if a == winner else "extinct",
                     {"rule": rule, "winner": winner, **{k: summary[a][k] for k in
                      ("battery_iqm_public", "battery_iqm_public_95ci", "promotion_vs_G2", "P_vs_G1_on_G1_tasks", "eligible")}})
    if winner:
        reg.freeze_reference(winner, quotients.fingerprint(G2_TASKS, T.EVAL_SEEDS, [anchors_pub[t] for t in G2_TASKS],
                                                           [T.TASKS[t].solved for t in G2_TASKS], eval_mode=G1_EVAL_MODE),
                             refreeze=refreeze)
    report = {"generation": "G3", "parents": "G2", "tasks": G2_TASKS, "steps": G2_STEPS, "block_steps": G2_BLOCK,
              "eval_mode": G1_EVAL_MODE, "arms": summary, "winner": winner, "selection_rule": rule,
              "secret_seed_commitment": asdict(commitment),
              "measured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "training_seconds": round(time.time() - t0, 1),
              "lineage_ledger": {"path": str(led.path), "verified": led.verify()[0], "code": code},
              "hardware": hw, "software": sw, **business.FIELDS}
    out = Path(__file__).resolve().parents[1] / "docs" / "learn" / "generation_g3_report.json"
    out.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({k: report[k] for k in ("arms", "winner", "training_seconds")}, indent=1))

if __name__ == "__main__":
    main()
