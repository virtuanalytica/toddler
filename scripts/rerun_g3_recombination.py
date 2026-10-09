"""Pre-register and evaluate a G3 recombination without changing the official lineage.

prepare freezes per-child routes from PUBLIC task scores and commits a fresh secret seed set.
run measures every frozen source expert on those seeds, compares the recombination with G2
and a random route through the same bank, and saves reviewable composite weights separately.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import torch

from toddler import resources
from toddler.learn import generations as G
from toddler.learn import multitask as M
from toddler.learn import scoring, secret_seeds as SS, tasks as T
from toddler.learn.routing import TaskExpertRouter

TASKS = ("cartpole", "acrobot", "empty5", "doorkey5", "doorkey8", "unlock",
         "unlockpickup", "keycorridor3", "lavacross9")
GENERATIONS = ("G2", "G2-scratch", "G3-sp", "G3-trunk", "G3-scratch")
PARENTS = tuple(f"t{2000 + i}" for i in range(1, 6))
MARGIN = 0.05
CONTROL_SEED = 734293
RULE = ("On fresh secret seeds, candidate must pass decide_promotion versus frozen G2 and an "
        "equal-expert-bank random route; all original G1 tasks stay routed to G2. Candidate "
        "aggregate IQM must exceed every single source generation. Five matched children per arm; "
        "no official promotion before review.")


def source_id(generation: str, parent: str) -> str:
    return parent if generation.startswith("G2") else f"t{int(parent[1:]) + 1000}"


def public_sources(reg: G.Registry) -> dict[str, dict[str, dict]]:
    rows = {}
    for generation in GENERATIONS:
        rows[generation] = {}
        records = {r.toddler_id: r for r in reg.generation(generation)}
        for parent in PARENTS:
            rec = records[source_id(generation, parent)]
            if set(rec.config["per_task"]) != set(TASKS):
                raise ValueError(f"{generation}/{rec.toddler_id} lacks a public task score")
            rows[generation][parent] = {"id": rec.toddler_id, "sha256": rec.weights_sha256,
                                         "public": {t: float(rec.config["per_task"][t]) for t in TASKS}}
    return rows


def routes(rows: dict) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, str]]]:
    rng = random.Random(CONTROL_SEED)
    candidate, control = {}, {}
    for parent in PARENTS:
        candidate[parent] = {t: "G2" for t in TASKS}
        control[parent] = {t: "G2" for t in TASKS}
        for task in TASKS[4:]:
            baseline = rows["G2"][parent]["public"][task]
            best = max(GENERATIONS, key=lambda g: rows[g][parent]["public"][task])
            if rows[best][parent]["public"][task] >= baseline + MARGIN:
                candidate[parent][task] = best
            control[parent][task] = rng.choice(GENERATIONS)
    return candidate, control


def prepare(root: Path, protocol_path: Path) -> None:
    if protocol_path.exists():
        raise FileExistsError(protocol_path)
    reg = G.Registry(root)
    rows = public_sources(reg)
    candidate, control = routes(rows)
    commitment = SS.new_set(30, date.today() + timedelta(days=7))
    protocol = {"title": "G3 recombination from frozen public-score experts",
                "tasks": TASKS, "parents": PARENTS, "source_generations": GENERATIONS,
                "source_weights": {g: {p: rows[g][p]["sha256"] for p in PARENTS} for g in GENERATIONS},
                "selection": "per-child public task mean; switch only when best exceeds G2 by 0.05",
                "control_seed": CONTROL_SEED, "margin": MARGIN,
                "candidate_routes": candidate, "random_control_routes": control,
                "promotion_rule": RULE, "secret_seed_set_id": commitment.set_id,
                "secret_seed_commitment": asdict(commitment)}
    protocol_path.parent.mkdir(parents=True, exist_ok=True)
    protocol_path.write_text(json.dumps(protocol, indent=2) + "\n")
    print(json.dumps({"protocol": str(protocol_path), "secret_seed_commitment": asdict(commitment),
                      "candidate_routes": candidate, "random_control_routes": control}, indent=2))


def load_protocol(path: Path, root: Path) -> dict:
    protocol = json.loads(path.read_text())
    if (tuple(protocol["tasks"]) != TASKS or tuple(protocol["parents"]) != PARENTS
            or tuple(protocol["source_generations"]) != GENERATIONS
            or protocol["promotion_rule"] != RULE or protocol["margin"] != MARGIN
            or protocol["control_seed"] != CONTROL_SEED):
        raise ValueError("protocol constants changed")
    rows = public_sources(G.Registry(root))
    weights = {g: {p: rows[g][p]["sha256"] for p in PARENTS} for g in GENERATIONS}
    candidate, control = routes(rows)
    if (protocol["source_weights"] != weights or protocol["candidate_routes"] != candidate
            or protocol["random_control_routes"] != control):
        raise ValueError("weights or routes changed since pre-registration")
    commitment, _, _ = SS.load_private(protocol["secret_seed_set_id"])
    if asdict(commitment) != protocol["secret_seed_commitment"]:
        raise ValueError("secret seed commitment changed")
    return protocol


def evaluate_source(job: tuple[str, str, str, tuple[int, ...], dict[str, float]]) -> dict:
    generation, parent, root, seeds, anchors = job
    torch.set_num_threads(2)
    toddler_id = source_id(generation, parent)
    net, rec = G.Registry(Path(root)).load(generation, toddler_id)
    scores = M.evaluate_multitask(net, list(TASKS), anchors, mode="sample", seeds=seeds)
    return {"generation": generation, "parent": parent, "sha256": rec.weights_sha256,
            "secret": {t: float(np.mean(scores[t])) for t in TASKS}}


def matrix(rows: dict, route: dict[str, dict[str, str]]) -> np.ndarray:
    return np.asarray([[rows[(route[p][t], p)]["secret"][t] for t in TASKS] for p in PARENTS])


def summary(scores: np.ndarray) -> dict:
    return {"aggregate_iqm": scoring.aggregate_iqm(scores),
            "per_child_mean": scores.mean(axis=1).tolist(),
            "per_task_iqm": {t: scoring.iqm(scores[:, i]) for i, t in enumerate(TASKS)}}


def run(root: Path, protocol_path: Path, out: Path) -> None:
    if (out / "report.json").exists():
        raise FileExistsError(out / "report.json")
    protocol_raw = protocol_path.read_bytes()
    protocol = load_protocol(protocol_path, root)
    _, seeds, _ = SS.load_private(protocol["secret_seed_set_id"])
    anchors = {t: T.random_anchor(t, seeds) for t in TASKS}
    host = resources.probe()
    workers = max(1, min(5, resources.cpu_threads(host.cores, host.load_1m) // 2))
    jobs = [(g, p, str(root), seeds, anchors) for g in GENERATIONS for p in PARENTS]
    print(f"Evaluating {len(jobs)} frozen G2/G3 experts, {workers} workers x 2 CPU threads", flush=True)
    start = time.time()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(evaluate_source, jobs))
    rows = {(r["generation"], r["parent"]): r for r in results}
    for g in GENERATIONS:
        for p in PARENTS:
            if rows[(g, p)]["sha256"] != protocol["source_weights"][g][p]:
                raise ValueError("a frozen source weight changed during evaluation")
    route = protocol["candidate_routes"]
    controls = protocol["random_control_routes"]
    scores = {"candidate": matrix(rows, route), "random_control": matrix(rows, controls)}
    scores.update({g: matrix(rows, {p: dict.fromkeys(TASKS, g) for p in PARENTS}) for g in GENERATIONS})
    versus_g2 = G.decide_promotion(scores["candidate"].mean(axis=1), scores["G2"].mean(axis=1))
    versus_control = G.decide_promotion(scores["candidate"].mean(axis=1), scores["random_control"].mean(axis=1))
    all_sources_beaten = all(summary(scores["candidate"])["aggregate_iqm"] > summary(scores[g])["aggregate_iqm"]
                             for g in GENERATIONS)
    preserves_g1 = all(route[p][t] == "G2" for p in PARENTS for t in TASKS[:4])
    eligible = bool(versus_g2.promote and versus_control.promote and all_sources_beaten and preserves_g1)
    out.mkdir(parents=True, exist_ok=True)
    trial = G.Registry(out / "registry")
    children = []
    official = G.Registry(root)
    for i, parent in enumerate(PARENTS, start=1):
        nets = {g: official.load(g, source_id(g, parent))[0] for g in GENERATIONS}
        for arm, arm_route in (("candidate", route[parent]), ("random_control", controls[parent])):
            net = TaskExpertRouter(nets, arm_route)
            per_task = {t: rows[(arm_route[t], parent)]["secret"][t] for t in TASKS}
            rec = G.ToddlerRecord("G3-recombined" if arm == "candidate" else "G3-random-route",
                                  f"t{5000 + i}", "task-router:" + "+".join(TASKS),
                                  {"route": arm_route, "protocol_sha256": hashlib.sha256(protocol_raw).hexdigest(),
                                   "secret_set_id": protocol["secret_seed_set_id"]}, 0,
                                  [f"{g}/{source_id(g, parent)}" for g in GENERATIONS],
                                  [per_task[t] for t in TASKS], G.hardware_fingerprint(), software=G.software_versions())
            trial.save(net, rec)
            children.append({"arm": arm, "id": rec.toddler_id, "weights_sha256": rec.weights_sha256})
    report = {"protocol_sha256": hashlib.sha256(protocol_raw).hexdigest(),
              "secret_seed_commitment": protocol["secret_seed_commitment"],
              "selection_rule": RULE, "routes": route,
              "secret": {name: summary(value) for name, value in scores.items()},
              "promotion_vs_G2": asdict(versus_g2), "promotion_vs_random_control": asdict(versus_control),
              "beats_all_source_IQMs": all_sources_beaten, "preserves_G1_tasks": preserves_g1,
              "eligible_for_lineage_review": eligible,
              "verdict": "eligible successor for review" if eligible else "no successor; G2 remains official",
              "children": children, "duration_s": round(time.time() - start, 1)}
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("promotion_vs_G2", "promotion_vs_random_control",
                                        "beats_all_source_IQMs", "eligible_for_lineage_review", "verdict",
                                        "duration_s")}, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("prepare", "run"))
    parser.add_argument("protocol", type=Path)
    parser.add_argument("--root", type=Path, default=Path("/media/knight2/EDS2/toddler-generations"))
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.root, args.protocol)
    else:
        if args.out is None:
            parser.error("run requires --out")
        run(args.root, args.protocol, args.out)


if __name__ == "__main__":
    main()
