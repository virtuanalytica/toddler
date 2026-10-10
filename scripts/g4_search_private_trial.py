"""Freeze and privately evaluate a searched five-child G4 cohort against ancestors.

The training process supplies only a hash-bearing manifest. This evaluator
creates its own secret seed sets and never passes their values to Teacher.
Raw per-child trial files stay outside Git; stdout contains only commitments
and aggregate promotion decisions.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import tempfile
from dataclasses import asdict
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import torch

from toddler.learn import generations as G, scoring, secret_seeds as SS, tasks as T
from toddler.learn.ancestor_gate import Trial, assess_lineage_replication, surviving_ancestors
from toddler.learn.lineage import Inheritance, Ledger, code_version
from toddler.learn.multitask import TaskView
from toddler.learn.routing import TaskExpertRouter

ROOT = Path("/media/knight2/EDS2/toddler-generations")
TASKS = ("cartpole", "acrobot", "empty5", "doorkey5", "doorkey8", "unlock",
         "unlockpickup", "keycorridor3", "lavacross9")
CHILDREN = tuple(f"t600{i}" for i in range(1, 6))
PARENTS = tuple(f"G3-recombined/t500{i}" for i in range(1, 6))
SELECTION_RULE = "best dev mean; use only if dev and independent public check exceed G3 by >= 0.005"
CONFIRMATORY_ALPHA = 0.01


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_private(path: Path, data: dict) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def _load_child(path: Path):
    blob = torch.load(io.BytesIO(path.read_bytes()), weights_only=True, map_location="cpu")
    net = TaskExpertRouter.from_spec(blob["spec"])
    net.load_state_dict(blob["state"])
    return net.eval()


def validate_cohort(manifest_path: Path, root: Path) -> tuple[dict, dict[str, object]]:
    """Verify all five children and byte-identical inherited expert parameters."""
    manifest = json.loads(manifest_path.read_text())
    nav = manifest.get("navigation", {})
    children = nav.get("children", [])
    if manifest.get("status") != "completed_candidate" or nav.get("status") != "trained_cohort":
        raise ValueError("nightly cohort is incomplete")
    if [row.get("id") for row in children] != list(CHILDREN):
        raise ValueError("cohort needs the five ordered G3-matched children")
    registry = G.Registry(root)
    loaded = {}
    for index, row in enumerate(children, start=1):
        expected_parent = PARENTS[index - 1]
        if row["g3_parent"] != expected_parent or row["g2_source"] != f"G2/t200{index}":
            raise ValueError(f"{row['id']} has a mismatched parent")
        path = Path(row["artifact"]).resolve()
        if path.parent != (manifest_path.parent / "navigation").resolve() or not path.is_file():
            raise ValueError("candidate artifact left its manifest directory")
        if digest(path) != row["sha256"]:
            raise ValueError(f"{row['id']} candidate hash differs")
        parent, parent_rec = registry.load("G3-recombined", f"t500{index}")
        _, g2_rec = registry.load("G2", f"t200{index}")
        if (parent_rec.weights_sha256 != row["g3_parent_sha256"]
                or g2_rec.weights_sha256 != row["g2_source_sha256"]):
            raise ValueError("ancestor weights changed")
        child = _load_child(path)
        if row["route"] != child.route:
            raise ValueError("manifest route differs from candidate weights")
        if row.get("selection_rule") != SELECTION_RULE or not isinstance(row.get("selected_specialist"), bool):
            raise ValueError("cohort used a different route selection rule")
        search = row.get("search", {})
        if (search.get("selection_rule") != SELECTION_RULE or search.get("selected_specialist") != row["selected_specialist"]
                or not isinstance(search.get("profiles"), list) or not search.get("winner")):
            raise ValueError("missing or mismatched architecture search provenance")
        winners = [profile for profile in search["profiles"] if profile.get("profile") == search["winner"]]
        if len(winners) != 1 or any("development_mean" not in profile for profile in search["profiles"]):
            raise ValueError("search winner is absent or duplicated")
        best = max(search["profiles"], key=lambda profile: (profile["development_mean"],
                                                             -profile["hidden"], profile["profile"]))
        if best["profile"] != search["winner"]:
            raise ValueError("search did not select its recorded development winner")
        dev, check = search.get("development", {}), search.get("public_check", {})
        try:
            expected_selection = (dev["candidate_mean"] >= dev["parent_mean"] + 0.005
                                  and check["candidate_mean"] >= check["parent_mean"] + 0.005)
        except (KeyError, TypeError) as exc:
            raise ValueError("search lacks public development or check scores") from exc
        if expected_selection != row["selected_specialist"]:
            raise ValueError("route selection contradicts the recorded public scores")
        expected_pickup = "g4_pickup" if row["selected_specialist"] else parent.route["unlockpickup"]
        if (set(child.route) != set(TASKS) or child.route["unlockpickup"] != expected_pickup
                or {task: expert for task, expert in child.route.items() if task != "unlockpickup"}
                != {task: expert for task, expert in parent.route.items() if task != "unlockpickup"}):
            raise ValueError("G4 changed an inherited task route")
        expected_experts = set(parent.experts) | ({"g4_pickup"} if row["selected_specialist"] else set())
        if set(child.experts) != expected_experts:
            raise ValueError("G4 altered its expert bank")
        for name in parent.experts:
            old, new = parent.experts[name].state_dict(), child.experts[name].state_dict()
            if old.keys() != new.keys() or any(not torch.equal(old[key], new[key]) for key in old):
                raise ValueError(f"G4 changed inherited expert {name}")
        loaded[row["id"]] = child
    return manifest, loaded


def prepare(manifest_path: Path, out: Path, root: Path, reveal_days: int = 7) -> dict:
    if reveal_days < 1:
        raise ValueError("reveal window must be at least one day")
    manifest, _ = validate_cohort(manifest_path, root)
    if not all(row["selected_specialist"] for row in manifest["navigation"]["children"]):
        raise ValueError("all five children need a public improvement before spending the sole private audit")
    ledger = Ledger(root)
    if ledger.verify() != (True, None):
        raise ValueError("official lineage chain is invalid")
    ancestors = surviving_ancestors(ledger, PARENTS)
    family_root = Path(os.environ.get("TODDLER_G4_SEARCH_TRIAL_ROOT",
                                   "~/.local/share/toddler/g4-search-private")).expanduser().resolve()
    if out.resolve().parent != family_root:
        raise ValueError(f"G4-search confirmatory trials must live under {family_root}")
    family_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    lock = family_root / "family-attempt.json"
    _write_private(lock, {"manifest_sha256": digest(manifest_path), "out": str(out.resolve()),
                          "policy": "one confirmatory G4-search audit; no repeated private peeks"})
    out.mkdir(mode=0o700, parents=True, exist_ok=False)
    commitments = [SS.new_set(30, date.today() + timedelta(days=reveal_days)) for _ in range(2)]
    protocol = {"schema": "toddler-g4-search-private-trial/v1", "generation": "G4-search",
                "manifest": str(manifest_path.resolve()), "manifest_sha256": digest(manifest_path),
                "evaluator_sha256": digest(Path(__file__)),
                "children": CHILDREN, "parents": PARENTS, "tasks": TASKS,
                "candidate_weights": {row["id"]: row["sha256"] for row in manifest["navigation"]["children"]},
                "ancestor_generations": ancestors, "control": "matched frozen G3 parent",
                "eval_mode": "sample", "seed_commitments": [asdict(c) for c in commitments],
                "promotion_gate": {"child_count": 5, "sets": 2, "seeds_per_set": 30,
                                   "mastery": 0.9, "max_regression": 0.1,
                                   "p_less_than": CONFIRMATORY_ALPHA, "prob_improvement_at_least": 0.75,
                                   "sequential_policy": "one prospective confirmatory G4-search audit; later attempts need a new preregistered family and larger cohort"}}
    _write_private(out / "protocol.json", protocol)
    return {"protocol": str(out / "protocol.json"), "protocol_sha256": digest(out / "protocol.json"),
            "seed_commitments": [asdict(c) for c in commitments]}


def _trial(raw: dict) -> Trial:
    return Trial(raw["seed_set_id"], raw["frozen_plan_sha256"], tuple(raw["child_ids"]),
                 raw["candidate_weights"],
                 {name: tuple(scores) for name, scores in raw["candidate"].items()},
                 {name: tuple(scores) for name, scores in raw["control"].items()},
                 {gen: {name: tuple(scores) for name, scores in tasks.items()}
                  for gen, tasks in raw["ancestors"].items()})


def confirmatory_pass(decision) -> bool:
    """One prospectively frozen G4-search attempt, stricter than the old trial."""
    return bool(decision.promote and all(
        row.passed and row.promotion_vs_ancestor.p_value < CONFIRMATORY_ALPHA
        and row.promotion_vs_control.p_value < CONFIRMATORY_ALPHA
        for row in decision.trials))


def run(protocol_path: Path, index: int, root: Path) -> dict:
    if index not in (0, 1):
        raise ValueError("trial index must be 0 or 1")
    protocol = json.loads(protocol_path.read_text())
    if (protocol["schema"] != "toddler-g4-search-private-trial/v1" or tuple(protocol["tasks"]) != TASKS
            or tuple(protocol["children"]) != CHILDREN or tuple(protocol["parents"]) != PARENTS
            or protocol["eval_mode"] != "sample"):
        raise ValueError("frozen protocol does not match the evaluator")
    if protocol["evaluator_sha256"] != digest(Path(__file__)):
        raise ValueError("evaluator code changed after protocol freeze")
    manifest_path = Path(protocol["manifest"])
    if digest(manifest_path) != protocol["manifest_sha256"]:
        raise ValueError("cohort manifest changed since freezing")
    manifest, children = validate_cohort(manifest_path, root)
    if {row["id"]: row["sha256"] for row in manifest["navigation"]["children"]} != protocol["candidate_weights"]:
        raise ValueError("candidate weights changed")
    commitment, seeds, _ = SS.load_private(protocol["seed_commitments"][index]["set_id"])
    if asdict(commitment) != protocol["seed_commitments"][index]:
        raise ValueError("secret seed commitment changed")
    out = protocol_path.parent / f"trial-{index + 1}.json"
    if out.exists():
        raise FileExistsError(out)
    torch.set_num_threads(2)
    anchors = {task: T.random_anchor(task, seeds) for task in TASKS}
    registry = G.Registry(root)
    rows = {arm: {task: [] for task in TASKS} for arm in ("candidate", "control", "G2", "G3-recombined")}
    rows["G1"] = {task: [] for task in TASKS[:4]}

    def score(net, task):
        raw = scoring.evaluate(TaskView(net, task), task, seeds, mode="sample")
        return float(np.mean([T.normalise(task, value, anchors[task]) for value in raw]))

    for child_index, child_id in enumerate(CHILDREN, start=1):
        parent, _ = registry.load("G3-recombined", f"t500{child_index}")
        g2, _ = registry.load("G2", f"t200{child_index}")
        g1_ref = next(ref for ref in registry.lineage("G2", f"t200{child_index}") if ref.startswith("G1/"))
        g1, _ = registry.load(*g1_ref.split("/", 1))
        for task in TASKS:
            parent_score = score(parent, task)
            rows["control"][task].append(parent_score)
            rows["G3-recombined"][task].append(parent_score)
            rows["candidate"][task].append(
                score(children[child_id], task)
                if task == "unlockpickup" and children[child_id].route[task] == "g4_pickup"
                else parent_score)
            rows["G2"][task].append(score(g2, task))
            if task in rows["G1"]:
                rows["G1"][task].append(score(g1, task))
    trial = {"seed_set_id": commitment.set_id, "frozen_plan_sha256": digest(protocol_path),
             "child_ids": CHILDREN, "candidate_weights": protocol["candidate_weights"],
             "candidate": rows.pop("candidate"), "control": rows.pop("control"), "ancestors": rows}
    _write_private(out, trial)
    return {"trial": str(out), "sha256": digest(out), "seed_set_id": commitment.set_id,
            "children": len(CHILDREN), "tasks": len(TASKS)}


def assess(protocol_path: Path, root: Path) -> dict:
    protocol = json.loads(protocol_path.read_text())
    if protocol.get("evaluator_sha256") != digest(Path(__file__)):
        raise ValueError("evaluator code changed after protocol freeze")
    paths = [protocol_path.parent / f"trial-{i}.json" for i in (1, 2)]
    trials = tuple(_trial(json.loads(path.read_text())) for path in paths)
    if [trial.seed_set_id for trial in trials] != [row["set_id"] for row in protocol["seed_commitments"]]:
        raise ValueError("trials do not match committed seed sets")
    if any(trial.frozen_plan_sha256 != digest(protocol_path)
           or trial.candidate_weights != protocol["candidate_weights"]
           or set(trial.candidate) != set(TASKS) for trial in trials):
        raise ValueError("trial rows differ from the frozen protocol")
    if digest(Path(protocol["manifest"])) != protocol["manifest_sha256"]:
        raise ValueError("cohort manifest changed after evaluation")
    ledger = Ledger(root)
    if ledger.verify() != (True, None):
        raise ValueError("official lineage chain is invalid")
    decision = assess_lineage_replication(ledger, PARENTS, trials)
    result = {"promote": confirmatory_pass(decision),
              "alpha": CONFIRMATORY_ALPHA, "required_ancestors": decision.required_ancestors,
              "protocol_sha256": digest(protocol_path),
              "trial_sha256": [digest(path) for path in paths],
              "trials": [{"seed_set_id": trial.seed_set_id, "passed": row.passed,
                          "strongest_ancestor": row.strongest_ancestor,
                          "vs_ancestor": asdict(row.promotion_vs_ancestor),
                          "vs_control": asdict(row.promotion_vs_control),
                          "retention_floors": row.retention_floors,
                          "regressions": row.regressions}
                         for trial, row in zip(trials, decision.trials)]}
    return result


def promote(protocol_path: Path, root: Path) -> dict:
    """Import only a replicated winner; repeat safely after an interrupted import."""
    result = assess(protocol_path, root)
    if not result["promote"]:
        raise ValueError("G4-search did not pass both private confirmation trials")
    protocol = json.loads(protocol_path.read_text())
    manifest, children = validate_cohort(Path(protocol["manifest"]), root)
    ledger = Ledger(root)
    if ledger.verify() != (True, None):
        raise ValueError("official lineage chain is invalid")
    generation = "G4-search"
    target = root / generation
    trial = _trial(json.loads((protocol_path.parent / "trial-1.json").read_text()))
    if not target.exists():
        staging = Path(tempfile.mkdtemp(prefix=".g4-import-", dir=root))
        try:
            registry = G.Registry(staging)
            for index, row in enumerate(manifest["navigation"]["children"]):
                child_id = row["id"]
                per_task = {task: trial.candidate[task][index] for task in TASKS}
                record = G.ToddlerRecord(
                    generation, child_id, "task-router:" + "+".join(TASKS),
                    {"route": row["route"], "method": "bounded-oracle-architecture-search",
                     "eval_mode": "sample", "per_task": per_task,
                     "score_basis": "private trial; aggregate only until seed reveal",
                     "protocol_sha256": digest(protocol_path)},
                    next(profile["oracle_imitation"]["oracle_steps"] for profile in row["search"]["profiles"]
                         if profile["profile"] == row["search"]["winner"]),
                    [row["g3_parent"], row["g2_source"]],
                    [per_task[task] for task in TASKS], G.hardware_fingerprint(),
                    software=G.software_versions())
                registry.save(children[child_id], record)
                if record.weights_sha256 != row["sha256"]:
                    raise ValueError(f"{child_id} changed during registry serialization")
            os.replace(staging / generation, target)
        finally:
            shutil.rmtree(staging)
    registry = G.Registry(root)
    for index, row in enumerate(manifest["navigation"]["children"]):
        child_id = row["id"]
        _, record = registry.load(generation, child_id)
        if record.weights_sha256 != row["sha256"]:
            raise ValueError(f"{child_id} official weights differ from the frozen cohort")
        ref = f"{generation}/{child_id}"
        birth = ledger.birth(ref)
        if birth:
            if birth["weights_sha256"] != row["sha256"]:
                raise ValueError(f"{ref} has a conflicting lineage birth")
            continue
        winner = row["search"]["winner"]
        g3_inherited = tuple("task:" + task for task in TASKS
                             if task != "unlockpickup" or not row["selected_specialist"]
                             or winner.startswith("g3-") or winner.startswith("previous-"))
        inherit = [Inheritance(row["g3_parent"], row["g3_parent_sha256"], g3_inherited)]
        if row["selected_specialist"] and winner.startswith("g2-transfer-"):
            inherit.append(Inheritance(row["g2_source"], row["g2_source_sha256"],
                                       ("task:unlockpickup",)))
        ledger.born(ref, row["sha256"], role="population", inherits=inherit,
                    code=code_version(),
                    data={"protocol_sha256": digest(protocol_path),
                          "trial_sha256": result["trial_sha256"],
                          "teacher_code_sha256": manifest["teacher_code_sha256"],
                          "training_first_seed": row["training_first_seed"],
                          "selection_rule": row["selection_rule"], "search_winner": winner,
                          "previous_candidate_sha256": row.get("previous_candidate_sha256")},
                    budget={"method": "bounded-oracle-architecture-search",
                            "oracle_steps": next(profile["oracle_imitation"]["oracle_steps"]
                                                 for profile in row["search"]["profiles"]
                                                 if profile["profile"] == winner)},
                    hardware=record.hardware, software=record.software)
    verdict = ledger.verdict(generation)
    if verdict:
        if verdict["verdict"] != "survived" or verdict["evidence"].get("protocol_sha256") != digest(protocol_path):
            raise ValueError("existing G4 verdict has another protocol")
    else:
        ledger.selected(generation, "survived", result)
    if ledger.verify() != (True, None):
        raise ValueError("lineage chain failed after G4 import")
    return {"generation": generation, "verdict": "survived", "children": len(CHILDREN),
            "ledger_verified": True, "protocol_sha256": digest(protocol_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("prepare", "run", "assess", "promote"))
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--index", type=int, choices=(0, 1))
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    if args.command == "prepare":
        if args.manifest is None or args.out is None:
            parser.error("prepare needs --manifest and --out")
        result = prepare(args.manifest, args.out, args.root)
    elif args.command == "run":
        if args.protocol is None or args.index is None:
            parser.error("run needs --protocol and --index")
        result = run(args.protocol, args.index, args.root)
    elif args.command == "assess":
        if args.protocol is None:
            parser.error("assess needs --protocol")
        result = assess(args.protocol, args.root)
    else:
        if args.protocol is None:
            parser.error("promote needs --protocol")
        result = promote(args.protocol, args.root)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
