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
import math
import os
import shutil
import tempfile
from dataclasses import asdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import torch
from scipy.stats import binomtest

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
SELECTION_RULE = ("best development recipe; trained update must beat G3 and unchanged prior "
                  "on development and fixed public check by >= 0.005; otherwise retain eligible prior or G3")
BASE_PROFILES = {
    "g2-transfer-h64-128x12": "oracle_imitation",
    "g3-continue-h64-128x12": "oracle_imitation",
    "g3-continue-h64-256x16": "oracle_imitation",
    "scratch-h32-128x12": "oracle_imitation",
    "scratch-h128-128x12": "oracle_imitation",
    "g3-self-ppo-4096": "self_reward_ppo",
    "g3-continue-h64-128x12-then-self-ppo-4096": "oracle_then_self_reward",
}
PRIOR_PROFILES = {"previous-unchanged": "unchanged_hash_verified_prior",
                  "previous-continue-128x12": "oracle_imitation",
                  "previous-self-ppo-4096": "self_reward_ppo"}
CONFIRMATORY_ALPHA = 0.01
# This one-family decision was frozen before the next nightly public split.
# Earlier G4-search cohorts repeatedly used the 2026-10-10 public maps.
PUBLIC_GATE_NOT_BEFORE = datetime(2026, 10, 10, 20, 0, tzinfo=timezone.utc)


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


def winner_training(row: dict) -> dict:
    """Validate provenance of the actually routed profile, including rollback."""
    selected = row["search"].get("selected_profile")
    if selected == "G3-parent":
        return {"method": "inherited_G3_parent"}
    profiles = [profile for profile in row["search"]["profiles"]
                if profile["profile"] == selected]
    if len(profiles) != 1:
        raise ValueError("missing or duplicated selected training profile")
    training = profiles[0].get("training") or profiles[0].get("oracle_imitation")
    if not isinstance(training, dict):
        raise ValueError("selected profile lacks training provenance")
    expected_method = (BASE_PROFILES | PRIOR_PROFILES).get(selected)
    if training.get("method") != expected_method:
        raise ValueError("selected profile method differs from the frozen search contract")

    def valid_ppo(evidence: dict) -> bool:
        return (evidence.get("method") == "self_reward_ppo" and evidence.get("steps") == 4096
                and evidence.get("uses_teacher_grid") is False)

    def valid_oracle(evidence: dict) -> bool:
        return (evidence.get("method") == "oracle_imitation" and evidence.get("oracle_steps", 0) >= 1
                and evidence.get("teacher_privileged_grid") is True)

    method = training.get("method")
    if selected == "previous-unchanged":
        if method != "unchanged_hash_verified_prior" or training.get("steps") != 0:
            raise ValueError("retained prior lacks unchanged-weight provenance")
    elif method == "self_reward_ppo":
        if not valid_ppo(training):
            raise ValueError("self-reward profile lacks the fixed CPU training contract")
    elif method == "oracle_then_self_reward":
        if (not isinstance(training.get("imitation"), dict)
                or not isinstance(training.get("own_interaction"), dict)
                or not valid_oracle(training["imitation"])
                or not valid_ppo(training["own_interaction"])
                or training.get("disjoint_training_seeds") is not True):
            raise ValueError("combined profile lacks imitation and own-reward provenance")
    elif not valid_oracle(training):
        raise ValueError("oracle profile lacks demonstration provenance")
    return training


def verify_search_selection(search: dict, selected_specialist: bool) -> str:
    """Recompute the predeclared public rollback from unrounded evidence."""
    if search.get("selection_rule") != SELECTION_RULE:
        raise ValueError("cohort used a different route selection rule")
    profiles = search.get("profiles")
    if not isinstance(profiles, list) or not profiles:
        raise ValueError("search lacks candidate profiles")
    if len({row.get("profile") for row in profiles}) != len(profiles):
        raise ValueError("search has duplicate profile names")
    if any(not isinstance(row.get("development_mean_raw"), (int, float))
           or not math.isfinite(row["development_mean_raw"])
           or round(row["development_mean_raw"], 5) != row.get("development_mean")
           or not isinstance(row.get("hidden"), int) for row in profiles):
        raise ValueError("search lacks raw development evidence")
    best = max(profiles, key=lambda row: (row["development_mean_raw"], -row["hidden"], row["profile"]))
    if best["profile"] != search.get("winner"):
        raise ValueError("search did not select its recorded development winner")
    evidence = search.get("selection_evidence", {})
    required = ("winner_development_mean", "winner_public_check_mean",
                "parent_development_mean", "parent_public_check_mean")
    if any(not isinstance(evidence.get(name), (int, float)) or not math.isfinite(evidence[name])
           for name in required):
        raise ValueError("search lacks raw public selection evidence")
    winner_dev, winner_check = evidence["winner_development_mean"], evidence["winner_public_check_mean"]
    parent_dev, parent_check = evidence["parent_development_mean"], evidence["parent_public_check_mean"]
    prior_dev, prior_check = evidence.get("previous_development_mean"), evidence.get("previous_public_check_mean")
    expected_profiles = set(BASE_PROFILES) | (set(PRIOR_PROFILES) if prior_dev is not None else set())
    if {row["profile"] for row in profiles} != expected_profiles:
        raise ValueError("search changed the frozen profile set")
    for profile in profiles:
        if profile.get("training", {}).get("method") != (BASE_PROFILES | PRIOR_PROFILES)[profile["profile"]]:
            raise ValueError("search profile method differs from the frozen contract")
    if (best["development_mean_raw"] != winner_dev
            or (prior_dev is None) != (prior_check is None)
            or (prior_dev is not None and
                (not all(isinstance(value, (int, float)) and math.isfinite(value)
                         for value in (prior_dev, prior_check))
                 or not any(row["profile"] == "previous-unchanged"
                            and row["development_mean_raw"] == prior_dev for row in profiles)))):
        raise ValueError("search winner or prior baseline differs from raw evidence")
    prior_display = search.get("previous_baseline")
    if ((prior_dev is None and prior_display is not None)
            or (prior_dev is not None and
                (not isinstance(prior_display, dict)
                 or prior_display.get("development_mean") != round(prior_dev, 5)
                 or prior_display.get("public_check_mean") != round(prior_check, 5)))):
        raise ValueError("displayed prior baseline differs from raw evidence")
    gain = 0.005
    beats_parent = winner_dev >= parent_dev + gain and winner_check >= parent_check + gain
    beats_prior = (prior_dev is None or
                   (winner_dev >= prior_dev + gain and winner_check >= prior_check + gain))
    prior_eligible = (prior_dev is not None and prior_dev >= parent_dev + gain
                      and prior_check >= parent_check + gain)
    choice = ("previous" if best["profile"] == "previous-unchanged" and beats_parent else
              "parent" if best["profile"] == "previous-unchanged" else
              "challenger" if beats_parent and beats_prior else
              "previous" if prior_eligible else "parent")
    expected_profile = (best["profile"] if choice == "challenger" else
                        "previous-unchanged" if choice == "previous" else "G3-parent")
    chosen_dev = winner_dev if choice == "challenger" else prior_dev if choice == "previous" else parent_dev
    chosen_check = (winner_check if choice == "challenger" else
                    prior_check if choice == "previous" else parent_check)
    development, check = search.get("development", {}), search.get("public_check", {})
    if (search.get("selected_profile") != expected_profile
            or selected_specialist != (choice != "parent")
            or round(chosen_dev, 5) != development.get("candidate_mean")
            or round(chosen_check, 5) != check.get("candidate_mean")
            or round(parent_dev, 5) != development.get("parent_mean")
            or round(parent_check, 5) != check.get("parent_mean")
            or round(winner_check, 5) != search.get("winner_public_check_mean")):
        raise ValueError("public route selection contradicts the recorded scores")
    return expected_profile


def prior_source(manifest_path: Path, row: dict):
    """Independently load the hash-bound previous expert, if one was compared."""
    source_sha = row.get("previous_candidate_sha256")
    has_prior = row["search"].get("previous_baseline") is not None
    if not has_prior:
        if source_sha is not None:
            raise ValueError("prior source hash exists without a prior baseline")
        return None
    if (not isinstance(source_sha, str) or len(source_sha) != 64
            or any(char not in "0123456789abcdef" for char in source_sha)):
        raise ValueError("prior baseline lacks a source weight hash")
    for previous in sorted(manifest_path.parent.parent.glob("*/manifest.json"), reverse=True):
        if previous.parent.name >= manifest_path.parent.name:
            continue
        earlier = json.loads(previous.read_text())
        if earlier.get("status") != "completed_candidate":
            continue
        sources = [candidate for candidate in earlier.get("navigation", {}).get("children", [])
                   if candidate.get("id") == row["id"] and candidate.get("sha256") == source_sha
                   and candidate.get("selected_specialist") is True]
        if not sources:
            continue
        source = sources[0]
        path = (previous.parent / "navigation" / f"{row['id']}.pt").resolve()
        if (Path(source["artifact"]).resolve() != path or not path.is_file()
                or digest(path) != source_sha
                or source.get("g3_parent") != row["g3_parent"]
                or source.get("g3_parent_sha256") != row["g3_parent_sha256"]):
            raise ValueError("prior source artifact or parent lineage differs")
        net = _load_child(path)
        if net.route != source.get("route") or net.route.get("unlockpickup") != "g4_pickup":
            raise ValueError("prior source route differs from its manifest")
        return net.experts["g4_pickup"]
    raise ValueError("prior baseline source was not found in archived cohorts")


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
        if search.get("selected_specialist") != row["selected_specialist"]:
            raise ValueError("missing or mismatched architecture search provenance")
        selected_profile = verify_search_selection(search, row["selected_specialist"])
        winner_training(row)
        previous_expert = prior_source(manifest_path, row)
        expected_pickup = "g4_pickup" if row["selected_specialist"] else parent.route["unlockpickup"]
        if (set(child.route) != set(TASKS) or child.route["unlockpickup"] != expected_pickup
                or {task: expert for task, expert in child.route.items() if task != "unlockpickup"}
                != {task: expert for task, expert in parent.route.items() if task != "unlockpickup"}):
            raise ValueError("G4 changed an inherited task route")
        expected_experts = set(parent.experts) | ({"g4_pickup"} if row["selected_specialist"] else set())
        if set(child.experts) != expected_experts:
            raise ValueError("G4 altered its expert bank")
        if selected_profile == "previous-unchanged":
            if previous_expert is None:
                raise ValueError("retained prior has no archived source")
            old, new = previous_expert.state_dict(), child.experts["g4_pickup"].state_dict()
            if old.keys() != new.keys() or any(not torch.equal(old[key], new[key]) for key in old):
                raise ValueError("retained prior expert differs from archived source")
        for name in parent.experts:
            old, new = parent.experts[name].state_dict(), child.experts[name].state_dict()
            if old.keys() != new.keys() or any(not torch.equal(old[key], new[key]) for key in old):
                raise ValueError(f"G4 changed inherited expert {name}")
        loaded[row["id"]] = child
    return manifest, loaded


def public_gate(manifest_path: Path, manifest: dict) -> dict:
    """Require a fresh prospective split and preserve one mastered G3 parent."""
    def ranges_from(navigation: dict) -> tuple[tuple[int, int], tuple[int, int]]:
        ranges = tuple(tuple(navigation[key]) for key in ("development_seeds", "public_check_seeds"))
        if (len(ranges) != 2 or any(len(pair) != 2 or
                any(type(value) is not int for value in pair) for pair in ranges)):
            raise ValueError("invalid public seed range")
        return ranges

    def overlaps(first: tuple[int, int], second: tuple[int, int]) -> bool:
        return max(first[0], second[0]) <= min(first[1], second[1])

    try:
        created = datetime.strptime(manifest["created_utc"].split("-", 1)[0],
                                    "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
        nav = manifest["navigation"]
        ranges = ranges_from(nav)
        counts = tuple(high - low + 1 for low, high in ranges)
        if (created < PUBLIC_GATE_NOT_BEFORE or counts != (50, 50)
                or not (0 <= ranges[0][0] <= ranges[0][1] < ranges[1][0] <= ranges[1][1] < 100_000)):
            return {"eligible": False, "reason": "needs a new, disjoint 50+50 public split after preregistration"}
        for previous in manifest_path.parent.parent.glob("*/manifest.json"):
            if previous.resolve() == manifest_path.resolve():
                continue
            try:
                older = json.loads(previous.read_text())
                old_ranges = ranges_from(older["navigation"])
            except (OSError, ValueError, KeyError, TypeError):
                return {"eligible": False, "reason": "cannot verify an earlier cohort's public seed provenance"}
            if any(overlaps(current, prior) for current in ranges for prior in old_ranges):
                return {"eligible": False, "reason": "public seeds overlap another cohort"}
        children = nav["children"]
        selected = sum(row["selected_specialist"] for row in children)
        if selected == 5:
            return {"eligible": True, "selected_children": 5, "mastered_fallback": []}
        if selected != 4:
            return {"eligible": False, "reason": "fewer than four children improved on both public sets",
                    "selected_children": selected}
        fallback = next(row for row in children if not row["selected_specialist"])
        search = fallback["search"]
        if any(search[key]["parent_successes"] != count or search[key]["parent_mean"] < 1.0
               for key, count in zip(("development", "public_check"), counts)):
            return {"eligible": False, "reason": "unchanged G3 parent has not mastered both public sets",
                    "selected_children": selected}
        return {"eligible": True, "selected_children": selected,
                "mastered_fallback": [fallback["id"]]}
    except (KeyError, TypeError, ValueError) as exc:
        return {"eligible": False, "reason": f"invalid public gate evidence: {type(exc).__name__}"}


def prepare(manifest_path: Path, out: Path, root: Path, reveal_days: int = 7) -> dict:
    if reveal_days < 1:
        raise ValueError("reveal window must be at least one day")
    manifest, _ = validate_cohort(manifest_path, root)
    gate = public_gate(manifest_path, manifest)
    if not gate["eligible"]:
        raise ValueError(f"public gate failed before private seeds: {gate['reason']}")
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
    protocol = {"schema": "toddler-g4-search-private-trial/v2", "generation": "G4-search",
                "manifest": str(manifest_path.resolve()), "manifest_sha256": digest(manifest_path),
                "evaluator_sha256": digest(Path(__file__)),
                "public_gate": gate,
                "children": CHILDREN, "parents": PARENTS, "tasks": TASKS,
                "candidate_weights": {row["id"]: row["sha256"] for row in manifest["navigation"]["children"]},
                "ancestor_generations": ancestors, "control": "matched frozen G3 parent",
                "eval_mode": "sample", "seed_commitments": [asdict(c) for c in commitments],
                "promotion_gate": {"child_count": 5, "sets": 2, "seeds_per_set": 30,
                                   "mastery": 0.9, "max_regression": 0.1,
                                   "p_less_than": CONFIRMATORY_ALPHA, "prob_improvement_at_least": 0.75,
                                   "test": "one-sided exact paired sign test on 30 matched seed means across five fixed children and nine tasks",
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


def _validated_seed_means(raw: dict) -> dict[str, np.ndarray]:
    """Recompute public-free trial aggregates from the private seed matrices."""
    expected = {"candidate": raw["candidate"], "control": raw["control"], **raw["ancestors"]}
    if raw.get("seed_count") != 30 or set(raw.get("per_seed", {})) != set(expected):
        raise ValueError("private trial lacks the frozen thirty-seed matrices")
    means = {}
    for arm, tasks in expected.items():
        matrices = raw["per_seed"][arm]
        if set(matrices) != set(tasks):
            raise ValueError(f"{arm} seed matrices differ from aggregate tasks")
        cols = []
        for task in TASKS:
            if task not in tasks:
                cols.append(np.zeros(30))
                continue
            matrix = np.asarray(matrices[task], dtype=float)
            if matrix.shape != (len(CHILDREN), 30) or not np.isfinite(matrix).all():
                raise ValueError(f"{arm}/{task} needs five finite thirty-seed rows")
            if not np.allclose(matrix.mean(axis=1), tasks[task], rtol=0, atol=1e-10):
                raise ValueError(f"{arm}/{task} aggregate differs from private seed rows")
            cols.append(matrix.mean(axis=0))
        means[arm] = np.mean(np.stack(cols), axis=0)
    return means


def paired_seed_evidence(candidate: np.ndarray, reference: np.ndarray) -> dict:
    """Test fixed-cohort improvement across matched, unseen environments."""
    if candidate.shape != (30,) or reference.shape != (30,):
        raise ValueError("paired test needs thirty matched seed means per arm")
    if not np.isfinite(candidate).all() or not np.isfinite(reference).all():
        raise ValueError("paired seed scores must be finite")
    difference = candidate - reference
    wins = int(np.count_nonzero(difference > 1e-12))
    losses = int(np.count_nonzero(difference < -1e-12))
    ties = 30 - wins - losses
    p_value = float(binomtest(wins, wins + losses, 0.5, alternative="greater").pvalue) if wins + losses else 1.0
    improvement_probability = (wins + 0.5 * ties) / 30
    gain = float(np.mean(difference))
    return {"passed": bool(p_value < CONFIRMATORY_ALPHA and improvement_probability >= 0.75 and gain > 0),
            "p_value": p_value, "prob_improvement": improvement_probability,
            "mean_paired_gain": gain, "wins": wins, "losses": losses, "ties": ties,
            "candidate_mean": float(np.mean(candidate)), "reference_mean": float(np.mean(reference))}


def run(protocol_path: Path, index: int, root: Path) -> dict:
    if index not in (0, 1):
        raise ValueError("trial index must be 0 or 1")
    protocol = json.loads(protocol_path.read_text())
    if (protocol["schema"] != "toddler-g4-search-private-trial/v2" or tuple(protocol["tasks"]) != TASKS
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
    per_seed = {arm: {task: [] for task in tasks} for arm, tasks in rows.items()}

    def score(net, task):
        raw = scoring.evaluate(TaskView(net, task), task, seeds, mode="sample")
        return [float(T.normalise(task, value, anchors[task])) for value in raw]

    def append(arm, task, values):
        if len(values) != len(seeds):
            raise ValueError("evaluation returned an incomplete seed vector")
        rows[arm][task].append(float(np.mean(values)))
        per_seed[arm][task].append(values)

    for child_index, child_id in enumerate(CHILDREN, start=1):
        parent, _ = registry.load("G3-recombined", f"t500{child_index}")
        g2, _ = registry.load("G2", f"t200{child_index}")
        g1_ref = next(ref for ref in registry.lineage("G2", f"t200{child_index}") if ref.startswith("G1/"))
        g1, _ = registry.load(*g1_ref.split("/", 1))
        for task in TASKS:
            parent_score = score(parent, task)
            append("control", task, parent_score)
            append("G3-recombined", task, parent_score)
            append("candidate", task,
                score(children[child_id], task)
                if task == "unlockpickup" and children[child_id].route[task] == "g4_pickup"
                else parent_score)
            append("G2", task, score(g2, task))
            if task in rows["G1"]:
                append("G1", task, score(g1, task))
    trial = {"seed_set_id": commitment.set_id, "frozen_plan_sha256": digest(protocol_path),
             "child_ids": CHILDREN, "candidate_weights": protocol["candidate_weights"],
             "candidate": rows.pop("candidate"), "control": rows.pop("control"), "ancestors": rows,
             "per_seed": per_seed, "seed_count": len(seeds)}
    _write_private(out, trial)
    return {"trial": str(out), "sha256": digest(out), "seed_set_id": commitment.set_id,
            "children": len(CHILDREN), "tasks": len(TASKS)}


def assess(protocol_path: Path, root: Path) -> dict:
    protocol = json.loads(protocol_path.read_text())
    if (protocol.get("schema") != "toddler-g4-search-private-trial/v2"
            or protocol.get("evaluator_sha256") != digest(Path(__file__))):
        raise ValueError("evaluator code changed after protocol freeze")
    paths = [protocol_path.parent / f"trial-{i}.json" for i in (1, 2)]
    raw_trials = [json.loads(path.read_text()) for path in paths]
    trials = tuple(_trial(raw) for raw in raw_trials)
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
    trial_rows = []
    for trial, raw, retention in zip(trials, raw_trials, decision.trials):
        means = _validated_seed_means(raw)
        ancestor = paired_seed_evidence(means["candidate"], means[retention.strongest_ancestor])
        control = paired_seed_evidence(means["candidate"], means["control"])
        passed = bool(not retention.regressions and ancestor["passed"] and control["passed"])
        trial_rows.append({"seed_set_id": trial.seed_set_id, "passed": passed,
                           "strongest_ancestor": retention.strongest_ancestor,
                           "vs_ancestor": ancestor, "vs_control": control,
                           "retention_floors": retention.retention_floors,
                           "regressions": retention.regressions})
    result = {"promote": all(row["passed"] for row in trial_rows),
              "alpha": CONFIRMATORY_ALPHA, "required_ancestors": decision.required_ancestors,
              "protocol_sha256": digest(protocol_path),
              "trial_sha256": [digest(path) for path in paths],
              "test": "one-sided exact paired sign test over thirty seed means",
              "trials": trial_rows}
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
                    winner_training(row).get("steps", winner_training(row).get("oracle_steps")),
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
                    budget={"method": winner_training(row).get("method", "oracle_imitation"),
                            "training_steps": winner_training(row).get("steps", winner_training(row).get("oracle_steps"))},
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
