"""Prospective, one-attempt G5 arithmetic gate with frozen G4 retention.

Evaluator-owned banks and navigation seeds stay outside Git. This command
prepares two fixed trials, resumes an interrupted run on the same commitments,
and assesses both. It cannot append a lineage promotion without separate
semantic review of the private question wording.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import date, timedelta
from pathlib import Path

import joblib
import numpy as np
from scipy.stats import binomtest

from scripts.g5_parent_contract import DEFAULT_ROOT, verify_contract
from toddler.g4_curriculum import PracticeItem
from toddler.g5_cognitive import G5CognitiveStudent, parse_form
from toddler.g5_system import G5Prototype
from toddler.learn import scoring, secret_seeds as SS
from toddler.learn.generations import Registry
from toddler.learn.lineage import code_version
from toddler.learn.multitask import TaskView

REPO = Path(__file__).resolve().parents[1]
TASKS = ("cartpole", "acrobot", "empty5", "doorkey5", "doorkey8", "unlock",
         "unlockpickup", "keycorridor3", "lavacross9")
LANGUAGES = ("english", "nederlands")
OPERATIONS = ("add", "subtract", "multiply")
SEEDS_PER_TRIAL = 8
ITEMS_PER_CELL = 12
SPENT_BANK_SHA256 = {
    "d2679f8c22303cc1be25ee1c065a2f68cc19a3d96ff687de1a3b3a414c640e9b",
    "4ff13d20cedfb9c748771150f9b7978dd7b3e0ecc24acee2557bcc3f5763ca43",
}
BASELINE_SHA256 = "2b8d37f87e33dea48bfe34492fc19b2842dd24b28487b9402e46b917c3e064be"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _private(path: Path) -> Path:
    path = path.resolve()
    if path.is_relative_to(REPO) or path.stat().st_mode & 0o077:
        raise ValueError("G5 audit input must be owner-only and outside Git")
    return path


def _write_private(path: Path, obj: dict) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as handle:
        json.dump(obj, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def _answer(operation: str, a: int, b: int) -> int:
    return {"add": a + b, "subtract": a - b, "multiply": a * b}[operation]


def _normal_question(value: str) -> str:
    return re.sub(r"\s+", " ", value.casefold()).strip()


def reject_question_overlap(banks: tuple[list[dict], list[dict]],
                            exclusion_files: tuple[Path, ...]) -> list[dict]:
    """Reject exact prompt reuse across training, selection and spent diagnostics."""
    if len(exclusion_files) < 4:
        raise ValueError("G5 promotion needs both Teacher lessons and both spent banks excluded")
    excluded, provenance = set(), []
    for path in exclusion_files:
        path = _private(path)
        payload = json.loads(path.read_text())
        questions = [row.get("question") for row in payload.get("items", [])]
        if not questions or any(not isinstance(value, str) for value in questions):
            raise ValueError("exclusion file lacks auditable question rows")
        excluded.update(_normal_question(value) for value in questions)
        provenance.append({"path": str(path), "sha256": digest(path), "items": len(questions)})
    proposed = {_normal_question(row["item"].question) for part in banks for row in part}
    if proposed & excluded:
        raise ValueError("promotion bank repeats a training, selection or spent diagnostic question")
    return provenance


def _verify_exclusions(protocol: dict) -> None:
    exclusions = protocol.get("excluded_question_files", [])
    if len(exclusions) < 4 or any(digest(_private(Path(row["path"]))) != row["sha256"]
                                  for row in exclusions):
        raise ValueError("training or spent-bank exclusion evidence changed")


def load_bank(path: Path, manifest_sha256: str) -> list[dict]:
    bank = json.loads(_private(path).read_text())
    if (bank.get("schema") != "toddler-g5-prospective-arithmetic/v1"
            or bank.get("status") != "sealed_confirmation"
            or bank.get("candidate_manifest_sha256") != manifest_sha256):
        raise ValueError("bank is not sealed to the exact candidate")
    rows, seen, counts = [], set(), Counter()
    for row in bank.get("items", []):
        language, operation = row.get("language"), row.get("operation")
        a, b = row.get("a"), row.get("b")
        if (language not in LANGUAGES or operation not in OPERATIONS
                or type(a) is not int or type(b) is not int
                or row.get("area") != "arithmetic"
                or row.get("key_source") != "deterministic_integer_arithmetic_v1"
                or not isinstance(row.get("id"), str)
                or not row["id"].startswith("private-g5-confirm-")):
            raise ValueError("private bank item lacks a valid key source or identity")
        item = PracticeItem("arithmetic", row["question"], tuple(row["options"]), row["answer"])
        if (re.findall(r"\d+", item.question) != [str(a), str(b)]
                or item.answer != str(_answer(operation, a, b))
                or parse_form(item) is not None):
            raise ValueError("private bank answer, operands or syntax failed verification")
        norm = _normal_question(item.question)
        if row["id"] in seen or norm in seen:
            raise ValueError("private bank has repeated IDs or questions")
        seen.update((row["id"], norm))
        counts[(language, operation)] += 1
        rows.append({"id": row["id"], "language": language,
                     "operation": operation, "item": item})
    if (len(rows) != 72 or any(counts[(language, operation)] != ITEMS_PER_CELL
                               for language in LANGUAGES for operation in OPERATIONS)):
        raise ValueError("confirmation bank needs six balanced 12-item cells")
    return rows


def _identical_base(current: G5CognitiveStudent, reference_path: Path) -> bool:
    if digest(reference_path) != BASELINE_SHA256:
        raise ValueError("historical G4 cognitive reference artifact changed")
    prior = joblib.load(_private(reference_path))
    if not isinstance(prior, G5CognitiveStudent):
        raise ValueError("historical cognitive reference has wrong type")
    a, b = prior.base.model, current.base.model
    return (a.named_steps["tfidfvectorizer"].vocabulary_
            == b.named_steps["tfidfvectorizer"].vocabulary_
            and np.array_equal(a.named_steps["tfidfvectorizer"].idf_,
                               b.named_steps["tfidfvectorizer"].idf_)
            and all(np.array_equal(getattr(a.named_steps["logisticregression"], field),
                                       getattr(b.named_steps["logisticregression"], field))
                    for field in ("classes_", "coef_", "intercept_")))


def prepare(*, manifest_path: Path, parent_contract: Path, parent_root: Path,
            reference_path: Path, banks: tuple[Path, Path],
            exclusion_files: tuple[Path, ...], out: Path) -> dict:
    version = code_version(REPO)
    if version.get("dirty") or not version.get("commit"):
        raise ValueError("prepare requires clean committed evaluator source")
    manifest_path, parent_contract = _private(manifest_path), parent_contract.resolve()
    manifest_sha = digest(manifest_path)
    if any(digest(path) in SPENT_BANK_SHA256 for path in banks):
        raise ValueError("a spent diagnostic bank cannot become a promotion bank")
    rows = [load_bank(path, manifest_sha) for path in banks]
    questions = [{_normal_question(row["item"].question) for row in part} for part in rows]
    if questions[0] & questions[1] or digest(banks[0]) == digest(banks[1]):
        raise ValueError("confirmation banks must have disjoint prompts and hashes")
    exclusions = reject_question_overlap((rows[0], rows[1]), exclusion_files)
    contract = json.loads(parent_contract.read_text())
    verify_contract(parent_root, contract)
    candidate = G5Prototype.load(manifest_path, "G5-semantic-public/t7101",
                                 parent_root=parent_root, parent_contract=parent_contract)
    if not _identical_base(candidate.cognitive, reference_path):
        raise ValueError("non-arithmetic G4 cognitive fallback changed")
    if not (candidate.cognitive.arithmetic and candidate.cognitive.semantic):
        raise ValueError("candidate lacks both frozen arithmetic experts")
    out = out.resolve()
    if out.is_relative_to(REPO) or out.exists():
        raise ValueError("audit directory must be new and outside Git")
    out.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    commitments = [SS.new_set(SEEDS_PER_TRIAL, date.today() + timedelta(days=7))
                   for _ in range(2)]
    protocol = {"schema": "toddler-g5-prospective-arithmetic/v1",
                "status": "prepared_unrun", "candidate_manifest": str(manifest_path),
                "candidate_manifest_sha256": manifest_sha,
                "parent_contract": str(parent_contract),
                "parent_contract_sha256": digest(parent_contract),
                "reference_cognitive": str(_private(reference_path)),
                "reference_cognitive_sha256": digest(reference_path),
                "banks": [{"path": str(_private(path)), "sha256": digest(path)} for path in banks],
                "excluded_question_files": exclusions,
                "navigation_seed_commitments": [asdict(c) for c in commitments],
                "evaluator_code": version, "evaluator_sha256": digest(Path(__file__)),
                "child_refs": [row["ref"] for row in json.loads(manifest_path.read_text())["children"]],
                "tasks": list(TASKS), "seeds_per_trial": SEEDS_PER_TRIAL,
                "scope": "English and Dutch arithmetic only; other 17 cognitive areas structurally retained",
                "gate": {"two_sets_required": True, "items_per_set": 72,
                         "min_net_gain_per_set": 12, "min_g5_correct_per_set": 44,
                         "max_g4_only_correct_per_set": 2,
                         "one_sided_p_less_than_per_set": 0.005,
                         "no_negative_language_operation_cell": True,
                         "navigation_per_seed_equal": True},
                "semantic_review": "pending_independent_review",
                "promotion_eligible": False}
    staging = Path(tempfile.mkdtemp(prefix=f".{out.name}-preparing-", dir=out.parent))
    staging.chmod(0o700)
    _write_private(staging / "protocol.json", protocol)
    staging.rename(out)
    return {"status": protocol["status"], "protocol_sha256": digest(out / "protocol.json"),
            "candidate_manifest_sha256": manifest_sha,
            "seed_commitments": [asdict(c) for c in commitments]}


def _strict(model, item: PracticeItem) -> bool:
    return all(model.choose(PracticeItem(item.area, item.question,
                                         item.options[offset:] + item.options[:offset],
                                         item.answer))[0] == item.answer
               for offset in range(4))


def run(protocol_path: Path, index: int, parent_root: Path) -> dict:
    protocol_path = _private(protocol_path)
    protocol = json.loads(protocol_path.read_text())
    if protocol.get("schema") != "toddler-g5-prospective-arithmetic/v1" or index not in (0, 1):
        raise ValueError("unknown prospective protocol or trial index")
    out = protocol_path.parent / f"trial-{index}.json"
    if out.exists():
        prior = json.loads(_private(out).read_text())
        if (prior.get("protocol_sha256") != digest(protocol_path)
                or prior.get("trial_index") != index
                or prior.get("bank_sha256") != protocol["banks"][index]["sha256"]
                or prior.get("seed_set_id")
                != protocol["navigation_seed_commitments"][index]["set_id"]):
            raise ValueError("existing trial result conflicts with frozen protocol")
        return {"status": "trial_already_complete", "index": index,
                "trial_sha256": digest(out)}
    if digest(Path(__file__)) != protocol["evaluator_sha256"]:
        raise ValueError("evaluator source changed after trial preparation")
    _verify_exclusions(protocol)
    bank_ref = protocol["banks"][index]
    bank_path = _private(Path(bank_ref["path"]))
    manifest_path = _private(Path(protocol["candidate_manifest"]))
    parent_contract = Path(protocol["parent_contract"])
    if (digest(bank_path) != bank_ref["sha256"]
            or digest(manifest_path) != protocol["candidate_manifest_sha256"]
            or digest(parent_contract) != protocol["parent_contract_sha256"]
            or digest(_private(Path(protocol["reference_cognitive"])))
            != protocol["reference_cognitive_sha256"]):
        raise ValueError("prospective input changed after freeze")
    rows = load_bank(bank_path, protocol["candidate_manifest_sha256"])
    commitment = protocol["navigation_seed_commitments"][index]
    c, seeds, _ = SS.load_private(commitment["set_id"])
    if asdict(c) != commitment:
        raise ValueError("secret navigation seed commitment differs from protocol")
    manifest = json.loads(manifest_path.read_text())
    first = G5Prototype.load(manifest_path, manifest["children"][0]["ref"],
                             parent_root=parent_root, parent_contract=parent_contract)
    cognitive = []
    for row in rows:
        cognitive.append({"id": row["id"], "language": row["language"],
                          "operation": row["operation"],
                          "g4_correct": _strict(first.cognitive.base, row["item"]),
                          "g5_correct": _strict(first.cognitive, row["item"])})
    nav = {}
    registry = Registry(parent_root)
    for child in manifest["children"]:
        candidate = G5Prototype.load(manifest_path, child["ref"],
                                     parent_root=parent_root, parent_contract=parent_contract)
        parent_gen, parent_id = child["navigation_parent"].split("/", 1)
        parent, _ = registry.load(parent_gen, parent_id)
        task_rows = {}
        for task in TASKS:
            old = scoring.evaluate(TaskView(parent, task), task, seeds, mode="sample").tolist()
            new = scoring.evaluate(TaskView(candidate.navigation, task), task, seeds,
                                   mode="sample").tolist()
            task_rows[task] = {"g4": old, "g5": new}
        nav[child["ref"]] = task_rows
    result = {"schema": "toddler-g5-prospective-arithmetic-result/v1",
              "protocol_sha256": digest(protocol_path), "trial_index": index,
              "bank_sha256": digest(bank_path), "seed_set_id": c.set_id,
              "cognitive": cognitive, "navigation": nav}
    _write_private(out, result)
    return {"status": "trial_complete", "index": index, "trial_sha256": digest(out),
            "items": len(cognitive), "navigation_child_count": len(nav)}


def assess(protocol_path: Path) -> dict:
    protocol_path = _private(protocol_path)
    protocol = json.loads(protocol_path.read_text())
    if protocol.get("schema") != "toddler-g5-prospective-arithmetic/v1":
        raise ValueError("unknown prospective protocol")
    if (digest(Path(__file__)) != protocol["evaluator_sha256"]
            or digest(Path(protocol["candidate_manifest"])) != protocol["candidate_manifest_sha256"]
            or digest(Path(protocol["parent_contract"])) != protocol["parent_contract_sha256"]):
        raise ValueError("frozen candidate or parent contract changed")
    _verify_exclusions(protocol)
    summaries = []
    for index in (0, 1):
        path = _private(protocol_path.parent / f"trial-{index}.json")
        trial = json.loads(path.read_text())
        if (trial.get("schema") != "toddler-g5-prospective-arithmetic-result/v1"
                or trial.get("protocol_sha256") != digest(protocol_path)
                or trial.get("trial_index") != index
                or trial.get("bank_sha256") != protocol["banks"][index]["sha256"]
                or trial.get("seed_set_id") != protocol["navigation_seed_commitments"][index]["set_id"]):
            raise ValueError("trial does not match frozen protocol")
        rows = trial["cognitive"]
        if len(rows) != 72 or len({row["id"] for row in rows}) != 72:
            raise ValueError("trial lacks 72 distinct paired items")
        bank_path = _private(Path(protocol["banks"][index]["path"]))
        if digest(bank_path) != protocol["banks"][index]["sha256"]:
            raise ValueError("confirmation bank changed after trial")
        bank_rows = load_bank(bank_path, protocol["candidate_manifest_sha256"])
        expected = [(row["id"], row["language"], row["operation"]) for row in bank_rows]
        recorded = [(row.get("id"), row.get("language"), row.get("operation")) for row in rows]
        if expected != recorded or any(type(row.get(flag)) is not bool
                                   for row in rows for flag in ("g4_correct", "g5_correct")):
            raise ValueError("trial item identities or correctness flags changed")
        wins = sum(row["g5_correct"] and not row["g4_correct"] for row in rows)
        losses = sum(row["g4_correct"] and not row["g5_correct"] for row in rows)
        g4 = sum(row["g4_correct"] for row in rows)
        g5 = sum(row["g5_correct"] for row in rows)
        p = binomtest(wins, wins + losses, p=0.5,
                      alternative="greater").pvalue if wins + losses else 1.0
        cells = defaultdict(lambda: [0, 0])
        for row in rows:
            cell = cells[(row["language"], row["operation"])]
            cell[0] += row["g4_correct"]
            cell[1] += row["g5_correct"]
        no_cell_loss = all(cells[(language, operation)][1] >= cells[(language, operation)][0]
                           for language in LANGUAGES for operation in OPERATIONS)
        nav_equal = (set(trial["navigation"]) == set(protocol["child_refs"])
                     and all(set(tasks) == set(TASKS)
                             and all(len(values["g4"]) == len(values["g5"]) == SEEDS_PER_TRIAL
                                     and values["g4"] == values["g5"] for values in tasks.values())
                             for tasks in trial["navigation"].values()))
        passed = (g5 >= 44 and g5 - g4 >= 12 and losses <= 2 and p < 0.005
                  and no_cell_loss and nav_equal)
        summaries.append({"trial_index": index, "trial_sha256": digest(path),
                          "g4_correct": g4, "g5_correct": g5, "g5_only": wins,
                          "g4_only": losses, "one_sided_p": p,
                          "no_negative_cell": no_cell_loss,
                          "navigation_exact_retention": nav_equal,
                          "evidence_passed": passed})
    result = {"schema": "toddler-g5-prospective-assessment/v1",
              "protocol_sha256": digest(protocol_path),
              "candidate_manifest_sha256": protocol["candidate_manifest_sha256"],
              "status": "evidence_passed_pending_semantic_review"
              if all(row["evidence_passed"] for row in summaries)
              else "failed_hold_no_promotion",
              "generation_promoted": False, "trials": summaries,
              "scope": protocol["scope"]}
    _write_private(protocol_path.parent / "assessment.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "assess"))
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--parent-contract", type=Path, default=REPO /
                        "reports/g5_frozen_g4_parents_20261011.json")
    parser.add_argument("--parent-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--bank", type=Path, action="append")
    parser.add_argument("--exclude", type=Path, action="append")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--index", type=int)
    args = parser.parse_args()
    if args.command == "prepare":
        if not (args.manifest and args.reference and args.out and args.bank and len(args.bank) == 2
                and args.exclude and len(args.exclude) >= 4):
            parser.error("prepare requires manifest, reference, two banks and four exclusion files")
        result = prepare(manifest_path=args.manifest, parent_contract=args.parent_contract,
                         parent_root=args.parent_root, reference_path=args.reference,
                         banks=tuple(args.bank), exclusion_files=tuple(args.exclude), out=args.out)
    elif args.command == "run":
        if args.protocol is None or args.index not in (0, 1):
            parser.error("run requires protocol and index 0 or 1")
        result = run(args.protocol, args.index, args.parent_root)
    else:
        if args.protocol is None:
            parser.error("assess requires protocol")
        result = assess(args.protocol)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
