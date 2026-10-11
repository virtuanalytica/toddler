"""Train an auditable G5 cognitive prototype over five frozen G4 parents.

This command uses only public, deterministically keyed practice. It does not
read the private G4 diagnostic or produce a G5 promotion verdict.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from scripts.g5_parent_contract import DEFAULT_ROOT, current_contract, verify_contract
from toddler.g4_curriculum import PracticeItem, practice_items
from toddler.g4_student import CognitiveStudent
from toddler.g5_cognitive import FORMS, FeedbackArithmetic, G5CognitiveStudent
from toddler.learn.lineage import code_version

ROOT = Path(__file__).resolve().parents[1]
OPERATIONS = ("add", "subtract", "multiply")
FORM_OPERATION = {form: OPERATIONS[index % 3] for index, form in enumerate(FORMS)}


def arithmetic_result(operation: str, a: int, b: int) -> int:
    return {"add": a + b, "subtract": a - b, "multiply": a * b}[operation]


def question(form: str, a: int, b: int) -> str:
    labels = {
        "english_plus": f"What is {a} plus {b}?",
        "english_minus": f"What is {a} minus {b}?",
        "english_times": f"What is {a} times {b}?",
        "dutch_plus": f"Wat is {a} plus {b}?",
        "dutch_minus": f"Wat is {a} min {b}?",
        "dutch_times": f"Wat is {a} keer {b}?",
        "cantonese_plus": f"計算 {a} 加 {b}。",
        "cantonese_minus": f"計算 {a} 減 {b}。",
        "cantonese_times": f"計算 {a} 乘 {b}。",
    }
    return labels[form]


def make_public_items(*, seed: int, per_form: int, low: int, high: int) -> list[tuple[str, PracticeItem]]:
    rng = random.Random(seed)
    rows = []
    for form in FORMS:
        for index in range(per_form):
            a, b = rng.randint(low, high), rng.randint(low, high)
            if a == b or b == 1:
                b = low + ((b + 2) % (high - low + 1))
            result = arithmetic_result(FORM_OPERATION[form], a, b)
            choices = [str(value) for value in (result, result + 1, result - 1, result + 3)]
            rng.shuffle(choices)
            item = PracticeItem("arithmetic", question(form, a, b), tuple(choices), str(result))
            rows.append((f"public-g5-{seed}-{form}-{index}", item))
    return rows


def load_teacher_curriculum(path: Path) -> list[tuple[str, PracticeItem]]:
    lesson = json.loads(path.read_text())
    if (lesson.get("schema") != "teacher-g5-public-feedback/v1"
            or lesson.get("status") != "public_training_only_not_promotion"
            or lesson.get("seed") != 7001 or lesson.get("examples_per_form") != 3
            or not isinstance(lesson.get("teacher_commit"), str)
            or len(lesson["teacher_commit"]) != 40):
        raise ValueError("Teacher lesson does not match the fixed public G5 contract")
    rows = []
    counts: Counter[str] = Counter()
    for row in lesson.get("items", []):
        item = PracticeItem(row["area"], row["question"], tuple(row["options"]), row["answer"])
        form = row.get("form")
        if (form not in FORMS or row.get("key_source") != "deterministic_integer_arithmetic_v1"
                or not row.get("source_id", "").startswith("public-g5-7001-")):
            raise ValueError("Teacher example lacks a public verified source")
        if row["source_id"] not in {f"public-g5-7001-{form}-{index}" for index in range(3)}:
            raise ValueError("Teacher source ID differs from form and lesson slot")
        parsed = FORMS[form][1].fullmatch(item.question)
        if parsed is None or item.area != FORMS[form][0]:
            raise ValueError("Teacher question differs from its declared form")
        a, b = int(parsed[1]), int(parsed[2])
        if item.answer != str(arithmetic_result(FORM_OPERATION[form], a, b)):
            raise ValueError("Teacher answer differs from independently recomputed key")
        rows.append((row["source_id"], item))
        counts[form] += 1
    if (len(rows) != 27 or len({source for source, _ in rows}) != 27
            or counts != Counter({form: 3 for form in FORMS})):
        raise ValueError("Teacher lesson needs three distinct examples in each of nine forms")
    return rows


def strict_accuracy(model, items: list[PracticeItem]) -> float:
    if not items:
        raise ValueError("public evaluation needs items")
    return sum(all(model.choose(PracticeItem(item.area, item.question,
                                              item.options[offset:] + item.options[:offset],
                                              item.answer))[0] == item.answer
                   for offset in range(4)) for item in items) / len(items)


def search_public(train: list[tuple[str, PracticeItem]] | None = None) -> tuple[G5CognitiveStudent, dict]:
    # Existing G4 public curriculum is retained as a fallback. Its repeated
    # fact templates are never reported as an independent generalization score.
    base = CognitiveStudent.train(practice_items(seed=20261009, arithmetic_count=240))
    train = train if train is not None else make_public_items(seed=7001, per_form=3, low=4, high=15)
    check = make_public_items(seed=7002, per_form=12, low=16, high=49)
    train_questions = {item.question for _, item in train}
    if train_questions & {item.question for _, item in check}:
        raise ValueError("public training and check questions overlap")
    check_items = [item for _, item in check]
    baseline = strict_accuracy(base, check_items)
    trials = []
    for examples_per_form in (1, 3):
        learner = G5CognitiveStudent(base)
        for index, (source_id, item) in enumerate(train):
            if index % 3 < examples_per_form:
                learner.arithmetic.learn_verified(item, source_id=source_id,
                                                  key_source="deterministic_integer_arithmetic_v1")
        score = strict_accuracy(learner, check_items)
        trials.append((learner, {"examples_per_form": examples_per_form,
                                 "public_check_strict_accuracy": round(score, 6),
                                 "feedback": learner.arithmetic.manifest()}))
    # Select on public check only. The private evaluator must never inherit this set.
    chosen, selected = max(trials, key=lambda pair: (pair[1]["public_check_strict_accuracy"],
                                                    -pair[1]["examples_per_form"]))
    retention = practice_items(seed=20261010, arithmetic_count=60)
    report = {"status": "public_development_only", "promotion_eligible": False,
              "baseline_public_check_strict_accuracy": round(baseline, 6),
              "selected": selected,
              "search": [row for _, row in trials],
              "check_items": len(check_items), "check_question_sha256": hashlib.sha256(
                  json.dumps(sorted(item.question for item in check_items)).encode()).hexdigest(),
              "g4_public_retention_accuracy": round(base.accuracy(retention), 6),
              "g5_public_retention_accuracy": round(chosen.accuracy(retention), 6),
              "retention_warning": "public overlapping templates; exact frozen navigation hashes are the only current navigation retention evidence"}
    return chosen, report


def build(*, parent_root: Path, parent_contract: Path, teacher_curriculum: Path,
          out: Path) -> dict:
    import joblib

    version = code_version(ROOT)
    if version.get("dirty") or not version.get("commit"):
        raise ValueError("G5 candidate requires committed, clean training source")
    frozen = json.loads(parent_contract.read_text())
    verify_contract(parent_root, frozen)
    if frozen != current_contract(parent_root):
        raise ValueError("promoted parent contract changed during candidate start")
    lesson = load_teacher_curriculum(teacher_curriculum)
    lesson_sha = hashlib.sha256(teacher_curriculum.read_bytes()).hexdigest()
    student, report = search_public(lesson)
    report["teacher_curriculum_sha256"] = lesson_sha
    out = out.resolve()
    if out.is_relative_to(ROOT):
        raise ValueError("candidate artifacts must stay outside the source repository")
    out.mkdir(parents=True, mode=0o700, exist_ok=False)
    out.chmod(0o700)
    model_path = out / "cognitive.joblib"
    joblib.dump(student, model_path)
    model_path.chmod(0o600)
    model_sha = hashlib.sha256(model_path.read_bytes()).hexdigest()
    verify_contract(parent_root, frozen)
    children = []
    for index, parent in enumerate(frozen["parents"], start=1):
        children.append({"ref": f"G5-public-candidate/t700{index}",
                         "navigation_parent": parent["ref"],
                         "navigation_parent_weights_sha256": parent["weights_sha256"],
                         "inherited_expert_sha256": parent["expert_sha256"],
                         "inherited_route_sha256": parent["route_sha256"],
                         "cognitive_artifact_sha256": model_sha})
    manifest = {"schema": "toddler-g5-public-candidate/v1",
                "status": "trained_public_candidate_unpromoted",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "code": version,
                "parent_contract_sha256": hashlib.sha256(parent_contract.read_bytes()).hexdigest(),
                "cognitive_artifact_sha256": model_sha,
                "cognitive_training": report,
                "children": children,
                "private_evaluation": "not_started",
                "lineage_registration": "candidate manifest only; no G5 survived verdict"}
    path = out / "manifest.json"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--parent-contract", type=Path, default=ROOT /
                        "reports/g5_frozen_g4_parents_20261011.json")
    parser.add_argument("--teacher-curriculum", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    manifest = build(parent_root=args.parent_root, parent_contract=args.parent_contract,
                     teacher_curriculum=args.teacher_curriculum, out=args.out)
    print(json.dumps({"status": manifest["status"], "children": len(manifest["children"]),
                      "public_check": manifest["cognitive_training"]["selected"]["public_check_strict_accuracy"],
                      "promotion_eligible": False}))


if __name__ == "__main__":
    main()
