"""Build a second G5 public candidate from verified multilingual feedback.

This searches two small operation-classifier architectures on a public check
split. It never reads evaluator-owned diagnostic or promotion questions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from scripts.g5_parent_contract import DEFAULT_ROOT, current_contract, verify_contract
from scripts.g5_public_candidate import (arithmetic_result, load_teacher_curriculum,
                                         strict_accuracy)
from toddler.g4_curriculum import PracticeItem, practice_items
from toddler.g4_student import CognitiveStudent
from toddler.g5_cognitive import (FeedbackArithmetic, G5CognitiveStudent,
                                  SemanticOperationExpert)
from toddler.learn.lineage import code_version

ROOT = Path(__file__).resolve().parents[1]
LANGUAGES = ("english", "nederlands", "cantonese")
OPERATIONS = ("add", "subtract", "multiply")


def _owner_lesson(path: Path) -> Path:
    path = path.resolve()
    if path.is_relative_to(ROOT) or path.stat().st_mode & 0o077:
        raise ValueError("Teacher lessons must be owner-only and outside Toddler Git")
    return path


def load_semantic_lesson(path: Path) -> tuple[list[tuple[str, PracticeItem]],
                                              list[tuple[str, str, PracticeItem]]]:
    lesson = json.loads(_owner_lesson(path).read_text())
    if (lesson.get("schema") != "teacher-g5-semantic-feedback/v1"
            or lesson.get("status") != "public_train_check_not_promotion"
            or lesson.get("seed") != 7003
            or lesson.get("train_examples") != 81 or lesson.get("check_examples") != 108
            or not re.fullmatch(r"[0-9a-f]{40}", lesson.get("teacher_commit", ""))):
        raise ValueError("Teacher semantic lesson does not match the public contract")
    train, check, seen_sources, seen_questions = [], [], set(), set()
    counts = Counter()
    for row in lesson.get("items", []):
        language, operation, split = row.get("language"), row.get("operation"), row.get("split")
        index, a, b = row.get("template_id"), row.get("a"), row.get("b")
        source = row.get("source_id", "")
        if (language not in LANGUAGES or operation not in OPERATIONS
                or split not in {"train", "check"}
                or type(index) is not int or index not in range(4)
                or (split == "train") != (index < 3)
                or type(a) is not int or type(b) is not int
                or row.get("area") != "arithmetic"
                or row.get("key_source") != "deterministic_integer_arithmetic_v1"
                or not re.fullmatch(
                    rf"public-g5-semantic-7003-{language}-{operation}-{index}-(?:[0-9]|1[01])",
                    source)):
            raise ValueError("semantic item has invalid identity or key provenance")
        item = PracticeItem(row["area"], row["question"], tuple(row["options"]), row["answer"])
        if (re.findall(r"\d+", item.question) != [str(a), str(b)]
                or item.answer != str(arithmetic_result(operation, a, b))):
            raise ValueError("Teacher semantic key fails independent recomputation")
        if source in seen_sources or item.question in seen_questions:
            raise ValueError("semantic lesson contains duplicate source or question")
        seen_sources.add(source)
        seen_questions.add(item.question)
        counts[(language, operation, index)] += 1
        if split == "train":
            train.append((source, item))
        else:
            check.append((language, operation, item))
    if (len(train) != 81 or len(check) != 108
            or any(counts[(language, operation, index)] != (3 if index < 3 else 12)
                   for language in LANGUAGES for operation in OPERATIONS for index in range(4))):
        raise ValueError("semantic lesson needs every balanced public train/check cell")
    return train, check


def _exact_student(base: CognitiveStudent, old_lesson: list[tuple[str, PracticeItem]]) -> G5CognitiveStudent:
    student = G5CognitiveStudent(base)
    for source, item in old_lesson:
        student.arithmetic.learn_verified(item, source_id=source,
                                          key_source="deterministic_integer_arithmetic_v1")
    return student


def search_public(old_lesson: list[tuple[str, PracticeItem]],
                  new_train: list[tuple[str, PracticeItem]],
                  check: list[tuple[str, str, PracticeItem]]) -> tuple[G5CognitiveStudent, dict]:
    base = CognitiveStudent.train(practice_items(seed=20261009, arithmetic_count=240))
    items = [item for _, _, item in check]
    old = _exact_student(base, old_lesson)
    baseline = strict_accuracy(base, items)
    old_score = strict_accuracy(old, items)
    trials = []
    for architecture in ("tfidf-char-logreg", "hashing-char-sgd"):
        for floor in (0.4, 0.5):
            student = _exact_student(base, old_lesson)
            expert = SemanticOperationExpert(architecture=architecture, confidence_floor=floor)
            for source, item in old_lesson + new_train:
                expert.learn_verified(item, source_id=source,
                                      key_source="deterministic_integer_arithmetic_v1")
            expert.fit()
            student.semantic = expert
            score = strict_accuracy(student, items)
            trials.append((student, {"architecture": architecture, "confidence_floor": floor,
                                     "strict_accuracy": round(score, 6),
                                     "semantic_manifest": expert.manifest()}))
    # Fixed tie preference avoids selecting on wall-clock noise.
    selected_student, selected = max(
        trials, key=lambda pair: (pair[1]["strict_accuracy"],
                                  pair[1]["architecture"] == "tfidf-char-logreg",
                                  pair[1]["confidence_floor"]))
    cell_rows = defaultdict(list)
    for language, operation, item in check:
        cell_rows[(language, operation)].append(
            all(selected_student.choose(PracticeItem(item.area, item.question,
                                                    item.options[offset:] + item.options[:offset],
                                                    item.answer))[0] == item.answer
                for offset in range(4)))
    retention = practice_items(seed=20261010, arithmetic_count=60)
    report = {"status": "public_development_only", "promotion_eligible": False,
              "check_items": len(items), "g4_strict_accuracy": round(baseline, 6),
              "first_g5_strict_accuracy": round(old_score, 6),
              "selected": selected, "trials": [entry for _, entry in trials],
              "cells": {f"{language}/{operation}": {"n": len(cell_rows[(language, operation)]),
                        "strict_correct": sum(cell_rows[(language, operation)])}
                        for language in LANGUAGES for operation in OPERATIONS},
              "g4_public_retention_accuracy": round(base.accuracy(retention), 6),
              "g5_public_retention_accuracy": round(selected_student.accuracy(retention), 6),
              "limitation": "Public check guided architecture selection; no transfer or promotion inference."}
    return selected_student, report


def build(*, parent_root: Path, parent_contract: Path, old_lesson_path: Path,
          semantic_lesson_path: Path, out: Path) -> dict:
    import joblib

    version = code_version(ROOT)
    if version.get("dirty") or not version.get("commit"):
        raise ValueError("G5 semantic candidate requires committed, clean Toddler source")
    frozen = json.loads(parent_contract.read_text())
    verify_contract(parent_root, frozen)
    if frozen != current_contract(parent_root):
        raise ValueError("promoted G4 parent contract changed")
    old_lesson = load_teacher_curriculum(_owner_lesson(old_lesson_path))
    train, check = load_semantic_lesson(semantic_lesson_path)
    student, report = search_public(old_lesson, train, check)
    report["original_teacher_lesson_sha256"] = hashlib.sha256(old_lesson_path.read_bytes()).hexdigest()
    report["semantic_teacher_lesson_sha256"] = hashlib.sha256(semantic_lesson_path.read_bytes()).hexdigest()
    out = out.resolve()
    if out.is_relative_to(ROOT):
        raise ValueError("candidate artifact must stay outside the repository")
    out.mkdir(parents=True, mode=0o700, exist_ok=False)
    out.chmod(0o700)
    cognitive_path = out / "cognitive.joblib"
    joblib.dump(student, cognitive_path)
    cognitive_path.chmod(0o600)
    cognitive_sha = hashlib.sha256(cognitive_path.read_bytes()).hexdigest()
    verify_contract(parent_root, frozen)
    children = [{"ref": f"G5-semantic-public/t710{index}",
                 "navigation_parent": parent["ref"],
                 "navigation_parent_weights_sha256": parent["weights_sha256"],
                 "inherited_expert_sha256": parent["expert_sha256"],
                 "inherited_route_sha256": parent["route_sha256"],
                 "cognitive_artifact_sha256": cognitive_sha}
                for index, parent in enumerate(frozen["parents"], start=1)]
    manifest = {"schema": "toddler-g5-semantic-public-candidate/v2",
                "status": "trained_public_semantic_candidate_unpromoted",
                "created_at": datetime.now(timezone.utc).isoformat(), "code": version,
                "parent_contract_sha256": hashlib.sha256(parent_contract.read_bytes()).hexdigest(),
                "cognitive_artifact_sha256": cognitive_sha, "children": children,
                "cognitive_training": report, "private_evaluation": "not_started",
                "promotion_eligible": False}
    fd = os.open(out / "manifest.json", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--parent-contract", type=Path, default=ROOT /
                        "reports/g5_frozen_g4_parents_20261011.json")
    parser.add_argument("--old-lesson", required=True, type=Path)
    parser.add_argument("--semantic-lesson", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    manifest = build(parent_root=args.parent_root, parent_contract=args.parent_contract,
                     old_lesson_path=args.old_lesson,
                     semantic_lesson_path=args.semantic_lesson, out=args.out)
    print(json.dumps({"status": manifest["status"],
                      "public_check": manifest["cognitive_training"]["selected"]["strict_accuracy"],
                      "children": len(manifest["children"]), "promotion_eligible": False}))


if __name__ == "__main__":
    main()
