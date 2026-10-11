"""Build the G5.1 public arithmetic candidate from the Teacher curriculum.

The public check split selects the architecture only. The resulting artifact is
unpromoted and cannot read evaluator-owned banks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from scripts.g5_parent_contract import DEFAULT_ROOT, current_contract, verify_contract
from scripts.g5_public_candidate import arithmetic_result, load_teacher_curriculum
from scripts.g5_semantic_candidate import _owner_lesson, _exact_student, search_public
from toddler.g4_curriculum import PracticeItem
from toddler.learn.lineage import code_version

ROOT = Path(__file__).resolve().parents[1]
LANGUAGES = ("english", "nederlands")
OPERATIONS = ("add", "subtract", "multiply")


def load_g5_1_lesson(path: Path) -> tuple[list[tuple[str, PracticeItem]],
                                          list[tuple[str, str, PracticeItem]]]:
    lesson = json.loads(_owner_lesson(path).read_text())
    if (lesson.get("schema") != "teacher-g5-1-public-curriculum/v1"
            or lesson.get("status") != "public_train_check_not_promotion"
            or lesson.get("seed") != 7011
            or lesson.get("train_examples") != 72
            or lesson.get("check_examples") != 72
            or lesson.get("dutch_review") != "developer_draft_pending_native_review"
            or not re.fullmatch(r"[0-9a-f]{40}", lesson.get("teacher_commit", ""))):
        raise ValueError("Teacher G5.1 lesson does not match the public contract")
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
                    rf"public-g5-1-7011-{language}-{operation}-{index}-(?:[0-9]|1[01])",
                    source)):
            raise ValueError("G5.1 item has invalid identity or key provenance")
        item = PracticeItem(row["area"], row["question"], tuple(row["options"]), row["answer"])
        numbers = re.findall(r"\d+", item.question)
        if sorted(numbers) != sorted([str(a), str(b)]) or item.answer != str(arithmetic_result(operation, a, b)):
            raise ValueError("G5.1 key fails independent recomputation")
        if source in seen_sources or item.question in seen_questions:
            raise ValueError("G5.1 lesson contains duplicate source or question")
        seen_sources.add(source)
        seen_questions.add(item.question)
        counts[(language, operation, index)] += 1
        if split == "train":
            train.append((source, item))
        else:
            check.append((language, operation, item))
    if (len(train) != 72 or len(check) != 72
            or any(counts[(language, operation, index)] != 4
                   for language in LANGUAGES for operation in OPERATIONS for index in range(3))
            or any(counts[(language, operation, 3)] != 12
                   for language in LANGUAGES for operation in OPERATIONS)):
        raise ValueError("G5.1 lesson needs six balanced train/check cells")
    return train, check


def build(*, parent_root: Path, parent_contract: Path, old_lesson_path: Path,
          semantic_lesson_path: Path, g5_1_lesson_path: Path, out: Path) -> dict:
    import joblib
    version = code_version(ROOT)
    if version.get("dirty") or not version.get("commit"):
        raise ValueError("G5.1 candidate requires committed, clean Toddler source")
    frozen = json.loads(parent_contract.read_text())
    verify_contract(parent_root, frozen)
    if frozen != current_contract(parent_root):
        raise ValueError("promoted G4 parent contract changed")
    old_lesson = load_teacher_curriculum(_owner_lesson(old_lesson_path))
    semantic_train, _ = __import__("scripts.g5_semantic_candidate", fromlist=["load_semantic_lesson"]).load_semantic_lesson(semantic_lesson_path)
    g5_1_train, g5_1_check = load_g5_1_lesson(g5_1_lesson_path)
    student, report = search_public(old_lesson + semantic_train, g5_1_train, g5_1_check)
    report["curriculum"] = "g5-1"
    report["g5_1_teacher_lesson_sha256"] = hashlib.sha256(g5_1_lesson_path.read_bytes()).hexdigest()
    report["semantic_teacher_lesson_sha256"] = hashlib.sha256(semantic_lesson_path.read_bytes()).hexdigest()
    report["original_teacher_lesson_sha256"] = hashlib.sha256(old_lesson_path.read_bytes()).hexdigest()
    report["dutch_review"] = "pending_native_review"
    report["promotion_eligible"] = False
    out = out.resolve()
    if out.is_relative_to(ROOT):
        raise ValueError("candidate artifact must stay outside the source repository")
    out.mkdir(parents=True, mode=0o700, exist_ok=False)
    out.chmod(0o700)
    cognitive_path = out / "cognitive.joblib"
    joblib.dump(student, cognitive_path)
    cognitive_path.chmod(0o600)
    cognitive_sha = hashlib.sha256(cognitive_path.read_bytes()).hexdigest()
    children = [{"ref": f"G5.1-public/t720{index}",
                 "navigation_parent": parent["ref"],
                 "navigation_parent_weights_sha256": parent["weights_sha256"],
                 "inherited_expert_sha256": parent["expert_sha256"],
                 "inherited_route_sha256": parent["route_sha256"],
                 "cognitive_artifact_sha256": cognitive_sha}
                for index, parent in enumerate(frozen["parents"], start=1)]
    manifest = {"schema": "toddler-g5-1-public-candidate/v1",
                "status": "trained_public_g5_1_candidate_unpromoted",
                "created_at": datetime.now(timezone.utc).isoformat(), "code": version,
                "parent_contract_sha256": hashlib.sha256(parent_contract.read_bytes()).hexdigest(),
                "cognitive_artifact_sha256": cognitive_sha, "children": children,
                "cognitive_training": report,
                "private_evaluation": "not_started",
                "dutch_review": "pending_native_review",
                "promotion_eligible": False}
    fd = os.open(out / "manifest.json", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--parent-contract", type=Path,
                        default=ROOT / "reports/g5_frozen_g4_parents_20261011.json")
    parser.add_argument("--old-lesson", required=True, type=Path)
    parser.add_argument("--semantic-lesson", required=True, type=Path)
    parser.add_argument("--g5-1-lesson", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    manifest = build(parent_root=args.parent_root, parent_contract=args.parent_contract,
                     old_lesson_path=args.old_lesson,
                     semantic_lesson_path=args.semantic_lesson,
                     g5_1_lesson_path=args.g5_1_lesson, out=args.out)
    print(json.dumps({"status": manifest["status"],
                      "public_check": manifest["cognitive_training"]["selected"]["strict_accuracy"],
                      "children": len(manifest["children"]), "promotion_eligible": False}))


if __name__ == "__main__":
    main()
