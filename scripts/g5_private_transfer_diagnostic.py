"""One-use, evaluator-owned G5 arithmetic transfer diagnostic.

The question bank belongs outside Git and is never used to select or train a
candidate. This diagnostic cannot promote a generation. It compares the frozen
G4 cognitive head with the exact G5 candidate on paired, rotated answers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import defaultdict
from pathlib import Path

from scipy.stats import binomtest

from scripts.g5_parent_contract import DEFAULT_ROOT
from toddler.g4_curriculum import PracticeItem, practice_items
from toddler.g5_cognitive import parse_form
from toddler.g5_system import G5Prototype

REPO = Path(__file__).resolve().parents[1]
LANGUAGES = ("english", "nederlands", "cantonese")
OPERATIONS = ("add", "subtract", "multiply")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _private_file(path: Path) -> Path:
    path = path.resolve()
    if path.is_relative_to(REPO):
        raise ValueError("private diagnostic files must be outside the repository")
    if path.stat().st_mode & 0o077:
        raise ValueError("private diagnostic files must be owner-only")
    return path


def _result(operation: str, a: int, b: int) -> int:
    if operation == "add":
        return a + b
    if operation == "subtract":
        return a - b
    if operation == "multiply":
        return a * b
    raise ValueError("unknown arithmetic operation")


def load_bank(path: Path, *, manifest_sha256: str, min_per_cell: int = 8) -> list[dict]:
    bank = json.loads(_private_file(path).read_text())
    if (bank.get("schema") != "toddler-g5-arithmetic-transfer-diagnostic/v1"
            or bank.get("status") != "sealed_diagnostic_not_promotion"
            or bank.get("candidate_manifest_sha256") != manifest_sha256):
        raise ValueError("bank is not sealed for this exact G5 candidate")
    public = {item.question.casefold().strip() for item in practice_items()}
    seen_ids, seen_questions, cells, rows = set(), set(), defaultdict(int), []
    for row in bank.get("items", []):
        source_id = row.get("id")
        language, operation = row.get("language"), row.get("operation")
        if (not isinstance(source_id, str) or not source_id.startswith("private-g5-")
                or source_id in seen_ids or language not in LANGUAGES
                or operation not in OPERATIONS or row.get("area") != "arithmetic"
                or row.get("key_source") != "deterministic_integer_arithmetic_v1"):
            raise ValueError("bank item has invalid identity, provenance or category")
        a, b = row.get("a"), row.get("b")
        if not isinstance(a, int) or isinstance(a, bool) or not isinstance(b, int) or isinstance(b, bool):
            raise ValueError("bank operands must be integers")
        item = PracticeItem(row["area"], row["question"], tuple(row["options"]), row["answer"])
        if (item.answer != str(_result(operation, a, b))
                or re.findall(r"\d+", item.question) != [str(a), str(b)]):
            raise ValueError("bank answer or visible operands fail deterministic verification")
        question = item.question.casefold().strip()
        if (question in seen_questions or question in public or parse_form(item) is not None):
            raise ValueError("bank repeats a public question or trained exact form")
        seen_ids.add(source_id)
        seen_questions.add(question)
        cells[(language, operation)] += 1
        rows.append({"id": source_id, "language": language, "operation": operation, "item": item})
    if any(cells[(language, operation)] < min_per_cell
           for language in LANGUAGES for operation in OPERATIONS):
        raise ValueError("each language-operation cell needs enough transfer items")
    return rows


def _strict(model, item: PracticeItem) -> bool:
    return all(model.choose(PracticeItem(item.area, item.question,
                                         item.options[offset:] + item.options[:offset],
                                         item.answer))[0] == item.answer
               for offset in range(4))


def evaluate(*, bank_path: Path, manifest_path: Path, parent_root: Path,
             parent_contract: Path, child_ref: str, out: Path) -> dict:
    manifest_sha = digest(manifest_path)
    rows = load_bank(bank_path, manifest_sha256=manifest_sha)
    if out.exists() or out.resolve().is_relative_to(REPO):
        raise ValueError("diagnostic report must be a new private file outside Git")
    candidate = G5Prototype.load(manifest_path, child_ref, parent_root=parent_root,
                                 parent_contract=parent_contract)
    cells = defaultdict(lambda: {"n": 0, "g4_correct": 0, "g5_correct": 0})
    details = []
    for row in rows:
        g4 = _strict(candidate.cognitive.base, row["item"])
        g5 = _strict(candidate.cognitive, row["item"])
        cell = cells[(row["language"], row["operation"])]
        cell["n"] += 1
        cell["g4_correct"] += int(g4)
        cell["g5_correct"] += int(g5)
        details.append({"id": row["id"], "g4_correct": g4, "g5_correct": g5})
    wins = sum(item["g5_correct"] and not item["g4_correct"] for item in details)
    losses = sum(item["g4_correct"] and not item["g5_correct"] for item in details)
    report = {"schema": "toddler-g5-arithmetic-transfer-result/v1",
              "status": "diagnostic_only_no_promotion",
              "bank_sha256": digest(bank_path), "candidate_manifest_sha256": manifest_sha,
              "child_ref": child_ref, "items": len(details),
              "g4_strict_correct": sum(item["g4_correct"] for item in details),
              "g5_strict_correct": sum(item["g5_correct"] for item in details),
              "g5_only_correct": wins, "g4_only_correct": losses,
              "paired_two_sided_p": binomtest(wins, wins + losses, p=0.5).pvalue
              if wins + losses else 1.0,
              "cells": {f"{language}/{operation}": cells[(language, operation)]
                        for language in LANGUAGES for operation in OPERATIONS},
              "items_private": details,
              "limitations": "Post-freeze transfer diagnostic only; no G5 promotion or other-domain claim."}
    out.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd = os.open(out, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return {key: report[key] for key in ("status", "bank_sha256", "candidate_manifest_sha256",
                                         "items", "g4_strict_correct", "g5_strict_correct",
                                         "g5_only_correct", "g4_only_correct", "paired_two_sided_p")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--parent-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--parent-contract", type=Path, default=REPO /
                        "reports/g5_frozen_g4_parents_20261011.json")
    parser.add_argument("--child-ref", default="G5-public-candidate/t7001")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(evaluate(bank_path=args.bank, manifest_path=args.manifest,
                              parent_root=args.parent_root, parent_contract=args.parent_contract,
                              child_ref=args.child_ref, out=args.out)))


if __name__ == "__main__":
    main()
