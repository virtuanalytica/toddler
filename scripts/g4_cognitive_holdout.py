"""Score a private G4 cognitive question bank without publishing its prompts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import Counter
from pathlib import Path

from toddler.g4_curriculum import KNOWLEDGE_AREAS, MONTESSORI_AREAS, PracticeItem, practice_items

AREAS = set(KNOWLEDGE_AREAS + MONTESSORI_AREAS)
REPO = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalized(value: str) -> str:
    return re.sub(r"\s+", " ", value.casefold()).strip()


def load_bank(path: Path, min_per_area: int = 3) -> list[PracticeItem]:
    """Keep independently authored prompts outside the training repository."""
    resolved = path.resolve()
    if resolved.is_relative_to(REPO):
        raise ValueError("private questions cannot live in the training repository")
    if resolved.stat().st_mode & 0o077:
        raise ValueError("private question bank must be owner-readable only")
    bank = json.loads(resolved.read_text())
    if bank.get("schema") != "toddler-g4-cognitive-holdout/v1":
        raise ValueError("unknown private question-bank schema")
    public_questions = {normalized(item.question) for item in practice_items(arithmetic_count=0)}
    questions, items = set(), []
    for row in bank.get("items", []):
        if row.get("area") not in AREAS:
            raise ValueError("unknown cognitive area")
        item = PracticeItem(row["area"], row["question"], tuple(row["options"]), row["answer"])
        question = normalized(item.question)
        if question in public_questions or question in questions:
            raise ValueError("duplicate or public-training question in private bank")
        if not row.get("key_source"):
            raise ValueError("each private answer needs an auditable key source")
        questions.add(question)
        items.append(item)
    counts = Counter(item.area for item in items)
    if set(counts) != AREAS or min(counts.values()) < min_per_area:
        raise ValueError(f"every cognitive area needs at least {min_per_area} private questions")
    return items


def evaluate(bank_path: Path, model_path: Path, out: Path) -> dict:
    import joblib

    items = load_bank(bank_path)
    model = joblib.load(model_path)
    area_rows: dict[str, list[tuple[bool, float]]] = {area: [] for area in AREAS}
    for item in items:
        # Four cyclic rotations expose any position-dependent shortcut.
        correct = []
        for offset in range(4):
            options = item.options[offset:] + item.options[:offset]
            rotated = PracticeItem(item.area, item.question, options, item.answer)
            correct.append(model.choose(rotated)[0] == item.answer)
        area_rows[item.area].append((all(correct), sum(correct) / 4))
    area_scores = {area: {"n": len(rows),
                          "strict_accuracy": round(sum(row[0] for row in rows) / len(rows), 4),
                          "rotation_mean": round(sum(row[1] for row in rows) / len(rows), 4)}
                   for area, rows in sorted(area_rows.items())}
    result = {"status": "diagnostic_only", "promotion": "not_evaluated",
              "bank_sha256": digest(bank_path), "model_sha256": digest(model_path),
              "items": len(items), "areas": area_scores,
              "strict_accuracy": round(sum(row[0] for rows in area_rows.values() for row in rows) / len(items), 4),
              "warning": "private pilot bank; answer provenance and larger matched trials required for promotion"}
    out.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd = os.open(out, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.bank, args.model, args.out), indent=2))


if __name__ == "__main__":
    main()
