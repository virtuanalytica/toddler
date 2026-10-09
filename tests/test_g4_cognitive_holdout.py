import json
import os

import pytest

from scripts import g4_cognitive_holdout as H


def _bank(path, question="Independent question"):
    rows = []
    for area in H.AREAS:
        for index in range(3):
            rows.append({"area": area, "question": f"{question} {area} {index}?",
                         "options": ["correct", "wrong A", "wrong B", "wrong C"],
                         "answer": "correct", "key_source": "test key"})
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "w") as handle:
        json.dump({"schema": "toddler-g4-cognitive-holdout/v1", "items": rows}, handle)


def test_private_bank_requires_every_area_and_unique_unseen_questions(tmp_path):
    path = tmp_path / "bank.json"
    _bank(path)
    assert len(H.load_bank(path)) == 54
    data = json.loads(path.read_text())
    data["items"][0]["question"] = data["items"][1]["question"]
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="duplicate"):
        H.load_bank(path)


def test_private_bank_rejects_public_training_prompt(tmp_path):
    path = tmp_path / "bank.json"
    _bank(path)
    data = json.loads(path.read_text())
    data["items"][0]["question"] = "Which word means a place to read books?"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="public-training"):
        H.load_bank(path)
