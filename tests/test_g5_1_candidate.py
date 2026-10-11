import json
import os
from pathlib import Path

import pytest

from scripts.g5_1_candidate import load_g5_1_lesson


def test_g5_1_lesson_loader_rejects_repository_and_accepts_owner_file(tmp_path):
    lesson = {
        "schema": "teacher-g5-1-public-curriculum/v1",
        "status": "public_train_check_not_promotion", "seed": 7011,
        "train_examples": 72, "check_examples": 72,
        "dutch_review": "developer_draft_pending_native_review",
        "teacher_commit": "a" * 40, "items": []}
    path = tmp_path / "lesson.json"
    path.write_text(json.dumps(lesson))
    path.chmod(0o600)
    with pytest.raises(ValueError, match="six balanced"):
        load_g5_1_lesson(path)

    repo_path = Path(__file__).resolve().parents[1] / "tests" / "_g5_1_tmp.json"
    repo_path.write_text(json.dumps(lesson))
    repo_path.chmod(0o600)
    try:
        with pytest.raises(ValueError, match="outside Toddler Git"):
            load_g5_1_lesson(repo_path)
    finally:
        repo_path.unlink()
