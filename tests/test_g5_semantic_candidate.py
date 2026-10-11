import json

import pytest

from scripts.g5_semantic_candidate import build, load_semantic_lesson
from toddler.g4_curriculum import PracticeItem
from toddler.g5_cognitive import SemanticOperationExpert, masked_question
from toddler.g5_system import G5Prototype


def _item(question, answer):
    value = int(answer)
    return PracticeItem("arithmetic", question,
                        tuple(str(x) for x in (value, value + 1, value - 1, value + 3)),
                        str(value))


def test_semantic_expert_infers_operations_from_answers_and_can_refit():
    expert = SemanticOperationExpert(confidence_floor=0.0)
    examples = [(_item("What is 31 plus 7?", 38), "public-g5-add"),
                (_item("What is 31 minus 7?", 24), "public-g5-sub"),
                (_item("What is 31 times 7?", 217), "public-g5-mul")]
    for item, source in examples:
        assert expert.learn_verified(item, source_id=source,
                                     key_source="deterministic_integer_arithmetic_v1")
    assert expert.choose(_item("What is 42 plus 6?", 48)) is None
    expert.fit()
    assert expert.choose(_item("What is 42 plus 6?", 48))[0] == "48"
    expert.learn_verified(_item("Add 50 and 6.", 56), source_id="public-g5-new",
                          key_source="deterministic_integer_arithmetic_v1")
    assert expert.choose(_item("What is 42 plus 6?", 48)) is None
    expert.fit()
    assert expert.manifest()["examples"] == 4
    assert masked_question("Add 50 and 6.") == "add <n> and <n>."
    with pytest.raises(ValueError, match="new public"):
        expert.learn_verified(_item("Add 50 and 6.", 56), source_id="public-g5-new",
                              key_source="deterministic_integer_arithmetic_v1")


def test_semantic_expert_rejects_ambiguous_or_unverified_feedback():
    expert = SemanticOperationExpert()
    with pytest.raises(ValueError, match="deterministic verifier"):
        expert.learn_verified(_item("What is 31 plus 7?", 38), source_id="public-g5-x",
                              key_source="judge")
    with pytest.raises(ValueError, match="one arithmetic operation"):
        expert.learn_verified(_item("What is 2 plus 2?", 4), source_id="public-g5-x",
                              key_source="deterministic_integer_arithmetic_v1")


def test_dirty_source_cannot_build_second_candidate(tmp_path, monkeypatch):
    monkeypatch.setattr("scripts.g5_semantic_candidate.code_version",
                        lambda root: {"commit": "a" * 40, "dirty": True})
    with pytest.raises(ValueError, match="committed, clean"):
        build(parent_root=tmp_path, parent_contract=tmp_path / "none",
              old_lesson_path=tmp_path / "none", semantic_lesson_path=tmp_path / "none",
              out=tmp_path / "candidate")


def test_semantic_lesson_and_runtime_reject_wrong_contract(tmp_path):
    lesson = tmp_path / "lesson.json"
    lesson.write_text(json.dumps({"schema": "teacher-g5-semantic-feedback/v1",
                                  "status": "public_train_check_not_promotion",
                                  "seed": 7003, "train_examples": 81,
                                  "check_examples": 108, "teacher_commit": "a" * 40,
                                  "items": []}))
    lesson.chmod(0o600)
    with pytest.raises(ValueError, match="balanced"):
        load_semantic_lesson(lesson)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"schema": "toddler-g5-semantic-public-candidate/v2",
                                    "status": "trained_public_semantic_candidate_unpromoted",
                                    "private_evaluation": "not_started",
                                    "code": {"dirty": True}}))
    with pytest.raises(ValueError, match="clean"):
        G5Prototype.load(manifest, "G5-semantic-public/t7101",
                         parent_root=tmp_path, parent_contract=tmp_path / "none")
