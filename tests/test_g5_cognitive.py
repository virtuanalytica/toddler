import json

import pytest

from scripts.g5_public_candidate import (FORM_OPERATION, arithmetic_result,
                                         build, load_teacher_curriculum,
                                         make_public_items, search_public)
from toddler.g4_curriculum import PracticeItem
from toddler.g5_cognitive import FeedbackArithmetic, parse_form
from toddler.g5_system import G5Prototype


def test_verified_feedback_learns_new_form_without_option_position_shortcut():
    learner = FeedbackArithmetic()
    row = make_public_items(seed=9, per_form=1, low=4, high=15)
    source_id, item = next((source, question) for source, question in row
                           if "english_minus" in source)
    assert learner.choose(item) is None
    assert learner.learn_verified(item, source_id=source_id,
                                  key_source="deterministic_integer_arithmetic_v1")
    assert learner.choose(item)[0] == item.answer
    for offset in range(4):
        rotated = PracticeItem(item.area, item.question,
                               item.options[offset:] + item.options[:offset], item.answer)
        assert learner.choose(rotated)[0] == item.answer
    with pytest.raises(ValueError, match="new public"):
        learner.learn_verified(item, source_id=source_id,
                               key_source="deterministic_integer_arithmetic_v1")


def test_unverified_or_contradictory_feedback_rejected():
    learner = FeedbackArithmetic()
    item = PracticeItem("arithmetic", "Wat is 11 keer 7?", ("77", "78", "76", "80"), "77")
    with pytest.raises(ValueError, match="verifier"):
        learner.learn_verified(item, source_id="public-g5-one", key_source="judge-says-so")
    learner.learn_verified(item, source_id="public-g5-one",
                           key_source="deterministic_integer_arithmetic_v1")
    incorrect = PracticeItem("arithmetic", item.question, item.options, "80")
    with pytest.raises(ValueError, match="contradicts"):
        learner.learn_verified(incorrect, source_id="public-g5-two",
                               key_source="deterministic_integer_arithmetic_v1")
    assert learner.accepted_examples == 1


def test_public_search_reports_development_only():
    _, report = search_public()
    assert report["status"] == "public_development_only"
    assert report["promotion_eligible"] is False
    assert report["check_items"] == 108
    assert report["selected"]["public_check_strict_accuracy"] > report["baseline_public_check_strict_accuracy"]
    assert report["g5_public_retention_accuracy"] >= report["g4_public_retention_accuracy"]


def test_teacher_lesson_keys_are_recomputed_before_learning(tmp_path):
    items = make_public_items(seed=7001, per_form=3, low=4, high=15)
    payload = {"schema": "teacher-g5-public-feedback/v1", "seed": 7001,
               "examples_per_form": 3, "status": "public_training_only_not_promotion",
               "teacher_commit": "a" * 40, "items": []}
    for source, item in items:
        form, a, b = parse_form(item)
        assert item.answer == str(arithmetic_result(FORM_OPERATION[form], a, b))
        payload["items"].append({"source_id": source, "form": form, "area": item.area,
                                 "question": item.question, "options": item.options,
                                 "answer": item.answer,
                                 "key_source": "deterministic_integer_arithmetic_v1"})
    path = tmp_path / "lesson.json"
    path.write_text(json.dumps(payload))
    assert len(load_teacher_curriculum(path)) == 27
    payload["items"][0]["answer"] = str(int(payload["items"][0]["answer"]) + 1)
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="recomputed key"):
        load_teacher_curriculum(path)


def test_dirty_training_source_cannot_create_candidate(tmp_path, monkeypatch):
    monkeypatch.setattr("scripts.g5_public_candidate.code_version",
                        lambda root: {"commit": "a" * 40, "dirty": True})
    with pytest.raises(ValueError, match="committed, clean"):
        build(parent_root=tmp_path, parent_contract=tmp_path / "none",
              teacher_curriculum=tmp_path / "none", out=tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_runtime_rejects_dirty_or_promoted_prototype_before_loading(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"schema": "toddler-g5-public-candidate/v1",
                                    "status": "trained_public_candidate_unpromoted",
                                    "private_evaluation": "not_started", "code": {"dirty": True}}))
    with pytest.raises(ValueError, match="clean"):
        G5Prototype.load(manifest, "G5-public-candidate/t7001",
                         parent_root=tmp_path, parent_contract=tmp_path / "contract.json")
