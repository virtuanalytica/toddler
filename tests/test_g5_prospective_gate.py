import json
from pathlib import Path

import pytest

from scripts.g5_prospective_gate import (LANGUAGES, OPERATIONS, TASKS, assess,
                                         digest, load_bank, reject_question_overlap, run)


def _write(path, data):
    path.write_text(json.dumps(data))
    path.chmod(0o600)
    return path


def _bank(path, manifest_sha, family):
    rows = []
    for language in LANGUAGES:
        for operation in OPERATIONS:
            for index in range(12):
                a, b = 40 + index, 4 + index
                answer = {"add": a + b, "subtract": a - b, "multiply": a * b}[operation]
                rows.append({"id": f"private-g5-confirm-{family}-{language}-{operation}-{index}",
                             "language": language, "operation": operation, "area": "arithmetic",
                             "a": a, "b": b,
                             "question": f"For {language} {operation}, calculate {a} with {b} in "
                                         f"{'alpha' if family == '0' else 'beta'}.",
                             "options": [str(x) for x in (answer, answer + 1,
                                                           answer - 1, answer + 3)],
                             "answer": str(answer),
                             "key_source": "deterministic_integer_arithmetic_v1"})
    _write(path, {"schema": "toddler-g5-prospective-arithmetic/v1",
                  "status": "sealed_confirmation", "candidate_manifest_sha256": manifest_sha,
                  "items": rows})
    return rows


def _fixture(tmp_path, *, pass_second=True, make_second=True):
    tmp_path.chmod(0o700)
    manifest = tmp_path / "manifest.json"
    parent = tmp_path / "parent.json"
    manifest.write_text("frozen candidate")
    parent.write_text("frozen parents")
    manifest.chmod(0o600)
    parent.chmod(0o600)
    banks = [tmp_path / f"bank-{i}.json" for i in range(2)]
    raw_rows = [_bank(path, digest(manifest), str(i)) for i, path in enumerate(banks)]
    exclusions = []
    for index in range(4):
        path = _write(tmp_path / f"excluded-{index}.json",
                      {"items": [{"question": f"Past public question {index}?"}]})
        exclusions.append({"path": str(path), "sha256": digest(path), "items": 1})
    refs = [f"G5-semantic-public/t710{i}" for i in range(1, 6)]
    protocol = tmp_path / "protocol.json"
    _write(protocol, {"schema": "toddler-g5-prospective-arithmetic/v1",
                      "candidate_manifest": str(manifest),
                      "candidate_manifest_sha256": digest(manifest),
                      "parent_contract": str(parent), "parent_contract_sha256": digest(parent),
                      "banks": [{"path": str(path), "sha256": digest(path)} for path in banks],
                      "excluded_question_files": exclusions,
                      "navigation_seed_commitments": [{"set_id": f"secret-{i}"} for i in range(2)],
                      "evaluator_sha256": digest(Path(__file__).parents[1] /
                                                   "scripts/g5_prospective_gate.py"),
                      "child_refs": refs, "scope": "English and Dutch arithmetic only"})
    for trial_index, bank_rows in enumerate(raw_rows):
        if trial_index == 1 and not make_second:
            continue
        correct_g5 = 8 if trial_index == 0 or pass_second else 4
        cognitive = [{"id": row["id"], "language": row["language"],
                      "operation": row["operation"],
                      "g4_correct": index % 12 < 4,
                      "g5_correct": index % 12 < correct_g5}
                     for index, row in enumerate(bank_rows)]
        navigation = {ref: {task: {"g4": [1.0] * 8, "g5": [1.0] * 8}
                            for task in TASKS} for ref in refs}
        _write(tmp_path / f"trial-{trial_index}.json",
               {"schema": "toddler-g5-prospective-arithmetic-result/v1",
                "protocol_sha256": digest(protocol), "trial_index": trial_index,
                "bank_sha256": digest(banks[trial_index]),
                "seed_set_id": f"secret-{trial_index}", "cognitive": cognitive,
                "navigation": navigation})
    return protocol


def test_bank_recomputes_keys_and_rejects_exact_training_form(tmp_path):
    manifest_sha = "a" * 64
    path = tmp_path / "bank.json"
    rows = _bank(path, manifest_sha, "one")
    assert len(load_bank(path, manifest_sha)) == 72
    rows[0]["answer"] = rows[0]["options"][1]
    _write(path, {"schema": "toddler-g5-prospective-arithmetic/v1",
                  "status": "sealed_confirmation", "candidate_manifest_sha256": manifest_sha,
                  "items": rows})
    with pytest.raises(ValueError, match="answer"):
        load_bank(path, manifest_sha)


def test_two_passed_trials_remain_pending_semantic_review(tmp_path):
    protocol = _fixture(tmp_path)
    result = assess(protocol)
    assert result["status"] == "evidence_passed_pending_semantic_review"
    assert result["generation_promoted"] is False
    assert all(row["evidence_passed"] and row["navigation_exact_retention"]
               for row in result["trials"])


def test_failure_and_interruption_cannot_promote(tmp_path):
    failed = tmp_path / "failed"
    failed.mkdir(mode=0o700)
    result = assess(_fixture(failed, pass_second=False))
    assert result["status"] == "failed_hold_no_promotion"
    interrupted = tmp_path / "interrupted"
    interrupted.mkdir(mode=0o700)
    protocol = _fixture(interrupted, make_second=False)
    with pytest.raises(FileNotFoundError):
        assess(protocol)
    assert not (interrupted / "assessment.json").exists()


def test_frozen_manifest_mutation_is_rejected(tmp_path):
    protocol = _fixture(tmp_path)
    (tmp_path / "manifest.json").write_text("changed")
    with pytest.raises(ValueError, match="frozen candidate"):
        assess(protocol)


def test_completed_trial_reuses_same_result(tmp_path):
    protocol = _fixture(tmp_path)
    cached = run(protocol, 0, tmp_path)
    assert cached["status"] == "trial_already_complete"
    assert cached["trial_sha256"] == digest(tmp_path / "trial-0.json")


def test_exact_question_contamination_is_rejected(tmp_path):
    manifest_sha = "b" * 64
    banks = []
    for index in range(2):
        path = tmp_path / f"bank-{index}.json"
        _bank(path, manifest_sha, str(index))
        banks.append(load_bank(path, manifest_sha))
    excluded = []
    for index in range(4):
        question = banks[0][0]["item"].question if index == 0 else f"unrelated question {index}"
        excluded.append(_write(tmp_path / f"exclude-{index}.json",
                               {"items": [{"question": question}]}))
    with pytest.raises(ValueError, match="repeats a training"):
        reject_question_overlap((banks[0], banks[1]), tuple(excluded))
