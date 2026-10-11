"""Contamination and provenance failures that must stop a whole SFT export."""

import copy

import pytest

from toddler.learn.preschooler_data import digest, normalized_prompt, prepare


def row(item_id="1", split="train", family="source-a"):
    return {"schema": "preschooler-evidence/v1", "item_id": item_id,
            "split": split, "domain": "software", "source_family": family,
            "source": {"access": "owned", "training_allowed": True,
                       "license_id": "own-work", "sha256": "a" * 64},
            "prompt": f"Explain this new code path {item_id}", "answer": "Result verified by test.",
            "model": {"id": "local-model", "access": "local", "weights_sha256": "b" * 64,
                      "license_id": "permissive-local", "output_training_allowed": True},
            "verification": {"passed": True, "method": "objective", "reviewer_id": "pytest",
                             "evidence_sha256": "c" * 64},
            "overlap_audit": {"status": "passed", "report_sha256": "d" * 64}}


def test_local_verified_examples_export_without_promotion_claim():
    output, manifest = prepare([row(), row("2", "development", "source-b")],
                               protected_hashes={"e" * 64}, holdout_commit="f" * 64)
    assert manifest["promotion_eligible"] is False
    assert manifest["counts"] == {"development/software": 1, "train/software": 1}
    assert output[0]["messages"][1]["content"] == "Result verified by test."


@pytest.mark.parametrize("change", [
    lambda x: x["model"].update(access="cloud"),
    lambda x: x["model"].update(output_training_allowed=False),
    lambda x: x["source"].update(training_allowed=False),
    lambda x: x["verification"].update(reviewer_id="local-model"),
    lambda x: x["verification"].update(passed=False),
    lambda x: x["overlap_audit"].update(status="unknown"),
    lambda x: x.update(split="private_holdout"),
])
def test_rejects_unlicensed_unverified_cloud_or_holdout(change):
    bad = copy.deepcopy(row())
    change(bad)
    with pytest.raises(ValueError):
        prepare([bad], protected_hashes={"e" * 64}, holdout_commit="f" * 64)


def test_protected_prompt_and_cross_split_source_are_rejected():
    current = row()
    protected = digest(normalized_prompt(current["prompt"]))
    with pytest.raises(ValueError, match="protected prompt overlap"):
        prepare([current], protected_hashes={protected}, holdout_commit="f" * 64)
    with pytest.raises(ValueError, match="source family crosses"):
        prepare([current, row("2", "development", "source-a")],
                protected_hashes={"e" * 64}, holdout_commit="f" * 64)
