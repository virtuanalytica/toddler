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


def mixture_row():
    item = row()
    item["schema"] = "preschooler-evidence/v2"
    del item["answer"]
    del item["model"]
    item["candidates"] = [
        {"candidate_id": "a", "answer": "Result verified by test.",
         "model": {"id": "qwen-instruct", "family": "qwen", "access": "local",
                   "revision": "a" * 40,
                   "weights_sha256": "1" * 64, "license_id": "apache-2.0",
                   "output_training_allowed": True}},
        {"candidate_id": "b", "answer": "A different implementation.",
         "model": {"id": "phi-mini", "family": "phi", "access": "local",
                   "revision": "b" * 40,
                   "weights_sha256": "2" * 64, "license_id": "mit",
                   "output_training_allowed": True}},
    ]
    item["selection"] = {"method": "chosen", "contributor_ids": ["a"],
                         "answer": "Result verified by test."}
    return item


def test_multi_family_candidates_retain_provenance_and_only_selected_answer():
    output, manifest = prepare([mixture_row()], protected_hashes={"e" * 64},
                               holdout_commit="f" * 64, min_teacher_models=2)
    assert output[0]["messages"][1]["content"] == "Result verified by test."
    assert len(output[0]["metadata"]["candidate_models"]) == 2
    assert output[0]["metadata"]["selected_candidate_ids"] == ["a"]
    assert len(manifest["model_weights_sha256"]) == 2


def test_multi_teacher_rights_selection_and_diversity_are_enforced():
    for mutate in (
        lambda x: x["candidates"][1]["model"].update(output_training_allowed=False),
        lambda x: x["candidates"][1]["model"].update(family="qwen"),
        lambda x: x["selection"].update(contributor_ids=["unknown"]),
        lambda x: x["selection"].update(answer="Unverified changed text"),
        lambda x: x["verification"].update(reviewer_id="phi-mini"),
    ):
        item = mixture_row()
        mutate(item)
        with pytest.raises(ValueError):
            prepare([item], protected_hashes={"e" * 64}, holdout_commit="f" * 64)
    with pytest.raises(ValueError, match="teacher models"):
        prepare([mixture_row()], protected_hashes={"e" * 64},
                holdout_commit="f" * 64, min_teacher_models=10)


def test_mixture_export_requires_matching_rights_approved_roster():
    item = mixture_row()
    roster = {
        "qwen-instruct": {"revision": "a" * 40, "license_id": "apache-2.0",
                            "family": "qwen", "output_training_rights": "approved"},
        "phi-mini": {"revision": "b" * 40, "license_id": "mit",
                     "family": "phi", "output_training_rights": "pending_per_model_review"},
    }
    with pytest.raises(ValueError, match="rights-approved roster"):
        prepare([item], protected_hashes={"e" * 64}, holdout_commit="f" * 64,
                teacher_roster=roster)
    roster["phi-mini"]["output_training_rights"] = "approved"
    output, _ = prepare([item], protected_hashes={"e" * 64}, holdout_commit="f" * 64,
                        teacher_roster=roster)
    assert output[0]["metadata"]["candidate_models"][1]["model_revision"] == "b" * 40
