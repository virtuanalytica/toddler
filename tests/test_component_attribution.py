"""Ablation accounting must refuse unmatched tasks and changed model profiles."""

import copy

import pytest

from toddler.learn.component_attribution import analyze


def fixture():
    profile = "a" * 64
    rows = lambda accepted: [{"item_id": f"item-{i}", "profile_sha256": profile,
                              "accepted": accepted if i % 2 else True,
                              "latency_s": 1.0, "gpu_board_wh": 0.01,
                              "system_kwh": 0.001} for i in range(30)]
    return {"schema": "toddler-component-arms/v1", "split": "public_development",
            "shared_profile_sha256": profile, "randomization_sha256": "b" * 64,
            "applicable_components": ["codegraph"],
            "arms": {"baseline": rows(False), "full": rows(True),
                     "plus/codegraph": rows(True), "minus/codegraph": rows(False)}}


def test_paired_effect_uses_same_tasks_and_keeps_nonpromotion_status():
    result = analyze(fixture())
    assert result["paired_items"] == 30
    assert result["promotion_eligible"] is False
    assert result["standalone_effects"]["codegraph"]["accepted"]["mean_delta"] == 0.5
    assert result["conditional_effects"]["codegraph"]["accepted"]["mean_delta"] == 0.5


def test_changed_task_or_profile_refuses_attribution():
    bad = copy.deepcopy(fixture())
    bad["arms"]["full"][0]["item_id"] = "different"
    with pytest.raises(ValueError, match="identical task IDs"):
        analyze(bad)
    bad = copy.deepcopy(fixture())
    bad["arms"]["full"][0]["profile_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="differs from shared"):
        analyze(bad)
