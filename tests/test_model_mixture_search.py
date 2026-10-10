"""Public-only route search and its contamination boundary."""

import copy

import pytest

from toddler.model_mixture_search import search


def report():
    def measured(a, b):
        return {"code": {"quality": a, "latency_s": 2, "gpu_board_wh_per_answer": .01,
                         "decode_tps": 40, "n": 40},
                "reasoning": {"quality": b, "latency_s": 2, "gpu_board_wh_per_answer": .01,
                              "decode_tps": 40, "n": 40}}

    return {"schema": "toddler-mom-public-dev/v1", "split": "public_development",
            "benchmark_policy": "anti_contamination_public_development",
            "promotion_eligible": False, "training_overlap_check": "passed",
            "item_bank_sha256": "a" * 64, "prompt_pack_sha256": "b" * 64,
            "decode_profile_sha256": "c" * 64,
            "tasks": ["code", "reasoning"],
            "models": [
                {"model": "coder", "weights_sha256": "d" * 64, "access": "local",
                 "energy_scope": "gpu_board", "resident_vram_gb": 10,
                 "tasks": measured(.9, .5)},
                {"model": "reasoner", "weights_sha256": "e" * 64, "access": "local",
                 "energy_scope": "gpu_board", "resident_vram_gb": 10,
                 "tasks": measured(.5, .9)},
            ]}


def test_search_finds_complementary_models_and_reports_baselines():
    result = search(report(), vram_budget_gb=20)  # default cap 3; only 2 measured
    assert result["status"] == "research_candidate_only"
    assert result["best_single"]["quality"] == .7
    assert result["random_router_expected_quality"] == .7
    assert result["random_router_expected"]["gpu_board_wh_per_answer"] == .01
    assert result["lowest_energy_at_least_best_single_quality"]["quality"] >= .7
    assert result["candidates"][0]["route"] == {"code": "coder", "reasoning": "reasoner"}
    assert result["candidates"][0]["quality"] == .9
    assert result["candidates"][0]["gpu_board_wh_per_answer"] == .01
    assert result["candidates"][0]["decode_tps_by_task"] == {"code": 40, "reasoning": 40}


def test_private_or_historical_scores_cannot_enter_search():
    for change in ({"split": "private_holdout"}, {"benchmark_policy": "public_full_suite"},
                   {"training_overlap_check": "unknown"}, {"promotion_eligible": True}):
        bad = report()
        bad.update(change)
        with pytest.raises(ValueError):
            search(bad)


def test_missing_energy_or_incompatible_sample_count_is_refused():
    bad = copy.deepcopy(report())
    bad["models"][1]["tasks"]["code"]["n"] = 2
    with pytest.raises(ValueError, match="at least 30"):
        search(bad)
    bad = copy.deepcopy(report())
    del bad["models"][1]["tasks"]["code"]["gpu_board_wh_per_answer"]
    with pytest.raises(ValueError, match="gpu_board_wh_per_answer"):
        search(bad)


def test_budget_can_exclude_two_model_mixture():
    result = search(report(), max_models=2, vram_budget_gb=10)
    assert result["candidates"][0]["quality"] == .7
    assert all(len(row["models"]) == 1 for row in result["candidates"])
