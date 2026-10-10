"""Public-only route search and its contamination boundary."""

import copy
import hashlib
import json

import pytest

from toddler.model_mixture_search import search


def report():
    def identity(task):
        ids = [f"{task}-{index:02d}" for index in range(40)]
        digest = hashlib.sha256(json.dumps(ids, ensure_ascii=False,
                                           separators=(",", ":")).encode()).hexdigest()
        return ids, digest

    def measured(a, b, reverse=False):
        def scores(quality):
            passed = [1.0] * round(quality * 40)
            values = passed + [0.0] * (40 - len(passed))
            return list(reversed(values)) if reverse else values

        code_ids, code_hash = identity("code")
        reasoning_ids, reasoning_hash = identity("reasoning")
        return {"code": {"quality": a, "latency_s": 2, "gpu_board_wh_per_answer": .01,
                         "item_ids": code_ids, "item_ids_sha256": code_hash,
                         "item_scores": scores(a), "decode_tps": 40, "n": 40},
                "reasoning": {"quality": b, "latency_s": 2, "gpu_board_wh_per_answer": .01,
                              "item_ids": reasoning_ids, "item_ids_sha256": reasoning_hash,
                              "item_scores": scores(b), "decode_tps": 40, "n": 40}}

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
                 "tasks": measured(.5, .9, reverse=True)},
            ]}


def test_search_finds_complementary_models_and_reports_baselines():
    result = search(report(), vram_budget_gb=20)  # default cap 3; only 2 measured
    assert result["status"] == "research_candidate_only"
    assert result["best_single"]["quality"] == .7
    assert result["oracle_router_ceiling"] == 1.0
    assert result["oracle_gain_over_best_single"] == .3
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
    bad = copy.deepcopy(report())
    bad["models"][1]["tasks"]["code"]["item_ids"] = list(reversed(bad["models"][1]["tasks"]["code"]["item_ids"]))
    with pytest.raises(ValueError, match="ordered hash"):
        search(bad)


def test_oracle_and_random_baseline_respect_resident_budget():
    data = report()
    data["models"][0]["resident_vram_gb"] = 10
    data["models"][1]["resident_vram_gb"] = 20
    result = search(data, vram_budget_gb=10)
    assert result["random_router_population"] == ["coder"]
    assert result["random_router_expected_quality"] == .7
    assert result["oracle_router_ceiling"] == .7
    assert result["oracle_router_ceiling_unconstrained"] == 1.0
    assert result["oracle_gain_over_best_single"] == 0


def test_no_feasible_model_has_no_feasible_baseline():
    result = search(report(), vram_budget_gb=1)
    assert result["status"] == "no_feasible_route"
    assert result["best_single"] is None
    assert result["random_router_expected_quality"] is None
    assert result["oracle_router_ceiling"] is None


def test_budget_can_exclude_two_model_mixture():
    result = search(report(), max_models=2, vram_budget_gb=10)
    assert result["candidates"][0]["quality"] == .7
    assert all(len(row["models"]) == 1 for row in result["candidates"])
