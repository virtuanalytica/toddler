import pytest

from toddler.work_pilot import WorkResult, score_paired


def row(task: str, arm: str, *, accepted: bool = True, review: float = 10.0,
        cost: float = 2.0, energy: float | None = 0.5) -> WorkResult:
    return WorkResult.from_dict({
        "experiment_id": "pilot-1", "task_set_id": "sealed-set-1", "task_id": task,
        "source": "sealed_private", "arm": arm, "worker_id": arm,
        "verifier_id": "independent-reviewer", "artifact_sha256": "a" * 64,
        "accepted": accepted, "critical_incident": False,
        "review_minutes": review, "elapsed_seconds": 120.0,
        "direct_cost_eur": cost, "gpu_board_wh": energy, "system_kwh": None,
    })


def test_paired_gate_uses_accepted_work_and_reports_missing_system_energy() -> None:
    records = [result for task in range(100)
               for result in (row(str(task), "incumbent"),
                              row(str(task), "toddler_teacher", review=5, cost=1))]
    result = score_paired(records, "toddler_teacher")
    assert result["gate"] == "passed"
    assert result["paired_acceptance_delta"] == 0
    assert result["candidate_metrics"]["direct_cost_eur_per_accepted"] == 1
    assert result["candidate_metrics"]["system_kwh_per_accepted"] is None
    assert result["candidate_metrics"]["system_energy_coverage"] == 0


def test_rejects_unpaired_tasks_and_development_evidence() -> None:
    with pytest.raises(ValueError, match="same task IDs"):
        score_paired([row("1", "incumbent"), row("2", "toddler_teacher")], "toddler_teacher")
    invalid = dict(row("1", "incumbent").__dict__)
    invalid["source"] = "public_development"
    with pytest.raises(ValueError, match="only sealed_private"):
        WorkResult.from_dict(invalid)


def test_rejects_self_verification_and_duplicate_result() -> None:
    invalid = dict(row("1", "incumbent").__dict__)
    invalid["verifier_id"] = invalid["worker_id"]
    with pytest.raises(ValueError, match="own result"):
        WorkResult.from_dict(invalid)
    records = [row("1", "incumbent"), row("1", "incumbent"), row("1", "toddler_teacher")]
    with pytest.raises(ValueError, match="duplicate"):
        score_paired(records, "toddler_teacher")
