"""Score paired, independently verified Toddler work pilots.

The input contains identifiers, digests and measurements, never task prompts or
customer artifacts. Development data are intentionally refused by this scorer.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path


ARMS = frozenset({"incumbent", "toddler_teacher", "toddler_teacher_agent"})
SOURCES = frozenset({"sealed_private", "customer_pilot"})


@dataclass(frozen=True)
class WorkResult:
    experiment_id: str
    task_set_id: str
    task_id: str
    source: str
    arm: str
    worker_id: str
    verifier_id: str
    artifact_sha256: str
    accepted: bool
    critical_incident: bool
    review_minutes: float
    elapsed_seconds: float
    direct_cost_eur: float
    gpu_board_wh: float | None
    system_kwh: float | None

    @classmethod
    def from_dict(cls, row: dict) -> "WorkResult":
        expected = set(cls.__dataclass_fields__)
        if set(row) != expected:
            raise ValueError(f"work result fields differ: missing={sorted(expected-set(row))}, extra={sorted(set(row)-expected)}")
        for key in ("experiment_id", "task_set_id", "task_id", "worker_id", "verifier_id"):
            if not isinstance(row[key], str) or not row[key].strip():
                raise ValueError(f"{key} must be a nonempty string")
        if not isinstance(row["source"], str) or row["source"] not in SOURCES:
            raise ValueError("only sealed_private or customer_pilot evidence is eligible")
        if not isinstance(row["arm"], str) or row["arm"] not in ARMS:
            raise ValueError(f"unknown arm: {row['arm']}")
        if row["worker_id"] == row["verifier_id"]:
            raise ValueError("worker cannot verify its own result")
        digest = row["artifact_sha256"]
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("artifact_sha256 must be a lowercase SHA-256 digest")
        for key in ("accepted", "critical_incident"):
            if type(row[key]) is not bool:
                raise ValueError(f"{key} must be boolean")
        for key in ("review_minutes", "elapsed_seconds", "direct_cost_eur", "gpu_board_wh", "system_kwh"):
            value = row[key]
            if value is None and key in ("gpu_board_wh", "system_kwh"):
                continue
            if type(value) not in (int, float) or not (0 <= value < float("inf")):
                raise ValueError(f"{key} must be a finite nonnegative measurement or allowed null")
        return cls(**row)


def load_results(path: Path) -> list[WorkResult]:
    results: list[WorkResult] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError("row must be an object")
            results.append(WorkResult.from_dict(row))
        except (json.JSONDecodeError, ValueError) as error:
            raise ValueError(f"{path}:{line_number}: {error}") from error
    return results


def _metric(rows: list[WorkResult]) -> dict:
    accepted = sum(row.accepted for row in rows)
    review_minutes = sum(row.review_minutes for row in rows)
    direct_cost = sum(row.direct_cost_eur for row in rows)
    measured_gpu = [row.gpu_board_wh for row in rows if row.gpu_board_wh is not None]
    measured_system = [row.system_kwh for row in rows if row.system_kwh is not None]
    return {
        "tasks": len(rows),
        "accepted": accepted,
        "acceptance_rate": accepted / len(rows),
        "critical_incidents": sum(row.critical_incident for row in rows),
        "review_minutes_per_accepted": review_minutes / accepted if accepted else None,
        "direct_cost_eur_per_accepted": direct_cost / accepted if accepted else None,
        "gpu_board_wh_per_accepted": sum(measured_gpu) / accepted if accepted and len(measured_gpu) == len(rows) else None,
        "system_kwh_per_accepted": sum(measured_system) / accepted if accepted and len(measured_system) == len(rows) else None,
        "gpu_energy_coverage": len(measured_gpu) / len(rows),
        "system_energy_coverage": len(measured_system) / len(rows),
    }


def score_paired(results: list[WorkResult], candidate: str) -> dict:
    if candidate not in ARMS - {"incumbent"}:
        raise ValueError("candidate must be a Toddler arm")
    if not results:
        raise ValueError("no pilot results")
    experiments = {r.experiment_id for r in results}
    task_sets = {r.task_set_id for r in results}
    sources = {r.source for r in results}
    if len(experiments) != 1 or len(task_sets) != 1 or len(sources) != 1:
        raise ValueError("one scorecard must use one experiment, task set and evidence source")
    rows: dict[tuple[str, str], WorkResult] = {}
    for result in results:
        key = (result.task_id, result.arm)
        if key in rows:
            raise ValueError(f"duplicate task/arm: {key}")
        rows[key] = result
    baseline_ids = {task for task, arm in rows if arm == "incumbent"}
    candidate_ids = {task for task, arm in rows if arm == candidate}
    if not baseline_ids or baseline_ids != candidate_ids:
        raise ValueError("incumbent and candidate must have exactly the same task IDs")
    if any(arm not in {"incumbent", candidate} for _, arm in rows):
        raise ValueError("score one candidate against incumbent at a time")
    ordered = sorted(baseline_ids)
    baseline = [rows[task, "incumbent"] for task in ordered]
    contender = [rows[task, candidate] for task in ordered]
    diffs = [int(c.accepted) - int(b.accepted) for b, c in zip(baseline, contender)]
    rng = random.Random(0)
    bootstrap = sorted(sum(rng.choices(diffs, k=len(diffs))) / len(diffs) for _ in range(5000))
    low, high = bootstrap[124], bootstrap[4874]
    base_metrics = _metric(baseline)
    candidate_metrics = _metric(contender)
    review_base = base_metrics["review_minutes_per_accepted"]
    review_candidate = candidate_metrics["review_minutes_per_accepted"]
    cost_base = base_metrics["direct_cost_eur_per_accepted"]
    cost_candidate = candidate_metrics["direct_cost_eur_per_accepted"]
    gate = "insufficient_pairs" if len(ordered) < 100 else "failed"
    if (len(ordered) >= 100 and low > -0.05 and
            candidate_metrics["critical_incidents"] == 0 and
            review_base is not None and review_candidate is not None and review_candidate <= 0.8 * review_base and
            cost_base is not None and cost_candidate is not None and cost_candidate <= cost_base):
        gate = "passed"
    manifest = json.dumps([r.__dict__ for r in baseline + contender], sort_keys=True, separators=(",", ":"))
    return {
        "schema": "toddler-work-pilot-scorecard/v1",
        "experiment_id": results[0].experiment_id,
        "task_set_id": results[0].task_set_id,
        "source": results[0].source,
        "candidate": candidate,
        "paired_tasks": len(ordered),
        "evidence_sha256": hashlib.sha256(manifest.encode()).hexdigest(),
        "incumbent": base_metrics,
        "candidate_metrics": candidate_metrics,
        "paired_acceptance_delta": sum(diffs) / len(diffs),
        "paired_acceptance_delta_ci95": [low, high],
        "gate": gate,
        "gate_rule": "N>=100; paired 95% bootstrap lower bound > -0.05; zero critical incidents; <=80% incumbent review minutes per accepted task; direct cost per accepted <= incumbent",
    }
