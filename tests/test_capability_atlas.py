"""The crowdsourcing atlas must retain its stable addresses and honest labels."""

import json
from pathlib import Path

from scripts.capability_atlas import ROOT, export_competencies, validate


def test_materialized_capability_atlas_is_complete():
    report = validate(ROOT)
    assert report["validated_slots"] == report["validated_competencies"] == 10_000
    assert len(list(Path(ROOT).rglob("lesson.md"))) >= 10


def test_specs_are_distinct_and_require_professional_evidence():
    def record(capability_id):
        return json.loads((ROOT / capability_id / "capability.json").read_text())

    a = record("reasoning_and_language/numeracy/implement__few_shot")
    b = record("reasoning_and_language/numeracy/diagnose__distribution_shift")
    c = record("systems_and_safety/deployment_gates/validate__real_time")
    assert len({a["competency"]["outcome"], b["competency"]["outcome"],
                c["competency"]["outcome"]}) == 3
    assert "units-aware" in a["competency"]["practice_task"]
    assert "shift" in b["competency"]["setting_constraint"]
    assert "rollback" in " ".join(c["competency"]["required_evidence"])
    for row in (a, b, c):
        spec = row["competency"]
        assert spec["status"] == "generated_unreviewed"
        assert spec["assessment"]["pass_rule"] == "all_four_criteria_required"
        assert len(spec["assessment"]["criteria"]) == 4
        assert row["mastery_claim"] is False


def test_teacher_export_is_complete_and_excludes_answers(tmp_path):
    destination = tmp_path / "teacher_competencies.jsonl"
    summary = export_competencies(destination)
    assert summary["competency_specs"] == 10_000
    rows = destination.read_text().splitlines()
    assert len(rows) == 10_000
    first = json.loads(rows[0])
    assert first["mastery_claim"] is False
    assert first["competency"]["assessment"]["mode"] == "independent_evaluator_new_case"
    assert "answer_key" not in first and "private_items" not in first
