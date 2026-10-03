import pytest

from toddler import config, specialize

CFG = {
    "schema": "toddler-config/v1",
    "role": {"name": "Data Steward", "family": "data-governance", "level": "senior", "tasks": ["dq-check"]},
    "guardrails": [{"id": "no-pii-export", "reason": "never export personal data", "owner": "compliance@acme.nl",
                    "forbidden_tags": ["export-pii"]}],
    "jev_rules": [{"qid": "spill_risk", "text": "Will it spill?", "slow_if_above": 0.2, "stop_if_above": 0.4}],
    "experts": [{"model_id": "dq-small", "domains": ["dq"], "quality": 0.82, "eur_per_1k_tokens": 0.001,
                 "joules_per_1k_tokens": 40, "evidence": "mlflow run 42"}],
    "audiences": [{"name": "compliance", "event_types": ["guardrails_applied", "stop_fired"], "role": None}],
}


def test_loads_gui_config():
    d = config.from_dict(CFG)
    assert d.spec.role.name == "Data Steward"
    assert any(r.rule_id == "company:no-pii-export" for r in d.spec.stop_rules())
    assert d.audiences[0].name == "compliance"


def test_rejects_wrong_schema():
    with pytest.raises(ValueError):
        config.from_dict({**CFG, "schema": "other"})


def test_rejects_weakened_base_question_even_from_a_modified_gui():
    bad = {**CFG, "jev_rules": [{"qid": "person_in_zone", "text": "x", "slow_if_above": 0.9, "stop_if_above": 0.95}]}
    with pytest.raises(specialize.UnsafeSpecialisation):
        config.from_dict(bad)


def test_rejects_inverted_thresholds():
    bad = {**CFG, "jev_rules": [{"qid": "q", "text": "x", "slow_if_above": 0.6, "stop_if_above": 0.4}]}
    with pytest.raises(specialize.UnsafeSpecialisation):
        config.from_dict(bad)
