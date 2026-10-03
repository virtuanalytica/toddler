import dataclasses

import pytest

from toddler import audit, objective, specialize as sp, stop


def _log():
    log = audit.AuditLog()
    log.append(1000, "toddler", "task_done", {"role": "data-steward", "task": "dq-check"})
    log.append(2000, "operator", "guardrails_applied", {"role": "data-steward", "rules": ["company:no-pii-export"]})
    log.append(3000, "toddler", "stop_fired", {"role": "finance", "rule": "no-secrets", "email": "x@y.nl"})
    return log


def test_chain_verifies_and_detects_tampering():
    log = _log()
    assert log.verify() == (True, None)
    log.entries[1] = dataclasses.replace(log.entries[1], details={"role": "data-steward", "rules": []})
    assert log.verify() == (False, 1)


def test_prune_keeps_chain_verifiable():
    log = _log()
    assert log.prune_before(2500) == 2
    assert log.verify() == (True, None)
    log.append(4000, "toddler", "task_done", {"role": "finance"})
    assert log.verify() == (True, None) and log.entries[-1].seq == 3


def test_secrets_never_enter_the_trail():
    with pytest.raises(ValueError):
        audit.AuditLog().append(1, "toddler", "x", {"note": "api_key=sk-123"})


def test_scoped_view_filters_and_masks():
    compliance = audit.Audience("compliance", frozenset({"guardrails_applied", "stop_fired"}))
    rows = audit.view(_log(), compliance)
    assert [r["event_type"] for r in rows] == ["guardrails_applied", "stop_fired"]
    assert rows[1]["details"]["email"] == "***" and len(rows[1]["row_hash"]) == 64
    deployer = audit.Audience("deployer", frozenset({"task_done", "stop_fired"}), role="data-steward")
    assert [r["event_type"] for r in audit.view(_log(), deployer)] == ["task_done"]


def test_company_guardrail_vetoes_and_is_recorded():
    g = sp.company_guardrail("no-pii-export", "never export personal data", "compliance@acme.nl",
                             frozenset({"export-pii"}))
    spec = sp.build(sp.Role("Data Steward", "data-governance"), extra_stop_rules=[g])
    cand = objective.Candidate(stop.Action("export", tags=frozenset({"export-pii"})), 5, 0, 0, 0, 0, 0)
    decision = objective.choose([cand], rules=spec.stop_rules())
    assert decision.chosen is None and decision.vetoed[0][1] == ("company:no-pii-export",)
    log = audit.AuditLog()
    e = audit.record_guardrails(log, 1, "operator", "data-steward", [g.rule_id])
    assert e.details["rules"] == ["company:no-pii-export"]
