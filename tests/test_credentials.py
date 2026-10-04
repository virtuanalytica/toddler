from datetime import datetime, timedelta, timezone

import pytest

from toddler import audit
from toddler.credentials import identities as ids
from toddler.credentials import providers, rotation, vault


def test_identity_requires_valid_email_and_approver():
    with pytest.raises(ValueError):
        ids.Identity("not-an-email", "manual", "x", "operator")
    with pytest.raises(ValueError):
        ids.Identity("a@b.nl", "manual", "x", "")


def test_one_identity_per_provider():
    reg = ids.IdentityRegistry()
    reg.add(ids.Identity("toddler-cf@example.nl", "transip-mailbox", "cloudflare", "operator"))
    reg.add(ids.Identity("toddler-oa@example.nl", "simplelogin", "openai", "operator"))
    reg.assign("cloudflare", "toddler-cf@example.nl")
    reg.assign("openai", "toddler-oa@example.nl")
    with pytest.raises(ValueError):
        reg.assign("google", "toddler-cf@example.nl")
    assert reg.for_provider("openai").backend == "simplelogin"


def test_plus_address():
    assert ids.plus_address("toddler+old@example.com", "cloudflare") == "toddler+cloudflare@example.com"


def test_secret_path_scheme_and_injection_guard():
    assert vault.secret_path("t@ex.nl", "openai", "svc-1") == "toddler/t@ex.nl/openai/svc-1"
    with pytest.raises(ValueError):
        vault.secret_path("t@ex.nl", "../etc", "x")


def test_vault_refuses_without_token(monkeypatch):
    monkeypatch.delenv("TODDLER_VAULT_TOKEN", raising=False)
    with pytest.raises(RuntimeError):
        vault.OpenBaoKV()


def test_key_creation_needs_operator_approval_and_refusal_is_audited():
    log = audit.AuditLog()
    with pytest.raises(providers.ApprovalRequired):
        providers.create_openai_service_account("proj", "toddler", admin_key="unused", approved=False,
                                                audit_log=log)
    assert [e.event_type for e in log.entries] == ["key_creation_refused"]
    assert log.entries[0].details["provider"] == "openai"


def test_key_creation_without_audit_log_is_refused_before_any_call():
    with pytest.raises(TypeError):
        providers.create_openai_service_account("proj", "toddler", admin_key="unused", approved=True)


def test_created_key_is_audited_without_its_secret():
    class IAM:                                   # stands in for boto3's IAM client; no network
        def create_access_key(self, UserName):
            return {"AccessKey": {"AccessKeyId": "key-id-1", "SecretAccessKey": "do-not-log-this-value"}}

    log = audit.AuditLog()
    key = providers.create_aws_access_key("toddler", approved=True, iam_client=IAM(), audit_log=log, actor="op")
    assert key.secret == "do-not-log-this-value"
    (entry,) = log.entries
    assert entry.event_type == "key_created" and entry.actor == "op"
    assert entry.details["key_id"] == "key-id-1" and entry.details["provider"] == "aws"
    assert "do-not-log-this-value" not in entry.to_json()


@pytest.mark.parametrize("bad", [".", "..", "a/b", ""])
def test_secret_path_rejects_traversal_segments(bad):
    with pytest.raises(ValueError):
        vault.secret_path("t@example.nl", bad, "k")
    assert vault.secret_path("t@example.nl", "openai", "k.v1") == "toddler/t@example.nl/openai/k.v1"


def test_reassigning_a_provider_must_be_explicit():
    reg = ids.IdentityRegistry()
    reg.add(ids.Identity("toddler-a@example.nl", "manual", "x", "operator"))
    reg.add(ids.Identity("toddler-b@example.nl", "manual", "x", "operator"))
    reg.assign("openai", "toddler-a@example.nl")
    reg.assign("openai", "toddler-a@example.nl")                 # same binding: idempotent
    with pytest.raises(ValueError):
        reg.assign("openai", "toddler-b@example.nl")
    reg.assign("openai", "toddler-b@example.nl", replace=True)
    assert reg.for_provider("openai").email == "toddler-b@example.nl"


def test_human_only_providers_become_tasks():
    task = providers.plan_creation("transip", "toddler@example.nl")
    assert isinstance(task, providers.HumanTask) and "control panel" in task.instruction
    assert providers.plan_creation("cloudflare", "toddler@example.nl") == "cloudflare"


def test_rotation_due_and_ordered():
    st = rotation.KeyState("cloudflare", "t@ex.nl", "k", datetime(2026, 1, 1, tzinfo=timezone.utc), timedelta(days=30))
    steps = rotation.plan(st, now=datetime(2026, 3, 1, tzinfo=timezone.utc))
    assert steps.index("verify") < steps.index("revoke_old")
    assert rotation.plan(st, now=datetime(2026, 1, 5, tzinfo=timezone.utc)) == ()


def test_rotation_stops_before_revoke_when_verify_fails():
    calls = []

    def ok(name):
        return lambda: calls.append(name) or name

    def fail():
        raise RuntimeError("new key rejected")

    handlers = {"create": ok("create"), "store": ok("store"), "verify": fail, "swap": ok("swap"), "revoke_old": ok("revoke_old")}
    res = rotation.execute(rotation.STEPS, handlers)
    assert [r.step for r in res] == ["create", "store", "verify"] and not res[-1].ok
    assert "revoke_old" not in calls


def test_rotation_refuses_revoke_without_verify_and_missing_handlers():
    with pytest.raises(ValueError, match="verify"):
        rotation.execute(("create", "revoke_old"), {"create": lambda: "", "revoke_old": lambda: ""})
    with pytest.raises(ValueError, match="no handler"):
        rotation.execute(("create",), {})


def test_created_key_repr_hides_secret():
    k = providers.CreatedKey("cloudflare", "id1", "super-secret-value", "endpoint")
    assert "super-secret-value" not in repr(k)
