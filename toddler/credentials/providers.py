"""Official API-key creation endpoints, and the providers where a human must create the key.

Every call is an outward action: it goes through toddler.stop with operator approval, and every
creator takes a required `audit_log` (toddler.audit.AuditLog): a created key is recorded as
`key_created` (provider, key_id, endpoint, expiry; never the secret) and a refused request as
`key_creation_refused`. Admin credentials come from the vault, never from prompts. Endpoints verified 2026-10-03:

  google      POST https://apikeys.googleapis.com/v2/projects/{project}/locations/global/keys
  cloudflare  POST https://api.cloudflare.com/client/v4/user/tokens   (revoke: DELETE .../tokens/{id})
  openai      POST https://api.openai.com/v1/organization/projects/{project}/service_accounts
  github-app  POST https://api.github.com/app/installations/{installation}/access_tokens (1 h token)
  aws         IAM CreateAccessKey / UpdateAccessKey / DeleteAccessKey (max 2 keys per user)
  anthropic   Admin API cannot create keys (list/update/archive only) -> human task
  transip     key pair only via the control panel; API exchanges it for a JWT -> human task
"""

from __future__ import annotations

import functools
import time
from dataclasses import dataclass, field
from typing import Callable

import requests

from toddler import audit, stop

HUMAN_ONLY = {
    "anthropic": "Create the key in the Anthropic Console; the Admin API can only list, update and archive keys.",
    "transip": "Generate the key pair in the control panel (https://www.transip.nl/cp/account/api); the API only exchanges it for a JWT.",
}


@dataclass(frozen=True)
class CreatedKey:
    provider: str
    key_id: str
    secret: str = field(repr=False)     # never appears in repr() or log lines
    created_via: str = ""
    expires_at: str = ""


@dataclass(frozen=True)
class HumanTask:
    provider: str
    identity_email: str
    instruction: str


class ApprovalRequired(PermissionError):
    pass


def _require_approval(provider: str, approved: bool) -> None:
    verdict = stop.check(stop.Action(f"create-key:{provider}", logs_into_account=True, operator_approved=approved))
    if not verdict.allowed:
        raise ApprovalRequired(f"operator approval required to create a {provider} key ({', '.join(verdict.violated)})")


def _audited(provider: str):
    """Make `audit_log` a required keyword of a creator and record its outcome (no secret)."""
    def wrap(fn: Callable[..., CreatedKey]) -> Callable[..., CreatedKey]:
        @functools.wraps(fn)
        def inner(*args, audit_log: audit.AuditLog, actor: str = "toddler", **kw) -> CreatedKey:
            try:
                key = fn(*args, **kw)
            except ApprovalRequired as exc:
                audit_log.append(int(time.time() * 1000), actor, "key_creation_refused",
                                 {"provider": provider, "reason": str(exc)})
                raise
            audit_log.append(int(time.time() * 1000), actor, "key_created",
                             {"provider": provider, "key_id": key.key_id, "created_via": key.created_via,
                              "expires_at": key.expires_at})
            return key
        return inner
    return wrap


def _post(url: str, headers: dict, json: dict, timeout: float = 20.0) -> dict:
    r = requests.post(url, headers=headers, json=json, timeout=timeout)
    r.raise_for_status()
    return r.json()


@_audited("google")
def create_google_key(project: str, display_name: str, bearer: str, approved: bool) -> CreatedKey:
    _require_approval("google", approved)
    url = f"https://apikeys.googleapis.com/v2/projects/{project}/locations/global/keys"
    op = _post(url, {"Authorization": f"Bearer {bearer}"}, {"displayName": display_name})
    resp = op.get("response", {})
    return CreatedKey("google", resp.get("uid", op.get("name", "")), resp.get("keyString", ""), url)


@_audited("cloudflare")
def create_cloudflare_token(name: str, policies: list[dict], bearer: str, approved: bool,
                            expires_on: str = "") -> CreatedKey:
    _require_approval("cloudflare", approved)
    url = "https://api.cloudflare.com/client/v4/user/tokens"
    body: dict = {"name": name, "policies": policies}
    if expires_on:
        body["expires_on"] = expires_on
    res = _post(url, {"Authorization": f"Bearer {bearer}"}, body)["result"]
    return CreatedKey("cloudflare", res["id"], res["value"], url, expires_on)


@_audited("openai")
def create_openai_service_account(project: str, name: str, admin_key: str, approved: bool) -> CreatedKey:
    _require_approval("openai", approved)
    url = f"https://api.openai.com/v1/organization/projects/{project}/service_accounts"
    res = _post(url, {"Authorization": f"Bearer {admin_key}"}, {"name": name})
    return CreatedKey("openai", res["api_key"]["id"], res["api_key"]["value"], url)


@_audited("github-app")
def create_github_installation_token(installation_id: int, app_jwt: str, approved: bool) -> CreatedKey:
    _require_approval("github-app", approved)
    url = f"https://api.github.com/app/installations/{installation_id}/access_tokens"
    res = _post(url, {"Authorization": f"Bearer {app_jwt}", "Accept": "application/vnd.github+json"}, {})
    return CreatedKey("github-app", str(installation_id), res["token"], url, res.get("expires_at", ""))


@_audited("aws")
def create_aws_access_key(user_name: str, approved: bool, iam_client=None) -> CreatedKey:
    _require_approval("aws", approved)
    if iam_client is None:
        import boto3            # optional dependency: only needed without an injected client

        iam_client = boto3.client("iam")
    iam = iam_client
    k = iam.create_access_key(UserName=user_name)["AccessKey"]
    return CreatedKey("aws", k["AccessKeyId"], k["SecretAccessKey"], "iam:CreateAccessKey")


CREATORS: dict[str, Callable[..., CreatedKey]] = {
    "google": create_google_key,
    "cloudflare": create_cloudflare_token,
    "openai": create_openai_service_account,
    "github-app": create_github_installation_token,
    "aws": create_aws_access_key,
}


def plan_creation(provider: str, identity_email: str) -> HumanTask | str:
    """Return the automated creator name, or a HumanTask when only a human can create the key."""
    if provider in HUMAN_ONLY:
        return HumanTask(provider, identity_email, HUMAN_ONLY[provider])
    if provider in CREATORS:
        return provider
    raise KeyError(f"no official key-creation route known for {provider}")
