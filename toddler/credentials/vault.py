"""Key storage in OpenBao (Vault-compatible, MPL-2.0) through the KV v2 HTTP API.

Path scheme: kv/toddler/<identity_email>/<provider>/<key_name>, with metadata
owner_identity, created_via, expires_at and rotation_policy. Infisical or HashiCorp Vault can
replace OpenBao behind the same interface. The token comes from the environment
(TODDLER_VAULT_TOKEN), never from code, prompts or memory.
"""

from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass
from typing import Protocol

import requests

_SAFE = re.compile(r"^[A-Za-z0-9@._+-]+$")


def secret_path(identity_email: str, provider: str, key_name: str, prefix: str = "toddler") -> str:
    for part in (identity_email, provider, key_name):
        if not _SAFE.match(part):
            raise ValueError(f"unsafe path segment: {part!r}")
    return f"{prefix}/{identity_email}/{provider}/{key_name}"


@dataclass(frozen=True)
class KeyMetadata:
    owner_identity: str
    created_via: str           # official endpoint, or "human-console"
    expires_at: str            # ISO 8601, or "" when the provider sets no expiry
    rotation_policy: str       # e.g. "30d", "on-demand"
    key_id: str = ""


class SecretStore(Protocol):
    def put(self, path: str, value: str, meta: KeyMetadata) -> int: ...
    def get(self, path: str) -> str: ...
    def destroy_version(self, path: str, version: int) -> None: ...


class OpenBaoKV:
    """Minimal KV v2 client: write a version with metadata, read latest, destroy a version."""

    def __init__(self, addr: str | None = None, token: str | None = None, mount: str = "kv",
                 timeout: float = 10.0) -> None:
        self.addr = (addr or os.environ.get("TODDLER_VAULT_ADDR", "http://127.0.0.1:8200")).rstrip("/")
        self._token = token or os.environ.get("TODDLER_VAULT_TOKEN", "")
        if not self._token:
            raise RuntimeError("TODDLER_VAULT_TOKEN is not set")
        self.mount = mount
        self.timeout = timeout

    def _url(self, kind: str, path: str) -> str:
        return f"{self.addr}/v1/{self.mount}/{kind}/{path}"

    def _call(self, method: str, url: str, **kw) -> dict:
        r = requests.request(method, url, headers={"X-Vault-Token": self._token}, timeout=self.timeout, **kw)
        r.raise_for_status()
        return r.json() if r.content else {}

    def put(self, path: str, value: str, meta: KeyMetadata) -> int:
        """Write the value, then its custom metadata (two KV v2 calls). If the metadata call fails
        the secret exists without ownership/expiry metadata; recover by calling put() again: KV v2
        stores a new version and the metadata write is idempotent."""
        data = self._call("POST", self._url("data", path), json={"data": {"value": value}})
        self._call("POST", self._url("metadata", path), json={"custom_metadata": asdict(meta)})
        return int(data["data"]["version"])

    def get(self, path: str) -> str:
        return self._call("GET", self._url("data", path))["data"]["data"]["value"]

    def destroy_version(self, path: str, version: int) -> None:
        self._call("POST", self._url("destroy", path), json={"versions": [version]})
