"""API keys for the own Jev server.

A key is shown once at creation; only its sha256 hash is stored (file mode 600). Verification
compares hashes in constant time. Keys can be revoked by id.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import time
from pathlib import Path

DEFAULT_PATH = Path(os.environ.get("TODDLER_JEV_KEYS", Path.home() / ".config/toddler-jev/keys.json"))
PREFIX = "tjev_"


def _hash(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


class KeyStore:
    def __init__(self, path: Path = DEFAULT_PATH) -> None:
        self.path = Path(path)

    def _load(self) -> dict:
        if not self.path.exists():
            return {"keys": []}
        return json.loads(self.path.read_text(encoding="utf-8"))

    def _save(self, data: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=1), encoding="utf-8")
        os.chmod(tmp, 0o600)
        tmp.replace(self.path)

    def create(self, label: str) -> tuple[str, str]:
        """Return (key_id, key). The key is not stored and cannot be recovered."""
        key = PREFIX + secrets.token_urlsafe(32)
        key_id = secrets.token_hex(4)
        data = self._load()
        data["keys"].append({"id": key_id, "label": label, "sha256": _hash(key), "created": int(time.time()), "revoked": False})
        self._save(data)
        return key_id, key

    def revoke(self, key_id: str) -> bool:
        data = self._load()
        hit = False
        for k in data["keys"]:
            if k["id"] == key_id and not k["revoked"]:
                k["revoked"] = True
                hit = True
        self._save(data)
        return hit

    def verify(self, key: str) -> str | None:
        """Return the key id for a valid, non-revoked key."""
        if not key.startswith(PREFIX):
            return None
        h = _hash(key)
        found = None
        for k in self._load()["keys"]:
            if hmac.compare_digest(k["sha256"], h) and not k["revoked"]:
                found = k["id"]
        return found

    def list(self) -> list[dict]:
        return [{kk: v for kk, v in k.items() if kk != "sha256"} for k in self._load()["keys"]]
