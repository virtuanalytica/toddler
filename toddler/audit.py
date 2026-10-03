"""Append-only, hash-chained audit trail with scoped views, modelled on the audit_log that
fillslava built for ClaudeClaw (docs/audit-log-redesign-2026-06.md in that repository).

Every row stores prev_hash and row_hash. A prune keeps a stored anchor so verification still
works after retention cleanup. Rows are immutable once appended (details are frozen), and an
optional JSONL file makes the trail persistent: each append is written and flushed, and
loading a file re-verifies the whole chain.

Secrets: values under credential-like keys, and values that look like credentials (bearer
tokens, provider key prefixes, key=value secrets), are replaced by "[redacted]" before
hashing; the row records which fields were redacted.

Readers get a sanitised projection (view) limited to the event types their audience may see.
It carries seq, prev_hash and row_hash, so a reader can check the chain LINKAGE of the rows
they see. A reader can recompute a row hash only for rows where nothing was masked for that
audience; masked rows list their masked fields so this limit is explicit.

Company guardrails (toddler.specialize) are recorded here too, so who changed which rule is
always provable.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Iterable, Mapping

GENESIS = "0" * 64
REDACTED = "[redacted]"
_CREDENTIAL_KEY = re.compile(r"(?i)(api[_-]?key|apikey|password|passwd|secret|token|authorization|bearer|"
                             r"private[_-]?key|access[_-]?key|credential|cookie|session[_-]?id)")
_CREDENTIAL_VALUE = re.compile(r"(?i)(bearer\s+\S+|\bsk-[A-Za-z0-9_-]{8,}|\bAKIA[0-9A-Z]{16}\b|"
                               r"\bgh[pousr]_[A-Za-z0-9]{20,}|"
                               r"(api[_-]?key|password|secret|token)\s*[:=]\s*\S+)")


def _row_hash(prev_hash: str, seq: int, ts_ms: int, actor: str, event_type: str, details: Mapping) -> str:
    payload = json.dumps([prev_hash, seq, ts_ms, actor, event_type, dict(details)], sort_keys=True,
                         separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _has_credential_key(v) -> bool:
    """True if a nested dict/list contains a credential-like key at any depth."""
    if isinstance(v, Mapping):
        return any(_CREDENTIAL_KEY.search(str(k)) or _has_credential_key(x) for k, x in v.items())
    if isinstance(v, (list, tuple)):
        return any(_has_credential_key(x) for x in v)
    return False


def _redact(details: Mapping) -> tuple[dict, list[str]]:
    """Redact a top-level field when its key, any nested key, or its value looks like a credential."""
    clean, redacted = {}, []
    for k, v in details.items():
        text = json.dumps(v, default=str)
        if _CREDENTIAL_KEY.search(str(k)) or _has_credential_key(v) or _CREDENTIAL_VALUE.search(text):
            clean[k] = REDACTED
            redacted.append(k)
        else:
            clean[k] = json.loads(json.dumps(v, default=str))   # deep copy, JSON-safe
    return clean, sorted(redacted)


@dataclass(frozen=True)
class Entry:
    seq: int
    ts_ms: int
    actor: str
    event_type: str
    details: Mapping              # read-only after append
    prev_hash: str
    row_hash: str

    def to_json(self) -> str:
        return json.dumps({"seq": self.seq, "ts_ms": self.ts_ms, "actor": self.actor, "event_type": self.event_type,
                           "details": dict(self.details), "prev_hash": self.prev_hash, "row_hash": self.row_hash},
                          sort_keys=True)


class AuditLog:
    """In-memory chain with optional append-only JSONL persistence (`path`)."""

    def __init__(self, path: Path | None = None) -> None:
        self._entries: list[Entry] = []
        self.anchor_seq = -1
        self.anchor_hash = GENESIS
        self.path = Path(path) if path else None

    @property
    def entries(self) -> tuple[Entry, ...]:
        return tuple(self._entries)

    def append(self, ts_ms: int, actor: str, event_type: str, details: Mapping) -> Entry:
        clean, redacted = _redact(details)
        if redacted:
            clean["_redacted_fields"] = redacted
        prev = self._entries[-1].row_hash if self._entries else self.anchor_hash
        seq = self._entries[-1].seq + 1 if self._entries else self.anchor_seq + 1
        e = Entry(seq, ts_ms, actor, event_type, MappingProxyType(clean), prev,
                  _row_hash(prev, seq, ts_ms, actor, event_type, clean))
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(e.to_json() + "\n")
                fh.flush()
                os.fsync(fh.fileno())
        self._entries.append(e)
        return e

    @classmethod
    def load(cls, path: Path) -> "AuditLog":
        """Load a JSONL trail and verify it; raises ValueError on the first broken row.

        The anchor is taken from the first remaining row, the same trust model as the in-memory
        prune anchor: deleting rows from the START of the file is not detectable from the file
        alone. Keep a copy of the latest anchor elsewhere (or the newest row hash) if that matters."""
        log = cls()
        rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
        if rows:
            log.anchor_seq, log.anchor_hash = rows[0]["seq"] - 1, rows[0]["prev_hash"]
        for r in rows:
            log._entries.append(Entry(r["seq"], r["ts_ms"], r["actor"], r["event_type"],
                                      MappingProxyType(r["details"]), r["prev_hash"], r["row_hash"]))
        ok, bad = log.verify()
        if not ok:
            raise ValueError(f"audit trail broken at seq {bad}")
        log.path = Path(path)
        return log

    def verify(self) -> tuple[bool, int | None]:
        """Walk from the anchor; returns (ok, first bad seq)."""
        prev = self.anchor_hash
        for e in self._entries:
            if e.prev_hash != prev or e.row_hash != _row_hash(e.prev_hash, e.seq, e.ts_ms, e.actor, e.event_type, e.details):
                return False, e.seq
            prev = e.row_hash
        return True, None

    def prune_before(self, ts_ms: int) -> int:
        """Retention: drop old rows but keep the chain verifiable by moving the anchor.
        A persistent trail is rewritten atomically with only the kept rows."""
        keep = [e for e in self._entries if e.ts_ms >= ts_ms]
        dropped = len(self._entries) - len(keep)
        if dropped:
            last = self._entries[dropped - 1]
            self.anchor_seq, self.anchor_hash = last.seq, last.row_hash
            self._entries = keep
            if self.path:
                tmp = self.path.with_suffix(".tmp")
                with tmp.open("w", encoding="utf-8") as fh:
                    fh.write("".join(e.to_json() + "\n" for e in keep))
                    fh.flush()
                    os.fsync(fh.fileno())
                tmp.replace(self.path)
        return dropped


@dataclass(frozen=True)
class Audience:
    """Who may read which part of the trail, e.g. the company's compliance officer sees
    guardrail and STOP events; a deployer sees only events of their own role."""

    name: str
    event_types: frozenset[str]
    role: str | None = None              # restrict to entries whose details.role matches
    hidden_fields: frozenset[str] = frozenset({"ip", "email"})


def view(log: AuditLog, audience: Audience) -> list[dict]:
    """Sanitised read-only projection, filtered by event type and role. Each row carries seq,
    prev_hash and row_hash (linkage is checkable) and lists the fields masked for this audience
    (the row hash is recomputable only when that list is empty)."""
    out = []
    for e in log.entries:
        if e.event_type not in audience.event_types:
            continue
        if audience.role is not None and e.details.get("role") != audience.role:
            continue
        masked = sorted(k for k in e.details if k in audience.hidden_fields)
        details = {k: ("***" if k in audience.hidden_fields else v) for k, v in e.details.items()}
        out.append({"seq": e.seq, "ts_ms": e.ts_ms, "actor": e.actor, "event_type": e.event_type,
                    "details": details, "prev_hash": e.prev_hash, "row_hash": e.row_hash,
                    "masked_fields": masked})
    return out


def record_guardrails(log: AuditLog, ts_ms: int, actor: str, role: str, rule_ids: Iterable[str]) -> Entry:
    return log.append(ts_ms, actor, "guardrails_applied", {"role": role, "rules": sorted(rule_ids)})
