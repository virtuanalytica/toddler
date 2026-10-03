"""Append-only, hash-chained audit trail with scoped views, modelled on the audit_log fillslava
built for ClaudeClaw (docs/audit-log-redesign-2026-06.md there): every row stores prev_hash and
row_hash; a prune keeps a stored anchor so verification still works after retention cleanup;
readers get a sanitised, read-only view limited to the event types their audience may see.

Company guardrails (toddler.specialize) are recorded here too, so who changed which rule is
always provable.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Iterable

GENESIS = "0" * 64
_SECRET = re.compile(r"(?i)(api[_-]?key|password|secret|token)\s*[:=]\s*\S+")


def _row_hash(prev_hash: str, seq: int, ts_ms: int, actor: str, event_type: str, details: dict) -> str:
    payload = json.dumps([prev_hash, seq, ts_ms, actor, event_type, details], sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Entry:
    seq: int
    ts_ms: int
    actor: str
    event_type: str
    details: dict
    prev_hash: str
    row_hash: str


@dataclass
class AuditLog:
    entries: list[Entry] = field(default_factory=list)
    anchor_seq: int = -1
    anchor_hash: str = GENESIS

    def append(self, ts_ms: int, actor: str, event_type: str, details: dict) -> Entry:
        if _SECRET.search(json.dumps(details)):
            raise ValueError("audit details must not contain secret values; record a reference instead")
        prev = self.entries[-1].row_hash if self.entries else self.anchor_hash
        seq = self.entries[-1].seq + 1 if self.entries else self.anchor_seq + 1
        e = Entry(seq, ts_ms, actor, event_type, dict(details), prev, _row_hash(prev, seq, ts_ms, actor, event_type, details))
        self.entries.append(e)
        return e

    def verify(self) -> tuple[bool, int | None]:
        """Walk from the anchor; returns (ok, first bad seq)."""
        prev = self.anchor_hash
        for e in self.entries:
            if e.prev_hash != prev or e.row_hash != _row_hash(e.prev_hash, e.seq, e.ts_ms, e.actor, e.event_type, e.details):
                return False, e.seq
            prev = e.row_hash
        return True, None

    def prune_before(self, ts_ms: int) -> int:
        """Retention: drop old rows but keep the chain verifiable by moving the anchor."""
        keep = [e for e in self.entries if e.ts_ms >= ts_ms]
        dropped = len(self.entries) - len(keep)
        if dropped:
            last = self.entries[dropped - 1]
            self.anchor_seq, self.anchor_hash = last.seq, last.row_hash
            self.entries = keep
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
    """Sanitised read-only projection: filtered by event type and role, fields masked,
    every row still carrying its hash so a reader can check it against the full chain."""
    out = []
    for e in log.entries:
        if e.event_type not in audience.event_types:
            continue
        if audience.role is not None and e.details.get("role") != audience.role:
            continue
        details = {k: ("***" if k in audience.hidden_fields else v) for k, v in e.details.items()}
        out.append({"seq": e.seq, "ts_ms": e.ts_ms, "actor": e.actor, "event_type": e.event_type,
                    "details": details, "row_hash": e.row_hash})
    return out


def record_guardrails(log: AuditLog, ts_ms: int, actor: str, role: str, rule_ids: Iterable[str]) -> Entry:
    return log.append(ts_ms, actor, "guardrails_applied", {"role": role, "rules": sorted(rule_ids)})
