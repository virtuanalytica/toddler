"""Evolution ledger: every toddler's birth, inheritance and selection, traceable back to its roots.

The generation register (generations.py) stores what a toddler IS (weights, scores). This ledger
records how it CAME TO BE, as a hash-chained, append-only log (toddler.audit.AuditLog persisted
to <registry root>/lineage.jsonl), so behaviour seen later in a toddler, or in an agent derived
from one, can be traced back to the generation, parent and training material it came from.

Events (Darwin's three ingredients plus derivation):
  born            variation   a toddler is created: parents and the exact modules inherited from each
                              (with the parent's weights sha256), code commit, training data (tasks
                              and their environment ids, seed sets / commitments), budget, hardware,
                              software, and its role ("population" or "control")
  selected        selection   a generation's verdict: "survived", "extinct" or "control", with the
                              evidence (promotion test, scores) behind it
  derived_agent   heredity    an agent (specialist, service, Jev model) built on a toddler, with the
                              toddler's weights sha256, so an agent maps to its Toddler foundation
  transmorphed    heredity    an agent becomes another agent through cross-skill transfer learning:
                              one or more sources (agents or toddlers), the modules transferred from
                              each, the skills gained, and the new weights sha256

Refs: toddlers are "<generation>/<toddler_id>", agents are "agent:<agent_id>". trace_agent walks
derivations and transmorphoses back to every Toddler foundation an agent rests on.
  backfilled      (any)       a record written after the fact for toddlers born before this ledger

Only surviving generations get a development report (scripts/build_generations.py --report).
"""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from toddler import audit

VERDICTS = ("survived", "extinct", "control")
ROLES = ("population", "control")


def code_version(repo: Path | None = None) -> dict:
    """Commit of the code that trains the toddler; `dirty` means uncommitted changes took part."""
    repo = Path(repo) if repo else Path(__file__).resolve().parents[2]

    def git(*args: str) -> str:
        return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True).stdout.strip()

    try:
        return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
                "remote": git("config", "--get", "remote.origin.url")}
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        return {"commit": "", "dirty": True, "error": f"{type(exc).__name__}"}


@dataclass(frozen=True)
class Inheritance:
    parent: str                 # "G1/t1001"
    parent_weights_sha256: str
    modules: tuple[str, ...]    # e.g. ("trunk", "task:cartpole", ...)

    def as_dict(self) -> dict:
        return {"parent": self.parent, "parent_weights_sha256": self.parent_weights_sha256,
                "modules": list(self.modules)}


class Ledger:
    def __init__(self, root: Path, actor: str = "toddler-evolution") -> None:
        self.path = Path(root) / "lineage.jsonl"
        self.log = audit.AuditLog.load(self.path) if self.path.exists() else audit.AuditLog(self.path)
        self.actor = actor

    def _append(self, event: str, details: dict) -> audit.Entry:
        e = self.log.append(int(time.time() * 1000), self.actor, event, details)
        if e.details.get("_redacted_fields"):     # the ledger must stay complete; never store credentials in it
            raise ValueError(f"ledger entry had fields redacted: {e.details['_redacted_fields']}")
        return e

    # -- writing ------------------------------------------------------------
    def born(self, ref: str, weights_sha256: str, *, role: str, inherits: Iterable[Inheritance] = (),
             code: dict, data: dict, budget: dict, hardware: dict, software: dict,
             backfilled: bool = False) -> audit.Entry:
        if role not in ROLES:
            raise ValueError(f"role must be one of {ROLES}")
        if self.birth(ref):
            raise ValueError(f"{ref} is already born in the ledger")
        inh = [i.as_dict() for i in inherits]
        for i in inh:
            p = self.birth(i["parent"])
            if p and p["weights_sha256"] != i["parent_weights_sha256"]:
                raise ValueError(f"{ref}: parent {i['parent']} weights differ from the parent's recorded birth")
        return self._append("backfilled" if backfilled else "born",
                            {"ref": ref, "weights_sha256": weights_sha256, "role": role, "inherits": inh,
                             "code": code, "data": data, "budget": budget, "hardware": hardware,
                             "software": software})

    def selected(self, generation: str, verdict: str, evidence: dict) -> audit.Entry:
        if verdict not in VERDICTS:
            raise ValueError(f"verdict must be one of {VERDICTS}")
        if not self.members(generation):
            raise ValueError(f"generation {generation} has no born toddlers in the ledger")
        return self._append("selected", {"generation": generation, "verdict": verdict, "evidence": evidence})

    def derived_agent(self, agent_id: str, toddler_ref: str, purpose: str, approved_by: str,
                      weights_sha256: str = "") -> audit.Entry:
        b = self.birth(toddler_ref)
        if not b:
            raise KeyError(f"{toddler_ref} is not in the ledger")
        if not approved_by.strip():
            raise ValueError("a derived agent needs a named approver")
        if self.agent(agent_id):
            raise ValueError(f"agent {agent_id} already exists in the ledger")
        return self._append("derived_agent", {"agent_id": agent_id, "toddler": toddler_ref,
                                              "toddler_weights_sha256": b["weights_sha256"],
                                              "weights_sha256": weights_sha256,
                                              "purpose": purpose, "approved_by": approved_by})

    def transmorphed(self, agent_id: str, sources: Iterable[Inheritance], skills: Iterable[str],
                     weights_sha256: str, purpose: str, approved_by: str, code: dict, data: dict) -> audit.Entry:
        """A new agent made from existing agents/toddlers by transfer learning. Every source must be in
        the ledger with the weights hash given here, so the transfer is provably from that exact model."""
        if not approved_by.strip():
            raise ValueError("a transmorphosis needs a named approver")
        if self.agent(agent_id):
            raise ValueError(f"agent {agent_id} already exists in the ledger")
        src = [i.as_dict() for i in sources]
        if not src:
            raise ValueError("a transmorphosis needs at least one source")
        for i in src:
            known = self.weights_of(i["parent"])
            if known is None:
                raise KeyError(f"source {i['parent']} is not in the ledger")
            if known != i["parent_weights_sha256"]:
                raise ValueError(f"source {i['parent']}: weights differ from the ledger record")
        return self._append("transmorphed", {"agent_id": agent_id, "sources": src, "skills": list(skills),
                                             "weights_sha256": weights_sha256, "purpose": purpose,
                                             "approved_by": approved_by, "code": code, "data": data})

    # -- reading ------------------------------------------------------------
    def _events(self, *kinds: str) -> list[dict]:
        return [dict(e.details) | {"_event": e.event_type, "_seq": e.seq, "_ts_ms": e.ts_ms}
                for e in self.log.entries if e.event_type in kinds]

    def birth(self, ref: str) -> dict | None:
        return next((b for b in self._events("born", "backfilled") if b["ref"] == ref), None)

    def members(self, generation: str) -> list[dict]:
        return [b for b in self._events("born", "backfilled") if b["ref"].split("/", 1)[0] == generation]

    def verdict(self, generation: str) -> dict | None:
        """The latest selection verdict of a generation (a later verdict supersedes, both stay logged)."""
        v = [s for s in self._events("selected") if s["generation"] == generation]
        return v[-1] if v else None

    def surviving_generations(self) -> list[str]:
        gens = []
        for s in self._events("selected"):
            if s["generation"] not in gens:
                gens.append(s["generation"])
        return [g for g in gens if self.verdict(g)["verdict"] == "survived"]

    def agent(self, agent_id: str) -> dict | None:
        a = [x for x in self._events("derived_agent", "transmorphed") if x["agent_id"] == agent_id]
        return a[-1] if a else None

    def weights_of(self, ref: str) -> str | None:
        if ref.startswith("agent:"):
            a = self.agent(ref[len("agent:"):])
            return a["weights_sha256"] if a else None
        b = self.birth(ref)
        return b["weights_sha256"] if b else None

    def ancestry(self, ref: str) -> list[dict]:
        """Every ancestor of a toddler (nearest first) with what was inherited from it."""
        out, frontier, seen = [], [ref], {ref}
        while frontier:
            b = self.birth(frontier.pop(0))
            if not b:
                continue
            for i in b["inherits"]:
                if i["parent"] not in seen:
                    seen.add(i["parent"])
                    out.append({"child": b["ref"], **i})
                    frontier.append(i["parent"])
        return out

    def descendants(self, ref: str) -> list[str]:
        out, frontier = [], [ref]
        births = self._events("born", "backfilled")
        while frontier:
            cur = frontier.pop(0)
            for b in births:
                if any(i["parent"] == cur for i in b["inherits"]) and b["ref"] not in out:
                    out.append(b["ref"])
                    frontier.append(b["ref"])
        return out

    def trace_agent(self, agent_id: str) -> dict:
        """From an agent back to every Toddler foundation it rests on: the chain of derivations and
        transmorphoses (nearest first), the foundation toddlers with their full ancestry, and the
        selection verdict of every generation on those paths."""
        if not self.agent(agent_id):
            raise KeyError(f"agent {agent_id} is not in the ledger")
        chain, foundations, frontier, seen = [], [], [agent_id], {agent_id}
        while frontier:
            a = self.agent(frontier.pop(0))
            chain.append(a)
            refs = [a["toddler"]] if a["_event"] == "derived_agent" else [s["parent"] for s in a["sources"]]
            for r in refs:
                if r.startswith("agent:"):
                    if r[6:] not in seen:
                        seen.add(r[6:])
                        frontier.append(r[6:])
                elif r not in foundations:
                    foundations.append(r)
        toddlers = {t: {"birth": self.birth(t), "ancestry": self.ancestry(t)} for t in foundations}
        gens = [t.split("/", 1)[0] for t in foundations]
        gens += [x["parent"].split("/", 1)[0] for v in toddlers.values() for x in v["ancestry"]]
        return {"chain": chain, "foundations": toddlers,
                "verdicts": {g: self.verdict(g) for g in dict.fromkeys(gens)}}

    def verify(self) -> tuple[bool, int | None]:
        return self.log.verify()
