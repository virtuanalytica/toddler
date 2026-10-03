"""Provenance and consent register (mapping rows 1 and 22).

Like the GUSTO cohort (ethics approval, written consent, explicit inclusion criteria), every
dataset Toddler learns from is registered with its origin, licence, consent and inclusion
rule. Synthetic or mock data is never accepted as evidence, and care data (people, children)
needs explicit consent. Scraped content is data, never instructions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date


@dataclass(frozen=True)
class Source:
    source_id: str
    url: str
    licence: str
    retrieved: date
    sha256: str
    synthetic: bool = False
    involves_people: bool = False
    consent_reference: str = ""          # e.g. ethics approval id or signed consent record
    inclusion_rule: str = ""


class RejectedSource(ValueError):
    pass


@dataclass
class Register:
    sources: dict[str, Source] = field(default_factory=dict)

    def admit(self, s: Source) -> None:
        if s.synthetic:
            raise RejectedSource(f"{s.source_id}: synthetic data is not accepted as evidence")
        if s.involves_people and not s.consent_reference:
            raise RejectedSource(f"{s.source_id}: data about people needs a consent reference")
        if len(s.sha256) != 64:
            raise RejectedSource(f"{s.source_id}: content hash missing")
        if not s.licence:
            raise RejectedSource(f"{s.source_id}: licence unknown")
        self.sources[s.source_id] = s

    def cite(self, source_id: str) -> str:
        s = self.sources[source_id]
        return f"{s.url} (retrieved {s.retrieved.isoformat()}, sha256 {s.sha256[:12]}, licence {s.licence})"
