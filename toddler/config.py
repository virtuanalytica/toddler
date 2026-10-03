"""Load a deployment configuration produced by the Toddler GUI (gui/index.html).

The GUI only edits JSON; this module is the authority that validates it through
toddler.specialize.build, so a configuration that would weaken a base rule is rejected here
even if a modified GUI produced it.

Schema (toddler-config/v1):
{
  "schema": "toddler-config/v1",
  "role": {"name": str, "family": str, "level": str, "tasks": [str]},
  "guardrails": [{"id": str, "reason": str, "owner": str, "forbidden_tags": [str]}],
  "jev_rules": [{"qid": str, "text": str, "slow_if_above": float, "stop_if_above": float}],
  "experts": [{"model_id": str, "domains": [str], "quality": float, "eur_per_1k_tokens": float,
               "joules_per_1k_tokens": float, "evidence": str}],
  "audiences": [{"name": str, "event_types": [str], "role": str | null}]
}
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from toddler import audit, fastpath, specialize

SCHEMA = "toddler-config/v1"


@dataclass(frozen=True)
class Deployment:
    spec: specialize.Specialisation
    audiences: tuple[audit.Audience, ...]


def from_dict(cfg: dict) -> Deployment:
    if cfg.get("schema") != SCHEMA:
        raise ValueError(f"expected schema {SCHEMA}")
    r = cfg["role"]
    role = specialize.Role(r["name"], r["family"], r.get("level", ""), tuple(r.get("tasks", ())))
    guardrails = [specialize.company_guardrail(g["id"], g["reason"], g["owner"], frozenset(g["forbidden_tags"]))
                  for g in cfg.get("guardrails", [])]
    questions = [fastpath.PhysicalQuestion(q["qid"], q["text"], float(q["stop_if_above"]), float(q["slow_if_above"]))
                 for q in cfg.get("jev_rules", [])]
    experts = [specialize.Expert(e["model_id"], frozenset(e["domains"]), float(e["quality"]),
                                 float(e["eur_per_1k_tokens"]), float(e["joules_per_1k_tokens"]), e["evidence"])
               for e in cfg.get("experts", [])]
    spec = specialize.build(role, questions, guardrails, experts)
    audiences = tuple(audit.Audience(a["name"], frozenset(a["event_types"]), a.get("role"))
                      for a in cfg.get("audiences", []))
    return Deployment(spec, audiences)


def load(path: Path) -> Deployment:
    return from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
