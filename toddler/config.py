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


class ConfigError(ValueError):
    """A configuration problem, with the JSON path of the offending field."""


def _get(d: dict, key: str, where: str, kind: type | tuple = object):
    if not isinstance(d, dict) or key not in d:
        raise ConfigError(f"{where}.{key}: missing")
    v = d[key]
    if not isinstance(v, kind):
        raise ConfigError(f"{where}.{key}: expected {getattr(kind, '__name__', kind)}")
    return v


def _num(d: dict, key: str, where: str) -> float:
    v = _get(d, key, where, (int, float))
    if isinstance(v, bool):
        raise ConfigError(f"{where}.{key}: expected a number")
    return float(v)


def from_dict(cfg: dict) -> Deployment:
    if not isinstance(cfg, dict) or cfg.get("schema") != SCHEMA:
        raise ConfigError(f"schema: expected {SCHEMA}")
    r = _get(cfg, "role", "$", dict)
    role = specialize.Role(_get(r, "name", "$.role", str), _get(r, "family", "$.role", str),
                           r.get("level", ""), tuple(r.get("tasks", ())))
    guardrails = []
    for i, g in enumerate(cfg.get("guardrails", [])):
        w = f"$.guardrails[{i}]"
        guardrails.append(specialize.company_guardrail(_get(g, "id", w, str), _get(g, "reason", w, str),
                                                       _get(g, "owner", w, str),
                                                       frozenset(_get(g, "forbidden_tags", w, list))))
    questions = []
    for i, q in enumerate(cfg.get("jev_rules", [])):
        w = f"$.jev_rules[{i}]"
        questions.append(fastpath.PhysicalQuestion(_get(q, "qid", w, str), _get(q, "text", w, str),
                                                   _num(q, "stop_if_above", w), _num(q, "slow_if_above", w)))
    experts = []
    for i, e in enumerate(cfg.get("experts", [])):
        w = f"$.experts[{i}]"
        experts.append(specialize.Expert(_get(e, "model_id", w, str), frozenset(_get(e, "domains", w, list)),
                                         _num(e, "quality", w), _num(e, "eur_per_1k_tokens", w),
                                         _num(e, "joules_per_1k_tokens", w), _get(e, "evidence", w, str)))
    spec = specialize.build(role, questions, guardrails, experts)
    audiences = []
    for i, a in enumerate(cfg.get("audiences", [])):
        w = f"$.audiences[{i}]"
        audiences.append(audit.Audience(_get(a, "name", w, str), frozenset(_get(a, "event_types", w, list)),
                                        a.get("role")))
    return Deployment(spec, tuple(audiences))


def load(path: Path) -> Deployment:
    return from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
