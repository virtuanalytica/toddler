"""Logic layer: hard STOP rules that veto an action before any learned component runs.

STOP rules are constraints, not penalties. No reward can buy an action that a rule
forbids, which is why they live outside the utility function (see docs/design/brain.md
section 1 and wee2017-mapping row 20, the inhibition layer wired to every action).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable


@dataclass(frozen=True)
class Action:
    """A candidate action. Flags describe what the action would do, not what it intends."""

    name: str
    reveals_secret: bool = False
    spends_money: bool = False
    sends_message: bool = False
    logs_into_account: bool = False
    physical_contact: bool = False
    contact_with_infant: bool = False
    operator_approved: bool = False
    caregiver_present: bool = False
    tags: frozenset[str] = field(default_factory=frozenset)


@dataclass(frozen=True)
class StopRule:
    rule_id: str
    reason: str
    violates: Callable[[Action], bool]


DEFAULT_RULES: tuple[StopRule, ...] = (
    StopRule("no-secrets", "never reveal a password, token or key", lambda a: a.reveals_secret),
    StopRule(
        "approval-for-outward-actions",
        "buying, sending or logging in needs explicit operator approval",
        lambda a: (a.spends_money or a.sends_message or a.logs_into_account) and not a.operator_approved,
    ),
    StopRule(
        "caregiver-for-contact",
        "physical contact with a person needs a caregiver present",
        lambda a: a.physical_contact and not a.caregiver_present,
    ),
    StopRule("no-infant-contact", "physical contact with infants is out of scope", lambda a: a.contact_with_infant),
)


@dataclass(frozen=True)
class Verdict:
    allowed: bool
    violated: tuple[str, ...]


def check(action: Action, rules: Iterable[StopRule] = DEFAULT_RULES) -> Verdict:
    violated = tuple(rule.rule_id for rule in rules if rule.violates(action))
    return Verdict(allowed=not violated, violated=violated)
