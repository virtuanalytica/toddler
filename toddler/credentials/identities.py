"""Toddler's email identities. A toddler has several addresses: one stable address per
service account, so a leaked or revoked key never exposes the other identities.

Alias backends (chosen from the 2026-10-03 market scan, all with an official API and
configurable addresses): SimpleLogin and addy.io (self-hostable alias services), mailboxes
or forwards on an own domain via the TransIP mail API or Cloudflare Email Routing, and
plus-addressing as a no-API fallback. Mass account creation and temp-mail farming are out of
scope (provider ToS); every identity is created on an account a human approved.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_EMAIL = re.compile(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$")
ALIAS_BACKENDS = ("simplelogin", "addy", "transip-mailbox", "cloudflare-routing", "plus-address", "manual")


@dataclass(frozen=True)
class Identity:
    email: str
    backend: str
    purpose: str
    approved_by: str            # the human who approved this identity

    def __post_init__(self) -> None:
        if not _EMAIL.match(self.email):
            raise ValueError(f"not an email address: {self.email!r}")
        if self.backend not in ALIAS_BACKENDS:
            raise ValueError(f"unknown alias backend: {self.backend}")
        if not self.approved_by:
            raise ValueError("every identity needs a human approver")


def plus_address(base: str, tag: str) -> str:
    """Gmail-style plus address; some providers reject '+', so prefer a real alias backend."""
    if not _EMAIL.match(base) or not re.fullmatch(r"[a-z0-9-]{1,32}", tag):
        raise ValueError("invalid base address or tag")
    local, domain = base.split("@")
    return f"{local.split('+')[0]}+{tag}@{domain}"


@dataclass
class IdentityRegistry:
    """Maps (provider) -> identity. One provider account per identity keeps blast radius small."""

    identities: dict[str, Identity] = field(default_factory=dict)
    assignments: dict[str, str] = field(default_factory=dict)  # provider -> email

    def add(self, identity: Identity) -> None:
        if identity.email in self.identities:
            raise ValueError(f"identity already registered: {identity.email}")
        self.identities[identity.email] = identity

    def assign(self, provider: str, email: str, *, replace: bool = False) -> None:
        """Bind a provider to an identity. Re-binding a provider to another identity must be explicit
        (`replace=True`): a silent overwrite would orphan the old identity's keys."""
        if email not in self.identities:
            raise KeyError(f"unknown identity: {email}")
        used_by = [p for p, e in self.assignments.items() if e == email and p != provider]
        if used_by:
            raise ValueError(f"{email} already serves {used_by[0]}; use a separate identity per provider")
        current = self.assignments.get(provider)
        if current and current != email and not replace:
            raise ValueError(f"{provider} is already assigned to {current}; pass replace=True to re-bind")
        self.assignments[provider] = email

    def for_provider(self, provider: str) -> Identity:
        return self.identities[self.assignments[provider]]
