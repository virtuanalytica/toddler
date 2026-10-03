"""Pulse relay: decide whether Toddler runs a job locally or relays it to a knitweb peer.

Pulse behaviours (docs/design/brain.md section 4): trustless relay to compute, node-to-node
sharing, network weather, pay per served byte. A peer result is only accepted after
sampled re-execution, so verification cost is part of the price, and a peer whose
collateral makes fraud profitable is never chosen.

knitweb is loaded from its source tree (TODDLER_KNITWEB_SRC, default below) because its
packaging is owned by the knitweb project.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from fractions import Fraction
from typing import Sequence

_KNITWEB_SRC = os.environ.get("TODDLER_KNITWEB_SRC", "/media/knight2/EDS2/projects/knitweb/src")
if _KNITWEB_SRC not in sys.path:
    sys.path.append(_KNITWEB_SRC)

from knitweb.pouw.collateral import fraud_is_profitable  # noqa: E402
from knitweb.pouw.sampling import required_samples  # noqa: E402


@dataclass(frozen=True)
class Job:
    name: str
    blocks: int                  # output blocks a verifier could re-execute
    local_energy_j: float
    local_seconds: float
    deadline_s: float


@dataclass(frozen=True)
class PeerOffer:
    peer_id: str
    price_units: int             # integer token units, as knitweb escrow uses
    eur_per_unit: float
    eta_seconds: float
    collateral_units: int
    load: float                  # network weather: 0 idle .. 1 saturated


@dataclass(frozen=True)
class Prices:
    eur_per_joule: float
    max_miss: Fraction = Fraction(1, 100)     # tolerated chance of missing a 1% corruption
    assumed_corrupt_share: Fraction = Fraction(1, 100)
    max_peer_load: float = 0.9


@dataclass(frozen=True)
class RelayChoice:
    where: str                   # "local", a peer_id, or "none"
    cost_eur: float
    verify_samples: int
    reasons: tuple[str, ...]


def local_cost(job: Job, prices: Prices) -> float:
    return job.local_energy_j * prices.eur_per_joule


def verification_samples(job: Job, prices: Prices) -> int:
    corrupt = max(1, int(job.blocks * prices.assumed_corrupt_share))
    return required_samples(job.blocks, corrupt, prices.max_miss)


def peer_cost(job: Job, offer: PeerOffer, prices: Prices) -> float:
    """Price paid to the peer plus the energy to re-run the sampled blocks locally."""
    k = verification_samples(job, prices)
    reexec = local_cost(job, prices) * k / job.blocks
    return offer.price_units * offer.eur_per_unit + reexec


def choose_relay(job: Job, offers: Sequence[PeerOffer], prices: Prices) -> RelayChoice:
    options: list[tuple[float, str]] = []
    reasons: list[str] = []
    k = verification_samples(job, prices)
    if job.local_seconds <= job.deadline_s:
        options.append((local_cost(job, prices), "local"))
    else:
        reasons.append("local misses deadline")
    for o in offers:
        if fraud_is_profitable(o.collateral_units, o.price_units):
            reasons.append(f"{o.peer_id}: collateral too low, fraud would pay")
        elif o.load > prices.max_peer_load:
            reasons.append(f"{o.peer_id}: network weather, load {o.load:.2f}")
        elif o.eta_seconds > job.deadline_s:
            reasons.append(f"{o.peer_id}: misses deadline")
        else:
            options.append((peer_cost(job, o, prices), o.peer_id))
    if not options:
        return RelayChoice("none", float("inf"), k, tuple(reasons))
    cost, where = min(options)
    return RelayChoice(where, cost, k if where != "local" else 0, tuple(reasons))
