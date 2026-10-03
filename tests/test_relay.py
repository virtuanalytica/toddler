import math

import pytest

relay = pytest.importorskip("toddler.relay", reason="knitweb not available")

PRICES = relay.Prices(eur_per_joule=0.30 / 3.6e6)  # 0.30 EUR per kWh


def _job(**kw):
    base = dict(name="compile", blocks=1000, local_energy_j=3.6e7, local_seconds=60, deadline_s=120)
    base.update(kw)
    return relay.Job(**base)


def _offer(pid, price=1, collateral=10, eta=30, load=0.2):
    return relay.PeerOffer(pid, price, 1.0, eta, collateral, load)


def test_verification_samples_use_knitweb_sizing():
    # knitweb: 368 sampled blocks catch a 1% corruption of 1000 blocks with >= 99% probability
    assert relay.verification_samples(_job(), PRICES) == 368


def test_cheaper_verified_peer_wins():
    choice = relay.choose_relay(_job(), [_offer("peer-a", price=1)], PRICES)
    assert choice.where == "peer-a" and choice.verify_samples == 368
    assert choice.cost_eur < relay.local_cost(_job(), PRICES)


def test_peer_with_profitable_fraud_is_rejected():
    choice = relay.choose_relay(_job(), [_offer("cheat", price=5, collateral=1)], PRICES)
    assert choice.where == "local"
    assert any("fraud would pay" in r for r in choice.reasons)


def test_network_weather_skips_saturated_peer():
    choice = relay.choose_relay(_job(), [_offer("busy", load=0.95)], PRICES)
    assert choice.where == "local"


def test_no_option_means_hand_off():
    choice = relay.choose_relay(_job(local_seconds=500), [_offer("slow", eta=500)], PRICES)
    assert choice.where == "none" and math.isinf(choice.cost_eur)
