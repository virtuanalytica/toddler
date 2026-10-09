"""A routed successor must retain the exact policies it claims to inherit."""

import hashlib
import json
from pathlib import Path

import pytest
import torch

from scripts.benchmark_routed_successor import measure, verify_sources
from toddler.learn.generations import Registry, ToddlerRecord
from toddler.learn.multitask import MultiTaskNet
from toddler.learn.play import _policy
from toddler.learn.routing import TaskExpertRouter


def _record(generation, toddler_id, parents=(), config=None):
    return ToddlerRecord(generation, toddler_id, "test", config or {}, 0,
                         list(parents), [0.0], {"device": "cpu"})


def test_router_round_trip_uses_selected_full_expert(tmp_path):
    dims = {"cartpole": (4, 2), "acrobot": (6, 3)}
    torch.manual_seed(1)
    a = MultiTaskNet(dims)
    torch.manual_seed(2)
    b = MultiTaskNet(dims)
    route = {"cartpole": "g2", "acrobot": "scratch"}
    reg = Registry(tmp_path)
    reg.save(TaskExpertRouter({"g2": a, "scratch": b}, route),
             _record("next", "t1", config={"route": route}))
    loaded, _ = reg.load("next", "t1")
    assert isinstance(loaded, TaskExpertRouter) and loaded.route == route
    assert torch.equal(_policy(loaded, "cartpole")(torch.zeros(4))[0],
                       a.forward_task("cartpole", torch.zeros(4))[0])
    assert torch.equal(_policy(loaded, "acrobot")(torch.zeros(6))[0],
                       b.forward_task("acrobot", torch.zeros(6))[0])


def test_source_verifier_rejects_changed_embedded_weights(tmp_path):
    dims = {"cartpole": (4, 2)}
    base = Registry(tmp_path / "base")
    torch.manual_seed(3)
    a = MultiTaskNet(dims)
    torch.manual_seed(4)
    b = MultiTaskNet(dims)
    base.save(a, _record("G2", "t2001"))
    base.save(b, _record("G3-sp", "t3001"))
    route = {"cartpole": "G3-sp"}
    protocol = {"source_weights": {
        "G2": {"t2001": base.generation("G2")[0].weights_sha256},
        "G3-sp": {"t2001": base.generation("G3-sp")[0].weights_sha256}}}
    raw = json.dumps(protocol).encode()
    digest = hashlib.sha256(raw).hexdigest()
    child = TaskExpertRouter({"G2": a, "G3-sp": b}, route)
    rec = _record("G3-recombined", "t5001", ("G2/t2001",),
                  {"route": route, "protocol_sha256": digest})
    verify_sources(child, rec, "t2001", base, protocol, digest)
    with torch.no_grad():
        next(child.experts["G3-sp"].parameters()).add_(1)
    with pytest.raises(ValueError, match="embedded weights differ"):
        verify_sources(child, rec, "t2001", base, protocol, digest)


def test_measure_refuses_too_small_probe():
    net = MultiTaskNet({"cartpole": (4, 2)})
    with pytest.raises(ValueError, match="at least"):
        measure(net, "cartpole", torch.zeros(128, 4), singles=1)


def test_review_packet_matches_both_frozen_proofs_and_is_unsigned():
    root = Path(__file__).resolve().parents[1]
    packet = json.loads((root / "docs/learn/G3_RECOMBINED_REVIEW.json").read_text())
    assert packet["approval"] == {"reviewed_by": None, "approved_at": None, "signature": None}
    assert len(packet["confirmations"]) == 2
    for proof in packet["confirmations"]:
        for key, digest_key in (("protocol", "protocol_sha256"), ("report", "report_sha256")):
            assert hashlib.sha256((root / proof[key]).read_bytes()).hexdigest() == proof[digest_key]
        assert proof["eligible_for_lineage_review"] is True
